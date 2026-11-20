"""Zbiorcza ewaluacja: tabela wyników, krzywe, jakość vs czas, confusion matrix, analiza błędów."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

import config as C
from data import class_names, prepare
from metrics import confusion

THRESHOLD = 0.84
ORDER = ["r18_frozen", "r18_last_block", "r18_full", "mnv2_frozen", "mnv2_full"]
LABELS = {"r18_frozen": "ResNet-18 zamrożony (baseline)", "r18_last_block": "ResNet-18 ostatni blok",
          "r18_full": "ResNet-18 pełny FT", "mnv2_frozen": "MobileNetV2 zamrożony",
          "mnv2_full": "MobileNetV2 pełny FT"}


def load_runs():
    runs = [r for r in ORDER if (C.RESULTS_DIR / f"{r}.json").exists()]
    summ = pd.DataFrame([json.loads((C.RESULTS_DIR / f"{r}.json").read_text()) for r in runs])
    hist = {r: pd.read_csv(C.RESULTS_DIR / f"{r}_history.csv") for r in runs}
    preds = {r: pd.read_csv(C.RESULTS_DIR / f"{r}_test_preds.csv") for r in runs}
    return runs, summ, hist, preds


def cpu_latency(arch, n=30):
    """Latencja inferencji na CPU (bs=1) — stabilniejsza niż pomiar na MPS; zależy tylko od architektury."""
    import time
    import torch
    from models import build_model
    torch.set_num_threads(4)
    m, *_ = build_model(arch, "full", pretrained=False)
    m.eval()
    x = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    with torch.no_grad():
        for _ in range(10):
            m(x)
        t = time.perf_counter()
        for _ in range(n):
            m(x)
    return (time.perf_counter() - t) / n * 1000


def summary_table(summ):
    lat = {a: cpu_latency(a) for a in summ.arch.unique()}
    t = pd.DataFrame({
        "wariant": summ.run.map(LABELS), "arch": summ.arch, "fine-tuning": summ.finetune,
        "trenowane param.": summ.trainable_params.map(lambda v: f"{v/1e6:.2f} M"),
        "wszystkie param.": summ.total_params.map(lambda v: f"{v/1e6:.2f} M"),
        "epoki": summ.epochs,
        "czas treningu [s]": summ.train_time_s.round(0).astype(int),
        "czas/epokę [s]": summ.mean_epoch_time_s.round(1),
        "latencja CPU bs=1 [ms]": summ.arch.map(lat).round(1),
        "test accuracy": summ.test_acc.round(4), "test macro-F1": summ.test_macro_f1.round(4),
        "test top-3": summ.test_top3.round(4), "test top-5": summ.test_top5.round(4),
        "próg 0.84": np.where(summ.test_acc >= THRESHOLD, "✅", "❌"),
    })
    t.to_csv(C.RESULTS_DIR / "summary.csv", index=False)
    (C.RESULTS_DIR / "summary.md").write_text(t.to_markdown(index=False))
    print(t.to_markdown(index=False))


def plot_curves(runs, hist):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for r in runs:
        h = hist[r]
        axes[0].plot(h.epoch, h.train_loss, "--", alpha=0.6)
        axes[0].plot(h.epoch, h.val_loss, label=LABELS[r], color=axes[0].lines[-1].get_color())
        axes[1].plot(h.epoch, h.val_acc, marker="o", ms=3, label=LABELS[r])
        axes[2].plot(h.epoch, h.val_macro_f1, marker="o", ms=3, label=LABELS[r])
    axes[0].set_title("Loss (przerywana: train, ciągła: val)")
    axes[1].set_title("Accuracy na walidacji")
    axes[2].set_title("Macro-F1 na walidacji")
    for ax in axes:
        ax.set_xlabel("epoka")
        ax.grid(alpha=0.3)
    axes[1].axhline(THRESHOLD, color="red", ls=":", lw=1)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "training_curves.png", dpi=130)
    plt.close(fig)

    # train vs val accuracy per run (overfitting)
    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.5), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        h = hist[r]
        ax.plot(h.epoch, h.train_acc, label="train")
        ax.plot(h.epoch, h.val_acc, label="val")
        ax.set_title(LABELS[r], fontsize=9)
        ax.set_xlabel("epoka")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend()
    np.atleast_1d(axes)[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "train_vs_val_accuracy.png", dpi=130)
    plt.close(fig)


def plot_quality_vs_time(summ):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for _, s in summ.iterrows():
        m = "o" if s.arch == "resnet18" else "s"
        for ax, x in zip(axes, (s.train_time_s / 60, s.trainable_params / 1e6)):
            ax.scatter(x, s.test_acc, s=80, marker=m)
            off = (-115, 6) if s.run == "mnv2_full" and ax is axes[0] else (6, -10)
            ax.annotate(LABELS[s.run], (x, s.test_acc), textcoords="offset points", xytext=off, fontsize=8)
    for ax, xl in zip(axes, ("łączny czas treningu [min]", "liczba trenowanych parametrów [M]")):
        ax.axhline(THRESHOLD, color="red", ls=":", lw=1, label="próg 0.84")
        ax.set_xlabel(xl)
        ax.set_ylabel("test accuracy")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Głębokość fine-tuningu i architektura: jakość vs koszt (● ResNet-18, ■ MobileNetV2)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "quality_vs_cost.png", dpi=130)
    plt.close(fig)

    # słupki metryk
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.25
    x = np.arange(len(summ))
    for i, (col, lab) in enumerate((("test_acc", "accuracy"), ("test_macro_f1", "macro-F1"), ("test_top3", "top-3"))):
        b = ax.bar(x + (i - 1) * w, summ[col], w, label=lab)
        ax.bar_label(b, fmt="%.3f", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(summ.run.map(LABELS), fontsize=8)
    ax.axhline(THRESHOLD, color="red", ls=":", lw=1)
    ax.set_ylim(0.5, 1.02)
    ax.legend()
    ax.set_title("Metryki na zbiorze testowym (819 zdjęć)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "test_metrics.png", dpi=130)
    plt.close(fig)


def plot_confusion(p, names, run, tag):
    cm = confusion(p.label, p.pred, C.NUM_CLASSES)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(C.NUM_CLASSES))
    ax.set_yticks(range(C.NUM_CLASSES))
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("predykcja")
    ax.set_ylabel("prawdziwa klasa")
    ax.set_title(f"Confusion matrix (znormalizowana wierszami) — {LABELS[run]}, test")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"confusion_matrix_{tag}.png", dpi=110)
    plt.close(fig)
    return cm


def confused_pairs(cm, names, k=15):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    sym = off + off.T  # para nieuporządkowana
    iu = np.triu_indices(len(cm), 1)
    order = np.argsort(sym[iu])[::-1][:k]
    pairs = []
    for o in order:
        a, b = iu[0][o], iu[1][o]
        if sym[a, b] == 0:
            break
        pairs.append(dict(a=int(a), b=int(b), name_a=names[a], name_b=names[b],
                          a_as_b=int(off[a, b]), b_as_a=int(off[b, a]), total=int(sym[a, b])))
    return pairs


def plot_pair_examples(df, pairs, names, n_pairs=6):
    fig, axes = plt.subplots(n_pairs, 4, figsize=(10, 2.6 * n_pairs))
    rng = np.random.default_rng(C.SEED)
    for r, pr in enumerate(pairs[:n_pairs]):
        for j, c in enumerate((pr["a"], pr["a"], pr["b"], pr["b"])):
            f = rng.choice(df[(df.label == c) & (df.split == "train")].file.values)
            axes[r, j].imshow(Image.open(C.CACHE_DIR / f))
            axes[r, j].set_title(names[c], fontsize=8, color="C0" if j < 2 else "C3")
            axes[r, j].axis("off")
    fig.suptitle("Najczęściej mylone pary gatunków (przykłady treningowe obu klas)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "confused_pairs_examples.jpg", dpi=90)
    plt.close(fig)


def plot_predictions(p, names, wrong: bool, fname, title, n=24):
    sel = p[(p.label != p.pred) if wrong else (p.label == p.pred)]
    sel = sel.sample(min(n, len(sel)), random_state=C.SEED)
    cols = 6
    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.3 * rows))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, (_, s) in zip(np.atleast_1d(axes).flat, sel.iterrows()):
        im = Image.open(C.CACHE_DIR / s.file)
        ax.imshow(im)
        top3 = [names[int(i)] for i in s.top5.split()[:3]]
        color = "red" if wrong else "green"
        ax.set_title(f"GT: {names[s.label]}\nPRED: {names[s.pred]} ({s.conf:.2f})\ntop3: {', '.join(top3[1:])}",
                     fontsize=7, color=color)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / fname, dpi=80)
    plt.close(fig)


def per_class_analysis(df, preds, runs, names, best):
    train_counts = df[df.split == "train"].label.value_counts().sort_index()
    out = pd.DataFrame({"class": names, "train_count": train_counts.values})
    for r in runs:
        p = preds[r]
        out[f"f1_{r}"] = f1_score(p.label, p.pred, average=None, labels=list(range(C.NUM_CLASSES)))
        out[f"recall_{r}"] = [((p.label == c) & (p.pred == c)).sum() / max((p.label == c).sum(), 1)
                              for c in range(C.NUM_CLASSES)]
    out.to_csv(C.RESULTS_DIR / "per_class_metrics.csv", index=False)

    corr = {}
    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.8), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        rho, pval = spearmanr(out.train_count, out[f"f1_{r}"])
        corr[r] = dict(spearman_rho=round(float(rho), 3), p_value=float(f"{pval:.3g}"))
        ax.scatter(out.train_count, out[f"f1_{r}"], s=12, alpha=0.7)
        ax.set_xscale("log")
        ax.set_title(f"{LABELS[r]}\nSpearman ρ={rho:.2f} (p={pval:.2g})", fontsize=8)
        ax.set_xlabel("liczba zdjęć treningowych klasy")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("F1 klasy (test)")
    fig.suptitle("Czy błędy wynikają z małej liczby przykładów? F1 klasy vs liczność")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "per_class_f1_vs_train_count.png", dpi=130)
    plt.close(fig)

    # najsłabsze klasy najlepszego modelu
    worst = out.sort_values(f"f1_{best}").head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(worst["class"] + " (n=" + worst.train_count.astype(str) + ")", worst[f"f1_{best}"], color="#C44E52")
    ax.invert_yaxis()
    ax.set_xlabel("F1 (test)")
    ax.set_title(f"15 najsłabszych klas — {LABELS[best]}")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "worst_classes.png", dpi=130)
    plt.close(fig)

    # porównanie liczności: klasy z błędami vs bez błędów (najlepszy model)
    err = out[f"recall_{best}"] < 1
    groups = dict(
        n_classes_with_errors=int(err.sum()),
        median_train_count_classes_with_errors=float(out[err].train_count.median()) if err.any() else None,
        median_train_count_classes_perfect=float(out[~err].train_count.median()),
    )
    return out, corr, groups


def main():
    df = prepare()
    names = class_names()
    runs, summ, hist, preds = load_runs()
    summary_table(summ)
    plot_curves(runs, hist)
    plot_quality_vs_time(summ)

    best = summ.loc[summ.test_acc.idxmax(), "run"]
    baseline = "r18_frozen"
    cm_best = plot_confusion(preds[best], names, best, "best")
    if baseline in preds:
        cm_base = plot_confusion(preds[baseline], names, baseline, "baseline")
    pairs_best = confused_pairs(cm_best, names)
    pairs_base = confused_pairs(cm_base, names) if baseline in preds else []
    plot_pair_examples(df, pairs_best if len(pairs_best) >= 6 else pairs_base, names)
    plot_predictions(preds[best], names, True, "errors_best_model.jpg",
                     f"Błędne predykcje — {LABELS[best]} (czerwone = pomyłka)")
    plot_predictions(preds[best], names, False, "correct_predictions_best_model.jpg",
                     f"Przykładowe poprawne predykcje — {LABELS[best]}", n=18)
    if baseline in preds:
        plot_predictions(preds[baseline], names, True, "errors_baseline.jpg",
                         f"Błędne predykcje — {LABELS[baseline]}")
    _, corr, groups = per_class_analysis(df, preds, runs, names, best)

    # czy model w ogóle się waha? błędne vs poprawne: pewność, trafienie w top-3
    p = preds[best]
    wrong = p[p.label != p.pred]
    in_top3 = np.mean([str(l) in s.split()[:3] for l, s in zip(wrong.label, wrong.top5)]) if len(wrong) else 0
    analysis = dict(
        best_run=best, n_test=len(p), n_errors_best=int(len(wrong)),
        errors_with_true_in_top3=round(float(in_top3), 3),
        mean_conf_correct=round(float(p[p.label == p.pred].conf.mean()), 3),
        mean_conf_wrong=round(float(wrong.conf.mean()), 3) if len(wrong) else None,
        spearman_f1_vs_train_count=corr, class_groups_best=groups,
        top_confused_pairs_best=pairs_best, top_confused_pairs_baseline=pairs_base,
    )
    (C.RESULTS_DIR / "error_analysis.json").write_text(json.dumps(analysis, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in analysis.items() if "pairs" not in k}, indent=1, ensure_ascii=False))
    print("Pary (best):", [(q["name_a"], q["name_b"], q["total"]) for q in pairs_best[:10]])
    print("Pary (baseline):", [(q["name_a"], q["name_b"], q["total"]) for q in pairs_base[:10]])


if __name__ == "__main__":
    main()
