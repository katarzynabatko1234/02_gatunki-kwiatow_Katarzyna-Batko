"""Proof of concept: cały pipeline projektu na małym podzbiorze w kilka minut.

Bierze N pierwszych klas z tego samego podziału 80/10/10 (seed 6102), trenuje ResNet-18
w dwóch wariantach (zamrożone ciało vs odmrożony ostatni blok) i raportuje accuracy,
macro-F1 i top-3 (własny licznik top-k) na teście. Reużywa kodu z src/.

    python test/poc.py [--classes 10] [--epochs 3] [--device auto|cpu]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

import config as C
from data import FlowersDataset, class_names, get_transforms, prepare
from models import build_model, count_params
from train import evaluate, set_train_mode
from utils import get_device, set_seed, sync

OUT = Path(__file__).resolve().parent / "poc_output"


def train_variant(finetune, loaders, n_classes, epochs, device):
    set_seed(C.SEED)
    cfg = C.VARIANTS[f"r18_{finetune}"]
    model, head, body, frozen = build_model("resnet18", finetune, num_classes=n_classes)
    model.to(device)
    groups = [{"params": head, "lr": cfg["lr_head"]}] + ([{"params": body, "lr": cfg["lr_body"]}] if body else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    val_curve, train_time = [], 0.0
    for epoch in range(1, epochs + 1):
        set_train_mode(model, frozen)
        sync(device)
        t0 = time.perf_counter()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sync(device)
        train_time += time.perf_counter() - t0
        val, _, _ = evaluate(model, loaders["val"], device, criterion)
        val_curve.append(val["acc"])
        print(f"  [{finetune}] ep {epoch} | val acc {val['acc']:.3f} | f1 {val['macro_f1']:.3f}", flush=True)
    test, logits, y_true = evaluate(model, loaders["test"], device, criterion)
    return dict(finetune=finetune, trainable_params=count_params(model)[1], train_time_s=round(train_time, 1),
                val_curve=val_curve, **{f"test_{k}": round(v, 4) for k, v in test.items()}), model, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    t_start = time.perf_counter()
    device = get_device() if args.device == "auto" else torch.device(args.device)
    df = prepare()
    names = class_names()[: args.classes]
    sub = df[df.label < args.classes]  # etykiety 0..N-1 bez przemapowania
    print(f"PoC: {args.classes} klas, device={device}, "
          f"train/val/test = {[int((sub.split == s).sum()) for s in ('train', 'val', 'test')]}")

    loaders = {s: DataLoader(FlowersDataset(sub[sub.split == s], train=(s == "train")), batch_size=32,
                             shuffle=(s == "train"), num_workers=2, persistent_workers=True,
                             generator=torch.Generator().manual_seed(C.SEED))
               for s in ("train", "val", "test")}

    results, best = [], None
    for finetune in ("frozen", "last_block"):
        res, model, logits = train_variant(finetune, loaders, args.classes, args.epochs, device)
        results.append(res)
        if best is None or res["test_acc"] >= best[0]["test_acc"]:
            best = (res, model)

    print(f"\n{'wariant':<12}{'param.':>12}{'czas [s]':>10}{'acc':>8}{'macro-F1':>10}{'top-3':>8}")
    for r in results:
        print(f"{r['finetune']:<12}{r['trainable_params']:>12,}{r['train_time_s']:>10}"
              f"{r['test_acc']:>8.3f}{r['test_macro_f1']:>10.3f}{r['test_top3']:>8.3f}")

    OUT.mkdir(exist_ok=True)
    summary = dict(classes=args.classes, class_names=names, epochs=args.epochs, device=str(device),
                   seed=C.SEED, total_time_s=round(time.perf_counter() - t_start, 1), results=results)
    (OUT / "poc_results.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    for r in results:
        ax.plot(range(1, args.epochs + 1), r["val_curve"], marker="o", label=r["finetune"])
    ax.set_xlabel("epoka")
    ax.set_ylabel("val accuracy")
    ax.set_title(f"PoC: ResNet-18 na {args.classes} klasach")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "poc_curves.png", dpi=120)
    plt.close(fig)

    # przykładowe predykcje najlepszego wariantu
    res, model = best
    model.eval()
    tf = get_transforms(train=False)
    sample = sub[sub.split == "test"].sample(8, random_state=C.SEED)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    with torch.no_grad():
        for ax, (_, row) in zip(axes.flat, sample.iterrows()):
            img = Image.open(C.CACHE_DIR / row.file).convert("RGB")
            probs = torch.softmax(model(tf(img).unsqueeze(0).to(device)), 1)[0].cpu()
            p, i = probs.topk(3)
            ok = int(i[0]) == row.label
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"GT: {names[row.label]}\n" + "\n".join(f"{names[j]}: {q:.2f}" for j, q in zip(i, p)),
                         fontsize=8, color="green" if ok else "red")
    fig.suptitle(f"PoC — predykcje top-3 ({res['finetune']})")
    fig.tight_layout()
    fig.savefig(OUT / "poc_predictions.jpg", dpi=80)
    plt.close(fig)
    print(f"\nGotowe w {summary['total_time_s']} s → {OUT}")


if __name__ == "__main__":
    main()
