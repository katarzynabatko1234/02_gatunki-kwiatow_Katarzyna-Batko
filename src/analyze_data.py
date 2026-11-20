"""Analiza danych (EDA): liczebności, rozkład klas, rozmiary obrazów, siatka przykładów."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config as C
from data import class_names, prepare


def main():
    df = prepare()
    names = class_names()
    C.FIG_DIR.mkdir(exist_ok=True)
    counts = df.label.value_counts().sort_index()
    stats = dict(
        n_images=len(df), n_classes=int(df.label.nunique()),
        split_sizes=df.split.value_counts().to_dict(),
        per_class_min=int(counts.min()), per_class_max=int(counts.max()),
        per_class_mean=round(float(counts.mean()), 1), per_class_median=float(counts.median()),
        imbalance_ratio=round(float(counts.max() / counts.min()), 2),
        smallest_classes={names[i]: int(c) for i, c in counts.nsmallest(5).items()},
        largest_classes={names[i]: int(c) for i, c in counts.nlargest(5).items()},
        width=df.orig_w.describe().round(1).to_dict(), height=df.orig_h.describe().round(1).to_dict(),
        aspect_ratio_mean=round(float((df.orig_w / df.orig_h).mean()), 3),
    )
    (C.RESULTS_DIR / "data_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))

    # rozkład klas (posortowany)
    s = counts.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(range(len(s)), s.values, color="#4C72B0")
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([names[i] for i in s.index], rotation=90, fontsize=6)
    ax.axhline(s.mean(), color="red", ls="--", lw=1, label=f"średnia = {s.mean():.0f}")
    ax.set_ylabel("liczba zdjęć")
    ax.set_title("Rozkład klas — Oxford Flowers 102 (8 189 zdjęć)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_class_distribution.png", dpi=130)
    plt.close(fig)

    # liczebność klas w poszczególnych splitach
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, col in zip(("train", "val", "test"), ("#4C72B0", "#DD8452", "#55A868")):
        ax.hist(df[df.split == split].label.value_counts(), bins=30, alpha=0.7, label=split, color=col)
    ax.set_xlabel("liczba zdjęć w klasie")
    ax.set_ylabel("liczba klas")
    ax.set_title("Liczebność klas w train / val / test (podział stratyfikowany)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_split_class_counts.png", dpi=130)
    plt.close(fig)

    # rozmiary obrazów
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(df.orig_w, df.orig_h, s=3, alpha=0.3)
    axes[0].set_xlabel("szerokość [px]")
    axes[0].set_ylabel("wysokość [px]")
    axes[0].set_title("Rozmiary oryginalnych obrazów")
    axes[1].hist(df.orig_w / df.orig_h, bins=50, color="#8172B2")
    axes[1].set_xlabel("proporcje (szer./wys.)")
    axes[1].set_title("Rozkład proporcji obrazów")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_image_sizes.png", dpi=130)
    plt.close(fig)

    # siatka: jeden przykład z każdej klasy
    rng = np.random.default_rng(C.SEED)
    fig, axes = plt.subplots(9, 12, figsize=(24, 19))
    for ax in axes.flat:
        ax.axis("off")
    for c, ax in zip(range(C.NUM_CLASSES), axes.flat):
        f = rng.choice(df[df.label == c].file.values)
        ax.imshow(Image.open(C.CACHE_DIR / f))
        ax.set_title(f"{c}: {names[c]}", fontsize=8)
    fig.suptitle("Po jednym przykładzie z każdej ze 102 klas", fontsize=16)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_class_grid.jpg", dpi=70)
    plt.close(fig)

    # zmienność wewnątrzklasowa: 8 zdjęć tej samej klasy dla 4 klas
    pick = [names.index(n) for n in ("sword lily", "petunia", "pink primrose", "water lily")]
    fig, axes = plt.subplots(len(pick), 8, figsize=(16, 2.2 * len(pick)))
    for r, c in enumerate(pick):
        files = rng.choice(df[df.label == c].file.values, 8, replace=False)
        for j, f in enumerate(files):
            axes[r, j].imshow(Image.open(C.CACHE_DIR / f))
            axes[r, j].axis("off")
        axes[r, 0].set_title(names[c], fontsize=10, loc="left")
    fig.suptitle("Zmienność wewnątrz klasy: tło, kadr, faza rozkwitu, oświetlenie")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_intra_class_variation.jpg", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
