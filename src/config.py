"""Stałe projektu: ziarno, ścieżki, hiperparametry wariantów."""
from pathlib import Path

SEED = 6102
NUM_CLASSES = 102
SPLIT = (0.8, 0.1, 0.1)  # train / val / test
IMG_SIZE = 224
RESIZE_SHORT = 256  # obrazy są wstępnie przeskalowane do krótszego boku 256 px (cache)
BATCH_SIZE = 64
NUM_WORKERS = 4

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "resized256"
LABELS_MAT = DATA_DIR / "imagelabels.mat"
LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
KAGGLE_ID = "waseemalastal/the-oxford-flowers-102-dataset"
SPLIT_CSV = ROOT / "splits" / f"split_seed{SEED}.csv"
CLASS_NAMES_JSON = ROOT / "splits" / "class_names.json"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"

WANDB_PROJECT = "oxford-flowers102-finetuning"

# Warianty: arch, głębokość fine-tuningu, lr głowy, lr ciała, epoki
VARIANTS = {
    "r18_frozen":     dict(arch="resnet18",     finetune="frozen",     lr_head=1e-3, lr_body=0.0,  epochs=10),
    "r18_last_block": dict(arch="resnet18",     finetune="last_block", lr_head=1e-3, lr_body=3e-4, epochs=10),
    "r18_full":       dict(arch="resnet18",     finetune="full",       lr_head=1e-3, lr_body=1e-4, epochs=12),
    "mnv2_frozen":    dict(arch="mobilenet_v2", finetune="frozen",     lr_head=1e-3, lr_body=0.0,  epochs=10),
    "mnv2_full":      dict(arch="mobilenet_v2", finetune="full",       lr_head=1e-3, lr_body=1e-4, epochs=12),
}
