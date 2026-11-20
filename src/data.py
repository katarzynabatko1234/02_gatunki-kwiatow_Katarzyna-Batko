"""Dane: pobranie (Kaggle), etykiety (oficjalny imagelabels.mat), podział 80/10/10, Dataset.

Kaggle-owa wersja zbioru ma obrazy testowe bez etykiet, dlatego etykiety dla
wszystkich 8 189 zdjęć bierzemy z oficjalnego pliku Oxford VGG (zweryfikowane:
zgodne z folderami train/valid w 100%). Oryginalny podział jest ignorowany —
robimy własny stratyfikowany 80/10/10 z ziarnem 6102.
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import config as C

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def download_raw() -> Path:
    import kagglehub
    root = Path(kagglehub.dataset_download(C.KAGGLE_ID)) / "flower_data"
    if not C.LABELS_MAT.exists():
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(C.LABELS_URL, C.LABELS_MAT)
    return root


def _build_cache(root: Path) -> pd.DataFrame:
    """Przeskalowanie wszystkich obrazów do krótszego boku 256 px (szybsze epoki)."""
    labels = scipy.io.loadmat(C.LABELS_MAT)["labels"][0]  # 1..102, indeks = nr obrazu - 1
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in tqdm(sorted(root.rglob("image_*.jpg")), desc="cache", mininterval=5):
        idx = int(f.stem.split("_")[1])
        dst = C.CACHE_DIR / f.name
        with Image.open(f) as im:
            w, h = im.size
            if not dst.exists():
                im = im.convert("RGB")
                s = C.RESIZE_SHORT / min(w, h)
                im.resize((round(w * s), round(h * s)), Image.BICUBIC).save(dst, quality=95)
        rows.append(dict(file=f.name, label=int(labels[idx - 1]) - 1, orig_w=w, orig_h=h))
    return pd.DataFrame(rows)


def prepare() -> pd.DataFrame:
    """Zwraca DataFrame [file, label, orig_w, orig_h, split]; tworzy plik splitu, jeśli go brak."""
    if C.SPLIT_CSV.exists() and C.CACHE_DIR.exists():
        return pd.read_csv(C.SPLIT_CSV)
    root = download_raw()
    df = _build_cache(root)
    assert len(df) == 8189 and df.label.nunique() == C.NUM_CLASSES
    _, val_frac, test_frac = C.SPLIT
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=val_frac + test_frac,
                                stratify=df.label, random_state=C.SEED)
    va, te = train_test_split(rest, test_size=test_frac / (val_frac + test_frac),
                              stratify=df.label.iloc[rest], random_state=C.SEED)
    df["split"] = ""
    df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"
    C.SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.SPLIT_CSV, index=False)
    names = json.loads((root / "cat_to_name.json").read_text())
    C.CLASS_NAMES_JSON.write_text(json.dumps(
        [names[str(i + 1)] for i in range(C.NUM_CLASSES)], indent=1))
    return df


def class_names() -> list:
    return json.loads(C.CLASS_NAMES_JSON.read_text())


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(C.IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.CenterCrop(C.IMG_SIZE),  # obrazy już mają krótszy bok 256
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class FlowersDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.files = df.file.tolist()
        self.labels = df.label.tolist()
        self.tf = get_transforms(train)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(C.CACHE_DIR / self.files[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.labels[i]


def get_loaders(df: pd.DataFrame, batch_size=C.BATCH_SIZE, subset=None):
    g = torch.Generator().manual_seed(C.SEED)
    loaders = {}
    for split in ("train", "val", "test"):
        part = df[df.split == split]
        if subset:
            part = part.iloc[:subset]
        ds = FlowersDataset(part, train=(split == "train"))
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=C.NUM_WORKERS, generator=g,
                                    persistent_workers=C.NUM_WORKERS > 0)
    return loaders


if __name__ == "__main__":
    df = prepare()
    print(df.split.value_counts())
    s = {k: set(df[df.split == k].file) for k in ("train", "val", "test")}
    assert not (s["train"] & s["val"] or s["train"] & s["test"] or s["val"] & s["test"])
    assert all(df[df.split == k].label.nunique() == C.NUM_CLASSES for k in s)
    print("Split OK: rozłączne części, wszystkie klasy w każdej części")
