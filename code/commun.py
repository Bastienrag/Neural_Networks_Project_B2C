"""Briques partagées par les notebooks : chargement des œuvres, découpage, normalisation."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

SEED = 0
DATA_DIR = Path(__file__).resolve().parent / "data" / "wikiart_style5"
UNKNOWN_ARTIST = "Unknown Artist"


def set_seed(seed: int = SEED) -> None:
    """Fixe les graines de Python, NumPy et PyTorch pour reproduire un run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_metadata() -> tuple[pd.DataFrame, list[str]]:
    """Lit metadata.csv et la liste des styles, dans l'ordre des label_idx."""
    if next(DATA_DIR.glob("*/*.jpg"), None) is None:
        raise FileNotFoundError(
            f"Aucune image dans {DATA_DIR} : générez le sous-ensemble avec la commande du README."
        )
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    classes = json.loads((DATA_DIR / "subset_info.json").read_text())["classes"]
    assert list(meta.groupby("label_idx")["label"].first()) == classes  # ordre d'ImageFolder
    return meta, classes


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def load_images(meta: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Charge toutes les œuvres : X en uint8 (N, C, H, W), y les indices de style."""
    X = torch.from_numpy(np.stack([read_rgb(DATA_DIR / p) for p in meta["path"]]))
    X = X.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)
    y = torch.tensor(meta["label_idx"].to_numpy())
    return X, y


def stratified_group_split(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    """Met de côté un pli (environ 1/n_splits des œuvres), stratifié par style, sans couper de groupe."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(sgkf.split(np.zeros(len(y)), y, groups))


def split_indices(meta: pd.DataFrame, seed: int = SEED) -> dict[str, np.ndarray]:
    """Découpage 60 / 20 / 20 stratifié par style, sans artiste nommé partagé entre deux splits.

    Chaque œuvre d'artiste inconnu forme son propre groupe.
    """
    unknown = meta["artist"] == UNKNOWN_ARTIST
    groups = np.where(unknown, "inconnu_" + meta["id"], meta["artist"])
    y = meta["label_idx"].to_numpy()

    rest, test_idx = stratified_group_split(y, groups, n_splits=5, seed=seed)
    train_pos, val_pos = stratified_group_split(y[rest], groups[rest], n_splits=4, seed=seed)
    splits = {"train": rest[train_pos], "val": rest[val_pos], "test": test_idx}

    # Contrôles : partition complète, aucun artiste nommé dans deux splits.
    assert sorted(np.concatenate(list(splits.values()))) == list(range(len(meta)))
    named_artists = {k: set(meta.loc[i, "artist"][~unknown[i]]) for k, i in splits.items()}
    assert not (named_artists["train"] & named_artists["val"]
                or named_artists["train"] & named_artists["test"]
                or named_artists["val"] & named_artists["test"])
    return splits


def normalization_stats(X: torch.Tensor, train_idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Moyenne et écart-type par canal, calculés sur le train seul, pixels ramenés dans [0, 1]."""
    X_train = X[train_idx].float() / 255
    return X_train.mean(dim=(0, 2, 3)), X_train.std(dim=(0, 2, 3))
