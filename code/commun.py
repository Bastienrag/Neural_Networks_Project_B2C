"""Briques partagées par les notebooks : chargement des œuvres, découpage, normalisation,
réseau dense et boucle d'entraînement."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 0
DATA_DIR = Path(__file__).resolve().parent / "data" / "wikiart_style5"
MODELS_DIR = Path(__file__).resolve().parent / "modeles"  # non versionné
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


def standardize(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Pixels uint8 (N, C, H, W) ramenés dans [0, 1], puis centrés-réduits canal par canal."""
    return (X.float() / 255 - mean[:, None, None]) / std[:, None, None]


class ArtworkDataset(Dataset):
    """Œuvres d'un split : image standardisée (C, H, W) et indice de style."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, idx: np.ndarray,
                 mean: torch.Tensor, std: torch.Tensor):
        self.X = standardize(X[idx], mean, std)
        self.y = y[idx]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


class DenseNet(nn.Module):
    """Réseau feed-forward sur les pixels aplatis : pour chaque couche cachée, Linear, batch norm
    (optionnelle), activation et dropout (optionnel), puis une couche Linear de sortie qui renvoie
    les logits. Les options désactivées n'ajoutent aucune couche."""

    def __init__(self, n_in: int, hidden: list[int], n_out: int,
                 activation: type[nn.Module] = nn.Sigmoid, dropout: float = 0.0,
                 batch_norm: bool = False, he_init: bool = False):
        super().__init__()
        self.config = {"n_in": n_in, "hidden": list(hidden), "n_out": n_out,
                       "activation": activation.__name__, "dropout": dropout, "batch_norm": batch_norm}
        layers, width = [nn.Flatten()], n_in
        for h in hidden:
            layers.append(nn.Linear(width, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = h
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)
        if he_init:  # variance 2 / n_in, adaptée à ReLU
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_dense(model: DenseNet, path: Path) -> None:
    """Sauvegarde les poids et la configuration, pour reconstruire le réseau sans la connaître."""
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_dense(path: Path) -> DenseNet:
    """Reconstruit un réseau sauvegardé par save_dense, en mode évaluation."""
    saved = torch.load(path)
    config = saved["config"] | {"activation": getattr(nn, saved["config"]["activation"])}
    model = DenseNet(**config)
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    """Une passe sur le train : forward, perte, zero_grad, backward, step.
    Renvoie la perte et l'accuracy moyennes de l'époque."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
             device: torch.device) -> tuple[float, float]:
    """Perte et accuracy moyennes, en mode évaluation et sans gradient."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Indices de style prédits pour tout un loader, dans l'ordre du loader."""
    model.eval()
    return torch.cat([model(xb.to(device)).argmax(1).cpu() for xb, _ in loader])


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer, n_epochs: int, device: torch.device,
        patience: int | None = None) -> dict[str, list[float]]:
    """Entraîne n_epochs époques et renvoie l'historique par époque (perte et accuracy,
    train et validation), pour tracer les courbes de suivi.

    Avec patience, arrêt précoce : l'entraînement s'arrête après patience époques sans baisse de
    la perte de validation, et le modèle reprend les poids de sa meilleure époque
    (history["best_epoch"], numérotée à partir de 1)."""
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        if patience is None:
            continue
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    if patience is not None:
        model.load_state_dict(best_state)
        history["best_epoch"] = best_epoch
    return history
