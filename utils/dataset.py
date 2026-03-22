"""
utils/dataset.py

Handles loading of HDF5 event datasets for both unsupervised pretraining
and supervised fine-tuning stages of the pipeline.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def preprocess(arr: np.ndarray) -> torch.Tensor:
    """
    Normalise a raw numpy array and convert to (C, H, W) float32 tensor.

    Handles three common layouts coming out of HDF5 files:
      - (H, W)      → treated as single-channel, expanded to (1, H, W)
      - (H, W, C)   → transposed to (C, H, W)
      - (C, H, W)   → used as-is
    """
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis]                        # (H,W) → (1,H,W)
    elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        arr = arr.transpose(2, 0, 1)                 # (H,W,C) → (C,H,W)
    # per-sample z-score normalisation
    mu    = arr.mean()
    sigma = arr.std() + 1e-8
    return torch.tensor((arr - mu) / sigma)


class UnlabelledDataset(Dataset):
    """
    Streams unlabelled event frames from an HDF5 file.
    Used in Step 1 (autoencoder pretraining) — no labels required.

    Args:
        path     : path to the HDF5 file
        data_key : name of the dataset inside the file (e.g. 'X', 'data')
    """

    def __init__(self, path: str, data_key: str = "X"):
        self.path     = path
        self.data_key = data_key
        with h5py.File(path, "r") as f:
            self.length = f[data_key].shape[0]
        print(f"[UnlabelledDataset]  {self.length:,} samples  |  key='{data_key}'")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.path, "r") as f:
            arr = np.array(f[self.data_key][idx])
        return preprocess(arr)


class LabelledDataset(Dataset):
    """
    Streams labelled event frames from an HDF5 file.
    Used in Step 2 (classifier fine-tuning).

    Args:
        path      : path to the HDF5 file
        data_key  : name of the feature dataset  (e.g. 'X', 'data')
        label_key : name of the label dataset    (e.g. 'y', 'labels')
    """

    def __init__(self, path: str, data_key: str = "X", label_key: str = "y"):
        self.path      = path
        self.data_key  = data_key
        self.label_key = label_key
        with h5py.File(path, "r") as f:
            self.length = f[data_key].shape[0]
            labels      = np.array(f[label_key])
        unique, counts = np.unique(labels, return_counts=True)
        print(f"[LabelledDataset]    {self.length:,} samples  |  "
              f"class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        with h5py.File(self.path, "r") as f:
            arr   = np.array(f[self.data_key][idx])
            label = float(f[self.label_key][idx])
        return preprocess(arr), torch.tensor(label)


def make_loaders(
    dataset,
    val_fraction: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 2,
):
    """
    Split a dataset into train/val DataLoaders.

    Returns:
        (train_loader, val_loader)
    """
    n_val   = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
