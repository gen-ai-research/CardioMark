"""
Sampling utilities
==================
WeightedRandomSampler builder that reads only .mat labels (no image I/O)
and corrects the ~12:1 Enlarged:Normal class imbalance in CardioMark.
"""

import logging
from collections import Counter

import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from tqdm import tqdm

from utils.vhs import calc_vhs, get_labels

logger = logging.getLogger(__name__)


def build_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Builds a class-balanced sampler by reading .mat labels only (fast).

    Args:
        dataset: DogHeartDataset instance

    Returns:
        WeightedRandomSampler with replacement
    """
    labels = []
    for mat_name in dataset.labels:
        mat = loadmat(os.path.join(dataset.root, "Labels", mat_name))
        pts = torch.tensor(mat["six_points"].astype(np.float32)).reshape(-1)
        vhs = calc_vhs(pts.unsqueeze(0)).view(-1)
        labels.append(int(get_labels(vhs).item()))

    counts = Counter(labels)
    logger.info(f"Sampler class counts: {dict(counts)}")

    cw = {c: 1.0 / n for c, n in counts.items()}
    sw = torch.tensor([cw[l] for l in labels], dtype=torch.double)
    return WeightedRandomSampler(sw, len(sw), replacement=True)


def compute_class_weights(
    loader: DataLoader,
    device: torch.device,
    save_path: str,
) -> torch.Tensor:
    """
    Computes inverse-frequency class weights and caches them to disk.

    Args:
        loader   : DataLoader over the training split
        device   : target device
        save_path: path to save/load the cached weight tensor

    Returns:
        Normalised weight tensor of shape (3,) on ``device``
    """
    import os
    if os.path.exists(save_path):
        logger.info(f"Loading class weights from {save_path}")
        return torch.load(save_path, map_location=device)

    counts = Counter()
    for _, _, _, vhs in tqdm(loader, desc="Computing class weights"):
        lbs = get_labels(vhs.squeeze()).cpu().tolist()
        if isinstance(lbs, int):
            lbs = [lbs]
        counts.update(lbs)

    c = torch.tensor(
        [counts.get(i, 0) for i in range(3)], dtype=torch.float32
    ).clamp_min(1.0)
    w = (1.0 / c) / (1.0 / c).sum()
    torch.save(w, save_path)
    logger.info(f"Class weights saved → {save_path}: {w.tolist()}")
    return w.to(device)


def summarize_label_distribution(loader: DataLoader, split_name: str) -> Counter:
    """Logs class counts and ratios for a data split."""
    counts = Counter()
    for _, _, _, vhs in loader:
        lbs = get_labels(vhs.squeeze()).cpu().tolist()
        if isinstance(lbs, int):
            lbs = [lbs]
        counts.update(lbs)
    total = sum(counts.values())
    ratios = {k: round(v / total, 4) for k, v in counts.items()}
    logger.info(f"{split_name}: {dict(counts)}  ratios: {ratios}")
    print(f"{split_name}: {dict(counts)}")
    return counts


# fix missing import
import os
