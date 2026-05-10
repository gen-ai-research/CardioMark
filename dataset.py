"""
CardioMark Dataset
==================
DogHeartDataset: loads lateral canine radiographs and .mat keypoint labels.
Ground truth VHS is computed from annotated keypoints (calc_vhs),
NOT from stored mat["VHS"] — ensuring a unified evaluation protocol
across all geometry-aware models.
"""

import os
import numpy as np
from PIL import Image
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from utils.vhs import calc_vhs


class DogHeartDataset(Dataset):
    """
    Lateral canine radiograph dataset with six cardiac keypoint annotations.

    Directory layout expected::

        root/
          Images/   *.png | *.jpg | *.jpeg
          Labels/   *.mat   (field: six_points, shape (6,2) in pixel coords)

    Args:
        root      : path to split folder (Train / Valid / Test_Images)
        transforms: torchvision transform applied to the PIL image
    """

    def __init__(self, root: str, transforms):
        self.root = root
        self.transforms = transforms

        self.imgs = sorted([
            f for f in os.listdir(os.path.join(root, "Images"))
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ])
        self.labels = sorted([
            f for f in os.listdir(os.path.join(root, "Labels"))
            if f.lower().endswith(".mat")
        ])

        if len(self.imgs) != len(self.labels):
            raise ValueError(
                f"Mismatch: {len(self.imgs)} images vs "
                f"{len(self.labels)} labels in {root}"
            )

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img_path    = os.path.join(self.root, "Images", self.imgs[idx])
        points_path = os.path.join(self.root, "Labels", self.labels[idx])

        # Load image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img = self.transforms(img)

        # Load and normalise keypoints
        mat = loadmat(points_path)
        pts = torch.tensor(mat["six_points"].astype(np.float32))
        pts[:, 0] /= float(w)   # normalise x by original width
        pts[:, 1] /= float(h)   # normalise y by original height
        pts = pts.clamp(0.0, 1.0).reshape(-1)   # (12,)

        # GT VHS computed from keypoints — unified protocol
        gt_vhs = calc_vhs(pts.unsqueeze(0)).reshape(1, 1)

        return idx, img, pts, gt_vhs


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transform(image_size: int = 512) -> T.Compose:
    """Geometry-safe augmentation: no horizontal flip (keypoints not mirrored)."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform(image_size: int = 512) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
