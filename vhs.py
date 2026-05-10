"""
VHS geometry utilities
======================
All VHS computation, label assignment, and keypoint geometry helpers.
These functions are the single source of truth used by the dataset,
loss functions, evaluation, and visualisation modules.
"""

import torch
import torch.nn.functional as F


# ── VHS computation ───────────────────────────────────────────────────────────

def calc_vhs(points12: torch.Tensor) -> torch.Tensor:
    """
    Computes Vertebral Heart Score from 6 normalised keypoints.

    Formula: VHS = 6 * (|AB| + |CD|) / |EF|

    Points layout (normalised [0,1] coordinates):
        A=0:2  B=2:4   long cardiac axis endpoints
        C=4:6  D=6:8   short cardiac axis endpoints
        E=8:10 F=10:12 vertebral reference endpoints

    Args:
        points12: (..., 12) tensor of normalised coordinates

    Returns:
        (...,) VHS scalar tensor
    """
    A  = points12[..., 0:2];  B  = points12[..., 2:4]
    C  = points12[..., 4:6];  D  = points12[..., 6:8]
    E  = points12[..., 8:10]; Fp = points12[..., 10:12]

    AB = torch.norm(A - B,  p=2, dim=-1)
    CD = torch.norm(C - D,  p=2, dim=-1)
    EF = torch.norm(E - Fp, p=2, dim=-1).clamp_min(1e-6)

    return 6.0 * (AB + CD) / EF


def get_labels(vhs: torch.Tensor) -> torch.Tensor:
    """
    Maps VHS scalar(s) to clinical class indices.

    Classes:
        0 — Normal        (VHS < 8.2)
        1 — Borderline    (8.2 ≤ VHS < 10.0)
        2 — Enlarged      (VHS ≥ 10.0)

    Args:
        vhs: scalar or batch tensor

    Returns:
        Long tensor of same leading shape
    """
    return ((vhs >= 10).long() - (vhs < 8.2).long() + 1).squeeze()


# ── Anchor / vector decomposition ────────────────────────────────────────────

def points12_to_geom(points12: torch.Tensor):
    """
    Decomposes (B,12) points into anchors (A,C,E) and vectors (AB,CD,EF).

    Returns:
        anchors : (B, 6)  — start points of each segment
        vectors : (B, 6)  — displacement vectors of each segment
    """
    A  = points12[:, 0:2];  B  = points12[:, 2:4]
    C  = points12[:, 4:6];  D  = points12[:, 6:8]
    E  = points12[:, 8:10]; Fp = points12[:, 10:12]
    anchors = torch.cat([A, C, E], dim=1)
    vectors = torch.cat([B - A, D - C, Fp - E], dim=1)
    return anchors, vectors


def geom_to_points12(anchors: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs (B,12) points from anchors and vectors.

    Returns:
        points12 : (B, 12) clamped to [0, 1]
    """
    A  = anchors[:, 0:2]; C  = anchors[:, 2:4]; E  = anchors[:, 4:6]
    AB = vectors[:, 0:2]; CD = vectors[:, 2:4]; EF = vectors[:, 4:6]
    return torch.cat([A, A+AB, C, C+CD, E, E+EF], dim=1).clamp(0.0, 1.0)


# ── Keypoint metrics ──────────────────────────────────────────────────────────

def keypoint_metrics(
    pred_pts12: torch.Tensor,
    true_pts12: torch.Tensor,
    image_size: int = 512,
) -> dict:
    """
    Computes pixel-space and normalised keypoint localization metrics.

    Args:
        pred_pts12: (B, 12) predicted normalised coordinates
        true_pts12: (B, 12) ground-truth normalised coordinates
        image_size: side length used to convert to pixel space

    Returns:
        dict with keys: nme, mean_px, per_point_px, pck_01, pck_02, pck_05
    """
    pred = pred_pts12.view(-1, 6, 2) * image_size
    true = true_pts12.view(-1, 6, 2) * image_size
    errs = torch.norm(pred - true, dim=-1)   # (B, 6) pixel errors
    diag = (2.0 ** 0.5) * image_size         # image diagonal

    return {
        "nme":          (errs / diag).mean().item(),
        "mean_px":      errs.mean().item(),
        "per_point_px": errs.mean(dim=0).tolist(),   # (6,)
        "pck_01":       (errs / diag <= 0.01).float().mean().item(),
        "pck_02":       (errs / diag <= 0.02).float().mean().item(),
        "pck_05":       (errs / diag <= 0.05).float().mean().item(),
    }


# ── Post-hoc calibration ──────────────────────────────────────────────────────

# Systematic bias offsets derived from validation set analysis.
# Apply ONLY at inference — never during training.
CALIBRATION_OFFSETS = {
    0: 0.089,    # Normal     — slight underprediction
    1: 0.758,    # Borderline — moderate underprediction
    2: 1.671,    # Enlarged   — severe underprediction
}


def calibrate_vhs(pred_vhs: torch.Tensor) -> torch.Tensor:
    """
    Applies per-class post-hoc bias correction to predicted VHS values.

    Args:
        pred_vhs: (N,) predicted VHS tensor

    Returns:
        Calibrated VHS tensor on the same device
    """
    arr = pred_vhs.detach().cpu().numpy().copy()
    for i, v in enumerate(arr):
        if v < 8.2:
            arr[i] += CALIBRATION_OFFSETS[0]
        elif v >= 10.0:
            arr[i] += CALIBRATION_OFFSETS[2]
        else:
            arr[i] += CALIBRATION_OFFSETS[1]
    return torch.tensor(arr, dtype=pred_vhs.dtype, device=pred_vhs.device)


# ── Perpendicular correction (post-processing for visualisation) ──────────────

def enforce_perpendicular(pts: "np.ndarray") -> "np.ndarray":
    """
    Enforces CD ⊥ AB as per VHS protocol.
    Keeps A, B, C fixed. Rotates D so CD ⊥ AB, preserving |CD|.

    Args:
        pts: (6, 2) numpy array in pixel coordinates

    Returns:
        Corrected (6, 2) numpy array
    """
    import numpy as np
    pts = pts.copy()
    A, B, C, D = pts[0], pts[1], pts[2], pts[3]
    AB = B - A
    AB_len = np.linalg.norm(AB)
    if AB_len < 1e-6:
        return pts
    AB_unit = AB / AB_len
    perp = np.array([-AB_unit[1], AB_unit[0]])
    CD = D - C
    CD_len = np.linalg.norm(CD)
    if CD_len < 1e-6:
        return pts
    if np.dot(CD, perp) < 0:
        perp = -perp
    pts[3] = C + CD_len * perp
    return pts
