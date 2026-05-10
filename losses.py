"""
Loss functions
==============
VHSAwareLoss    — clinical threshold-aware regression loss with boundary penalty
GeometryLoss    — prevents segment collapse and enforces near-perpendicularity
CoordLoss       — SmoothL1 on normalised and pixel-space coordinates
VectorLengths   — segment length supervision
VectorDirection — segment direction cosine loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.vhs import get_labels


# ── Coordinate loss (shared instance) ────────────────────────────────────────

coord_loss_fn = nn.SmoothL1Loss(beta=0.01)


# ── VHS-aware loss ────────────────────────────────────────────────────────────

class VHSAwareLoss(nn.Module):
    """
    Combines L1 regression, mismatch penalty, soft margin, and a boundary
    proximity multiplier that up-weights samples within ``boundary_zone`` VU
    of either clinical threshold (8.2 or 10.0).

    Args:
        class_weights        : (3,) inverse-frequency weights for Normal/Border/Enlarged
        margin               : soft margin around each threshold (VU)
        middle_class_multiplier: extra weight for Borderline margin violations
        boundary_weight      : loss multiplier for near-boundary samples
        boundary_zone        : distance (VU) defining "near boundary"
    """

    THRESHOLD_LOW  = 8.2
    THRESHOLD_HIGH = 10.0

    def __init__(
        self,
        class_weights=None,
        margin: float = 0.02,
        middle_class_multiplier: float = 2.0,
        boundary_weight: float = 3.0,
        boundary_zone: float = 0.5,
    ):
        super().__init__()
        self.margin               = margin
        self.mcm                  = middle_class_multiplier
        self.boundary_weight      = boundary_weight
        self.boundary_zone        = boundary_zone
        self.l1 = nn.L1Loss(reduction="none")
        self.cw = (class_weights if class_weights is not None
                   else torch.tensor([1.0, 1.0, 1.0]))

    def forward(self, vhs_pred: torch.Tensor, vhs_true: torch.Tensor) -> torch.Tensor:
        vhs_pred = vhs_pred.reshape(-1)
        vhs_true = vhs_true.reshape(-1)

        tc = get_labels(vhs_true)
        pc = get_labels(vhs_pred)

        # Base L1 per sample
        l1_per = self.l1(vhs_pred, vhs_true)

        # Boundary proximity multiplier
        dist = torch.min(
            torch.abs(vhs_true - self.THRESHOLD_LOW),
            torch.abs(vhs_true - self.THRESHOLD_HIGH),
        )
        bm = torch.where(
            dist < self.boundary_zone,
            torch.full_like(vhs_pred, self.boundary_weight),
            torch.ones_like(vhs_pred),
        )

        # Mismatch penalty
        mis = (pc != tc).float()

        # Soft margin penalty
        sp = torch.zeros_like(vhs_pred)

        m0 = tc == 0
        if m0.any():
            sp[m0] = F.relu(vhs_pred[m0] - (self.THRESHOLD_LOW + self.margin))

        m1 = tc == 1
        if m1.any():
            t = self.margin / self.mcm
            sp[m1] = self.mcm * (
                F.relu(self.THRESHOLD_LOW  - vhs_pred[m1]) +
                F.relu(vhs_pred[m1] - (self.THRESHOLD_HIGH + t))
            )

        m2 = tc == 2
        if m2.any():
            sp[m2] = F.relu((self.THRESHOLD_HIGH - self.margin) - vhs_pred[m2])

        cw = self.cw.to(vhs_pred.device)[tc]
        return (bm * l1_per + cw * (mis + sp)).mean()


# ── Geometry loss ─────────────────────────────────────────────────────────────

def geometry_loss(points12: torch.Tensor) -> torch.Tensor:
    """
    Penalises degenerate geometry:
      - collapse: segments shorter than 0.02 normalised units
      - parallel: AB and CD being parallel (should be perpendicular)

    Args:
        points12: (B, 12) normalised keypoint coordinates

    Returns:
        Scalar loss
    """
    pts = points12.view(-1, 6, 2)
    A, B, C, D, E, Fp = (pts[:, i] for i in range(6))

    AB = B - A; CD = D - C; EF = Fp - E
    AL = torch.norm(AB, dim=-1).clamp_min(1e-6)
    CL = torch.norm(CD, dim=-1).clamp_min(1e-6)
    EL = torch.norm(EF, dim=-1).clamp_min(1e-6)

    collapse  = (F.relu(0.02 - AL) + F.relu(0.02 - CL) + F.relu(0.02 - EL)).mean()
    ABu = AB / AL.unsqueeze(-1); CDu = CD / CL.unsqueeze(-1)
    parallel  = (1.0 - (ABu * CDu).sum(dim=-1).abs()).mean()

    return collapse + 0.25 * parallel


# ── Vector length / direction losses ─────────────────────────────────────────

def vector_length_loss(pred_v: torch.Tensor, true_v: torch.Tensor) -> torch.Tensor:
    """SmoothL1 on segment lengths (3 segments × batch)."""
    p = pred_v.view(-1, 3, 2); t = true_v.view(-1, 3, 2)
    return F.smooth_l1_loss(
        torch.norm(p, dim=-1), torch.norm(t, dim=-1), beta=0.01
    )


def vector_direction_loss(pred_v: torch.Tensor, true_v: torch.Tensor) -> torch.Tensor:
    """Cosine direction loss: 1 − cos(angle) for each segment."""
    p = pred_v.view(-1, 3, 2); t = true_v.view(-1, 3, 2)
    pu = p / torch.norm(p, dim=-1, keepdim=True).clamp_min(1e-6)
    tu = t / torch.norm(t, dim=-1, keepdim=True).clamp_min(1e-6)
    return (1.0 - (pu * tu).sum(dim=-1)).mean()
