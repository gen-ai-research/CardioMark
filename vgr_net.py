"""
VectorGeoRefineNet
==================
EfficientNet-B7 backbone with a geometry-aware keypoint head and
a local feature patch refiner.

Architecture:
    1. EfficientNet-B7 encoder  →  (B, 2560, 16, 16) feature map
    2. Global average pool      →  (B, 2560)
    3. Shared trunk             →  (B, 512)
    4. Anchor head              →  (B, 6)   normalised anchor coords (A, C, E)
    5. Vector head              →  (B, 6)   displacement vectors (AB, CD, EF)
    6. geom_to_points12         →  (B, 12)  coarse keypoints
    7. FeaturePatchRefiner       →  (B, 12)  refined keypoints
    8. calc_vhs                 →  (B,)     predicted VHS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils.vhs import calc_vhs, geom_to_points12


class FeaturePatchRefiner(nn.Module):
    """
    Refines coarse keypoint predictions using local feature patches
    extracted from the backbone feature map.

    For each keypoint, a (patch_size × patch_size) region is sampled
    from the feature map centred on the coarse prediction.  A small
    CNN processes all patches and outputs a (Δx, Δy) correction.

    Args:
        in_ch      : number of input feature channels (2560 for EffNet-B7)
        patch_size : spatial size of each sampled patch (default 7)
        hidden_dim : intermediate channel count in the refinement CNN
        num_points : number of keypoints to refine (6)
    """

    def __init__(
        self,
        in_ch: int,
        patch_size: int = 7,
        hidden_dim: int = 128,
        num_points: int = 6,
    ):
        super().__init__()
        self.in_ch      = in_ch
        self.patch_size = patch_size
        self.num_points = num_points

        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.delta_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 2),
        )

    def _sample_patches(self, feat_map: torch.Tensor, pts12: torch.Tensor):
        B, C, Hf, Wf = feat_map.shape
        pts  = pts12.view(B, self.num_points, 2)
        half = self.patch_size // 2
        all_patches = []
        for b in range(B):
            pb = []
            for k in range(self.num_points):
                x = int(pts[b, k, 0].item() * (Wf - 1))
                y = int(pts[b, k, 1].item() * (Hf - 1))
                x1 = max(0, x - half); x2 = min(Wf, x + half + 1)
                y1 = max(0, y - half); y2 = min(Hf, y + half + 1)
                patch = feat_map[b:b+1, :, y1:y2, x1:x2]
                patch = F.interpolate(
                    patch, (self.patch_size, self.patch_size),
                    mode="bilinear", align_corners=False,
                )
                pb.append(patch)
            all_patches.append(torch.cat(pb, dim=0))
        return torch.stack(all_patches, dim=0)

    def forward(self, feat_map: torch.Tensor, coarse: torch.Tensor):
        B = coarse.size(0)
        patches = self._sample_patches(feat_map, coarse)
        patches = patches.view(
            B * self.num_points, self.in_ch,
            self.patch_size, self.patch_size,
        )
        deltas  = self.delta_head(self.refine(patches)).view(B, self.num_points, 2)
        deltas  = 0.03 * torch.tanh(deltas)          # ≤ ~15 px at 512
        refined = (coarse.view(B, self.num_points, 2) + deltas).clamp(0.0, 1.0)
        return refined.view(B, -1), deltas


class VectorGeoRefineNet(nn.Module):
    """
    Geometry-aware VHS estimation model.

    The model predicts anchor points and displacement vectors rather than
    raw coordinates, then refines predictions using local feature patches.

    Args:
        pretrained: whether to initialise EfficientNet-B7 with ImageNet weights
        dropout   : dropout rate in the shared trunk
    """

    def __init__(self, pretrained: bool = False, dropout: float = 0.3):
        super().__init__()

        weights  = models.EfficientNet_B7_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b7(weights=weights)

        self.features = backbone.features
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        feat_dim      = backbone.classifier[1].in_features   # 2560

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 1024), nn.SiLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(1024, 512),      nn.SiLU(inplace=True), nn.Dropout(dropout * 0.5),
        )

        self.anchor_head  = nn.Sequential(nn.Linear(512, 6), nn.Sigmoid())
        self.vector_head  = nn.Sequential(nn.Linear(512, 6), nn.Tanh())
        self.vector_scale = nn.Parameter(torch.tensor(0.25))

        self.refiner = FeaturePatchRefiner(
            in_ch=feat_dim, patch_size=7, hidden_dim=128, num_points=6
        )

    def forward(self, x: torch.Tensor) -> dict:
        feat_map = self.features(x)
        pooled   = self.avgpool(feat_map)
        shared   = self.shared(pooled)

        anchors = self.anchor_head(shared)
        vectors = self.vector_head(shared) * self.vector_scale

        coarse_pts              = geom_to_points12(anchors, vectors)
        refined_pts, deltas     = self.refiner(feat_map, coarse_pts)
        pred_vhs                = calc_vhs(refined_pts)

        return {
            "anchors":        anchors,
            "vectors":        vectors,
            "coarse_points":  coarse_pts,
            "refined_points": refined_pts,
            "deltas":         deltas,
            "vhs":            pred_vhs,
        }
