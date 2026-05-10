"""
Evaluation
==========
Standalone evaluation functions — no training logic.
Used by both train.py (per-epoch validation) and evaluate.py (inference only).
"""

import logging
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, f1_score,
    balanced_accuracy_score, classification_report,
)

from utils.vhs import get_labels, keypoint_metrics, calibrate_vhs, calc_vhs
from utils.losses import (
    coord_loss_fn, geometry_loss,
    vector_length_loss, vector_direction_loss,
)
from utils.vhs import points12_to_geom

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Normal (<8.2)", "Borderline (8.2–10)", "Enlarged (≥10)"]


def _loss_weights(epoch):
    """Returns (w_anchor, w_vector, w_points, w_len, w_dir, w_geo, w_vhs)."""
    if epoch is not None and epoch < 10:
        return 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.05
    return 1.0, 1.2, 1.5, 0.4, 0.4, 0.2, 0.2


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    vhs_loss_fn,
    epoch=None,
    calibrate: bool = False,
    image_size: int = 512,
) -> dict:
    """
    Full validation pass — returns metrics dict.

    Args:
        model      : VectorGeoRefineNet (or any model with same output dict)
        loader     : validation DataLoader
        device     : torch device
        vhs_loss_fn: VHSAwareLoss instance
        epoch      : current epoch (controls loss weight schedule)
        calibrate  : whether to apply post-hoc VHS calibration
        image_size : used for pixel-space keypoint metrics

    Returns:
        dict with keys: loss, acc, mae, macro_f1, balanced_acc,
                        per_class_acc, nme, mean_px, pck_01/02/05, cm
    """
    model.eval()
    wa, wv, wp, wl, wd, wg, wvhs = _loss_weights(epoch)

    tot    = Counter()
    all_tl = []; all_pl = []
    correct = 0

    bar = tqdm(loader, desc="Val", leave=False)
    for _, images, pts12, vhs in bar:
        images = images.to(device, non_blocking=True)
        pts12  = pts12.to(device,  non_blocking=True)
        vhs    = vhs.to(device,    non_blocking=True).view(-1)

        true_anchors, true_vectors = points12_to_geom(pts12)

        out = model(images)
        pred_pts = out["refined_points"]
        pred_vhs = out["vhs"].view(-1)
        pred_vhs_eval = calibrate_vhs(pred_vhs) if calibrate else pred_vhs

        l_a   = coord_loss_fn(out["anchors"], true_anchors)
        l_v   = coord_loss_fn(out["vectors"], true_vectors)
        l_p   = coord_loss_fn(pred_pts, pts12)
        l_len = vector_length_loss(out["vectors"], true_vectors)
        l_dir = vector_direction_loss(out["vectors"], true_vectors)
        l_geo = geometry_loss(pred_pts)
        l_vhs = vhs_loss_fn(pred_vhs, vhs)
        loss  = wa*l_a + wv*l_v + wp*l_p + wl*l_len + wd*l_dir + wg*l_geo + wvhs*l_vhs

        bs = images.size(0)
        tot["loss"] += loss.item() * bs
        tot["mae"]  += torch.abs(pred_vhs_eval - vhs).sum().item()
        tot["n"]    += bs

        km = keypoint_metrics(pred_pts, pts12, image_size)
        for k, v in km.items():
            if k != "per_point_px":
                tot[k] += v * bs

        pc = get_labels(pred_vhs_eval).cpu().numpy()
        tc = get_labels(vhs).cpu().numpy()
        all_tl.extend(tc.tolist()); all_pl.extend(pc.tolist())
        correct += (pc == tc).sum()

    n   = tot["n"]
    cm  = confusion_matrix(all_tl, all_pl, labels=[0, 1, 2])
    pca = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    f1  = f1_score(all_tl, all_pl, average="macro")
    bal = balanced_accuracy_score(all_tl, all_pl)

    tag = " [cal]" if calibrate else ""
    logger.info(
        f"Val{tag} loss:{tot['loss']/n:.4f} acc:{correct/n:.4f} "
        f"mae:{tot['mae']/n:.4f} f1:{f1:.4f} bal:{bal:.4f} "
        f"nme:{tot['nme']/n:.5f} px:{tot['mean_px']/n:.2f} "
        f"pck05:{tot['pck_05']/n:.4f} "
        f"pc:{[round(x,4) for x in pca.tolist()]}"
    )
    logger.warning(
        f"Val{tag} report:\n" +
        classification_report(all_tl, all_pl,
                               target_names=CLASS_NAMES,
                               digits=4, zero_division=1)
    )

    return {
        "loss":         tot["loss"] / n,
        "acc":          correct / n,
        "mae":          tot["mae"] / n,
        "macro_f1":     f1,
        "balanced_acc": bal,
        "per_class_acc": pca.tolist(),
        "nme":          tot["nme"] / n,
        "mean_px":      tot["mean_px"] / n,
        "pck_01":       tot["pck_01"] / n,
        "pck_02":       tot["pck_02"] / n,
        "pck_05":       tot["pck_05"] / n,
        "cm":           cm.tolist(),
    }


@torch.no_grad()
def evaluate_test(
    model,
    loader,
    device,
    vhs_loss_fn,
    calibrate: bool = False,
    image_size: int = 512,
) -> dict:
    """
    Test-set evaluation — identical to validate() but with no epoch schedule.
    Logs under 'Test' tag.
    """
    return validate(
        model, loader, device, vhs_loss_fn,
        epoch=999,        # triggers full-weight schedule
        calibrate=calibrate,
        image_size=image_size,
    )
