"""
evaluate.py — CardioMark evaluation script
===========================================
Runs full evaluation on any split and saves all five experiment JSONs
used by extract_table1.py to populate the paper tables.

Usage::

    # Evaluate best checkpoint on test set
    python evaluate.py \\
        --checkpoint /path/to/best_test_acc.pth \\
        --data_root  /path/to/data \\
        --split      Test_Images \\
        --out_dir    eval_results_vgr

    # Evaluate with calibration
    python evaluate.py --checkpoint ... --calibrate

    # Evaluate on DogHeart (cross-institution transfer)
    python evaluate.py \\
        --checkpoint /path/to/best_test_acc.pth \\
        --data_root  /path/to/dogheart \\
        --split      Test_Images \\
        --out_dir    eval_results_dogheart
"""

import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from tqdm import tqdm

from data.dataset    import DogHeartDataset, get_eval_transform
from models.vgr_net  import VectorGeoRefineNet
from utils.vhs       import (
    calc_vhs, get_labels, keypoint_metrics,
    calibrate_vhs, CALIBRATION_OFFSETS,
)
from utils.losses    import VHSAwareLoss

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cardiomark.eval")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root",  required=True)
    p.add_argument("--split",      default="Test_Images",
                   help="Subfolder name: Train | Valid | Test_Images")
    p.add_argument("--out_dir",    default="eval_results")
    p.add_argument("--gpu",        default="3")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--calibrate",  action="store_true",
                   help="Apply post-hoc VHS calibration")
    p.add_argument("--model_name", default="VectorGeoRefineNet (EfficientNet-B7)")
    return p.parse_args()


@torch.no_grad()
def run_eval(model, loader, device, image_size, calibrate):
    model.eval()

    all_pred_vhs, all_gt_vhs         = [], []
    all_pred_pts,  all_gt_pts         = [], []
    all_pred_labels, all_gt_labels    = [], []
    all_idx, all_fnames               = [], []

    for idx, images, pts12, vhs in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        pts12  = pts12.to(device,  non_blocking=True)
        vhs    = vhs.to(device,    non_blocking=True).view(-1)

        out      = model(images)
        pred_pts = out["refined_points"]
        pred_vhs = out["vhs"].view(-1)

        if calibrate:
            pred_vhs = calibrate_vhs(pred_vhs)

        all_pred_vhs.extend(pred_vhs.cpu().tolist())
        all_gt_vhs.extend(vhs.cpu().tolist())
        all_pred_pts.append(pred_pts.cpu())
        all_gt_pts.append(pts12.cpu())
        all_pred_labels.extend(get_labels(pred_vhs).cpu().tolist())
        all_gt_labels.extend(get_labels(vhs).cpu().tolist())
        all_idx.extend(idx.tolist())

    pred_pts_cat = torch.cat(all_pred_pts)
    gt_pts_cat   = torch.cat(all_gt_pts)
    km = keypoint_metrics(pred_pts_cat, gt_pts_cat, image_size)

    return {
        "pred_vhs":    all_pred_vhs,
        "gt_vhs":      all_gt_vhs,
        "pred_labels": all_pred_labels,
        "gt_labels":   all_gt_labels,
        "km":          km,
        "idx":         all_idx,
        "pred_pts":    pred_pts_cat.numpy().tolist(),
        "gt_pts":      gt_pts_cat.numpy().tolist(),
    }


def build_exp1(data, model_name) -> dict:
    """exp1_per_class.json — aggregate + per-class accuracy."""
    tl = data["gt_labels"]; pl = data["pred_labels"]
    cm  = confusion_matrix(tl, pl, labels=[0, 1, 2])
    pca = (cm.diagonal() / cm.sum(axis=1).clip(min=1)).tolist()
    f1  = f1_score(tl, pl, average="macro")
    bal = balanced_accuracy_score(tl, pl)
    ov  = sum(t == p for t, p in zip(tl, pl)) / len(tl)
    worst = int(np.argmin(pca))
    return {
        "model":              model_name,
        "overall_accuracy":   ov,
        "per_class_accuracy": pca,
        "worst_group_acc":    pca[worst],
        "worst_group_class":  worst,
        "macro_f1":           f1,
        "balanced_acc":       bal,
        "confusion_matrix":   cm.tolist(),
    }


def build_exp2(data) -> dict:
    """exp2_boundary.json — Far / Mid / Near accuracy."""
    gv = np.array(data["gt_vhs"])
    tl = np.array(data["gt_labels"])
    pl = np.array(data["pred_labels"])

    dist = np.minimum(np.abs(gv - 8.2), np.abs(gv - 10.0))

    def bucket(mask):
        if mask.sum() == 0:
            return {"accuracy": None, "n": 0, "flip_rate": None, "mae": None}
        acc = (tl[mask] == pl[mask]).mean()
        mae = np.abs(np.array(data["pred_vhs"])[mask] - gv[mask]).mean()
        # Flip rate: fraction of near-boundary that would change class
        # under ±0.1 VU perturbation
        pv  = np.array(data["pred_vhs"])[mask]
        flip = (
            (get_labels(torch.tensor(pv + 0.1)) != get_labels(torch.tensor(pv))) |
            (get_labels(torch.tensor(pv - 0.1)) != get_labels(torch.tensor(pv)))
        ).float().mean().item()
        return {"accuracy": float(acc), "n": int(mask.sum()),
                "flip_rate": float(flip), "mae": float(mae)}

    return {
        "Far (>1.0 VU)":     bucket(dist > 1.0),
        "Mid (0.5-1.0 VU)":  bucket((dist >= 0.5) & (dist <= 1.0)),
        "Near (<0.5 VU)":    bucket(dist < 0.5),
    }


def build_exp3(data) -> dict:
    """exp3_keypoints.json — NME / PCK metrics."""
    km = data["km"]
    return {
        "nme_overall": km["nme"],
        "mean_px":     km["mean_px"],
        "per_point_px": km["per_point_px"],
        "pck": {
            "PCK@0.01": km["pck_01"],
            "PCK@0.02": km["pck_02"],
            "PCK@0.05": km["pck_05"],
        },
    }


def build_exp4(data) -> dict:
    """exp4_rawrr.json — right-answer-wrong-reason analysis."""
    km   = data["km"]
    tl   = np.array(data["gt_labels"])
    pl   = np.array(data["pred_labels"])
    diag = (2.0 ** 0.5) * 512

    # Keypoint correct = NME < 0.05 per sample
    pred_pts = torch.tensor(data["pred_pts"]).view(-1, 6, 2)
    gt_pts   = torch.tensor(data["gt_pts"]).view(-1, 6, 2)
    errs     = torch.norm(pred_pts - gt_pts, dim=-1).mean(dim=1) * 512
    kp_ok    = (errs / diag < 0.05).numpy()

    kp_ok_class_wrong = (kp_ok & (tl != pl)).mean()
    kp_ok_class_right = (kp_ok & (tl == pl)).mean()

    return {
        "kp_correct_class_wrong_rate": float(kp_ok_class_wrong),
        "kp_correct_class_right_rate": float(kp_ok_class_right),
        "kp_threshold_nme":            0.05,
        "n_total":                     len(tl),
    }


def build_exp5(data) -> dict:
    """exp5_bland_altman.json — clinical agreement."""
    diff = np.array(data["pred_vhs"]) - np.array(data["gt_vhs"])
    mae  = np.abs(np.array(data["pred_vhs"]) - np.array(data["gt_vhs"])).mean()
    return {
        "mean_bias":  float(diff.mean()),
        "sd":         float(diff.std()),
        "loa_lower":  float(diff.mean() - 1.96 * diff.std()),
        "loa_upper":  float(diff.mean() + 1.96 * diff.std()),
        "mae":        float(mae),
        "n":          len(diff),
    }


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Dataset
    ds = DogHeartDataset(
        os.path.join(args.data_root, args.split),
        get_eval_transform(args.image_size),
    )
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=8, pin_memory=True)
    logger.info(f"Evaluating {len(ds)} images from {args.split}")

    # Model
    model = VectorGeoRefineNet(pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info(f"Loaded: {args.checkpoint}")

    # Run inference
    data = run_eval(model, loader, device, args.image_size, args.calibrate)

    # Build and save experiment JSONs
    def save(obj, name):
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        logger.info(f"Saved → {path}")

    save(build_exp1(data, args.model_name), "exp1_per_class.json")
    save(build_exp2(data),                  "exp2_boundary.json")
    save(build_exp3(data),                  "exp3_keypoints.json")
    save(build_exp4(data),                  "exp4_rawrr.json")
    save(build_exp5(data),                  "exp5_bland_altman.json")

    # Save predictions CSV
    import pandas as pd
    rows = []
    gv = data["gt_vhs"]; pv = data["pred_vhs"]
    tl = data["gt_labels"]; pl = data["pred_labels"]
    for i in range(len(gv)):
        dist = min(abs(gv[i]-8.2), abs(gv[i]-10.0))
        rows.append({
            "idx":         data["idx"][i],
            "gt_vhs":      round(gv[i], 4),
            "pred_vhs":    round(pv[i], 4),
            "vhs_err":     round(abs(pv[i]-gv[i]), 4),
            "gt_label":    tl[i],
            "pred_label":  pl[i],
            "correct":     int(tl[i] == pl[i]),
            "dist_to_thr": round(dist, 4),
            "near_boundary": dist < 0.5,
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(args.out_dir, "predictions.csv"), index=False
    )

    # Print summary
    e1 = build_exp1(data, args.model_name)
    print(f"\n{'='*55}")
    print(f"Model  : {args.model_name}")
    print(f"Split  : {args.split}  N={len(ds)}")
    print(f"Overall: {e1['overall_accuracy']*100:.2f}%")
    print(f"Normal : {e1['per_class_accuracy'][0]*100:.2f}%")
    print(f"Border : {e1['per_class_accuracy'][1]*100:.2f}%")
    print(f"Enlarged: {e1['per_class_accuracy'][2]*100:.2f}%")
    print(f"F1     : {e1['macro_f1']:.4f}")
    print(f"NME    : {data['km']['nme']:.5f}")
    print(f"MAE    : {build_exp5(data)['mae']:.4f} VU")
    print(f"{'='*55}")
    print(f"Results saved → {args.out_dir}/")


if __name__ == "__main__":
    main()
