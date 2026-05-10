"""
train.py — CardioMark training script
======================================
Trains VectorGeoRefineNet on the CardioMark dataset.

Usage::

    python train.py --config configs/default.yaml

    # or override individual fields:
    python train.py --data_root /path/to/data --epochs 200 --gpu 0
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import platform
from datetime import datetime
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── project imports ────────────────────────────────────────────────────────────
from data.dataset    import DogHeartDataset, get_train_transform, get_eval_transform
from data.sampling   import build_weighted_sampler, compute_class_weights, summarize_label_distribution
from models.vgr_net  import VectorGeoRefineNet
from utils.vhs       import points12_to_geom, get_labels
from utils.losses    import (
    VHSAwareLoss, coord_loss_fn, geometry_loss,
    vector_length_loss, vector_direction_loss,
)
from utils.evaluation import validate


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train VectorGeoRefineNet on CardioMark")
    p.add_argument("--data_root",   default="/dog_heart/models/data")
    p.add_argument("--checkpoint",  default="", help="Resume from checkpoint path")
    p.add_argument("--gpu",         default="3", help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--epochs",      type=int, default=2000)
    p.add_argument("--image_size",  type=int, default=512)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--lr",          type=float, default=1.5e-5)
    p.add_argument("--seed",        type=int, default=113)
    p.add_argument("--out_dir",     default="", help="Output dir (auto-named if empty)")
    return p.parse_args()


# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging(out_dir: str) -> logging.Logger:
    logger = logging.getLogger("cardiomark")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    class InfoOnly(logging.Filter):
        def filter(self, r): return r.levelno == logging.INFO

    ih = logging.FileHandler(f"{out_dir}/info_log.txt",    mode="w")
    wh = logging.FileHandler(f"{out_dir}/warning_log.txt", mode="w")
    ch = logging.StreamHandler()

    ih.setLevel(logging.INFO);    ih.addFilter(InfoOnly())
    wh.setLevel(logging.WARNING); ch.setLevel(logging.INFO)
    for h in [ih, wh, ch]: h.setFormatter(fmt); logger.addHandler(h)
    return logger


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # Output directory
    out_dir = args.out_dir or datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{out_dir}/models").mkdir(exist_ok=True)
    print(f"Output dir: {out_dir}")

    logger = setup_logging(out_dir)
    logger.info(f"Args: {vars(args)}")
    logger.info(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_tf = get_train_transform(args.image_size)
    eval_tf  = get_eval_transform(args.image_size)

    ds_train = DogHeartDataset(f"{args.data_root}/Train",       train_tf)
    ds_valid = DogHeartDataset(f"{args.data_root}/Valid",       eval_tf)

    logger.info(f"Train: {len(ds_train)}  Valid: {len(ds_valid)}")

    sampler = build_weighted_sampler(ds_train)

    train_loader = DataLoader(
        ds_train,
        batch_size=max(1, args.batch_size // args.accum_steps),
        sampler=sampler, num_workers=8, pin_memory=True,
    )
    valid_loader = DataLoader(
        ds_valid, batch_size=32,
        shuffle=False, num_workers=8, pin_memory=True,
    )

    # Plain loader for class weight computation (no sampler)
    plain_loader = DataLoader(ds_train, batch_size=32, shuffle=False, num_workers=4)

    summarize_label_distribution(train_loader, "Train (sampled)")
    summarize_label_distribution(valid_loader, "Valid")

    train_cw = compute_class_weights(plain_loader, device, f"{out_dir}/train_cw.pt")
    val_cw   = compute_class_weights(valid_loader, device, f"{out_dir}/val_cw.pt")

    vhs_train_loss = VHSAwareLoss(class_weights=train_cw, margin=0.02, boundary_weight=3.0).to(device)
    vhs_valid_loss = VHSAwareLoss(class_weights=val_cw,   margin=0.02, boundary_weight=3.0).to(device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = VectorGeoRefineNet(pretrained=False, dropout=0.3).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.info("Training from scratch")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Params: {total:,}  Trainable: {trainable:,}")

    # Save config
    cfg = vars(args)
    cfg.update({"total_params": total, "out_dir": out_dir})
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Optimiser / scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"val_loss": float("inf"), "val_acc": -1, "val_f1": -1}
    epoch_log = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        running = Counter()

        # Loss weight schedule
        if epoch < 10:
            wa, wv, wp, wl, wd, wg, wvhs = 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.05
        else:
            wa, wv, wp, wl, wd, wg, wvhs = 1.0, 1.2, 1.5, 0.4, 0.4, 0.2, 0.2

        from tqdm import tqdm
        bar = tqdm(train_loader, desc=f"E{epoch+1}/{args.epochs}", leave=False)
        optimizer.zero_grad(set_to_none=True)

        for bidx, (_, images, pts12, vhs) in enumerate(bar):
            images = images.to(device, non_blocking=True)
            pts12  = pts12.to(device,  non_blocking=True)
            vhs    = vhs.to(device,    non_blocking=True).view(-1)

            true_anchors, true_vectors = points12_to_geom(pts12)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out      = model(images)
                pred_pts = out["refined_points"]
                pred_vhs = out["vhs"].view(-1)

                l_a   = coord_loss_fn(out["anchors"], true_anchors)
                l_v   = coord_loss_fn(out["vectors"], true_vectors)
                l_p   = coord_loss_fn(pred_pts, pts12)
                l_len = vector_length_loss(out["vectors"], true_vectors)
                l_dir = vector_direction_loss(out["vectors"], true_vectors)
                l_geo = geometry_loss(pred_pts)
                l_vhs = vhs_train_loss(pred_vhs, vhs)
                loss  = wa*l_a + wv*l_v + wp*l_p + wl*l_len + wd*l_dir + wg*l_geo + wvhs*l_vhs

            scaler.scale(loss / args.accum_steps).backward()
            running["total"] += loss.item(); running["pts"] += l_p.item()
            running["vhs"]   += l_vhs.item()

            do_step = (
                ((bidx + 1) % args.accum_steps == 0) or
                ((bidx + 1) == len(train_loader))
            )
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bar.set_postfix({"L": f"{loss.item():.4f}",
                             "Pts": f"{l_p.item():.4f}",
                             "VHS": f"{l_vhs.item():.3f}"})
        bar.close()

        val = validate(model, valid_loader, device, vhs_valid_loss, epoch=epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Checkpointing
        if val["loss"] < best["val_loss"]:
            best["val_loss"] = val["loss"]
            torch.save(model.state_dict(), f"{out_dir}/models/best_val_loss.pth")
            logger.info(f"✅ best_val_loss E{epoch+1} | {val['loss']:.4f}")

        if val["acc"] > best["val_acc"]:
            best["val_acc"] = val["acc"]
            torch.save(model.state_dict(), f"{out_dir}/models/best_val_acc.pth")
            logger.info(f"✅ best_val_acc  E{epoch+1} | {val['acc']:.4f}")

        if val["macro_f1"] > best["val_f1"]:
            best["val_f1"] = val["macro_f1"]
            torch.save(model.state_dict(), f"{out_dir}/models/best_val_f1.pth")
            logger.info(f"✅ best_val_f1   E{epoch+1} | {val['macro_f1']:.4f}")

        torch.save(model.state_dict(), f"{out_dir}/models/epoch_latest.pth")

        nb = len(train_loader)
        epoch_log.append({
            "epoch": epoch + 1,
            "train_loss": running["total"] / nb,
            "val": val, "lr": current_lr,
            "epoch_sec": time.time() - t0,
        })

        logger.info(
            f"E{epoch+1}/{args.epochs} | lr:{current_lr:.2e} | "
            f"train:{running['total']/nb:.4f} | "
            f"val_acc:{val['acc']:.4f} nme:{val['nme']:.5f} | "
            f"best_acc:{best['val_acc']:.4f} | "
            f"{time.time()-t0:.1f}s"
        )

    # Final save
    torch.save(model.state_dict(), f"{out_dir}/models/final.pth")
    with open(f"{out_dir}/epoch_metrics.json", "w") as f:
        json.dump(epoch_log, f, indent=2)
    logger.info(f"Done. Best val acc: {best['val_acc']:.4f}")
    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
