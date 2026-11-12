#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Binary image classification training script (Two folders: 0, 1) with ViT / Swin / MaxViT backbones.

OFFLINE-FRIENDLY: loads pretrained weights **from local files** (safetensors/.pth) and never tries to download.

- Data root example: /home/qxtech/data/yisheng/data_set (contains class folders: 0, 1)
- Automatic 8:2 stratified split per class (no file moves)
- Auto-switch to Focal Loss when class imbalance is detected
- Per-epoch validation with metrics saved: ACC, Precision, Recall, F1, AUC, TP, FP, TN, FN, plus ROC curve
- Checkpoints saved every epoch with filename pattern: epoch{epoch}_acc{acc:.4f}.pth

Local checkpoints expected (already downloaded by user):
  /home/qxtech/tmh/yisheng/checkpoint/
    ├── maxvit_tiny_tf_224.in1k.safetensors
    ├── swin_tiny_patch4_window7_224.ms_in1k.safetensors
    └── vit_base_patch16_224.augreg2_in21k_ft_in1k.safetensors

Dependencies:
  pip install timm scikit-learn matplotlib torchvision torch safetensors

Example usages:
  python train_transformer_cls.py \
    --data_dir /home/qxtech/data/yisheng/data_set \
    --model vit_base_patch16_224 \
    --local_ckpt_dir /home/qxtech/tmh/yisheng/checkpoint

  python train_transformer_cls.py \
    --data_dir /home/qxtech/data/yisheng/data_set \
    --model swin_tiny_patch4_window7_224 \
    --freeze

  python train_transformer_cls.py \
    --data_dir /home/qxtech/data/yisheng/data_set \
    --model maxvit_tiny_tf_224 --epochs 50
"""
from __future__ import annotations
import os
import argparse
import time
import json
import random
from datetime import datetime
import warnings
warnings.filterwarnings("ignore",
    message=".*torch\\.cuda\\.amp\\.autocast.*",
    category=FutureWarning)
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import timm
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt

# ------------------------- Utilities -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nowstamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ------------------------- Dataset & Split -------------------------
def build_splits(
    data_dir: str,
    img_size: int = 224,
    val_ratio: float = 0.2,
    interpolation: str = "bilinear",
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, list[int], list[int], Dict[str, int]]:
    """Create stratified train/val indices on the fly for ImageFolder (two-class 0/1)."""
    assert os.path.isdir(data_dir), f"Data directory not found: {data_dir}"

    # Base dataset (no transform yet), used to compute indices/targets
    base_ds = datasets.ImageFolder(data_dir)
    targets = np.array(base_ds.targets)
    classes = base_ds.classes
    assert len(classes) == 2, f"Expected exactly 2 classes (found {classes})."

    # Stratified split per class
    train_idx, val_idx = [], []
    for c in np.unique(targets):
        c_idx = np.where(targets == c)[0]
        np.random.shuffle(c_idx)
        n_val = int(round(len(c_idx) * val_ratio))
        val_idx.extend(c_idx[:n_val])
        train_idx.extend(c_idx[n_val:])

    # Transforms (lightweight and safe defaults)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if interpolation.lower() == "bicubic":
        interp = transforms.InterpolationMode.BICUBIC
    else:
        interp = transforms.InterpolationMode.BILINEAR

    train_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.14), interpolation=interp),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.14), interpolation=interp),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(data_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir, transform=val_tfms)

    return train_ds, val_ds, train_idx, val_idx, base_ds.class_to_idx


# ------------------------- Focal Loss -------------------------
class FocalLoss(nn.Module):
    """Multi-class Focal Loss (for logits), with optional class-wise alpha.
    alpha: Tensor of shape [C] or float, used as class weight.
    gamma: focusing parameter.
    """

    def __init__(self, alpha: torch.Tensor | float | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, torch.Tensor):
                self.register_buffer("alpha", alpha.float())
            else:
                self.register_buffer("alpha", torch.tensor([alpha], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy per-sample (no reduction)
        if hasattr(self, "alpha") and self.alpha is not None:
            # If alpha provided per-class, feed as weight into CE
            ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction="none")
        # pt = prob of the ground-truth class
        pt = F.softmax(logits, dim=1)[torch.arange(logits.size(0)), targets]
        focal_factor = (1.0 - pt).pow(self.gamma)
        loss = focal_factor * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def choose_loss(base_targets: np.ndarray, imbalance_threshold: float = 1.2) -> Tuple[nn.Module, Dict[str, Any]]:
    """Choose CrossEntropy or FocalLoss based on class imbalance.
    If (max_count / min_count) >= threshold, use FocalLoss with class-wise alpha.
    """
    counts = np.bincount(base_targets)
    assert len(counts) == 2, "This helper assumes exactly two classes (0/1)."
    min_c, max_c = counts.min(), counts.max()
    ratio = (max_c / max_c) if min_c == 0 else (max_c / min_c)

    use_focal = ratio >= imbalance_threshold and min_c > 0

    meta = {
        "counts": {"class0": int(counts[0]), "class1": int(counts[1])},
        "imbalance_ratio": float(ratio),
        "use_focal": bool(use_focal),
    }

    if use_focal:
        # alpha inversely proportional to frequency (normalized)
        freq = counts.astype(np.float32)
        alpha = (freq.sum() - freq) / freq.sum()  # minority gets larger weight
        alpha = torch.tensor(alpha, dtype=torch.float32)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion, meta


# ------------------------- Local Weights -------------------------
WEIGHT_FILES = {
    "vit_base_patch16_224": "vit_base_patch16_224.augreg2_in21k_ft_in1k.safetensors",
    "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224.ms_in1k.safetensors",
    "maxvit_tiny_tf_224": "maxvit_tiny_tf_224.in1k.safetensors",
}


def load_local_weights(model: nn.Module, weight_path: str) -> None:
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Local weight file not found: {weight_path}")
    ext = os.path.splitext(weight_path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file as safe_load_file
        sd = safe_load_file(weight_path)
    else:
        ckpt = torch.load(weight_path, map_location="cpu")
        if isinstance(ckpt, dict):
            sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        else:
            sd = ckpt
    # strip potential 'module.' prefixes
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # --- Robust head filtering: drop keys whose shape mismatches current model ---
    model_sd = model.state_dict()
    pruned_sd = {}
    dropped = []
    for k, v in sd.items():
        if k in model_sd:
            if isinstance(v, torch.Tensor) and model_sd[k].shape != v.shape:
                dropped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                continue
            pruned_sd[k] = v
        else:
            # key not used in this model definition
            continue

    missing, unexpected = model.load_state_dict(pruned_sd, strict=False)
    print(f"[INFO] Loaded local weights from {weight_path}. Missing: {len(missing)}, Unexpected: {len(unexpected)}, Dropped-mismatch: {len(dropped)}")
    if dropped:
        # Print first few for transparency
        for i, (k, s_src, s_dst) in enumerate(dropped[:8]):
            print(f"    - dropped: {k}: src{tuple(s_src)} -> dst{tuple(s_dst)}")


# ------------------------- Model -------------------------
def build_model(model_name: str, num_classes: int) -> nn.Module:
    """Create a timm model and set classifier to num_classes.
    We do NOT request pretrained weights online. We'll load local weights instead.
    """
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    return model


# ------------------------- Train / Eval -------------------------
def train_one_epoch(model, loader, optimizer, device, scaler=None, criterion=None) -> Tuple[float, float]:
    model.train()
    running_loss, correct, n = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets)
        if scaler is not None:
            # Correct AMP pattern: backward -> step -> update
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        n += images.size(0)

    return running_loss / max(n, 1), correct / max(n, 1)


def evaluate(model, loader, device) -> Dict[str, Any]:
    model.eval()
    ys, yprobs = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # prob of class 1
            yprobs.append(probs)
            ys.append(targets.numpy())

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(yprobs)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # ROC/AUC (guard against single-class edge cases)
    try:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        fpr, tpr, thr = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])
        auc = float("nan")

    # Confusion matrix (tn, fp, fn, tp)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "acc": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thr.tolist(),
    }


def save_roc_plot(fpr: np.ndarray, tpr: np.ndarray, auc: float, out_png: str, title: str):
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}" if not np.isnan(auc) else "ROC")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ------------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser(description="Binary classification with ViT/Swin/MaxViT (offline local weights)")
    parser.add_argument("--data_dir", type=str, default="/home/qxtech/data/yisheng/data_set",
                        help="Root containing class folders 0 and 1")
    parser.add_argument("--model", type=str, default="swin_tiny_patch4_window7_224",
                        choices=[
                            "vit_base_patch16_224",
                            "swin_tiny_patch4_window7_224",
                            "maxvit_tiny_tf_224",
                        ], help="Backbone architecture")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imbalance_threshold", type=float, default=1.2,
                        help="If max/min >= threshold, switch to FocalLoss")
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 2))
    parser.add_argument("--freeze", action="store_true", help="Freeze all backbone layers (linear probe)")
    parser.add_argument("--outdir", type=str, default="runs", help="Directory to save logs/ckpts")
    parser.add_argument("--local_ckpt_dir", type=str, default="/home/qxtech/tmh/yisheng/checkpoint",
                        help="Directory containing offline weights (.safetensors/.pth)")
    parser.add_argument("--local_weight_filename", type=str, default="",
                        help="Optional filename override; if empty, auto-select by model name.")
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data & split
    train_ds, val_ds, train_idx, val_idx, class_to_idx = build_splits(
        args.data_dir, img_size=args.img_size, val_ratio=args.val_ratio
    )

    # Loss selection based on base dataset distribution (before split)
    base_all = datasets.ImageFolder(args.data_dir)
    criterion, loss_meta = choose_loss(np.array(base_all.targets), args.imbalance_threshold)
    criterion = criterion.to(device)
    print("Loss selection:", loss_meta)

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model (no online pretrained) + load local weights
    model = build_model(args.model, num_classes=2)

    # Resolve local weight file
    if args.local_weight_filename:
        weight_file = args.local_weight_filename
    else:
        if args.model not in WEIGHT_FILES:
            raise KeyError(f"No default local weight filename for model: {args.model}")
        weight_file = WEIGHT_FILES[args.model]
    weight_path = os.path.join(args.local_ckpt_dir, weight_file)
    print(f"[INFO] Using local weights: {weight_path}")
    load_local_weights(model, weight_path)

    if args.freeze:
        for n, p in model.named_parameters():
            p.requires_grad = False
        # Unfreeze final classifier head (commonly named classifier/head/fc)
        for name in ["classifier", "head", "fc"]:
            if hasattr(model, name):
                for p in getattr(model, name).parameters():
                    p.requires_grad = True
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Output directory
    tag = f"{nowstamp()}_{args.model}"
    out_dir = os.path.join(args.outdir, tag)
    os.makedirs(out_dir, exist_ok=True)

    # Save config/meta
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({**vars(args), "class_to_idx": class_to_idx, "loss_meta": loss_meta}, f, ensure_ascii=False, indent=2)

    # Training loop
    best_acc = -1.0
    metrics_csv = os.path.join(out_dir, "metrics.csv")
    with open(metrics_csv, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,train_acc,val_acc,precision,recall,f1,auc,tp,fp,tn,fn,lr\n"
        )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler, criterion=criterion)
        metrics = evaluate(model, val_loader, device)
        scheduler.step()

        # Save per-epoch ROC plot
        fpr = np.array(metrics["fpr"], dtype=float)
        tpr = np.array(metrics["tpr"], dtype=float)
        auc = metrics["auc"]
        roc_png = os.path.join(out_dir, f"roc_epoch{epoch:03d}.png")
        save_roc_plot(fpr, tpr, auc, roc_png, title=f"ROC - epoch {epoch}")

        # Log row
        lr_cur = optimizer.param_groups[0]['lr']
        with open(metrics_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_acc:.6f},{metrics['acc']:.6f},{metrics['precision']:.6f},{metrics['recall']:.6f},{metrics['f1']:.6f},{auc if not np.isnan(auc) else ''},{metrics['tp']},{metrics['fp']},{metrics['tn']},{metrics['fn']},{lr_cur:.6f}\n"
            )

        # Save checkpoint named with epoch and acc
        ckpt_name = f"epoch{epoch:03d}_acc{metrics['acc']:.4f}.pth"
        ckpt_path = os.path.join(out_dir, ckpt_name)
        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "metrics": metrics,
            }, ckpt_path)

        # Track best
        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))

        dt = time.time() - t0
        auc_str = "nan" if np.isnan(auc) else f"{auc:.4f}"
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={metrics['acc']:.4f} AUC={auc_str} | {dt:.1f}s"
        )

    print("Training complete. Outputs saved to:", out_dir)


if __name__ == "__main__":
    main()
