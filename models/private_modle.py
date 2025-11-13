#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a 7-class facial emotion recognition model.

Expected directory layout under --data_root:
    data_root/
        train/<class>/*.jpg|png|jpeg
        val/<class>/*.jpg|png|jpeg
        test/<class>/*.jpg|png|jpeg   (optional)

Classes typically:
    angry, disgust, fear, happy, sad, surprise, neutral
"""

import os
import json
import math
import argparse
import random
from collections import Counter
from datetime import datetime
import platform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, transforms, models

from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # good default for vision
    torch.backends.cudnn.deterministic = False


def class_weights_from_counts(counts, mode="inv_log"):
    """
    counts: dict {class_index: count}
    mode: 'inv_log' or 'inv'
    """
    weights = []
    for i in range(len(counts)):
        c = counts.get(i, 1)
        if mode == "inv_log":
            w = 1.0 / math.log(1.0 + c)
        else:
            w = 1.0 / float(c)
        weights.append(w)
    s = sum(weights)
    weights = [w * len(weights) / s for w in weights]  # normalize around 1.0
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(dataset):
    """
    For WeightedRandomSampler (optional).
    """
    targets = [y for _, y in dataset.samples]
    counts = Counter(targets)
    class_w = {k: 1.0 / v for k, v in counts.items()}
    sample_w = [class_w[y] for y in targets]
    return torch.DoubleTensor(sample_w)


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y", "t")


# -----------------------------
# Losses
# -----------------------------

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional class weights.
    gamma: focusing parameter
    alpha: tensor of per-class weights or scalar
    label_smoothing: in [0, 1)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        if self.label_smoothing > 0.0:
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            true_dist = torch.zeros_like(log_probs)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0)

        pt = (probs * true_dist).sum(dim=1)  # p_t
        focal_term = (1 - pt).pow(self.gamma)

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                at = self.alpha.to(logits.device).gather(0, target)
            else:
                at = torch.tensor(self.alpha, device=logits.device)
        else:
            at = 1.0

        loss = -(at * focal_term * (true_dist * log_probs).sum(dim=1))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# -----------------------------
# MixUp
# -----------------------------

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0.0:
        return x, y, 1.0, None
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam, index


def mixup_criterion(criterion, preds, targets, lam):
    y_a, y_b = targets
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


# -----------------------------
# Model Factory
# -----------------------------

def build_model(arch: str, num_classes: int = 7, dropout: float = 0.3, pretrained: bool = True):
    arch = arch.lower()
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, num_classes))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = m.classifier[-1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(in_feats, num_classes)
        )
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = m.classifier[-1].in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, num_classes))
    else:
        raise ValueError(f"Unknown model arch: {arch}")
    return m


# -----------------------------
# Data
# -----------------------------

def build_transforms(img_size: int, erasing_p: float):
    """
    Windows-safe (no lambdas). RandomErasing constructed unconditionally; p controls application frequency.
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=erasing_p)  # p=0.0 => disabled, but still picklable
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return train_tf, eval_tf


def load_datasets(data_root, img_size, erasing_p):
    train_tf, eval_tf = build_transforms(img_size, erasing_p)
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError("Expect subfolders 'train' and 'val' under --data_root.")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=eval_tf)
    test_ds  = datasets.ImageFolder(test_dir,  transform=eval_tf) if os.path.isdir(test_dir) else None

    # Sanity check / helpful message
    expected = ['angry','disgusted','fearful','happy','sad','surprised','neutral']
    ds_classes_lower = [c.lower() for c in train_ds.classes]
    if sorted(ds_classes_lower) != sorted(expected):
        print("Warning: class names differ from the expected seven. Using whatever is in the folders.")
        if "surprise" in ds_classes_lower and "surprised" not in ds_classes_lower:
            print("Hint: You have a folder named 'surprise'. Rename it to 'surprised' in train/val[/test] to align labels.")

    return train_ds, val_ds, test_ds


# -----------------------------
# Metrics / Evaluation
# -----------------------------

def evaluate(model, loader, device, num_classes, return_cm=False):
    from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=True):
                logits = model(x)
            pred = logits.argmax(1)
            ys.append(y.cpu().numpy())
            ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = None
    if return_cm:
        from sklearn.metrics import confusion_matrix as cm_fn
        cm = cm_fn(y_true, y_pred, labels=list(range(num_classes)))
    return acc, macro_f1, cm


# -----------------------------
# Train
# -----------------------------

def train(args):
    set_seed(args.seed)

    # Windows users sometimes benefit from spawn context; default is already spawn on Windows
    if platform.system() == "Windows" and args.workers > 0:
        # Nothing to change; just a reminder that workers>0 uses spawn/pickling
        pass

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds, val_ds, test_ds = load_datasets(args.data_root, args.img_size, args.random_erasing)

    num_classes = len(train_ds.classes)
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Class weights for loss
    train_counts = Counter([y for _, y in train_ds.samples])
    weights = class_weights_from_counts({k: train_counts.get(k, 1) for k in range(num_classes)},
                                        mode="inv_log").to(device)

    # Sampler (optional): either standard shuffle or weighted sampler
    if args.weighted_sampler:
        sample_weights = compute_sample_weights(train_ds)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=4 if args.workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=4 if args.workers > 0 else None
    )
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=(args.workers > 0),
            prefetch_factor=4 if args.workers > 0 else None
        )


    # Model
    model = build_model(args.model, num_classes=num_classes, dropout=args.dropout, pretrained=not args.no_pretrain)
    model.to(device)

    # Freeze backbone for a few warmup epochs if requested
    if args.freeze_epochs > 0:
        for name, p in model.named_parameters():
            p.requires_grad = False
        # unfreeze last linear(s)
        if isinstance(model, models.ResNet):
            for p in model.fc.parameters():
                p.requires_grad = True
        else:
            for p in model.classifier.parameters():
                p.requires_grad = True

    # Optimizer & Scheduler
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=args.weight_decay)
    total_epochs = args.epochs
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, epochs=total_epochs, steps_per_epoch=steps_per_epoch,
        pct_start=args.onecycle_pct_start
    )

    # Loss
    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=weights if args.use_class_weights else None,
                              label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=(weights if args.use_class_weights else None),
            label_smoothing=args.label_smoothing
        )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_f1 = -1.0
    bad_epochs = 0
    history = []

    for epoch in range(total_epochs):
        # Unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs:
            for p in model.parameters():
                p.requires_grad = True

        model.train()
        running_loss = 0.0

        # NEW: live progress bar per epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", ncols=100)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # MixUp
            xb_mixed, yb_mix, lam, _ = mixup_data(xb, yb, alpha=args.mixup_alpha)

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=True):
                logits = model(xb_mixed)
                if args.mixup_alpha > 0.0:
                    loss = mixup_criterion(criterion, logits, yb_mix, lam)
                else:
                    loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * xb.size(0)
            current_lr = optim.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.1e}"})

        # Validation
        val_acc, val_f1, cm = evaluate(model, val_loader, device, num_classes, return_cm=True)
        train_loss = running_loss / len(train_loader.dataset)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
            "lr": float(optim.param_groups[0]["lr"])
        })

        print(f"Epoch {epoch+1}/{total_epochs} | train_loss={train_loss:.4f} | "
              f"val_acc={val_acc:.4f} | val_macroF1={val_f1:.4f}")

        # Checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            bad_epochs = 0
            ckpt = {
                "model_state": model.state_dict(),
                "arch": args.model,
                "num_classes": num_classes,
                "classes": [idx_to_class[i] for i in range(num_classes)],
                "args": vars(args),
                "val_macro_f1": float(val_f1),
                "val_acc": float(val_acc),
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            if cm is not None:
                np.save(os.path.join(args.out_dir, "best_confusion_matrix.npy"), cm)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    # Save training history
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation on test set (if provided)
    if test_loader is not None:
        print("Evaluating best checkpoint on test set...")
        ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_acc, test_f1, test_cm = evaluate(model, test_loader, device, num_classes, return_cm=True)
        print(f"TEST | acc={test_acc:.4f} | macroF1={test_f1:.4f}")
        np.save(os.path.join(args.out_dir, "test_confusion_matrix.npy"), test_cm)
        with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
            json.dump({"test_acc": float(test_acc), "test_macro_f1": float(test_f1)}, f, indent=2)


# -----------------------------
# Argparse
# -----------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train a 7-class FER model (PyTorch).")
    p.add_argument("--data_root", type=str, required=True, help="Path with train/val[/test] subfolders.")
    p.add_argument("--out_dir", type=str, default="./runs/exp1", help="Directory to save checkpoints/metrics.")
    p.add_argument("--model", type=str, default="resnet18",
                   choices=["resnet18", "efficientnet_b0", "mobilenet_v3_small"])
    p.add_argument("--no_pretrain", action="store_true", help="Do not use ImageNet-pretrained weights.")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--freeze_epochs", type=int, default=0, help="Warmup: train head only for N epochs.")
    p.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--onecycle_pct_start", type=float, default=0.3)
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--use_class_weights", type=str, default="true", help="true/false")
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--random_erasing", type=float, default=0.0, help="p for RandomErasing in train tf.")
    p.add_argument("--weighted_sampler", type=str, default="false", help="true/false (oversample minority).")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=7, help="Early stopping patience on macro-F1.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = p.parse_args()

    args.use_class_weights = str2bool(args.use_class_weights)
    args.weighted_sampler = str2bool(args.weighted_sampler)
    return args


# -----------------------------
# Entry
# -----------------------------

if __name__ == "__main__":
    args = get_args()
    train(args)
