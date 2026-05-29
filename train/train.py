"""
Training script — ConvNeXt-SplitBand-GAPGMP for DCASE 2026 Task 1.

Usage:
  python train.py --audio_dir /data/bsd10k/audio \
                  --csv_path  /data/bsd10k/metadata.csv \
                  --epochs 80 --batch_size 16 --seed 42

To use BSD35k-CS or both datasets together, pass multiple --csv_path / --audio_dir
arguments (see --help).
"""

import os, json, argparse
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from model   import build_model, BST_CLASSES
from dataset import BSTDataset, bst_collate, get_kfold_split, DEFAULT_NUM_WORKERS, SAMPLE_RATE
from metrics import hierarchical_precision_recall_f, top_level_accuracy, second_level_accuracy


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def mixup_batch(mels: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4):
    """Apply Mixup to a batch of mel spectrograms.

    Returns:
        mixed_mels  : linearly interpolated mel tensors
        labels_a    : original labels
        labels_b    : shuffled labels (mix target)
        lam         : mixing coefficient sampled from Beta(alpha, alpha)
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(mels.size(0), device=mels.device)
    return lam * mels + (1 - lam) * mels[idx], labels, labels[idx], lam


def mixup_criterion(criterion, logits, labels_a, labels_b, lam):
    """Compute mixed cross-entropy loss: λ·CE(a) + (1-λ)·CE(b)."""
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch_train(model, loader, criterion, optimizer, scaler, device, clip_norm,
                    mixup_alpha: float = 0.0):
    model.train()
    total_loss = 0.0
    skipped = 0
    for inputs, labels, confidences, _ in tqdm(loader, desc="  train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            if mixup_alpha > 0.0:
                inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels, alpha=mixup_alpha)
                logits = model(inputs)
                loss   = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                logits = model(inputs)
                loss   = criterion(logits, labels)
        # Guard: skip corrupt batches rather than letting NaN infect model weights
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad()
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    if skipped:
        print(f"  ⚠  Skipped {skipped} NaN/Inf batches this epoch.")
    torch.cuda.empty_cache()
    n_valid = len(loader) - skipped
    return total_loss / max(n_valid, 1)


@torch.no_grad()
def run_epoch_eval(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for inputs, labels, _, _ in tqdm(loader, desc="  eval ", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs)
            total_loss += criterion(logits, labels).item()
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    torch.cuda.empty_cache()
    return total_loss / len(loader), preds, labels


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {args.seed}")

    # ---- Dataset (stratified k-fold split) ---------------------------------
    csv_paths  = args.csv_path  if isinstance(args.csv_path,  list) else [args.csv_path]
    audio_dirs = args.audio_dir if isinstance(args.audio_dir, list) else [args.audio_dir]

    return_waveform = (args.model_type == "panns")
    train_ds, val_ds = get_kfold_split(
        csv_paths  = csv_paths,
        audio_dirs = audio_dirs,
        fold       = args.fold,
        n_folds    = args.n_folds,
        seed       = args.seed,
        confidence_threshold = args.confidence_threshold,
        return_waveform = return_waveform,
    )

    # Crop frames for training (reduces VRAM; val always uses full clip)
    crop_frames = int(np.ceil(args.crop_secs * SAMPLE_RATE / 256)) if args.crop_secs else None
    crop_samples = int(args.crop_secs * SAMPLE_RATE) if args.crop_secs else None
    train_collate = partial(bst_collate, crop_frames=crop_frames, crop_samples=crop_samples)
    val_collate   = bst_collate   # full 30s for validation

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=train_collate,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=val_collate,
                              num_workers=args.num_workers, pin_memory=True)

    # ---- Model ----
    model = build_model(
        model_type=args.model_type,
        num_classes=len(BST_CLASSES),
        freeze_backbone=not args.unfreeze_backbone
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Class weights derived from the training fold only
    weights = train_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    # AMP scaler
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6 / args.lr,
        end_factor=1.0, total_iters=args.warmup_epochs
    )

    if args.sched_mode == "plateau":
        # Halve LR when hF stops improving — responds directly to oscillation
        post_warmup_sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=args.lr_patience, min_lr=1e-6
        )
    else:
        # Single cosine cycle over remaining epochs (original behaviour)
        post_warmup_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-6
        )

    warmup_scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched,
                    optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)],
        milestones=[args.warmup_epochs]
    )

    # ---- Output dirs ----
    out_dir = os.path.join(args.output_dir, f"seed{args.seed}_fold{args.fold}")
    os.makedirs(out_dir, exist_ok=True)
    best_path   = os.path.join(out_dir, "best_model.pth")
    ckpt_path   = os.path.join(out_dir, "checkpoint_last.pth")
    metrics_path= os.path.join(out_dir, "metrics.json")

    start_epoch = 0
    best_hf     = 0.0
    history     = {"train_loss": [], "val_loss": [], "val_hF": [], "val_acc": [], "val_top_acc": []}

    # ---- Resume ----
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        warmup_scheduler.load_state_dict(ckpt["warmup_scheduler"])
        post_warmup_sched.load_state_dict(ckpt["post_warmup_sched"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])   # restore loss scale
        start_epoch = ckpt["epoch"] + 1
        best_hf     = ckpt["best_hf"]
        history     = ckpt["history"]
        print(f"Resumed from epoch {start_epoch}. Best hF: {best_hf:.4f}")

    # ---- Training ----
    for epoch in range(start_epoch, args.epochs):
        # Smooth clip_norm: 0.1 during warmup, linearly ramps to 1.0 over 10 epochs post-warmup
        ramp_epochs = 10
        if epoch < args.warmup_epochs:
            clip_norm = 0.1
        elif epoch < args.warmup_epochs + ramp_epochs:
            t = (epoch - args.warmup_epochs) / ramp_epochs
            clip_norm = 0.1 + 0.9 * t
        else:
            clip_norm = 1.0
        print(f"\nEpoch {epoch+1}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}")

        train_loss = run_epoch_train(model, train_loader, criterion, optimizer, scaler,
                                     device, clip_norm, mixup_alpha=args.mixup_alpha)
        val_loss, preds, labels = run_epoch_eval(model, val_loader, criterion, device)

        hmetrics = hierarchical_precision_recall_f(labels, preds, lam=0.75)
        acc      = second_level_accuracy(labels, preds)
        top_acc  = top_level_accuracy(labels, preds)

        print(f"  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")
        print(f"  hP={hmetrics['hP']:.4f}  hR={hmetrics['hR']:.4f}  hF={hmetrics['hF']:.4f}")
        print(f"  Acc(2nd)={acc:.4f}  Acc(top)={top_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_hF"].append(hmetrics["hF"])
        history["val_acc"].append(acc)
        history["val_top_acc"].append(top_acc)

        if hmetrics["hF"] > best_hf:
            best_hf = hmetrics["hF"]
            torch.save(model.state_dict(), best_path)
            print(f"  >>> New best hF: {best_hf:.4f}  (saved)")

        # Step schedulers
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        elif args.sched_mode == "plateau":
            post_warmup_sched.step(hmetrics["hF"])   # ReduceLROnPlateau needs the metric
        else:
            post_warmup_sched.step()                  # CosineAnnealingLR steps on epoch
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "warmup_scheduler": warmup_scheduler.state_dict(),
            "post_warmup_sched": post_warmup_sched.state_dict(),
            "scaler": scaler.state_dict(),
            "best_hf": best_hf, "history": history,
        }, ckpt_path)

        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best hF: {best_hf:.4f}")
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_all_folds(args):
    """Run all n_folds folds sequentially and report the average hF."""
    fold_results = []
    for fold in range(args.n_folds):
        print(f"\n{'='*60}")
        print(f" FOLD {fold+1} / {args.n_folds}")
        print(f"{'='*60}")
        args.fold = fold
        history = train(args)
        best_hf = max(history["val_hF"])
        fold_results.append(best_hf)
        print(f"  Fold {fold+1} best hF: {best_hf:.4f}")

    mean_hf = np.mean(fold_results)
    std_hf  = np.std(fold_results)
    print(f"\n{'='*60}")
    print(f" {args.n_folds}-Fold CV Results")
    print(f"{'='*60}")
    for i, h in enumerate(fold_results):
        print(f"  Fold {i+1}: hF={h:.4f}")
    print(f"  Mean hF: {mean_hf:.4f} ± {std_hf:.4f}")

    summary = {
        "fold_hF": fold_results,
        "mean_hF": mean_hf,
        "std_hF":  std_hf,
        "n_folds": args.n_folds,
        "seed":    args.seed,
    }
    summary_path = os.path.join(args.output_dir, "cv_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved → {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train ConvNeXt-SplitBand-GAPGMP on DCASE 2026 Task 1")
    p.add_argument("--csv_path",   nargs="+", required=True,
                   help="Path(s) to metadata CSV (BSD10k and/or BSD35k).")
    p.add_argument("--audio_dir",  nargs="+", required=True,
                   help="Root audio dir(s) corresponding to each CSV.")
    p.add_argument("--output_dir", default="results/ConvNeXt_SB_GAPGMP_BST")
    p.add_argument("--model_type", type=str, default="convnext",
                   choices=["convnext", "panns"],
                   help="Which architecture to train: 'convnext' (scratch) or 'panns' (pre-trained).")
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--unfreeze_backbone", action="store_true", 
                        help="If set, fine-tunes the entire pre-trained backbone.")
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--confidence_threshold", type=float, default=0.0,
                   help="Drop BSD10k samples below this confidence score (0 = keep all).")
    p.add_argument("--crop_secs", type=float, default=10.0,
                   help="Randomly crop training clips to this many seconds. 0 = use full 30s clip.")
    p.add_argument("--num_workers",  type=int,   default=DEFAULT_NUM_WORKERS,
                   help=f"DataLoader workers (default: {DEFAULT_NUM_WORKERS}, 0 on Windows).")
    p.add_argument("--mixup_alpha",  type=float, default=0.4,
                   help="Mixup alpha parameter. 0 = disabled. (default: 0.4)")
    p.add_argument("--sched_mode",   type=str,   default="plateau",
                   choices=["plateau", "cosine"],
                   help="LR schedule after warmup: 'plateau' (ReduceLROnPlateau, default) "
                        "or 'cosine' (CosineAnnealingLR).")
    p.add_argument("--lr_patience",  type=int,   default=5,
                   help="Epochs without hF improvement before LR is halved "
                        "(only used with --sched_mode plateau, default: 5).")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--resume",       action="store_true")
    # K-fold CV args
    p.add_argument("--fold",   type=int, default=0,
                   help="Which fold to use as validation (0-indexed). Default: 0.")
    p.add_argument("--n_folds",type=int, default=5,
                   help="Total number of folds. Default: 5.")
    p.add_argument("--all_folds", action="store_true",
                   help="Run all n_folds folds sequentially and report CV average hF.")
    args = p.parse_args()

    if len(args.csv_path) != len(args.audio_dir):
        p.error("Number of --csv_path and --audio_dir arguments must match.")

    if args.all_folds:
        run_all_folds(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
