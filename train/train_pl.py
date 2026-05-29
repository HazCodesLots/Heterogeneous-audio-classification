import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import warnings

# Suppress benign internal PyTorch Lightning warnings to keep the console clean
warnings.filterwarnings("ignore", ".*isinstance(treespec, LeafSpec).*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Precision 16-mixed is not supported.*")

from model import build_model, BST_CLASSES
from mel-spectrogram.lightning_data import BSTDataModule
from lightning_module import BSTLightningModule
import json

class JSONMetricsCallback(pl.Callback):
    """
    Automatically saves all training and validation metrics to a clean,
    dynamically updating JSON file at the end of every epoch.
    """
    def __init__(self, output_dir, fold):
        super().__init__()
        self.output_file = os.path.join(output_dir, f"fold{fold}", "metrics.json")
        self.metrics_history = []
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # PyTorch Lightning does a 'sanity check' validation before training starts. We ignore it.
        if trainer.sanity_checking:
            return
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Grab all accumulated metrics (train loss, val loss, hF, learning rate, etc)
        current_metrics = {"epoch": trainer.current_epoch}
        for k, v in trainer.callback_metrics.items():
            # Convert tensors to standard Python floats for JSON serialization
            current_metrics[k] = v.item() if hasattr(v, "item") else v
            
        self.metrics_history.append(current_metrics)
        
        # Overwrite the JSON file with the newly updated list
        with open(self.output_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)

def main():
    import torch
    torch.set_float32_matmul_precision("medium")
    
    parser = argparse.ArgumentParser(description="PyTorch Lightning Trainer for DCASE 2026")
    parser.add_argument("--csv_path",   type=str, nargs='+', required=True)
    parser.add_argument("--audio_dir",  type=str, nargs='+', required=True)
    parser.add_argument("--output_dir", type=str, default="results/Lightning_Run")
    parser.add_argument("--model_type", type=str, default="convnext", choices=["convnext", "panns", "clap"])
    
    parser.add_argument("--epochs",          type=int,   default=120)
    parser.add_argument("--warmup_epochs",   type=int,   default=10)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup_alpha",     type=float, default=0.0)
    parser.add_argument("--crop_secs",       type=float, default=10.0)
    
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--n_folds",              type=int,   default=5)
    parser.add_argument("--fold",                 type=int,   default=0)
    parser.add_argument("--num_workers",          type=int,   default=0)
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    
    parser.add_argument("--unfreeze_backbone", action="store_true", help="Fine-tune entire backbone")
    parser.add_argument("--spec_augment",     action="store_true", help="Apply SpecAugment time masking during training")
    parser.add_argument("--coarse_weight",    type=float, default=0.3, help="Weight for the auxiliary coarse-level loss (0=disabled)")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Setup DataModule
    return_waveform = (args.model_type in ["panns", "clap"])
    datamodule = BSTDataModule(
        csv_paths=args.csv_path,
        audio_dirs=args.audio_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fold=args.fold,
        n_folds=args.n_folds,
        seed=args.seed,
        confidence_threshold=args.confidence_threshold,
        crop_secs=args.crop_secs,
        return_waveform=return_waveform
    )
    
    # We must explicitly call setup to access the train dataset for class weights
    datamodule.setup()
    class_weights = datamodule.train_ds.class_weights()

    # 2. Build Model Architecture
    net = build_model(
        model_type=args.model_type,
        num_classes=len(BST_CLASSES),
        freeze_backbone=not args.unfreeze_backbone
    )
    print(f"Trainable Parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

    # 3. Setup Lightning Module
    module = BSTLightningModule(
        model=net,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing,
        class_weights=class_weights,
        spec_augment=args.spec_augment,
        coarse_weight=args.coarse_weight,
    )

    # 4. Callbacks & Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, f"fold{args.fold}"),
        filename="best_model-{epoch:02d}-{val_hF:.4f}",
        save_top_k=1,
        monitor="val_hF",
        mode="max"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    json_logger = JSONMetricsCallback(output_dir=args.output_dir, fold=args.fold)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto", 
        devices="auto",
        callbacks=[checkpoint_callback, lr_monitor, json_logger],
        gradient_clip_val=1.0,
        precision="16-mixed",
        # Enable default TensorBoard logger (could easily swap for WandbLogger)
        logger=True,
    )

    # 5. Train
    trainer.fit(
        model=module, 
        datamodule=datamodule,
        ckpt_path=args.resume_checkpoint
    )

if __name__ == "__main__":
    main()
