import os
import argparse
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_LOGS"] = "-all"
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
import torch.nn as nn
from train.model import BST_CLASSES

from dataset_embeddings import EmbeddingDataModule
from lightning_module import BSTLightningModule

class AttentionFusionClassifier(nn.Module):
    def __init__(self, use_panns=False, use_text=False, use_ast=False, use_wavlm=False, hidden_dim=512, num_classes=23, dropout=0.3):
        super().__init__()
        self.use_panns = use_panns
        self.use_text = use_text
        self.use_ast = use_ast
        self.use_wavlm = use_wavlm
        
        self.proj_clap = nn.Linear(512, hidden_dim)
        if use_panns:
            self.proj_panns = nn.Linear(2048, hidden_dim)
        if use_text:
            self.proj_text = nn.Linear(512, hidden_dim)
        if use_ast:
            self.proj_ast = nn.Linear(768, hidden_dim)
        if use_wavlm:
            self.proj_wavlm = nn.Linear(1024, hidden_dim)
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        idx = 512
        tokens = [self.proj_clap(x[:, :idx]).unsqueeze(1)]
        
        if self.use_panns:
            tokens.append(self.proj_panns(x[:, idx:idx+2048]).unsqueeze(1))
            idx += 2048
            
        if self.use_text:
            tokens.append(self.proj_text(x[:, idx:idx+512]).unsqueeze(1))
            idx += 512
            
        if self.use_ast:
            tokens.append(self.proj_ast(x[:, idx:idx+768]).unsqueeze(1))
            idx += 768
            
        if self.use_wavlm:
            tokens.append(self.proj_wavlm(x[:, idx:idx+1024]).unsqueeze(1))
            
        seq = torch.cat(tokens, dim=1)
        
        # Pass through Transformer
        seq = self.transformer(seq)
        
        # Mean Pooling over all modalities (Extremely robust regularizer)
        pooled = seq.mean(dim=1)
        
        out = self.act(self.ln1(self.fc1(pooled)))
        out = out + self.res1(out)
        out = self.act(out)
        out = out + self.res2(out)
        out = self.act(out)
        out = self.dropout(out)
        return self.head(out)

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
        if trainer.sanity_checking:
            return
            
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        current_metrics = {"epoch": trainer.current_epoch}
        for k, v in trainer.callback_metrics.items():
            current_metrics[k] = v.item() if hasattr(v, "item") else v
            
        self.metrics_history.append(current_metrics)
        
        with open(self.output_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)

def main():
    torch.set_float32_matmul_precision("medium")
    
    parser = argparse.ArgumentParser(description="PyTorch Lightning Trainer for Pre-computed Embeddings")
    parser.add_argument("--csv_path",   type=str, nargs='+', required=True)
    parser.add_argument("--emb_dir",  type=str, nargs='+', required=True, help="CLAP embeddings directories")
    parser.add_argument("--panns_dir", type=str, nargs='+', default=None, help="PANNs embeddings directories (optional)")
    parser.add_argument("--text_dir", type=str, nargs='+', default=None, help="CLAP text embeddings directories (optional)")
    parser.add_argument("--ast_dir", type=str, nargs='+', default=None, help="AST audio embeddings directories (optional)")
    parser.add_argument("--wavlm_dir", type=str, nargs='+', default=None, help="Paths to WavLM (BEATs) embeddings")
    parser.add_argument("--output_dir", type=str, default="results/Embeddings_Run")
    
    parser.add_argument("--epochs",          type=int,   default=120)
    parser.add_argument("--batch_size",      type=int,   default=256)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--coarse_weight",   type=float, default=0.3)
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--noise_std",        type=float, default=0.0,  help="Gaussian noise std dev for embedding augmentation")
    parser.add_argument("--mask_prob", type=float, default=0.1, help="Probability of zeroing out elements")
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Alpha for Mixup augmentation (e.g. 0.3)")
    parser.add_argument("--holdout_ratio",    type=float, default=-1.0, help="If >0, bypass K-fold and use this fraction as holdout val set (e.g. 0.1 = 90/10 split)")
    
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--fold",                 type=int,   default=0)
    parser.add_argument("--num_workers",          type=int,   default=4)
    
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    datamodule = EmbeddingDataModule(
        csv_paths=args.csv_path,
        clap_dirs=args.emb_dir,
        panns_dirs=args.panns_dir,
        text_dirs=args.text_dir,
        ast_dirs=args.ast_dir,
        wavlm_dirs=args.wavlm_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fold=args.fold,
        seed=args.seed,
        noise_std=args.noise_std,
        mask_prob=args.mask_prob,
        holdout_ratio=args.holdout_ratio,
    )
    
    datamodule.setup()
    class_weights = datamodule.class_weights

    use_panns = args.panns_dir is not None
    use_text = args.text_dir is not None
    use_ast = args.ast_dir is not None
    use_wavlm = args.wavlm_dir is not None
        
    net = AttentionFusionClassifier(
        use_panns=use_panns, 
        use_text=use_text, 
        use_ast=use_ast,
        use_wavlm=use_wavlm,
        hidden_dim=512, 
        num_classes=len(BST_CLASSES), 
        dropout=args.dropout
    )
    print(f"Trainable Parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

    module = BSTLightningModule(
        model=net,
        lr=args.lr,
        spec_augment=False,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing,
        class_weights=class_weights,
        coarse_weight=args.coarse_weight,
    )

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
        logger=True,
    )

    trainer.fit(model=module, datamodule=datamodule)

if __name__ == "__main__":
    main()
