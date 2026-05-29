import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from train.model import BST_CLASSES

class EmbeddingDataset(Dataset):
    def __init__(self, csv_paths, clap_dirs, panns_dirs=None, text_dirs=None, ast_dirs=None, confidence_threshold=0.0):
        super().__init__()
        
        # Load and combine metadata
        dfs = []
        for csv_path, clap_dir in zip(csv_paths, clap_dirs):
            df = pd.read_csv(csv_path)
            
            # Filter low confidence
            if "confidence" in df.columns and confidence_threshold > 0:
                df = df[df["confidence"] >= confidence_threshold]
                
            # Discard top-level/other
            s = df['class_idx'].astype(str)
            df = df[~((s.str.len() == 3) & (s.str.endswith('99') | s.str.endswith('00')))].copy()
            
            # Add CLAP embedding path
            df['clap_path'] = df['sound_id'].astype(str).str.strip().apply(lambda x: os.path.join(clap_dir, f"{x}.npy"))
            
            # Add PANNs embedding path if provided
            if panns_dirs is not None:
                panns_dir = panns_dirs[csv_paths.index(csv_path)]
                df['panns_path'] = df['sound_id'].astype(str).str.strip().apply(lambda x: os.path.join(panns_dir, f"{x}.pt"))
            else:
                df['panns_path'] = None
                
            # Add Text embedding path if provided
            if text_dirs is not None:
                text_dir = text_dirs[csv_paths.index(csv_path)]
                df['text_path'] = df['sound_id'].astype(str).str.strip().apply(lambda x: os.path.join(text_dir, f"{x}.npy"))
            else:
                df['text_path'] = None
                
            # Add AST embedding path if provided
            if ast_dirs is not None:
                ast_dir = ast_dirs[csv_paths.index(csv_path)]
                df['ast_path'] = df['sound_id'].astype(str).str.strip().apply(lambda x: os.path.join(ast_dir, f"{x}.pt"))
            else:
                df['ast_path'] = None
            
            # Keep only existing files
            valid_mask = df['clap_path'].apply(os.path.exists)
            if panns_dirs is not None:
                valid_mask = valid_mask & df['panns_path'].apply(os.path.exists)
            if text_dirs is not None:
                valid_mask = valid_mask & df['text_path'].apply(os.path.exists)
            if ast_dirs is not None:
                valid_mask = valid_mask & df['ast_path'].apply(os.path.exists)
            df = df[valid_mask]
            dfs.append(df)
            
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Build mapping
        self.classes = BST_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # For cross-validation split
        self.original_labels = self.df['class'].map(self.class_to_idx).values
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load CLAP embedding directly from disk (fast)
        emb = np.load(row['clap_path']).astype(np.float32)
        emb = torch.from_numpy(emb)
        
        # Load PANNs embedding if available and concatenate
        if pd.notna(row['panns_path']):
            panns_emb = torch.load(row['panns_path'], weights_only=True)
            emb = torch.cat([emb, panns_emb], dim=0)
            
        # Load Text embedding if available and concatenate
        if pd.notna(row['text_path']):
            text_emb = np.load(row['text_path']).astype(np.float32)
            text_emb = torch.from_numpy(text_emb)
            emb = torch.cat([emb, text_emb], dim=0)
            
        # Load AST embedding if available and concatenate
        if pd.notna(row['ast_path']):
            ast_emb = torch.load(row['ast_path'], weights_only=True).float()
            emb = torch.cat([emb, ast_emb], dim=0)
        
        label = self.class_to_idx[row['class']]
        conf = float(row.get("confidence", 1.0))
        sound_id = str(row.get("sound_id", ""))

        # Return 4-tuple to match BSTLightningModule's (inputs, labels, confidences, _) contract
        return emb, label, conf, sound_id


class AugmentedEmbeddingDataset(Dataset):
    """
    Wrapper that applies Gaussian noise and random dimension masking to
    embeddings at training time. Mirrors the augmentation strategy used
    by the official DCASE 2026 Task 1 baseline (HATR model).
    """
    def __init__(self, subset, noise_std=0.05, mask_prob=0.1):
        self.subset = subset
        self.noise_std = noise_std
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        emb, label, conf, sound_id = self.subset[idx]

        # 1. Gaussian noise
        if self.noise_std > 0:
            emb = emb + self.noise_std * torch.randn_like(emb)

        # 2. Random dimension masking (zero-out random embedding dims)
        if self.mask_prob > 0:
            mask = torch.bernoulli(torch.full_like(emb, 1.0 - self.mask_prob))
            emb = emb * mask

        return emb, label, conf, sound_id

class EmbeddingDataModule(pl.LightningDataModule):
    def __init__(self, csv_paths, clap_dirs, panns_dirs=None, text_dirs=None, ast_dirs=None, batch_size=256, num_workers=4,
                 fold=0, n_folds=5, seed=42, confidence_threshold=0.0,
                 noise_std=0.0, mask_prob=0.0, holdout_ratio=-1.0):
        super().__init__()
        self.csv_paths = csv_paths
        self.clap_dirs = clap_dirs
        self.panns_dirs = panns_dirs
        self.text_dirs = text_dirs
        self.ast_dirs = ast_dirs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold
        self.n_folds = n_folds
        self.seed = seed
        self.confidence_threshold = confidence_threshold
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.holdout_ratio = holdout_ratio  # if >0, bypass K-fold and use a simple split
        
    def setup(self, stage=None):
        from sklearn.model_selection import train_test_split
        full_dataset = EmbeddingDataset(self.csv_paths, self.clap_dirs, self.panns_dirs, self.text_dirs, self.ast_dirs, self.confidence_threshold)

        if self.holdout_ratio > 0:
            # Simple stratified split — use more data for training
            all_idx = np.arange(len(full_dataset))
            train_idx, val_idx = train_test_split(
                all_idx,
                test_size=self.holdout_ratio,
                stratify=full_dataset.original_labels,
                random_state=self.seed,
            )
            print(f"Holdout split: {len(train_idx)} train / {len(val_idx)} val ({self.holdout_ratio*100:.0f}% holdout)")
        else:
            # K-fold cross-validation
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            splits = list(skf.split(np.zeros(len(full_dataset)), full_dataset.original_labels))
            train_idx, val_idx = splits[self.fold]
            print(f"K-fold split (fold {self.fold}/{self.n_folds}): {len(train_idx)} train / {len(val_idx)} val")
        
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        self.val_ds = torch.utils.data.Subset(full_dataset, val_idx)

        # Wrap train split with augmentation if requested
        if self.noise_std > 0 or self.mask_prob > 0:
            self.train_ds = AugmentedEmbeddingDataset(
                train_subset, noise_std=self.noise_std, mask_prob=self.mask_prob
            )
            print(f"Embedding augmentation: noise_std={self.noise_std}, mask_prob={self.mask_prob}")
        else:
            self.train_ds = train_subset
        
        # We need class weights
        train_labels = [full_dataset.original_labels[i] for i in train_idx]
        counts = np.bincount(train_labels, minlength=len(BST_CLASSES))
        self.class_weights = torch.FloatTensor(1.0 / (counts + 1e-6))
        self.class_weights = self.class_weights / self.class_weights.sum() * len(BST_CLASSES)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
