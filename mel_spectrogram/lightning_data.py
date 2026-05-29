import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
import torch

from mel_spectrogram.dataset import get_kfold_split, bst_collate, SAMPLE_RATE

class BSTDataModule(pl.LightningDataModule):
    def __init__(self, csv_paths, audio_dirs, batch_size=16, num_workers=0, fold=0, n_folds=5, seed=42, confidence_threshold=0.0, crop_secs=None, return_waveform=False):
        super().__init__()
        self.csv_paths = csv_paths if isinstance(csv_paths, list) else [csv_paths]
        self.audio_dirs = audio_dirs if isinstance(audio_dirs, list) else [audio_dirs]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold
        self.n_folds = n_folds
        self.seed = seed
        self.confidence_threshold = confidence_threshold
        self.crop_secs = crop_secs
        self.return_waveform = return_waveform

    def setup(self, stage=None):
        self.train_ds, self.val_ds = get_kfold_split(
            csv_paths=self.csv_paths,
            audio_dirs=self.audio_dirs,
            fold=self.fold,
            n_folds=self.n_folds,
            seed=self.seed,
            confidence_threshold=self.confidence_threshold,
            return_waveform=self.return_waveform,
        )

    def train_dataloader(self):
        crop_frames = int(np.ceil(self.crop_secs * SAMPLE_RATE / 256)) if self.crop_secs else None
        crop_samples = int(self.crop_secs * SAMPLE_RATE) if self.crop_secs else None
        train_collate = partial(bst_collate, crop_frames=crop_frames, crop_samples=crop_samples)
        
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=train_collate, 
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=bst_collate, 
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
