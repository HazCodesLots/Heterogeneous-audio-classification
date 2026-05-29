"""
Dataset loader for DCASE 2026 Task 1 — Heterogeneous Audio Classification.
Audio backend: torchaudio (faster, lower RAM, no num_workers issues on Windows).

CSV schema: sound_id, class, class_idx, class_top, confidence, uploader,
            license, title, tags, description
NOTE: class_idx is non-contiguous (101-504). Labels are remapped to 0-22
      via CLASS_TO_IDX using the 'class' string column.
"""

import os
import platform
import torch
import torch.nn.functional as F
import torchaudio.transforms as TA
import soundfile as sf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from train.model import BST_CLASSES, CLASS_TO_IDX

# On Windows, num_workers > 0 requires __main__ guard and causes RAM/deadlock issues.
DEFAULT_NUM_WORKERS = 0 if platform.system() == "Windows" else 4

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DURATION_S = 30.0
SAMPLE_RATE    = 32_000
MAX_SAMPLES    = int(MAX_DURATION_S * SAMPLE_RATE)             # 960,000 samples
MAX_FRAMES     = int(np.ceil(MAX_DURATION_S * SAMPLE_RATE / 256))  # 3,750 mel frames
MIN_SAMPLES    = 1024   # n_fft — clips shorter than this get zero-padded

# ---------------------------------------------------------------------------
# Waveform loader (torchaudio)
# ---------------------------------------------------------------------------

def _load_waveform(path: str) -> torch.Tensor:
    """Load audio → mono → resample to SAMPLE_RATE → truncate to MAX_SAMPLES.

    Uses soundfile for I/O (WAV/FLAC; works on all platforms with torchaudio 2.x)
    and torchaudio.functional.resample for resampling (pure tensor, no backend).

    Returns:
        (T,) float32 CPU tensor, MIN_SAMPLES <= T <= MAX_SAMPLES.
    """
    # soundfile returns (T,) for mono, (T, C) for multi-channel; always float64
    data, sr = sf.read(path, dtype="float32", always_2d=False)

    waveform = torch.from_numpy(data)          # (T,) or (T, C)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)        # (T,)  mono-mix

    # Resample using torchaudio functional (no backend required)
    if sr != SAMPLE_RATE:
        import torchaudio.functional as TAF
        waveform = TAF.resample(waveform.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)

    # Truncate to max 30 s
    waveform = waveform[:MAX_SAMPLES]

    # Pad clips shorter than n_fft (avoids mel computation warning)
    if waveform.shape[0] < MIN_SAMPLES:
        waveform = F.pad(waveform, (0, MIN_SAMPLES - waveform.shape[0]))

    return waveform  # (T,)


# ---------------------------------------------------------------------------
# Audio transforms
# ---------------------------------------------------------------------------

class MelSpectrogramTransform:
    """Log-mel spectrogram using torchaudio (pure tensor, no numpy)."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_fft: int = 1024,
                 hop_length: int = 256, n_mels: int = 128,
                 fmin: float = 20.0, fmax: float = 16_000.0,
                 top_db: float = 80.0, eps: float = 1e-6):
        self.sample_rate = sample_rate
        self.eps = eps
        self.mel   = TA.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0,
        )
        self.to_db = TA.AmplitudeToDB(stype="power", top_db=top_db)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (T,) → returns (1, n_mels, T_frames) normalised."""
        mel_db = self.to_db(self.mel(waveform.unsqueeze(0)))  # (1, F, T)
        # Clamp before normalisation: prevents -inf from silent clips
        # turning (mel - mean) / std into NaN via (-inf - -inf)
        mel_db = mel_db.clamp(min=-80.0)
        return (mel_db - mel_db.mean()) / (mel_db.std() + self.eps)


class SpecAugment:
    """Time and frequency masking applied per-sample."""

    def __init__(self, time_mask_param: int = 50, freq_mask_param: int = 20,
                 num_time_masks: int = 2, num_freq_masks: int = 2):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks  = num_time_masks
        self.num_freq_masks  = num_freq_masks

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        _, F, T = mel.shape
        mel = mel.clone()
        for _ in range(self.num_freq_masks):
            f  = np.random.randint(0, max(self.freq_mask_param, 1))
            f0 = np.random.randint(0, max(F - f, 1))
            mel[:, f0:f0 + f, :] = 0
        for _ in range(self.num_time_masks):
            t  = np.random.randint(0, max(self.time_mask_param, 1))
            t0 = np.random.randint(0, max(T - t, 1))
            mel[:, :, t0:t0 + t] = 0
        return mel


class WaveformAugment:
    """Waveform-level augmentations applied before mel extraction (train only).

    - Random gain:  uniformly samples ±gain_db dB, simulating mic level variation.
    - Gaussian noise: additive white noise at a target SNR (noise_snr_db).
    """

    def __init__(self, gain_db: float = 6.0, noise_snr_db: float = 30.0):
        self.gain_db      = gain_db
        self.noise_snr_db = noise_snr_db

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # Random gain ±gain_db dB
        gain = 10 ** (np.random.uniform(-self.gain_db, self.gain_db) / 20.0)
        waveform = waveform * float(gain)

        # Additive Gaussian noise at noise_snr_db SNR
        signal_power = waveform.pow(2).mean().clamp(min=1e-9)
        noise_power  = signal_power / (10 ** (self.noise_snr_db / 10.0))
        waveform     = waveform + torch.randn_like(waveform) * noise_power.sqrt()

        return waveform


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BSTDataset(Dataset):
    """
    Loads BSD10k-v1.2 or BSD35k-CS for DCASE 2026 Task 1.

    Labels are remapped from the non-contiguous CSV class_idx (101-504)
    to contiguous 0-22 integers via CLASS_TO_IDX.
    """

    def __init__(self, csv_path: str, audio_dir: str,
                 augment: bool = False,
                 confidence_threshold: float = 0.0,
                 limit: int | None = None,
                 return_waveform: bool = False):
        self.audio_dir = audio_dir
        self.augment   = augment
        self.return_waveform = return_waveform

        df = pd.read_csv(csv_path)
        missing = {"sound_id", "class", "class_top"} - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        unknown = set(df["class"].unique()) - set(CLASS_TO_IDX)
        if unknown:
            print(f"  ⚠ Dropping {df['class'].isin(unknown).sum():,} rows "
                  f"with unrecognised classes: {unknown}")
        df = df[df["class"].isin(CLASS_TO_IDX)].copy()
        df["label"] = df["class"].map(CLASS_TO_IDX).astype(int)

        if confidence_threshold > 0.0 and "confidence" in df.columns:
            mask = df["confidence"].isna() | (df["confidence"] >= confidence_threshold)
            dropped = (~mask).sum()
            if dropped:
                print(f"  Confidence filter: dropping {dropped:,} rows below {confidence_threshold}")
            df = df[mask].copy()

        if limit is not None:
            df = df.head(limit)

        self.df = df.reset_index(drop=True)

        print(f"Building audio file index in {audio_dir} …")
        self.file_map: dict[str, str] = {}
        for root, _, files in os.walk(audio_dir):
            for fn in files:
                if fn.lower().endswith((".wav", ".flac", ".ogg", ".mp3")):
                    self.file_map[fn] = os.path.join(root, fn)
        print(f"  Index built: {len(self.file_map):,} audio files found.")
        print(f"  Dataset size: {len(self.df):,} samples across "
              f"{self.df['class'].nunique()} classes.")

        self.mel_transform   = MelSpectrogramTransform()
        self.spec_augment    = SpecAugment()
        self.wave_augment    = WaveformAugment()
        self._has_confidence = (
            "confidence" in self.df.columns and self.df["confidence"].notna().any()
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        sound_id = str(row["sound_id"])
        label    = int(row["label"])
        confidence = (
            float(row["confidence"]) if self._has_confidence and pd.notna(row.get("confidence", np.nan))
            else 1.0
        )

        waveform = _load_waveform(self._find_audio(sound_id))
        if self.augment:
            waveform = self.wave_augment(waveform)  # gain + noise before mel

        if self.return_waveform:
            return waveform, label, confidence, sound_id

        mel = self.mel_transform(waveform)
        if self.augment:
            mel = self.spec_augment(mel)            # freq/time masking after mel

        return mel, label, confidence, sound_id

    def _find_audio(self, sound_id: str) -> str:
        for ext in (".wav", ".flac", ".ogg", ".mp3"):
            key = f"{sound_id}{ext}"
            if key in self.file_map:
                return self.file_map[key]
        raise FileNotFoundError(
            f"No audio file found for sound_id={sound_id} in {self.audio_dir}"
        )

    def class_weights(self) -> torch.Tensor:
        counts  = np.bincount(self.df["label"].values, minlength=len(BST_CLASSES)).astype(float)
        counts  = np.maximum(counts, 1.0)
        weights = 1.0 / counts
        return torch.tensor(weights / weights.sum() * len(BST_CLASSES), dtype=torch.float32)

    def class_distribution(self) -> dict[str, int]:
        return self.df.groupby("class")["label"].count().to_dict()


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def bst_collate(batch, max_frames: int = MAX_FRAMES, crop_frames: int | None = None,
                max_samples: int = MAX_SAMPLES, crop_samples: int | None = None):
    """Pad / truncate mel spectrograms or waveforms to max length.

    Args:
        crop_frames: Randomly crop Mels.
        crop_samples: Randomly crop waveforms.
    """
    items, labels, confidences, ids = zip(*batch)
    padded = []
    is_waveform = items[0].ndim == 1

    for item in items:
        T = item.shape[-1]
        target_max = max_samples if is_waveform else max_frames
        crop_len = crop_samples if is_waveform else crop_frames

        if T > target_max:
            item = item[..., :target_max]
        elif T < target_max:
            item = F.pad(item, (0, target_max - T))
        
        if crop_len is not None and item.shape[-1] > crop_len:
            t0 = np.random.randint(0, item.shape[-1] - crop_len + 1)
            item = item[..., t0:t0 + crop_len]
            
        padded.append(item)
    return (
        torch.stack(padded),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(confidences, dtype=torch.float32),
        list(ids),
    )


# ---------------------------------------------------------------------------
# K-fold cross-validation split
# ---------------------------------------------------------------------------

class _SubsetBSTDataset(BSTDataset):
    """BSTDataset constructed from a pre-filtered DataFrame (no disk walk)."""

    def __init__(self, df: pd.DataFrame, audio_dirs: list[str],
                 augment: bool, has_confidence: bool, return_waveform: bool = False):
        object.__init__(self)
        self.df              = df.reset_index(drop=True)
        self.audio_dir       = audio_dirs[0]   # kept for API compat
        self.audio_dirs      = audio_dirs       # all dirs (used in error messages)
        self.augment         = augment
        self._has_confidence = has_confidence
        self.return_waveform = return_waveform
        self.mel_transform   = MelSpectrogramTransform()
        self.spec_augment    = SpecAugment()
        self.wave_augment    = WaveformAugment()
        self.file_map: dict[str, str] = {}

    def _find_audio(self, sound_id: str) -> str:
        for ext in (".wav", ".flac", ".ogg", ".mp3"):
            key = f"{sound_id}{ext}"
            if key in self.file_map:
                return self.file_map[key]
        raise FileNotFoundError(
            f"No audio file found for sound_id={sound_id} "
            f"in dirs: {self.audio_dirs}"
        )


def get_kfold_split(
    csv_paths: list[str],
    audio_dirs: list[str],
    fold: int,
    n_folds: int = 5,
    seed: int = 42,
    confidence_threshold: float = 0.0,
    return_waveform: bool = False,
) -> tuple["BSTDataset", "BSTDataset"]:
    """
    Stratified k-fold train/val split — DCASE 2026 baseline CV protocol.
    Walks the audio directory only once regardless of n_folds.
    """
    assert 0 <= fold < n_folds, f"fold must be in [0, {n_folds - 1}]"

    frames = []
    for csv_p, aud_d in zip(csv_paths, audio_dirs):
        df = pd.read_csv(csv_p)
        df = df[df["class"].isin(CLASS_TO_IDX)].copy()
        df["label"] = df["class"].map(CLASS_TO_IDX).astype(int)
        if confidence_threshold > 0.0 and "confidence" in df.columns:
            mask = df["confidence"].isna() | (df["confidence"] >= confidence_threshold)
            df = df[mask].copy()
        df["_audio_dir"] = aud_d
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_idx, val_idx = list(skf.split(merged, merged["label"]))[fold]
    train_df = merged.iloc[train_idx].copy()
    val_df   = merged.iloc[val_idx].copy()

    has_conf = (
        "confidence" in merged.columns and merged["confidence"].notna().any()
    )

    print(f"[Fold {fold + 1}/{n_folds}] Building shared audio index …")
    shared_file_map: dict[str, str] = {}
    for aud_d in audio_dirs:
        for root, _, files in os.walk(aud_d):
            for fn in files:
                if fn.lower().endswith((".wav", ".flac", ".ogg", ".mp3")):
                    shared_file_map[fn] = os.path.join(root, fn)
    print(f"  {len(shared_file_map):,} files indexed. "
          f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    train_ds = _SubsetBSTDataset(train_df, audio_dirs, augment=True,  has_confidence=has_conf, return_waveform=return_waveform)
    val_ds   = _SubsetBSTDataset(val_df,   audio_dirs, augment=False, has_confidence=has_conf, return_waveform=return_waveform)
    train_ds.file_map = shared_file_map
    val_ds.file_map   = shared_file_map

    return train_ds, val_ds
