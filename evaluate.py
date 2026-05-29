"""
Inference + official CSV output for DCASE 2026 Task 1 submission.

Usage:
  python evaluate.py --audio_dir /data/eval/audio \
                     --model_path results/ConvNeXt_SB_GAPGMP_BST/seed42/best_model.pth \
                     --output_csv submission/Font_UPF_task1_1.output.csv
"""

import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model   import build_model, BST_CLASSES
from dataset import MelSpectrogramTransform, MAX_FRAMES, SAMPLE_RATE, _load_waveform
from metrics import hierarchical_precision_recall_f, top_level_accuracy, second_level_accuracy


# ---------------------------------------------------------------------------
# Eval-only dataset (no labels required)
# ---------------------------------------------------------------------------

class EvalDataset(Dataset):
    """Loads unlabelled audio files from a directory for inference."""
    def __init__(self, audio_dir: str, return_waveform: bool = False):
        self.return_waveform = return_waveform
        self.mel_transform = MelSpectrogramTransform()
        self.files = []
        for root, _, fns in os.walk(audio_dir):
            for fn in fns:
                if fn.lower().endswith((".wav", ".flac", ".ogg", ".mp3")):
                    self.files.append(os.path.join(root, fn))
        self.files.sort()
        print(f"Found {len(self.files)} audio files in {audio_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        sound_id = os.path.splitext(os.path.basename(path))[0]
        waveform = _load_waveform(path)
        
        if self.return_waveform:
            T = waveform.shape[-1]
            # MAX_SAMPLES is 32000 * 30 = 960000. It's imported as MAX_SAMPLES? No, not yet.
            # Let's import MAX_SAMPLES from dataset
            from dataset import MAX_SAMPLES
            if T > MAX_SAMPLES:
                waveform = waveform[:MAX_SAMPLES]
            elif T < MAX_SAMPLES:
                waveform = torch.nn.functional.pad(waveform, (0, MAX_SAMPLES - T))
            return waveform, sound_id

        mel = self.mel_transform(waveform)
        # Pad / truncate
        T = mel.shape[-1]
        if T > MAX_FRAMES:
            mel = mel[..., :MAX_FRAMES]
        elif T < MAX_FRAMES:
            mel = torch.nn.functional.pad(mel, (0, MAX_FRAMES - T))
        return mel, sound_id


def eval_collate(batch):
    items, ids = zip(*batch)
    return torch.stack(items), list(ids)


# ---------------------------------------------------------------------------
# Labelled evaluation (when ground truth CSV is available)
# ---------------------------------------------------------------------------

class LabelledEvalDataset(EvalDataset):
    def __init__(self, csv_path: str, audio_dir: str, return_waveform: bool = False):
        super().__init__(audio_dir, return_waveform=return_waveform)
        df = pd.read_csv(csv_path)
        from model import CLASS_TO_IDX
        # Map the class string to contiguous 0-22 label
        labels = df["class"].map(CLASS_TO_IDX).astype(int)
        self.label_map = dict(zip(df["sound_id"].astype(str), labels))

    def __getitem__(self, idx):
        item, sound_id = super().__getitem__(idx)
        label = self.label_map.get(sound_id, -1)
        return item, sound_id, label


def labelled_collate(batch):
    items, ids, labels = zip(*batch)
    return torch.stack(items), list(ids), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_ids, all_preds, all_scores = [], [], []
    for batch in tqdm(loader, desc="Inference"):
        mels = batch[0].to(device)
        ids  = batch[1]
        logits = model(mels)
        probs  = F.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).cpu().numpy()
        pred_scores = probs.max(dim=1).values.cpu().numpy()
        all_ids.extend(ids)
        all_preds.extend(pred_idx.tolist())
        all_scores.extend(pred_scores.tolist())
    return all_ids, np.array(all_preds), np.array(all_scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="DCASE 2026 Task 1 — inference & evaluation")
    p.add_argument("--audio_dir",  required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_csv", default="output.csv",
                   help="Submission CSV path.")
    p.add_argument("--model_type", type=str, default="convnext",
                   choices=["convnext", "panns"],
                   help="Which architecture to load: 'convnext' or 'panns'.")
    p.add_argument("--csv_path",   default=None,
                   help="If provided, evaluate against ground-truth labels in this CSV.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--lam",        type=float, default=0.75)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(model_type=args.model_type, num_classes=len(BST_CLASSES)).to(device)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded model from {args.model_path}")

    return_waveform = (args.model_type == "panns")

    if args.csv_path:
        ds = LabelledEvalDataset(args.csv_path, args.audio_dir, return_waveform=return_waveform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=labelled_collate, num_workers=args.num_workers)
        all_ids, all_preds, all_scores, all_labels = [], [], [], []
        model.eval()
        with torch.no_grad():
            for inputs, ids, labels in tqdm(loader, desc="Eval"):
                logits = model(inputs.to(device))
                probs  = F.softmax(logits, dim=1)
                preds  = probs.argmax(dim=1).cpu().numpy()
                scores = probs.max(dim=1).values.cpu().numpy()
                all_ids.extend(ids)
                all_preds.extend(preds.tolist())
                all_scores.extend(scores.tolist())
                all_labels.extend(labels.numpy().tolist())

        gt = np.array(all_labels)
        pr = np.array(all_preds)
        valid = gt >= 0
        hmetrics = hierarchical_precision_recall_f(gt[valid], pr[valid], lam=args.lam)
        acc      = second_level_accuracy(gt[valid], pr[valid])
        top_acc  = top_level_accuracy(gt[valid], pr[valid])

        print(f"\n=== Evaluation Results (λ={args.lam}) ===")
        print(f"  hP={hmetrics['hP']:.4f}  hR={hmetrics['hR']:.4f}  hF={hmetrics['hF']:.4f}")
        print(f"  Accuracy (2nd level): {acc:.4f}")
        print(f"  Accuracy (top level): {top_acc:.4f}")
        print("\n  Per-class hF:")
        for cls, val in hmetrics["hF_per_class"].items():
            print(f"    {cls}: {val:.4f}")

        report_path = args.output_csv.replace(".csv", "_report.json")
        with open(report_path, "w") as f:
            json.dump({"overall": hmetrics, "second_level_accuracy": acc,
                       "top_level_accuracy": top_acc}, f, indent=2)
        print(f"\nReport saved → {report_path}")

        ids_out, preds_out, scores_out = all_ids, all_preds, all_scores

    else:
        ds = EvalDataset(args.audio_dir, return_waveform=return_waveform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=eval_collate, num_workers=args.num_workers)
        
        all_ids, all_preds, all_scores = [], [], []
        model.eval()
        with torch.no_grad():
            for inputs, ids in tqdm(loader, desc="Eval"):
                logits = model(inputs.to(device))
                probs  = F.softmax(logits, dim=1)
                preds  = probs.argmax(dim=1).cpu().numpy()
                scores = probs.max(dim=1).values.cpu().numpy()
                all_ids.extend(ids)
                all_preds.extend(preds.tolist())
                all_scores.extend(scores.tolist())
        
        ids_out, preds_out, scores_out = all_ids, all_preds, all_scores

    # Write submission CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    submission = pd.DataFrame({
        "id":                           ids_out,
        "predicted_bst_second_level_class": [BST_CLASSES[i] for i in preds_out],
        "prediction_score":             [round(float(s), 6) for s in scores_out],
    })
    submission.to_csv(args.output_csv, index=False)
    print(f"\nSubmission CSV saved → {args.output_csv}")


if __name__ == "__main__":
    main()
