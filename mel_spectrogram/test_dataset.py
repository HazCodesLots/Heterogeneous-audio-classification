import sys
import pandas as pd
import numpy as np
from model import BST_CLASSES, CLASS_TO_IDX

def run_smoke_test():
    csv_bsd10k = "C:/Users/HazCodes/Documents/Datasets/DCASE/19868804/metadata/BSD10k_metadata.csv"
    csv_bsd35k = "C:/Users/HazCodes/Documents/Datasets/DCASE/19187100/metadata/BSD35k-CS_metadata.csv"

    print("--- DCASE 2026 Dataset Loading Smoke Test ---")
    
    for name, path in [("BSD10k-v1.2", csv_bsd10k), ("BSD35k-CS", csv_bsd35k)]:
        try:
            df = pd.read_csv(path)
            # Filter to known classes
            df = df[df["class"].isin(CLASS_TO_IDX)].copy()
            # Remap to contiguous labels
            df["label"] = df["class"].map(CLASS_TO_IDX).astype(int)

            has_conf = "confidence" in df.columns and df["confidence"].notna().any()
            counts   = np.bincount(df["label"].values, minlength=23).astype(float)
            weights  = 1.0 / np.maximum(counts, 1.0)
            weights  = weights / weights.sum() * 23

            print(f"\n{name}:")
            print(f"  Rows retained : {len(df):,}")
            print(f"  Label range   : {df['label'].min()} - {df['label'].max()}")
            print(f"  Unique classes: {df['class'].nunique()} / 23")
            print(f"  Has confidence: {has_conf}")
            print(f"  Weight vector (first 5): {weights[:5].round(4)}")
        except Exception as e:
            print(f"\n{name}: FAILED to load ({e})")

    print("\n--- Model Constants Check ---")
    print("CLASS_TO_IDX sample:", list(CLASS_TO_IDX.items())[:5])
    print("All 23 classes covered:", len(CLASS_TO_IDX) == 23)

if __name__ == "__main__":
    run_smoke_test()
