import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from train.model import BST_CLASSES
from train_embeddings import AttentionFusionClassifier
from lightning_module import BSTLightningModule

class InferenceEmbeddingDataset(Dataset):
    """
    Loads embeddings for a blind evaluation set.
    """
    def __init__(self, csv_path, clap_dir, panns_dir=None, text_dir=None, ast_dir=None, wavlm_dir=None):
        super().__init__()
        
        self.df = pd.read_csv(csv_path)
        
        self.df['clap_path'] = self.df['sound_id'].astype(str).str.strip().apply(lambda x: os.path.join(clap_dir, f"{x}.npy"))
        
        if panns_dir is not None:
            self.df['panns_path'] = self.df['sound_id'].astype(str).str.strip().apply(lambda x: os.path.join(panns_dir, f"{x}.pt"))
        else:
            self.df['panns_path'] = None
            
        if text_dir is not None:
            self.df['text_path'] = self.df['sound_id'].astype(str).str.strip().apply(lambda x: os.path.join(text_dir, f"{x}.npy"))
        else:
            self.df['text_path'] = None
            
        if ast_dir is not None:
            self.df['ast_path'] = self.df['sound_id'].astype(str).str.strip().apply(lambda x: os.path.join(ast_dir, f"{x}.pt"))
        else:
            self.df['ast_path'] = None
            
        if wavlm_dir is not None:
            self.df['wavlm_path'] = self.df['sound_id'].astype(str).str.strip().apply(lambda x: os.path.join(wavlm_dir, f"{x}.pt"))
        else:
            self.df['wavlm_path'] = None
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        emb = np.load(row['clap_path']).astype(np.float32)
        emb = torch.from_numpy(emb)
        
        if pd.notna(row['panns_path']):
            panns_emb = torch.load(row['panns_path'], weights_only=True)
            emb = torch.cat([emb, panns_emb], dim=0)
            
        if pd.notna(row['text_path']):
            text_emb = np.load(row['text_path']).astype(np.float32)
            text_emb = torch.from_numpy(text_emb)
            emb = torch.cat([emb, text_emb], dim=0)
            
        if pd.notna(row['ast_path']):
            ast_emb = torch.load(row['ast_path'], weights_only=True).float()
            emb = torch.cat([emb, ast_emb], dim=0)
            
        if pd.notna(row['wavlm_path']):
            wavlm_emb = torch.load(row['wavlm_path'], weights_only=True).float()
            emb = torch.cat([emb, wavlm_emb], dim=0)
            
        sound_id = str(row['sound_id'])
        return emb, sound_id

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing the 5 fold .ckpt files")
    parser.add_argument("--csv_path", type=str, required=True, help="Blind evaluation set metadata CSV")
    
    parser.add_argument("--emb_dir",   type=str, required=True)
    parser.add_argument("--panns_dir", type=str, default=None)
    parser.add_argument("--text_dir",  type=str, default=None)
    parser.add_argument("--ast_dir",   type=str, default=None)
    parser.add_argument("--wavlm_dir", type=str, default=None)
    
    parser.add_argument("--output_csv", type=str, default="submission.csv")
    args = parser.parse_args()

    dataset = InferenceEmbeddingDataset(
        args.csv_path, args.emb_dir, args.panns_dir, args.text_dir, args.ast_dir, args.wavlm_dir
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

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
        dropout=0.0 # eval mode
    )
    
    import glob
    # 3. Find and Load all Checkpoints in Directory (recursively)
    ckpt_files = glob.glob(os.path.join(args.ckpt_dir, "**", "*.ckpt"), recursive=True)
    if not ckpt_files:
        raise ValueError(f"No .ckpt files found in {args.ckpt_dir}")
        
    print(f"Found {len(ckpt_files)} checkpoints for ensembling.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    
    for ckpt in ckpt_files:
        module = BSTLightningModule.load_from_checkpoint(ckpt, model=net)
        model = module.model.to(device)
        model.eval()
        models.append(model)

    # 4. Ensemble Inference Loop
    all_sound_ids = []
    all_preds = []
    
    for emb, sound_id in tqdm(loader, desc="Generating Ensemble Predictions"):
        emb = emb.to(device)
        
        # Average the raw logits across all models
        ensemble_logits = torch.zeros(emb.size(0), len(BST_CLASSES), device=device)
        for model in models:
            ensemble_logits += model(emb)
        ensemble_logits /= len(models)
        
        preds = ensemble_logits.argmax(dim=-1).cpu().numpy()
        
        all_sound_ids.extend(sound_id)
        all_preds.extend(preds)
        
    pred_classes = [BST_CLASSES[p] for p in all_preds]
    
    submission_df = pd.DataFrame({
        "sound_id": all_sound_ids,
        "class": pred_classes
    })
    
    submission_df.to_csv(args.output_csv, index=False)
    print(f"\nSubmission successfully saved to {args.output_csv}")

if __name__ == "__main__":
    main()
