import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import ClapTextModelWithProjection, ClapProcessor

def extract_text_embeddings(csv_paths, output_dirs, device="cuda"):
    print(f"Loading CLAP Text model on {device}...")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-fused").to(device)
    model.eval()
    
    for csv_path, output_dir in zip(csv_paths, output_dirs):
        print(f"\nProcessing {csv_path} -> {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(csv_path)
        
        batch_size = 64
        
        with torch.no_grad():
            for i in tqdm(range(0, len(df), batch_size)):
                batch_df = df.iloc[i:i+batch_size]
                
                texts = []
                sound_ids = []
                valid_indices = []
                
                for idx, row in batch_df.iterrows():
                    sound_id = str(row['sound_id']).strip()
                    out_file = os.path.join(output_dir, f"{sound_id}.npy")
                    
                    if os.path.exists(out_file):
                        continue
                        
                    title = str(row.get('title', '')).strip()
                    tags = str(row.get('tags', '')).replace(',', ' ').strip()
                    desc = str(row.get('description', '')).strip()
                    
                    text = f"{title}. {tags}. {desc}"[:512]
                    
                    texts.append(text)
                    sound_ids.append(sound_id)
                    valid_indices.append(idx)
                
                if not texts:
                    continue
                    
                inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                text_embeds = outputs.text_embeds.cpu().numpy()
                
                for idx_in_batch, sound_id in enumerate(sound_ids):
                    out_file = os.path.join(output_dir, f"{sound_id}.npy")
                    np.save(out_file, text_embeds[idx_in_batch])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_paths", nargs="+", required=True)
    parser.add_argument("--output_dirs", nargs="+", required=True)
    args = parser.parse_args()
    
    if len(args.csv_paths) != len(args.output_dirs):
        raise ValueError("Number of paths must match")
        
    extract_text_embeddings(
        csv_paths=args.csv_paths,
        output_dirs=args.output_dirs,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
