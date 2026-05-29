import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import torchaudio.functional as F_audio
from transformers import ASTModel, ASTFeatureExtractor

def extract_ast(csv_paths, audio_dirs, output_dirs, device="cuda"):
    print(f"Loading AST model on {device}...")
    # Load AST Feature Extractor and Model
    checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = ASTFeatureExtractor.from_pretrained(checkpoint)
    model = ASTModel.from_pretrained(checkpoint).to(device)
    model.eval()
    
    for csv_path, audio_dir, output_dir in zip(csv_paths, audio_dirs, output_dirs):
        print(f"\nProcessing {csv_path} -> {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(csv_path)
        
        with torch.no_grad():
            for _, row in tqdm(df.iterrows(), total=len(df)):
                sound_id = str(row['sound_id']).strip()
                out_file = os.path.join(output_dir, f"{sound_id}.pt")
                
                if os.path.exists(out_file):
                    continue
                    
                audio_path = os.path.join(audio_dir, f"{sound_id}.wav")
                if not os.path.exists(audio_path):
                    print(f"Missing {audio_path}")
                    continue
                    
                # Load audio
                try:
                    wav, sr = sf.read(audio_path, dtype='float32')
                    # AST expects mono audio
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                        
                    wav_tensor = torch.from_numpy(wav)
                    
                    # AST strictly expects 16kHz audio
                    if sr != 16000:
                        wav_tensor = F_audio.resample(wav_tensor, sr, 16000)
                        
                    # AST works best with standard ~10 second chunks, but handles variable length
                    inputs = feature_extractor(wav_tensor.numpy(), sampling_rate=16000, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Extract embeddings
                    outputs = model(**inputs)
                    
                    # The pooler_output (or mean of last hidden state) represents the entire clip
                    # Shape: (1, 768)
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                    
                    # Save tensor
                    torch.save(emb.cpu(), out_file)
                    
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_paths", nargs="+", required=True)
    parser.add_argument("--audio_dirs", nargs="+", required=True)
    parser.add_argument("--output_dirs", nargs="+", required=True)
    args = parser.parse_args()
    
    if len(args.csv_paths) != len(args.output_dirs) or len(args.audio_dirs) != len(args.output_dirs):
        raise ValueError("Number of paths must match")
        
    extract_ast(
        csv_paths=args.csv_paths,
        audio_dirs=args.audio_dirs,
        output_dirs=args.output_dirs,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
