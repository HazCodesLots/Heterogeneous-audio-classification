import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import torchaudio.functional as F_audio
from panns_inference import AudioTagging

def extract_panns(csv_paths, audio_dirs, output_dirs, device="cuda"):
    print(f"Loading PANNs model on {device}...")
    # This automatically downloads CNN14 checkpoint and loads it
    model = AudioTagging(checkpoint_path="C:/Users/HazCodes/panns_data/Cnn14_mAP=0.431.pth", device=device)
    
    for csv_path, audio_dir, output_dir in zip(csv_paths, audio_dirs, output_dirs):
        print(f"\nProcessing {csv_path} -> {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(csv_path)
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sound_id = str(row['sound_id']).strip()
            out_file = os.path.join(output_dir, f"{sound_id}.pt")
            
            if os.path.exists(out_file):
                continue
                
            try:
                audio_path = os.path.join(audio_dir, f"{sound_id}.wav")
                if not os.path.exists(audio_path):
                    continue
                # Use soundfile (fast)
                wav, sr = sf.read(audio_path, dtype='float32')
                if len(wav.shape) > 1:
                    wav = wav.mean(axis=1) # to mono
                    
                wav_tensor = torch.tensor(wav).unsqueeze(0) # (1, samples)
                
                # Resample to 32000 if needed (PANNs uses 32kHz)
                if sr != 32000:
                    wav_tensor = F_audio.resample(wav_tensor, orig_freq=sr, new_freq=32000)
                    
                # Pad to at least 1 second (32000 samples) to avoid CNN14 crashing on tiny files
                if wav_tensor.shape[1] < 32000:
                    pad_len = 32000 - wav_tensor.shape[1]
                    wav_tensor = torch.nn.functional.pad(wav_tensor, (0, pad_len))
                    
                wav_tensor = wav_tensor.to(device)
                
                with torch.no_grad():
                    # The inference method returns (clipwise_output, embedding)
                    _, embedding = model.inference(wav_tensor)
                
                # Save the 2048-d embedding
                # embedding shape is (1, 2048)
                emb_tensor = torch.from_numpy(embedding[0])
                torch.save(emb_tensor, out_file)
                
            except Exception as e:
                print(f"\nError processing {audio_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_paths", nargs="+", required=True, help="Paths to metadata CSVs")
    parser.add_argument("--audio_dirs", nargs="+", required=True, help="Paths to audio dirs")
    parser.add_argument("--output_dirs", nargs="+", required=True, help="Output directories for embeddings")
    args = parser.parse_args()
    
    if len(args.csv_paths) != len(args.output_dirs) or len(args.csv_paths) != len(args.audio_dirs):
        raise ValueError("Number of paths must match")
        
    extract_panns(
        csv_paths=args.csv_paths,
        audio_dirs=args.audio_dirs,
        output_dirs=args.output_dirs,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
