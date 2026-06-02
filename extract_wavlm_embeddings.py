import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import soundfile as sf
import torchaudio.functional as F_audio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

def extract_wavlm(csv_paths, audio_dirs, output_dirs, device="cuda"):
    print(f"Loading WavLM (BEATs equivalent) model on {device}...")
    
    # WavLM is Microsoft's natively supported equivalent to BEATs in HuggingFace
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
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
                    continue
                    
                try:
                    wav, sr = sf.read(audio_path, dtype='float32')
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                        
                    wav_tensor = torch.tensor(wav).unsqueeze(0)
                    if sr != 16000:
                        wav_tensor = F_audio.resample(wav_tensor, sr, 16000)
                        
                    inputs = processor(wav_tensor.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    outputs = model(**inputs)
                    
                    # Shape: (1024,)
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                    torch.save(emb.cpu(), out_file)
                    
                except Exception as e:
                    pass

if __name__ == "__main__":
    extract_wavlm(
        csv_paths=[
            "C:/Users/HazCodes/Documents/Datasets/DCASE/19868804/metadata/BSD10k_metadata.csv", 
            "C:/Users/HazCodes/Documents/Datasets/DCASE/19187100/metadata/BSD35k-CS_metadata.csv"
        ],
        audio_dirs=[
            "C:/Users/HazCodes/Documents/Datasets/DCASE/19868804/audio", 
            "C:/Users/HazCodes/Documents/Datasets/DCASE/19187100/audio"
        ],
        output_dirs=[
            "data/BSD10k_WavLM_Embeddings", 
            "data/BSD35k_WavLM_Embeddings"
        ]
    )
