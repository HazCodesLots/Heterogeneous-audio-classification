import os
import warnings
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import ClapAudioModelWithProjection, ClapProcessor
import soundfile as sf
import torchaudio.functional as F_audio

warnings.filterwarnings("ignore", message=".*`audios` is deprecated.*")

def extract_embeddings(csv_path, audio_dir, output_dir, device="cuda"):
    print(f"Extracting CLAP embeddings to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused").to(device)
    model.eval()
    
    df = pd.read_csv(csv_path)
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sound_id = str(row['sound_id']).strip()
            out_path = os.path.join(output_dir, f"{sound_id}.npy")
            
            if os.path.exists(out_path):
                continue
                
            audio_path = os.path.join(audio_dir, f"{sound_id}.wav")
            if not os.path.exists(audio_path):
                print(f"Warning: {audio_path} not found.")
                continue
            try:
                audio_np, sr = sf.read(audio_path, dtype='float32', always_2d=True)
            except Exception as e:
                print(f"Corrupted file {audio_path}: {e}")
                np.save(out_path, np.zeros(512, dtype=np.float32))
                continue
                
            waveform = torch.from_numpy(audio_np.T)  # (channels, samples)
            
            if sr != 48000:
                waveform = F_audio.resample(waveform, orig_freq=sr, new_freq=48000)
                
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            inputs = processor(audio=waveform[0].numpy(), sampling_rate=48000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)

            emb = outputs.audio_embeds.squeeze(0).cpu().numpy()  # (512,)
            
            np.save(out_path, emb)
            
if __name__ == "__main__":

    extract_embeddings(
        csv_path="C:/Users/HazCodes/Documents/Datasets/DCASE/19868804/metadata/BSD10k_metadata.csv",
        audio_dir="C:/Users/HazCodes/Documents/Datasets/DCASE/19868804/audio",
        output_dir="data/BSD10k_CLAP_Embeddings"
    )
    
    extract_embeddings(
        csv_path="C:/Users/HazCodes/Documents/Datasets/DCASE/19187100/metadata/BSD35k-CS_metadata.csv",
        audio_dir="C:/Users/HazCodes/Documents/Datasets/DCASE/19187100/audio",
        output_dir="data/BSD35k_CLAP_Embeddings"
    )
    print("Done!")
