import os
from extract_clap_embeddings import extract_embeddings as extract_clap
from extract_panns_embeddings import extract_panns
from extract_text_embeddings import extract_text_embeddings as extract_text

eval_csv = r"C:\Users\HazCodes\Documents\Datasets\DCASE\eval_metadata.csv"
audio_dir = r"C:\Users\HazCodes\Documents\Datasets\DCASE\audio"

print("1. Extracting CLAP Audio...")
extract_clap(eval_csv, audio_dir, "data/Blind_CLAP_Embeddings")

print("\n2. Extracting PANNs Audio...")
extract_panns([eval_csv], [audio_dir], ["data/Blind_PANNs_Embeddings"])

print("\n3. Extracting CLAP Text (Empty strings)...")
extract_text([eval_csv], ["data/Blind_Text_Embeddings"])

print("\nAll Evaluation Embeddings Extracted Successfully!")
