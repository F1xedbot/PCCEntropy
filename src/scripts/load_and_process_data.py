#!/usr/bin/env python3
"""
Script to load code data, compute embeddings and entropy features,
and save the processed results for downstream tasks.
"""

import torch
from utils.loader import load_pickles_split
from utils.builder import build_entropy_features, build_embedding_features
from models.encoder import LMEncoder
from utils.enums import LMModels

# ----------------- CONFIG ----------------- #
DATA_PATH = "" # <-- edit this
TARGET_FILES = ["data_code.pkl"]
EMBEDDINGS_SAVE_PATH = "../data/patch_train_embeddings.pt"
ENTROPY_SAVE_PATH = "../data/patch_train_entropy.csv"
LM_MODEL = LMModels.DEFAULT
# ------------------------------------------ #

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    encoder = LMEncoder(LM_MODEL, device)

    print(f"Loading pickles from: {DATA_PATH}")
    dataset_df = load_pickles_split(base_path=DATA_PATH, target_files=TARGET_FILES)
    if dataset_df.empty:
        print("No data loaded. Exiting.")
        return
    print(f"Loaded {len(dataset_df)} rows.")

    print("Computing embeddings...")
    dataset_embs = build_embedding_features(dataset_df, encoder, save_path=EMBEDDINGS_SAVE_PATH)
    print(f"Embeddings saved to: {EMBEDDINGS_SAVE_PATH}")

    print("Computing entropy features...")
    dataset_feats = build_entropy_features(dataset_df, encoder, save_path=ENTROPY_SAVE_PATH)
    print(f"Entropy features saved to: {ENTROPY_SAVE_PATH}")

if __name__ == "__main__":
    main()