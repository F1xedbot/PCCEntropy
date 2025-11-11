#!/usr/bin/env python3
"""
Train and evaluate a base LightGBM model using precomputed entropy features
and embeddings. Saves evaluation results, trained model, and embeddings optionally.
"""

import os
import json
import torch
import pandas as pd
from models.base.lightgbm_model import LightGBMClassifierModel
import numpy as np
from utils.analysis import platt_scaled_predictions

FEATURES_PATH = "src/data/patch_entropy.csv"
EMBEDDINGS_PATH = "src/data/patch_embeddings.pt"
MODEL_SAVE_PATH = "src/output/entropy_lgbm_model.pkl"
EMBEDDINGS_SAVE_PATH = "src/data/train_split_embeddings.npz"
RESULT_SAVE_PATH = "src/output/base_result.json"

# For analysis
OOF_PROBS_SAVE_PATH = "src/output/oof_probs.npy"
PLATT_SCALED_PROBS_SAVE_PATH = "src/output/platt_scaled_probs.npy"

def main():
    print(f"Loading features from: {FEATURES_PATH}")
    dataset_feats = pd.read_csv(FEATURES_PATH)

    print(f"Loading embeddings from: {EMBEDDINGS_PATH}")
    dataset_embs = torch.load(EMBEDDINGS_PATH, weights_only=False)

    if isinstance(dataset_embs, torch.Tensor):
        dataset_embs = dataset_embs.numpy()

    print("Initializing LightGBM model...")
    model = LightGBMClassifierModel(ds_feats=dataset_feats, ds_embs=dataset_embs)

    print("Fitting model...")
    model.fit()

    print("Evaluating model...")
    results = model.evaluate()
    print(json.dumps(results, indent=2))

    if RESULT_SAVE_PATH:
        os.makedirs(os.path.dirname(RESULT_SAVE_PATH), exist_ok=True)
        with open(RESULT_SAVE_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation results to: {RESULT_SAVE_PATH}")

    if MODEL_SAVE_PATH:
        model.save_model(MODEL_SAVE_PATH)
        print(f"Saved trained model to: {MODEL_SAVE_PATH}")

    if EMBEDDINGS_SAVE_PATH:
        model.save_embeddings(EMBEDDINGS_SAVE_PATH)
        print(f"Saved train/test embeddings to: {EMBEDDINGS_SAVE_PATH}")

    # Get OOF Predictions and Temp Scaled Predictions for later analysis
    # Fit a single-parameter logistic (temperature) on a validation set
    oof_probs = model.oof_predict()

    if OOF_PROBS_SAVE_PATH:
        np.save(OOF_PROBS_SAVE_PATH, oof_probs)
        print(f"Saved out-of-fold predictions to: {OOF_PROBS_SAVE_PATH}")

    calibrated_probs = platt_scaled_predictions(oof_probs, model.y_train)
    if PLATT_SCALED_PROBS_SAVE_PATH:
        np.save(PLATT_SCALED_PROBS_SAVE_PATH, calibrated_probs)
        print(f"Saved platt scaled predictions to: {PLATT_SCALED_PROBS_SAVE_PATH}")

if __name__ == "__main__":
    main()
