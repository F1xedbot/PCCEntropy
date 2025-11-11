#!/usr/bin/env python3
"""
Script to apply counterfactual corrective adjustment on model predictions
using FAISS neighbors and evaluate the final performance.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from services.vectorstore import FAISSStore
from models.base.lightgbm_model import LightGBMClassifierModel
from models.corrector import CounterfactualCorrector
from utils.analysis import evaluate

FEATURES_PATH = "src/data/patch_entropy.csv"
EMBEDDINGS_PATH = "src/data/patch_embeddings.pt"
SCALED_PROBS_PATH = "src/output/platt_scaled_probs.npy"
RESULT_SAVE_PATH = "src/output/corrective_result.json"


def main() -> None:
    store = FAISSStore()
    store.load_index()

    print(f"[INFO] Loading features from: {FEATURES_PATH}")
    dataset_feats = pd.read_csv(FEATURES_PATH)

    print(f"[INFO] Loading embeddings from: {EMBEDDINGS_PATH}")
    dataset_embs = torch.load(EMBEDDINGS_PATH, weights_only=False)

    print(f"[INFO] Loading calibrated probabilities from: {SCALED_PROBS_PATH}")
    calibrated_probs = np.load(SCALED_PROBS_PATH)

    base_model = LightGBMClassifierModel(ds_feats=dataset_feats, ds_embs=dataset_embs)
    base_model.split_data()
    base_model.load_model("src/output/entropy_lgbm_model.pkl")

    y_pred = base_model.model.predict(base_model.X_test)

    ps = []
    neigh_ps = []
    sims_arr = []

    print("[INFO] Searching nearest neighbors in FAISS index...")
    for i, emb in enumerate(tqdm(base_model.emb_test, desc="Searching neighbors")):
        sim, ids, probs = store.search_neighbors(emb, calibrated_probs)
        sims_arr.append(sim)
        neigh_ps.append(probs)
        ps.append(y_pred[i])

    ps = np.array(ps)
    neigh_ps = np.array(neigh_ps)
    sims_arr = np.array(sims_arr)

    cf_engine = CounterfactualCorrector()
    print("[INFO] Adjusting predictions with counterfactual correction...")
    cf_results = cf_engine.compute_batch(ps, neigh_ps, sims_arr)

    adjusted_y_proba = np.array([r['adjusted_proba'] for r in cf_results])
    adjusted_y_pred = (adjusted_y_proba > 0.5).astype(int)

    results = evaluate(adjusted_y_pred, adjusted_y_proba, base_model.y_test)
    print("[INFO] Evaluation results:")
    print(json.dumps(results, indent=2))

    if RESULT_SAVE_PATH:
        os.makedirs(os.path.dirname(RESULT_SAVE_PATH), exist_ok=True)
        with open(RESULT_SAVE_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Saved evaluation results to: {RESULT_SAVE_PATH}")


if __name__ == "__main__":
    main()