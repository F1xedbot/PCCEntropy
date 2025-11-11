from utils.builder import build_entropy_features, build_embedding_features
from models.encoder import LMEncoder
from utils.enums import LMModels
import torch
from utils.loader import load_pickles_split
from models.base.lightgbm_model import LightGBMClassifierModel
from services.vectorstore import FAISSStore
from tqdm import tqdm
import numpy as np
from models.corrector import CounterfactualCorrector

DATA_PATH = "src/input" # <-- edit this
TARGET_FILES = ["sample_data.pkl"]
LM_MODEL = LMModels.DEFAULT
MODEL_SAVE_PATH = "src/output/entropy_lgbm_model.pkl"
SCALED_PROBS_PATH = "src/output/platt_scaled_probs.npy"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    store = FAISSStore()
    store.load_index()

    encoder = LMEncoder(LM_MODEL, device)
    data_df = load_pickles_split(base_path=DATA_PATH, target_files=TARGET_FILES)

    if data_df.empty:
        print("No data loaded. Exiting.")
        return
    
    print(f"Loaded {len(data_df)} rows.")
    
    print("Computing embeddings...")
    data_embs = build_embedding_features(data_df, encoder)

    print("Computing entropy features...")
    data_feats = build_entropy_features(data_df, encoder)

    model = LightGBMClassifierModel(ds_feats=data_feats)
    model.load_model(MODEL_SAVE_PATH)
    y_pred, y_pred_proba = model.predict(model.X)

    print(f"[INFO] Loading calibrated probabilities from: {SCALED_PROBS_PATH}")
    calibrated_probs = np.load(SCALED_PROBS_PATH)

    print("[INFO] Searching nearest neighbors in FAISS index...")

    sims_arr = []
    neigh_ps = []
    ps = []

    for i, emb in enumerate(tqdm(data_embs, desc="Searching neighbors")):
        sim, ids, probs = store.search_neighbors(emb, calibrated_probs)
        sims_arr.append(sim)
        neigh_ps.append(probs)
        ps.append(y_pred_proba[i])

    ps = np.array(ps)
    neigh_ps = np.array(neigh_ps)
    sims_arr = np.array(sims_arr)

    cf_engine = CounterfactualCorrector()
    print("[INFO] Adjusting predictions with counterfactual correction...")
    cf_results = cf_engine.compute_batch(ps, neigh_ps, sims_arr)

    adjusted_y_proba = np.array([r['adjusted_proba'] for r in cf_results])
    adjusted_y_pred = (adjusted_y_proba > 0.5).astype(int)

    print(adjusted_y_pred) # <-- or save this

if __name__ == "__main__":
    main()
