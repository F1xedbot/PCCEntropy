import torch
import numpy as np
import pandas as pd
from models.encoder import LMEncoder

def build_embedding_features(
    df: pd.DataFrame,
    model: LMEncoder,
    column: str = "text2",
    save_path: str | None = None
) -> np.ndarray:
    """
    Compute embeddings for a DataFrame of code snippets.

    Args:
        df: DataFrame with columns ["text1", "text2", "label"]
        model: LMEncoder instance
        column: which text column to compute embeddings from ("text1" or "text2")
        save_path: optional .pt file to save results

    Returns:
        embeddings: np.ndarray of shape [len(df), embedding_dim]
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    embeddings = []

    for i, row in enumerate(df.itertuples(index=False)):
        text = getattr(row, column)
        emb = model.compute_embeddings(text)["embeddings"].numpy()
        embeddings.append(emb)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(df)} rows")

    embeddings = np.stack(embeddings)

    if save_path:
        torch.save(torch.from_numpy(embeddings), save_path)
        print(f"Saved embeddings to {save_path}")

    return embeddings

def build_entropy_features(
    df: pd.DataFrame, 
    model: LMEncoder, 
    save_path: str | None = None
) -> pd.DataFrame:
    """
    Compute entropy features for a DataFrame of code snippet pairs.

    Args:
        df: DataFrame with columns ["text1", "text2", "label"]
        model: LMEncoder instance
        save_path: optional CSV/PKL file to save results

    Returns:
        features_df: DataFrame with entropy features + label
    """
    rows = []

    for i, row in enumerate(df.itertuples(index=False)):
        ent1 = model.compute_entropy(row.text1)
        ent2 = model.compute_entropy(row.text2)

        rows.append({
            "text1": row.text1,
            "text2": row.text2,
            "label": row.label,
            "text1_mean_entropy": ent1["mean_entropy"],
            "text1_sum_entropy": ent1["sum_entropy"],
            "text1_n_tokens": ent1["n_tokens"],
            "text2_mean_entropy": ent2["mean_entropy"],
            "text2_sum_entropy": ent2["sum_entropy"],
            "text2_n_tokens": ent2["n_tokens"],
            "delta_mean_entropy": ent1["mean_entropy"] - ent2["mean_entropy"],
            "delta_sum_entropy": ent1["sum_entropy"] - ent2["sum_entropy"],
            "delta_n_tokens": ent1["n_tokens"] - ent2["n_tokens"],
        })

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(df)} rows")

    features_df = pd.DataFrame(rows)

    if save_path:
        if save_path.endswith(".pkl"):
            features_df.to_pickle(save_path)
        else:
            features_df.to_csv(save_path, index=False)
        print(f"Saved features to {save_path}")

    return features_df