import os
import pandas as pd
from typing import Union, List

def load_pickles_split(base_path: str, target_files: Union[str, List[str]]) -> pd.DataFrame:
    """
    Load one or more target pickle files from a directory and flip labels (1 -> 0, 0 -> 1).

    Args:
        base_path: Directory containing .pkl files
        target_files: Filename or list of filenames of pickles to load

    Returns:
        df: DataFrame with columns ["text1", "text2", "label", "source_file"]
    """
    if isinstance(target_files, str):
        target_files = [target_files]

    frames = []

    for file in target_files:
        path = os.path.join(base_path, file)
        if not os.path.exists(path) or not file.endswith(".pkl"):
            print(f"Warning: Pickle file '{file}' not found in '{base_path}' or not a .pkl file. Skipping.")
            continue

        with open(path, "rb") as f:
            try:
                texts_1, texts_2, labels = pd.read_pickle(f)
            except Exception as e:
                print(f"Error loading '{file}': {e}")
                continue

        # Flip labels
        flipped_labels = [1 - l for l in labels]

        df = pd.DataFrame({
            "text1": texts_1,
            "text2": texts_2,
            "label": flipped_labels,
            "source_file": file
        })

        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
