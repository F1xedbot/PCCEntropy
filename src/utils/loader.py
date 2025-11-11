import os
import pandas as pd

def load_pickles_split(base_path: str, target_file: str) -> pd.DataFrame:
    frames = []

    for file in os.listdir(base_path):
        if not file.endswith(".pkl"):
            continue
        if file == target_file: 
            path = os.path.join(base_path, file)
            with open(path, "rb") as f:
                texts_1, texts_2, labels = pd.read_pickle(f)
    
            # Flip labels
            flipped_labels = [1 - l for l in labels]
    
            df = pd.DataFrame({
                "text1": texts_1,
                "text2": texts_2,
                "label": flipped_labels,
                "source_file": file
            })
    
            frames.append(df)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df