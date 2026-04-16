import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from preprocessing import load_and_preprocess
from features import extract_features
from tqdm import tqdm

AGE_MAP = {
    "teens": 0,
    "twenties": 1,
    "thirties": 2,
    "fourties": 3,
    "fifties": 4,
    "sixties": 5,
    "seventies": 6,
    # drop eighties if removed earlier
}

def build_dataset(metadata_path,
                  audio_dir,
                  feature_set="all",
                  task="classification"):

    df = pd.read_csv(metadata_path)

    # Remove rows with missing age
    df = df[df["age"].notna()]

    # Map age categories to integers
    df = df[df["age"].isin(AGE_MAP.keys())]
    df["age_label"] = df["age"].map(AGE_MAP)

    feature_rows = []
    valid_indices = []

    cache_path = os.path.join(audio_dir, "..", "extracted_features.csv")
    valid_indices_path = os.path.join(audio_dir, "..", "valid_indices.npy")
    
    if os.path.exists(cache_path) and os.path.exists(valid_indices_path):
        print("Loading cached features...")
        X = pd.read_csv(cache_path)
        valid_indices = np.load(valid_indices_path).tolist()
        df = df.loc[valid_indices].reset_index(drop=True)
    else:
        print("Extracting features from scratch...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files", file=sys.stdout):
            file_path = os.path.join(audio_dir, row["path"])

            if not os.path.exists(file_path):
                continue

            try:
                y, sr = load_and_preprocess(file_path)
                feats = extract_features(y, sr)
                feature_rows.append(feats)
                valid_indices.append(idx)
            except Exception:
                continue

        X = pd.DataFrame(feature_rows)
        df = df.loc[valid_indices].reset_index(drop=True)
        # Save cache
        X.to_csv(cache_path, index=False)
        np.save(valid_indices_path, np.array(valid_indices))

    # Feature selection
    if feature_set == "pitch":
        X = X[[c for c in X.columns if c.startswith("f0")]]
    elif feature_set == "spectral":
        X = X[[c for c in X.columns if c.startswith("mfcc")]]

    # Target
    if task == "classification":
        y = df["age_label"]
    else:
        raise ValueError("Regression not supported for categorical age labels.")

    # Speaker-independent split
    splitter = GroupShuffleSplit(
        test_size=0.2,
        n_splits=1,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=df["client_id"])
    )

    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx]
    )