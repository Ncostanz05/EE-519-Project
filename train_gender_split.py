import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
from models import get_model

def main():
    metadata_path = "cv_au_subset/metadata.csv"
    audio_dir = "cv_au_subset/audio"
    
    print("=========================================")
    print("       EFFECT OF GENDER ISOLATION        ")
    print("=========================================")
    
    AGE_MAP = {
        "teens": 0, "twenties": 1, "thirties": 2, "fourties": 3,
        "fifties": 4, "sixties": 5, "seventies": 6,
    }
    
    # 1. Load and filter Data
    df = pd.read_csv(metadata_path)
    df = df[df["age"].notna()]
    df = df[df["age"].isin(AGE_MAP.keys())]
    df["age_label"] = df["age"].map(AGE_MAP)
    
    # FILTER FOR MALE ONLY
    df = df[df["gender"] == "male_masculine"]
    
    # Store the original index as a column before filtering out non-existent files
    df['original_idx'] = df.index
    
    # Filter out files that don't exist on disk
    valid_indices = []
    for idx, row in df.iterrows():
        if os.path.exists(os.path.join(audio_dir, row["path"])):
            valid_indices.append(idx)
    df = df.loc[valid_indices].reset_index(drop=True)
    
    print(f"Isolated {len(df)} verifiable male audio files.\n")
    
    # 2. Get Features (Load Cache directly for speed)
    print("Loading pre-calculated handcrafted features...")
    all_valid_indices = np.load("cv_au_subset/valid_indices.npy")
    X_cache = pd.read_csv("cv_au_subset/extracted_features.csv")
    
    # We need to find which row in X_cache corresponds to our male files
    # all_valid_indices tells us which original metadata.csv row each X_cache row belongs to.
    
    # Create a dictionary to instantly map an original_idx to an X_cache row number
    cache_row_map = {orig_id: row_num for row_num, orig_id in enumerate(all_valid_indices)}
    
    # Build our X for the male dataset
    male_cache_rows = []
    for orig_id in df['original_idx']:
        if orig_id in cache_row_map:
            male_cache_rows.append(cache_row_map[orig_id])
            
    X_handcrafted = X_cache.iloc[male_cache_rows].reset_index(drop=True)
    y_handcrafted = df["age_label"]
    
    # 3. Speaker-Independent Split (80/20)
    print("Splitting Male-Only data (80% Train, 20% Test)...")
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X_handcrafted, y_handcrafted, groups=df["client_id"]))
    
    X_train = X_handcrafted.iloc[train_idx]
    y_train = y_handcrafted.iloc[train_idx]
    X_test = X_handcrafted.iloc[test_idx]
    y_test = y_handcrafted.iloc[test_idx]
    
    # 4. Train Models
    print("  -> Training Logistic Regression (Male Baseline)...")
    log_reg = get_model(model_type="baseline", task="classification")
    log_reg.fit(X_train, y_train)
    log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test))
    
    print("  -> Training Random Forest (Male)...")
    rf = get_model(model_type="rf", task="classification")
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    
    print("\n=========================================")
    print("     MALE-ONLY GENDER ISOLATION RESULTS   ")
    print("=========================================")
    print(f"Test Set Size: {len(test_idx)} specific unseen male speakers")
    print(f"Logistic Regression : {log_reg_acc:.3%}")
    print(f"Random Forest       : {rf_acc:.3%}")
    print("=========================================")

if __name__ == "__main__":
    main()
