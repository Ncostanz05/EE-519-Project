"""
Train and save models needed for the live demo:
  1. Wav2Vec2 Embeddings → RF age classifier  (saved as embeddings_age_model.pkl + embeddings_scaler.pkl)
  2. Wav2Vec2 Embeddings → Gender classifier   (saved as gender_model.pkl + gender_scaler.pkl)
  3. Gender-conditioned age classifiers         (saved as age_model_male.pkl + age_model_female.pkl)

Uses cached wav2vec2 embeddings from cv_au_subset/wav2vec2_embeddings.npy
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

AGE_MAP = {
    "teens": 0, "twenties": 1, "thirties": 2, "fourties": 3,
    "fifties": 4, "sixties": 5, "seventies": 6,
}

GENDER_MAP = {
    "male_masculine": 0,
    "female_feminine": 1,
}

def main():
    metadata_path = "cv_au_subset/metadata.csv"
    audio_dir = "cv_au_subset/audio"
    embeddings_cache = "cv_au_subset/wav2vec2_embeddings.npy"
    indices_cache = "cv_au_subset/wav2vec2_indices.npy"

    print("=" * 50)
    print("  TRAINING DEMO MODELS")
    print("=" * 50)

    # ── Load metadata ──
    df = pd.read_csv(metadata_path, engine="python")
    df = df[df["age"].notna() & df["age"].isin(AGE_MAP.keys())]
    df["age_label"] = df["age"].map(AGE_MAP)
    df["original_idx"] = df.index

    # Filter valid files
    valid_indices = []
    for idx, row in df.iterrows():
        if os.path.exists(os.path.join(audio_dir, row["path"])):
            valid_indices.append(idx)
    df = df.loc[valid_indices].reset_index(drop=True)

    # ── Load cached embeddings ──
    if not os.path.exists(embeddings_cache):
        print("ERROR: Cached embeddings not found. Run train_embeddings.py first!")
        return

    print("Loading cached Wav2Vec2 embeddings...")
    embeddings = np.load(embeddings_cache)
    cached_indices = np.load(indices_cache)

    # Align embeddings with filtered dataframe
    cache_row_map = {orig_id: row_num for row_num, orig_id in enumerate(cached_indices)}
    aligned_rows = []
    valid_df_indices = []
    for i, orig_id in enumerate(df["original_idx"]):
        if orig_id in cache_row_map:
            aligned_rows.append(cache_row_map[orig_id])
            valid_df_indices.append(i)
    df = df.iloc[valid_df_indices].reset_index(drop=True)
    X_emb = embeddings[aligned_rows]

    print(f"Dataset: {len(df)} samples, Embedding dim: {X_emb.shape[1]}")

    # ── Split (80/20, speaker-independent) ──
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X_emb, df["age_label"].values, groups=df["client_id"]))

    X_train, X_test = X_emb[train_idx], X_emb[test_idx]
    y_age_train = df["age_label"].values[train_idx]
    y_age_test = df["age_label"].values[test_idx]

    # ────────────────────────────────────────────────────
    # MODEL 1: Wav2Vec2 Embeddings → Age (RF)
    # ────────────────────────────────────────────────────
    print("\n[1/3] Training Embeddings → Age classifier...")
    emb_scaler = StandardScaler()
    X_train_scaled = emb_scaler.fit_transform(X_train)
    X_test_scaled = emb_scaler.transform(X_test)

    emb_age_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    emb_age_rf.fit(X_train_scaled, y_age_train)
    emb_age_acc = accuracy_score(y_age_test, emb_age_rf.predict(X_test_scaled))
    print(f"  Embeddings Age RF Accuracy: {emb_age_acc:.3%}")

    joblib.dump(emb_age_rf, "embeddings_age_model.pkl")
    joblib.dump(emb_scaler, "embeddings_scaler.pkl")
    print("  ✓ Saved embeddings_age_model.pkl + embeddings_scaler.pkl")

    # ────────────────────────────────────────────────────
    # MODEL 2: Wav2Vec2 Embeddings → Gender
    # ────────────────────────────────────────────────────
    print("\n[2/3] Training Embeddings → Gender classifier...")
    # Filter to samples with known gender
    gender_mask_train = df.iloc[train_idx]["gender"].isin(GENDER_MAP.keys()).values
    gender_mask_test = df.iloc[test_idx]["gender"].isin(GENDER_MAP.keys()).values

    X_train_g = X_train_scaled[gender_mask_train]
    y_gender_train = df.iloc[train_idx][gender_mask_train]["gender"].map(GENDER_MAP).values
    X_test_g = X_test_scaled[gender_mask_test]
    y_gender_test = df.iloc[test_idx][gender_mask_test]["gender"].map(GENDER_MAP).values

    gender_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    gender_rf.fit(X_train_g, y_gender_train)
    gender_acc = accuracy_score(y_gender_test, gender_rf.predict(X_test_g))
    print(f"  Gender RF Accuracy: {gender_acc:.3%}")

    joblib.dump(gender_rf, "gender_model.pkl")
    print("  ✓ Saved gender_model.pkl")

    # ────────────────────────────────────────────────────
    # MODEL 3: Gender-conditioned Age classifiers
    # ────────────────────────────────────────────────────
    print("\n[3/3] Training Gender-conditioned Age classifiers...")

    for gender_name, gender_label in GENDER_MAP.items():
        mask_train = df.iloc[train_idx]["gender"].values == gender_name
        mask_test = df.iloc[test_idx]["gender"].values == gender_name

        if mask_train.sum() < 10:
            print(f"  Skipping {gender_name} — too few samples ({mask_train.sum()})")
            continue

        X_tr = X_train_scaled[mask_train]
        y_tr = y_age_train[mask_train]
        X_te = X_test_scaled[mask_test]
        y_te = y_age_test[mask_test]

        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)

        if len(y_te) > 0:
            acc = accuracy_score(y_te, clf.predict(X_te))
            print(f"  {gender_name}: Accuracy = {acc:.3%} (train={len(y_tr)}, test={len(y_te)})")
        else:
            print(f"  {gender_name}: trained on {len(y_tr)} samples (no test data for this gender)")

        short_name = "male" if "male" in gender_name else "female"
        joblib.dump(clf, f"age_model_{short_name}.pkl")
        print(f"  ✓ Saved age_model_{short_name}.pkl")

    print("\n" + "=" * 50)
    print("  ALL DEMO MODELS TRAINED AND SAVED")
    print("=" * 50)

if __name__ == "__main__":
    main()
