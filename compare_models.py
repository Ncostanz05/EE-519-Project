import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset

from dataset import build_dataset
from models import get_model
from cnn_dataset import CommonVoiceSpectrogramDataset
from cnn_model import AgeAudioCNN

def main():
    metadata_path = "cv_au_subset/metadata.csv"
    audio_dir = "cv_au_subset/audio"
    
    print("=========================================")
    print("      AUDIO AGE PREDICTION BENCHMARK     ")
    print("=========================================")
    
    # ---------------------------------------------------------
    # 1. ESTABLISH THE EXACT SAME SPLIT FOR BOTH PIPELINES
    # ---------------------------------------------------------
    # To compare them fairly, we must evaluate on the exact same audio files.
    # We will compute the splits once based on the original metadata.
    
    AGE_MAP = {
        "teens": 0, "twenties": 1, "thirties": 2, "fourties": 3,
        "fifties": 4, "sixties": 5, "seventies": 6,
    }
    
    df = pd.read_csv(metadata_path)
    df = df[df["age"].notna()]
    df = df[df["age"].isin(AGE_MAP.keys())]
    df["age_label"] = df["age"].map(AGE_MAP)
    
    # Filter out files that don't exist
    valid_indices = []
    for idx, row in df.iterrows():
        if os.path.exists(os.path.join(audio_dir, row["path"])):
            valid_indices.append(idx)
    df = df.loc[valid_indices].reset_index(drop=True)
    
    client_ids = df["client_id"].values
    
    # First split: 70% Train, 30% Temp (Val + Test)
    splitter_main = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
    dummy_X = np.zeros(len(client_ids))
    train_idx, temp_idx = next(splitter_main.split(dummy_X, groups=client_ids))
    
    # Second split: 15% Val, 15% Test
    splitter_temp = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
    temp_client_ids = client_ids[temp_idx]
    dummy_X_temp = np.zeros(len(temp_client_ids))
    val_rel_idx, test_rel_idx = next(splitter_temp.split(dummy_X_temp, groups=temp_client_ids))
    
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]
    
    print(f"Dataset securely split into: {len(train_idx)} Train, {len(val_idx)} Val, {len(test_idx)} Test files.")
    
    # ---------------------------------------------------------
    # PIPELINE A: HANDCRAFTED FEATURES (MFCC, F0, RMS, etc.)
    # ---------------------------------------------------------
    print("\n[PIPELINE A] Loading Handcrafted Features Pipeline...")
    # build_dataset automatically loads from the cached extracted_features.csv
    # Note: build_dataset currently does an internal 80/20 split. 
    # To bypass it and use our exact splits, we just load the cache directly.
    X_handcrafted = pd.read_csv("cv_au_subset/extracted_features.csv")
    y_handcrafted = df["age_label"] # matches length since we dropped invalid earlier
    
    # We combine Train and Val for the classical ML models since they don't do Early Stopping
    train_val_idx = np.concatenate((train_idx, val_idx))
    
    X_train_classical = X_handcrafted.iloc[train_val_idx]
    y_train_classical = y_handcrafted.iloc[train_val_idx]
    X_test_classical = X_handcrafted.iloc[test_idx]
    y_test_classical = y_handcrafted.iloc[test_idx]
    
    # Train Logistic Regression
    print("  -> Training Logistic Regression (Baseline)...")
    log_reg = get_model(model_type="baseline", task="classification")
    log_reg.fit(X_train_classical, y_train_classical)
    log_reg_acc = accuracy_score(y_test_classical, log_reg.predict(X_test_classical))
    
    # Train Random Forest
    print("  -> Training Random Forest...")
    rf = get_model(model_type="rf", task="classification")
    rf.fit(X_train_classical, y_train_classical)
    rf_acc = accuracy_score(y_test_classical, rf.predict(X_test_classical))


    # ---------------------------------------------------------
    # PIPELINE B: PYTORCH CNN ON MEL-SPECTROGRAMS
    # ---------------------------------------------------------
    print("\n[PIPELINE B] Loading Deep Learning CNN Pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    full_dl_dataset = CommonVoiceSpectrogramDataset(metadata_path, audio_dir)
    test_dl_dataset = Subset(full_dl_dataset, test_idx)
    test_loader = DataLoader(test_dl_dataset, batch_size=32, shuffle=False)
    
    # Load the best model weights saved from train_cnn.py's early stopping
    cnn_model = AgeAudioCNN(num_classes=7).to(device)
    try:
        cnn_model.load_state_dict(torch.load("cnn_age_model.pth"))
        cnn_model.eval()
        
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for batch in test_loader:
                mel_spec, labels, _ = batch
                mel_spec = mel_spec.to(device)
                labels = labels.to(device)
                
                outputs = cnn_model(mel_spec)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
        cnn_acc = correct_test / total_test
    except FileNotFoundError:
        print("  -> cnn_age_model.pth not found! Please run train_cnn.py first.")
        cnn_acc = 0.0

    print("\n=========================================")
    print("          FINAL BENCHMARK RESULTS        ")
    print("=========================================")
    print(f"Test Set Size: {len(test_idx)} specific unseen speakers\n")
    print(f"1. Logistic Regression : {log_reg_acc:.3%}")
    print(f"2. Random Forest       : {rf_acc:.3%}")
    print(f"3. Mel-Spectrogram CNN : {cnn_acc:.3%}")
    print("=========================================")

if __name__ == "__main__":
    main()
