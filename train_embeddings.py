import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from models import get_model

import librosa

def load_audio(file_path, target_sr=16000, max_duration=3.0):
    try:
        if not file_path.endswith('.mp3'):
            file_path += '.mp3'
            
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        waveform = torch.from_numpy(y)
        
        # Pad or truncate
        target_length = int(target_sr * max_duration)
        if waveform.shape[0] > target_length:
            waveform = waveform[:target_length]
        else:
            padding = target_length - waveform.shape[0]
            if padding > 0:
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform.numpy()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    metadata_path = "cv_au_subset/metadata.csv"
    audio_dir = "cv_au_subset/audio"
    embeddings_cache_file = "cv_au_subset/wav2vec2_embeddings.npy"
    cache_indices_file = "cv_au_subset/wav2vec2_indices.npy"
    
    print("=========================================")
    print("   PIPELINE C: WAV2VEC 2.0 EMBEDDINGS    ")
    print("=========================================")
    
    AGE_MAP = {
        "teens": 0, "twenties": 1, "thirties": 2, "fourties": 3,
        "fifties": 4, "sixties": 5, "seventies": 6,
    }
    
    # 1. Load Data
    df = pd.read_csv(metadata_path)
    df = df[df["age"].notna()]
    df = df[df["age"].isin(AGE_MAP.keys())]
    df["age_label"] = df["age"].map(AGE_MAP)
    df['original_idx'] = df.index
    
    # Filter valid files
    valid_indices = []
    for idx, row in df.iterrows():
        if os.path.exists(os.path.join(audio_dir, row["path"])):
            valid_indices.append(idx)
    df = df.loc[valid_indices].reset_index(drop=True)
    
    print(f"Total verifiable audio files: {len(df)}")
    
    # 2. Extract or Load Embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    if os.path.exists(embeddings_cache_file) and os.path.exists(cache_indices_file):
        print(f"Loading cached embeddings from {embeddings_cache_file}...")
        embeddings = np.load(embeddings_cache_file)
        cached_indices = np.load(cache_indices_file)
        
        # Ensure alignment
        cache_row_map = {orig_id: row_num for row_num, orig_id in enumerate(cached_indices)}
        
        aligned_rows = []
        valid_df_indices = []
        for i, orig_id in enumerate(df['original_idx']):
            if orig_id in cache_row_map:
                aligned_rows.append(cache_row_map[orig_id])
                valid_df_indices.append(i)
                
        # Filter df to match available cache instances
        df = df.iloc[valid_df_indices].reset_index(drop=True)
        X = embeddings[aligned_rows]
    else:
        print(f"Extracting embeddings using Wav2Vec 2.0 on {device}...")
        processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
        model.eval()
        
        X_list = []
        successful_indices = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            file_path = os.path.join(audio_dir, row["path"])
            waveform = load_audio(file_path)
            
            if waveform is not None:
                inputs = processor(waveform, sampling_rate=16000, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # average pooling over the time dimension
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                X_list.append(embedding)
                successful_indices.append(row["original_idx"])
        
        X = np.stack(X_list)
        
        # Cache them
        np.save(embeddings_cache_file, X)
        np.save(cache_indices_file, np.array(successful_indices))
        
        # keep dataframe in sync
        df = df[df["original_idx"].isin(successful_indices)].reset_index(drop=True)

    y = df["age_label"].values
    
    print(f"Final shape of X: {X.shape}, y: {y.shape}")
    
    # 3. Scale Embeddings
    print("Normalizing embeddings...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 4. Train/Test Split
    print("Splitting data (80% Train, 20% Test)...")
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=df["client_id"]))
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # 5. Train Classical Models
    print("  -> Training Logistic Regression on Embeddings...")
    log_reg = get_model(model_type="baseline", task="classification")
    log_reg.fit(X_train, y_train)
    log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test))
    
    print("  -> Training Random Forest on Embeddings...")
    rf = get_model(model_type="rf", task="classification")
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    
    print("\n=========================================")
    print("           EMBEDDINGS RESULTS            ")
    print("=========================================")
    print(f"Test Set Size: {len(test_idx)} specific unseen speakers")
    print(f"Logistic Regression : {log_reg_acc:.3%}")
    print(f"Random Forest       : {rf_acc:.3%}")
    print("=========================================")

if __name__ == "__main__":
    main()
