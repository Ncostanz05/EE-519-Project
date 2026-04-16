import librosa
import numpy as np

TARGET_SR = 16000

def load_and_preprocess(path, target_sr=TARGET_SR):
    y, sr = librosa.load(path, sr=None, mono=True)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Normalize amplitude
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    return y, sr
