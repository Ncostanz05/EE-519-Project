import numpy as np
import librosa
import parselmouth

def extract_features(y, sr):
    features = {}
    
    # Pitch (F0)
    f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]
    features["f0_mean"] = np.mean(f0) if len(f0) else 0
    features["f0_std"] = np.std(f0) if len(f0) else 0
    features["f0_min"] = np.min(f0) if len(f0) else 0
    features["f0_max"] = np.max(f0) if len(f0) else 0

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i}_mean"] = mfcc[i].mean()
        features[f"mfcc_{i}_std"] = mfcc[i].std()

    # Energy
    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = rms.mean()
    features["rms_std"] = rms.std()

    # Formants (Praat)
    sound = parselmouth.Sound(y, sr)
    formant = sound.to_formant_burg()
    for i, name in zip([1, 2, 3], ["f1", "f2", "f3"]):
        values = [
            formant.get_value_at_time(i, t)
            for t in np.linspace(0, sound.duration, num=50)
            if formant.get_value_at_time(i, t) is not None
        ]
        features[f"{name}_mean"] = np.mean(values) if values else 0

    return features
