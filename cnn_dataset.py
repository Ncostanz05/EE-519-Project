import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import librosa

AGE_MAP = {
    "teens": 0,
    "twenties": 1,
    "thirties": 2,
    "fourties": 3,
    "fifties": 4,
    "sixties": 5,
    "seventies": 6,
}

class CommonVoiceSpectrogramDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, target_sr=16000, max_duration=3.0, n_mels=64, n_fft=1024, hop_length=512):
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.max_length = int(target_sr * max_duration)
        
        df = pd.read_csv(metadata_path)
        df = df[df["age"].notna()]
        df = df[df["age"].isin(AGE_MAP.keys())]
        df["age_label"] = df["age"].map(AGE_MAP)
        
        # Filter out files that don't exist
        valid_indices = []
        for idx, row in df.iterrows():
            if os.path.exists(os.path.join(audio_dir, row["path"])):
                valid_indices.append(idx)
        
        self.df = df.loc[valid_indices].reset_index(drop=True)
        
        # Spectrogram Transformer
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, row["path"])
        
        # Load audio using librosa (already installed and robust)
        y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        waveform = torch.tensor(y).unsqueeze(0) # (1, Time)
        
        # Pad or truncate to max_length
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif waveform.shape[1] < self.max_length:
            pad_amount = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        # Generate Mel-Spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Normalize to roughly [0, 1] range for network stability
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
        
        label = row["age_label"]
        client_id = row["client_id"]
        
        return mel_spec_db, torch.tensor(label, dtype=torch.long), client_id
