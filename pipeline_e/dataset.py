import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import librosa

from pipeline_e.augmentation import AugmentationPipeline
from shared.constants import TARGET_SR, MAX_SAMPLES


class PipelineEDataset(Dataset):
    """
    Loads raw waveforms for Pipeline E.
    Returns dict with: waveform (numpy), age_label (tensor), gender_label (tensor),
    client_id (str), audio_path (str).
    The HuggingFace feature extractor is applied in collate_fn (batched).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        augment: AugmentationPipeline | None = None,
        training: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.training = training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        try:
            y, _ = librosa.load(row["audio_path"], sr=TARGET_SR, mono=True)
        except Exception:
            y = np.zeros(MAX_SAMPLES, dtype=np.float32)

        if self.augment is not None:
            y = self.augment(y, training=self.training)
        else:
            # Center crop / pad to MAX_SAMPLES
            if len(y) >= MAX_SAMPLES:
                start = (len(y) - MAX_SAMPLES) // 2
                y = y[start: start + MAX_SAMPLES]
            else:
                y = np.pad(y, (0, MAX_SAMPLES - len(y))).astype(np.float32)

        gender = row.get("gender_label", float("nan"))
        gender_t = torch.tensor(float(gender), dtype=torch.float32)

        return {
            "waveform": y,                                          # np.ndarray (MAX_SAMPLES,)
            "age_label": torch.tensor(int(row["age_label"]), dtype=torch.long),
            "gender_label": gender_t,
            "client_id": str(row.get("client_id", "")),
            "audio_path": str(row["audio_path"]),
        }


class SPSDataset(Dataset):
    """Cross-domain evaluation dataset using SPS corpus labeled samples."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        try:
            y, _ = librosa.load(row["audio_path"], sr=TARGET_SR, mono=True)
        except Exception:
            y = np.zeros(MAX_SAMPLES, dtype=np.float32)

        if len(y) >= MAX_SAMPLES:
            start = (len(y) - MAX_SAMPLES) // 2
            y = y[start: start + MAX_SAMPLES]
        else:
            y = np.pad(y, (0, MAX_SAMPLES - len(y))).astype(np.float32)

        return {
            "waveform": y,
            "age_label": torch.tensor(int(row["age_label"]), dtype=torch.long),
            "gender_label": torch.tensor(float(row.get("gender_label", float("nan")))),
            "audio_path": str(row["audio_path"]),
        }


def make_collate_fn(feature_extractor):
    """
    Returns a collate_fn that runs the HuggingFace feature extractor on a batch
    of raw waveform numpy arrays and stacks label tensors.
    """
    def collate_fn(batch: list[dict]) -> dict:
        waveforms = [b["waveform"] for b in batch]
        encoded = feature_extractor(
            waveforms,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_SAMPLES,
            truncation=True,
        )
        return {
            "input_values": encoded.input_values,
            "attention_mask": encoded.get("attention_mask"),
            "age_label": torch.stack([b["age_label"] for b in batch]),
            "gender_label": torch.stack([b["gender_label"] for b in batch]),
            "audio_paths": [b.get("audio_path", "") for b in batch],
        }
    return collate_fn
