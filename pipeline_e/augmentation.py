import numpy as np
import torch
import torchaudio.functional as AF

from pipeline_e.config import PipelineEConfig
from shared.constants import TARGET_SR, MAX_SAMPLES


class AugmentationPipeline:
    """
    Waveform-level augmentation for Pipeline E training.
    All transforms operate on raw float32 numpy arrays at TARGET_SR.
    At inference, only center_crop is applied.
    """

    def __init__(self, config: PipelineEConfig, rng: np.random.Generator | None = None):
        self.config = config
        self.rng = rng or np.random.default_rng()

    # ── Crop helpers ──────────────────────────────────────────────────────────

    def random_crop(self, y: np.ndarray, target_len: int = MAX_SAMPLES) -> np.ndarray:
        if len(y) >= target_len:
            start = self.rng.integers(0, len(y) - target_len + 1)
            return y[start: start + target_len]
        pad = target_len - len(y)
        return np.pad(y, (0, pad))

    def center_crop(self, y: np.ndarray, target_len: int = MAX_SAMPLES) -> np.ndarray:
        if len(y) >= target_len:
            start = (len(y) - target_len) // 2
            return y[start: start + target_len]
        pad = target_len - len(y)
        return np.pad(y, (pad // 2, pad - pad // 2))

    # ── Individual augmentations ──────────────────────────────────────────────

    def speed_perturb(self, y: np.ndarray) -> np.ndarray:
        lo, hi = self.config.speed_range
        rate = float(self.rng.uniform(lo, hi))
        if abs(rate - 1.0) < 1e-3:
            return y
        orig_sr = TARGET_SR
        new_sr = int(round(orig_sr * rate))
        t = torch.from_numpy(y).unsqueeze(0)
        resampled = AF.resample(t, orig_sr, new_sr).squeeze(0).numpy()
        return resampled

    def random_gain(self, y: np.ndarray) -> np.ndarray:
        lo, hi = self.config.gain_db_range
        gain_db = float(self.rng.uniform(lo, hi))
        return y * (10 ** (gain_db / 20.0))

    def add_gaussian_noise(self, y: np.ndarray) -> np.ndarray:
        lo, hi = self.config.noise_std_range
        std = float(self.rng.uniform(lo, hi))
        return y + self.rng.standard_normal(len(y)).astype(np.float32) * std

    def time_mask(self, y: np.ndarray) -> np.ndarray:
        """Zero out random contiguous time blocks (waveform-level SpecAugment)."""
        y = y.copy()
        for _ in range(self.config.num_time_masks):
            max_w = min(self.config.time_mask_max_frames, len(y) - 1)
            if max_w <= 0:
                break
            width = self.rng.integers(0, max_w + 1)
            start = self.rng.integers(0, max(1, len(y) - width))
            y[start: start + width] = 0.0
        return y

    # ── Main entry points ─────────────────────────────────────────────────────

    def __call__(self, y: np.ndarray, training: bool = True) -> np.ndarray:
        """Full training augmentation chain → normalized float32 of MAX_SAMPLES."""
        if not training or not self.config.use_augmentation:
            return self.center_crop(y).astype(np.float32)

        if self.config.use_speed_perturb:
            y = self.speed_perturb(y)
        if self.config.use_random_gain:
            y = self.random_gain(y)
        if self.config.use_gaussian_noise:
            y = self.add_gaussian_noise(y)
        y = self.random_crop(y)
        if self.config.use_time_masking:
            y = self.time_mask(y)

        # Amplitude normalize after augmentation
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        return y.astype(np.float32)
