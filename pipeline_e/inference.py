import os

import numpy as np
import torch
from transformers import AutoFeatureExtractor

from pipeline_e.calibration import TemperatureScaler
from pipeline_e.config import PipelineEConfig
from pipeline_e.model import WavLMMultiTaskModel
from shared.constants import AGE_LABELS, GENDER_LABELS, MAX_SAMPLES, TARGET_SR


class PipelineE:
    """
    Inference wrapper for Pipeline E.
    Loads model + optional calibrator; exposes predict() for demo and evaluate_all.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: torch.device | str | None = None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = torch.device(device) if isinstance(device, str) else device

        ckpt = torch.load(os.path.join(checkpoint_dir, "best_model.pt"), map_location=self.device)
        self.config: PipelineEConfig = ckpt["config"]

        self.model = WavLMMultiTaskModel(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.config.backbone_name)

        cal_path = os.path.join(checkpoint_dir, "calibrator.pt")
        self.calibrator = None
        if os.path.exists(cal_path):
            self.calibrator = TemperatureScaler()
            self.calibrator.load_state_dict(torch.load(cal_path, map_location=self.device))
            self.calibrator.to(self.device).eval()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _prep_waveform(self, y: np.ndarray) -> torch.Tensor:
        if len(y) >= MAX_SAMPLES:
            start = (len(y) - MAX_SAMPLES) // 2
            y = y[start: start + MAX_SAMPLES]
        else:
            y = np.pad(y, (0, MAX_SAMPLES - len(y)))
        y = y.astype(np.float32)
        encoded = self.feature_extractor(
            [y],
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_SAMPLES,
            truncation=True,
        )
        return encoded.input_values.to(self.device)

    def _ordinal_to_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """K-1 cumulative logits → 7-class probability simplex."""
        cum = torch.sigmoid(logits)     # (B, K-1)
        b, k = cum.shape
        p = torch.zeros(b, k + 1, device=logits.device)
        p[:, 0] = 1.0 - cum[:, 0]
        for i in range(1, k):
            p[:, i] = cum[:, i - 1] - cum[:, i]
        p[:, -1] = cum[:, -1]
        return p.clamp(min=0)

    def _age_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Logits → calibrated/softmax probability tensor (B, 7)."""
        if self.config.age_head_type == "ordinal":
            return self._ordinal_to_proba(logits)
        if self.calibrator is not None:
            return self.calibrator(logits)
        return torch.softmax(logits, dim=-1)

    # ── Public API ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        y: np.ndarray,
        sr: int = TARGET_SR,
        use_mc_dropout: bool = False,
    ) -> dict:
        """
        Predict age and gender for a single raw waveform.

        Returns:
          age_pred (int), age_label (str), age_proba (list[float] len 7),
          top3_ages ([{bin, label, prob}]),
          gender_pred (int), gender_label (str), gender_proba (float),
          uncertainty (float | None), embedding (list[float] len 1024)
        """
        if sr != TARGET_SR:
            import torchaudio.functional as AF
            t = torch.from_numpy(y).float().unsqueeze(0)
            y = AF.resample(t, sr, TARGET_SR).squeeze(0).numpy()

        input_values = self._prep_waveform(y)

        uncertainty = None
        if use_mc_dropout:
            mc = self.model.predict_with_mc_dropout(input_values, n_samples=self.config.mc_dropout_samples)
            age_proba_t = mc["mean_proba"][0]
            uncertainty = float(mc["uncertainty"][0].item())
            out = self.model(input_values)
        else:
            out = self.model(input_values)
            age_proba_t = self._age_proba(out["age_logits"])[0]

        age_pred = int(age_proba_t.argmax().item())
        age_proba = age_proba_t.cpu().float().tolist()

        top3_idx = age_proba_t.topk(min(3, len(age_proba))).indices.tolist()
        top3_ages = [
            {"bin": i, "label": AGE_LABELS[i], "prob": round(age_proba[i], 4)}
            for i in top3_idx
        ]

        gender_proba_val = float(out["gender_proba"][0].item())
        gender_pred = int(gender_proba_val > 0.5)

        return {
            "age_pred": age_pred,
            "age_label": AGE_LABELS[age_pred],
            "age_proba": age_proba,
            "top3_ages": top3_ages,
            "gender_pred": gender_pred,
            "gender_label": GENDER_LABELS[gender_pred],
            "gender_proba": round(gender_proba_val, 4),
            "uncertainty": round(uncertainty, 6) if uncertainty is not None else None,
            "embedding": out["embedding"][0].cpu().float().tolist(),
        }

    @torch.no_grad()
    def predict_batch(self, batch: dict) -> dict:
        """
        Batch inference from a pre-processed collate_fn batch.
        Used by evaluate_all.py. Returns numpy arrays.
        """
        iv = batch["input_values"].to(self.device)
        am = batch.get("attention_mask")
        if am is not None:
            am = am.to(self.device)

        out = self.model(iv, am)
        age_proba = self._age_proba(out["age_logits"])

        return {
            "age_pred": age_proba.argmax(dim=-1).cpu().numpy(),
            "age_proba": age_proba.cpu().float().numpy(),
            "gender_pred": (out["gender_proba"].squeeze(-1) > 0.5).long().cpu().numpy(),
            "gender_proba": out["gender_proba"].squeeze(-1).cpu().float().numpy(),
            "embedding": out["embedding"].cpu().float().numpy(),
        }
