import torch
import torch.nn as nn
from transformers import AutoModel

from pipeline_e.config import PipelineEConfig


class ExpertAgeHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_outputs: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GenderHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WavLMMultiTaskModel(nn.Module):
    """
    Backbone: WavLM-large or wav2vec2-large-robust (configurable).
    Shared projection → gender head + soft MoE age head.

    forward() returns dict:
      age_logits:    (B, num_age_outputs)
      gender_logits: (B, 1)
      gender_proba:  (B, 1)  — sigmoid, no detach (gradient flows through gate)
      embedding:     (B, embedding_dim)  — pooled backbone output for UMAP/saliency
    """

    def __init__(self, config: PipelineEConfig):
        super().__init__()
        self.config = config

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = AutoModel.from_pretrained(config.backbone_name)

        if config.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        elif config.use_lora:
            try:
                from peft import get_peft_model, LoraConfig
                lora_cfg = LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    target_modules=config.lora_target_modules,
                    lora_dropout=config.lora_dropout,
                    bias="none",
                )
                self.backbone = get_peft_model(self.backbone, lora_cfg)
            except ImportError:
                raise ImportError("Install peft: pip install peft")

        # ── Shared projection ─────────────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # ── Gender head ───────────────────────────────────────────────────────
        self.gender_head = GenderHead(config.hidden_dim, config.hidden_dim // 2)

        # ── Age head(s) ───────────────────────────────────────────────────────
        num_age_out = config.num_age_outputs()
        if config.use_gender_moe:
            self.male_expert = ExpertAgeHead(config.hidden_dim, config.hidden_dim // 2, num_age_out)
            self.female_expert = ExpertAgeHead(config.hidden_dim, config.hidden_dim // 2, num_age_out)
        else:
            self.age_head = ExpertAgeHead(config.hidden_dim, config.hidden_dim // 2, num_age_out)

    def _pool(self, backbone_out) -> torch.Tensor:
        return backbone_out.last_hidden_state.mean(dim=1)  # (B, embedding_dim)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        out = self.backbone(input_values=input_values, attention_mask=attention_mask)
        embedding = self._pool(out)          # (B, 1024)
        feat = self.proj(embedding)          # (B, 256)

        gender_logit = self.gender_head(feat)          # (B, 1)
        gender_proba = torch.sigmoid(gender_logit)     # soft gate, no detach

        if self.config.use_gender_moe:
            p_female = gender_proba                    # (B, 1)
            p_male = 1.0 - p_female
            male_out = self.male_expert(feat)
            female_out = self.female_expert(feat)
            age_logit = p_male * male_out + p_female * female_out
        else:
            age_logit = self.age_head(feat)

        return {
            "age_logits": age_logit,
            "gender_logits": gender_logit,
            "gender_proba": gender_proba,
            "embedding": embedding,
        }

    @torch.no_grad()
    def predict_with_mc_dropout(
        self, input_values: torch.Tensor, n_samples: int = 10
    ) -> dict:
        """MC dropout uncertainty estimate. Enables dropout at inference."""
        self.train()
        preds = []
        for _ in range(n_samples):
            out = self.forward(input_values)
            preds.append(torch.softmax(out["age_logits"], dim=-1))
        self.eval()
        preds = torch.stack(preds, dim=0)   # (n, B, 7)
        mean_proba = preds.mean(dim=0)      # (B, 7)
        variance = preds.var(dim=0).sum(-1) # (B,) scalar uncertainty
        return {"mean_proba": mean_proba, "uncertainty": variance}
