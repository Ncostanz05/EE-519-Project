import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline_e.config import PipelineEConfig


def build_adjacency_smooth_targets(
    labels: torch.Tensor,
    num_classes: int = 7,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Soft targets with adjacency smoothing: mass shifted toward neighboring bins.
    Each row sums to 1.
    """
    batch_size = labels.size(0)
    targets = torch.zeros(batch_size, num_classes, device=labels.device)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)

    smooth = torch.zeros_like(targets)
    smooth[:, :-1] += targets[:, 1:]   # left neighbor gets mass from right
    smooth[:, 1:] += targets[:, :-1]   # right neighbor gets mass from left
    # normalize neighbor contributions (each target bin has at most 2 neighbors)
    neighbor_count = (smooth > 0).float().sum(dim=1, keepdim=True).clamp(min=1)
    smooth = smooth / neighbor_count

    soft = (1 - alpha) * targets + alpha * smooth
    # re-normalize rows
    return soft / soft.sum(dim=1, keepdim=True)


class LDLLoss(nn.Module):
    """Label Distribution Learning: KL divergence + auxiliary CE for stability."""

    def __init__(self, num_classes: int = 7, alpha: float = 0.1, aux_weight: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.aux_weight = aux_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        soft = build_adjacency_smooth_targets(targets, self.num_classes, self.alpha)
        log_probs = F.log_softmax(logits, dim=-1)
        kl = F.kl_div(log_probs, soft, reduction="batchmean")
        ce = F.cross_entropy(logits, targets)
        return kl + self.aux_weight * ce


class OrdinalLoss(nn.Module):
    """
    CORN-style ordinal regression.
    Model outputs K-1 cumulative logits: P(y > k) for k in 0..K-2.
    Loss is sum of binary cross-entropies across K-1 tasks.
    """

    def __init__(self, num_classes: int = 7, alpha: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, K-1) cumulative sigmoid scores
        # Build binary targets: for sample i with label c, P(y>k)=1 for k<c, 0 otherwise
        batch_size = logits.size(0)
        k = self.num_classes - 1
        binary_targets = torch.zeros(batch_size, k, device=logits.device)
        for threshold in range(k):
            binary_targets[:, threshold] = (targets > threshold).float()
        return F.binary_cross_entropy_with_logits(logits, binary_targets)


class SoftmaxAgeLoss(nn.Module):
    """Plain cross-entropy with optional adjacency label smoothing."""

    def __init__(self, num_classes: int = 7, alpha: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        soft = build_adjacency_smooth_targets(targets, self.num_classes, self.alpha)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft * log_probs).sum(dim=-1).mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, config: PipelineEConfig):
        super().__init__()
        self.w_age = config.age_loss_weight
        self.w_gender = config.gender_loss_weight

        if config.age_head_type == "ldl":
            self.age_loss = LDLLoss(
                num_classes=7,
                alpha=config.adjacency_smoothing_alpha,
                aux_weight=config.ldl_aux_ce_weight,
            )
        elif config.age_head_type == "ordinal":
            self.age_loss = OrdinalLoss(
                num_classes=7, alpha=config.adjacency_smoothing_alpha
            )
        else:
            self.age_loss = SoftmaxAgeLoss(
                num_classes=7, alpha=config.adjacency_smoothing_alpha
            )

    def forward(
        self,
        age_logits: torch.Tensor,
        gender_logits: torch.Tensor,
        age_targets: torch.Tensor,
        gender_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        age_loss = self.age_loss(age_logits, age_targets)

        # Gender loss — mask out NaN entries
        gender_mask = ~torch.isnan(gender_targets)
        if gender_mask.any():
            gender_loss = F.binary_cross_entropy_with_logits(
                gender_logits.squeeze(-1)[gender_mask],
                gender_targets[gender_mask],
            )
        else:
            gender_loss = torch.tensor(0.0, device=age_logits.device)

        total = self.w_age * age_loss + self.w_gender * gender_loss
        return total, {"age_loss": age_loss.item(), "gender_loss": gender_loss.item()}
