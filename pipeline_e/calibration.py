import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration via a single learned temperature scalar.
    Fit on validation logits; apply at inference time.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits / self.temperature.clamp(min=0.05), dim=-1)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """LBFGS optimization of NLL on validation set. Returns final temperature."""
        logits = logits.detach()
        labels = labels.detach()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_fn():
            optimizer.zero_grad()
            scaled = logits / self.temperature.clamp(min=0.05)
            loss = criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        return self.temperature.item()
