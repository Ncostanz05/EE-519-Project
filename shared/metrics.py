import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix,
)


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= 1))


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error (confidence of argmax vs accuracy)."""
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / len(y_true))


def compute_brier_score(
    y_true: np.ndarray, y_proba: np.ndarray, num_classes: int = 7
) -> float:
    one_hot = np.zeros((len(y_true), num_classes), dtype=float)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    gender_true: np.ndarray | None = None,
    gender_pred: np.ndarray | None = None,
    num_age_classes: int = 7,
) -> dict:
    results = {
        "exact_acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "adjacent_acc": adjacent_accuracy(y_true, y_pred),
        "ece": compute_ece(y_true, y_proba),
        "brier_score": compute_brier_score(y_true, y_proba, num_age_classes),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_age_classes))
        ).tolist(),
    }

    if gender_true is not None and gender_pred is not None:
        mask = ~np.isnan(gender_true.astype(float))
        gt = gender_true[mask].astype(int)
        gp = gender_pred[mask].astype(int)
        results["gender_confusion"] = confusion_matrix(gt, gp, labels=[0, 1]).tolist()
        results["gender_acc"] = float(accuracy_score(gt, gp))

    # Subgroup breakdown by gender
    if gender_true is not None:
        subgroup = {}
        for g, name in enumerate(["male", "female"]):
            mask = gender_true == g
            if mask.sum() == 0:
                continue
            subgroup[name] = {
                "exact_acc": float(accuracy_score(y_true[mask], y_pred[mask])),
                "macro_f1": float(
                    f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
                ),
                "adjacent_acc": adjacent_accuracy(y_true[mask], y_pred[mask]),
                "n": int(mask.sum()),
            }
        results["subgroup"] = subgroup

    return results


def bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = (1 - ci) / 2
    hi = 1 - lo
    return float(np.quantile(scores, lo)), float(np.quantile(scores, hi))
