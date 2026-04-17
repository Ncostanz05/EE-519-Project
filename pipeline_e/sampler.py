import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(
    df: pd.DataFrame,
    stratify_joint: bool = True,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that up-samples rare age×gender buckets.
    If stratify_joint=True, uses (age_label, gender_label) joint bucket;
    otherwise uses age_label only.
    """
    if stratify_joint and "gender_label" in df.columns:
        # Drop rows with unknown gender for joint stratification
        valid = df["gender_label"].notna()
        keys = df.loc[valid, "age_label"].astype(str) + "_" + df.loc[valid, "gender_label"].astype(str)
        # For rows with unknown gender, fall back to age-only key
        all_keys = df["age_label"].astype(str).copy()
        all_keys[valid] = keys
    else:
        all_keys = df["age_label"].astype(str)

    class_counts = all_keys.value_counts().to_dict()
    weights = all_keys.map(lambda k: 1.0 / class_counts[k]).values
    weights = weights.astype(float)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
