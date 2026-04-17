import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from shared.constants import AGE_MAP, GENDER_MAP


def load_commonvoice_df(
    csv_path: str = "commonvoice-v24_en-AU/commonvoice-v24_en-AU.csv",
    audio_dir: str = "commonvoice-v24_en-AU/audio_files",
    require_gender: bool = True,
) -> pd.DataFrame:
    """
    Load full 40K Common Voice AU CSV. Filters to valid age/gender labels and
    existing audio files. Adds age_label (int), gender_label (int or NaN),
    audio_path (full path). Preserves client_id for speaker-independent splits.
    """
    df = pd.read_csv(csv_path)

    # Filter valid age labels
    df = df[df["age"].notna() & df["age"].isin(AGE_MAP.keys())].copy()
    df["age_label"] = df["age"].map(AGE_MAP).astype(int)

    # Map gender (NaN for unknowns)
    df["gender_label"] = df["gender"].map(GENDER_MAP)  # NaN for non-binary/missing
    if require_gender:
        df = df[df["gender_label"].notna()].copy()
        df["gender_label"] = df["gender_label"].astype(int)

    # Resolve audio paths and filter to existing files
    df["audio_path"] = df["path"].apply(lambda p: os.path.join(audio_dir, p))
    df = df[df["audio_path"].apply(os.path.exists)].copy()

    df = df.reset_index(drop=True)
    return df


def speaker_independent_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-stage GroupShuffleSplit on client_id for a strict 70/15/15 split.
    Returns (train_idx, val_idx, test_idx) as index arrays into df.
    """
    n = len(df)
    groups = df["client_id"].values
    dummy = np.zeros(n)

    # Stage 1: carve out test+val (30%)
    split1 = GroupShuffleSplit(n_splits=1, test_size=1 - train_frac, random_state=seed)
    train_idx, temp_idx = next(split1.split(dummy, groups=groups))

    # Stage 2: split temp into val / test (50/50 of the 30%)
    temp_groups = groups[temp_idx]
    dummy_temp = np.zeros(len(temp_idx))
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel, test_rel = next(split2.split(dummy_temp, groups=temp_groups))

    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    return train_idx, val_idx, test_idx


def load_sps_df(sps_dir: str = "sps-corpus-1.0-2025-11-25-en") -> pd.DataFrame:
    """
    Load labeled samples from SPS corpus. Returns df with:
    audio_path, age_label, gender_label, age (str), gender (str).
    Returns empty DataFrame if corpus has no labeled samples.
    """
    tsv_path = os.path.join(sps_dir, "ss-corpus-en.tsv")
    audio_dir = os.path.join(sps_dir, "audios")

    if not os.path.exists(tsv_path):
        print(f"  SPS manifest not found at {tsv_path}")
        return pd.DataFrame()

    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["age"].notna() & df["age"].isin(AGE_MAP.keys())].copy()
    df = df[df["gender"].notna() & df["gender"].isin(GENDER_MAP.keys())].copy()

    df["age_label"] = df["age"].map(AGE_MAP).astype(int)
    df["gender_label"] = df["gender"].map(GENDER_MAP).astype(int)
    df["audio_path"] = df["audio_file"].apply(lambda p: os.path.join(audio_dir, p))
    df = df[df["audio_path"].apply(os.path.exists)].copy()

    return df.reset_index(drop=True)
