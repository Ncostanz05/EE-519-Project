#!/usr/bin/env python3
"""
Fast Pipeline-E-only evaluation on the held-out test split.

Designed for the frozen-vs-LoRA ablation: runs ONLY Pipeline E on the test set
using the same speaker-independent split (seed=42) as evaluate_all.py, and
skips pipelines A-D entirely (those numbers don't change between checkpoints).

Usage:
    # Frozen:
    python evaluate_pipeline_e_only.py \
        --pe-checkpoint /content/drive/MyDrive/pipeline_e_checkpoints/seed_0/frozen \
        --cv-csv /content/cv-data/cv-corpus-25.0-2026-03-09/en/commonvoice_subset.csv \
        --cv-audio-dir /content/cv-data/cv-corpus-25.0-2026-03-09/en/clips \
        --tag frozen

    # LoRA (for direct comparison):
    python evaluate_pipeline_e_only.py \
        --pe-checkpoint /content/drive/MyDrive/pipeline_e_checkpoints/seed_0/lora \
        --cv-csv /content/cv-data/cv-corpus-25.0-2026-03-09/en/commonvoice_subset.csv \
        --cv-audio-dir /content/cv-data/cv-corpus-25.0-2026-03-09/en/clips \
        --tag lora

    # Smoke test:
    python evaluate_pipeline_e_only.py --pe-checkpoint ... --limit 100 --tag smoke

Expected runtime: ~30-45 min on A100 for 15K test samples.
"""

import argparse
import os
import sys
import time

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from shared.constants import AGE_LABELS, TARGET_SR
from shared.data_utils import load_commonvoice_df, speaker_independent_split
from shared.metrics import bootstrap_ci, compute_all_metrics


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Skip MPS: WavLM-large degenerate on Apple Metal (same as inference.py).
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pe-checkpoint", required=True, dest="pe_checkpoint",
                        help="Path to Pipeline E checkpoint dir (contains best_model.pt + calibrator.pt)")
    parser.add_argument("--cv-csv", required=True,
                        help="Path to commonvoice_subset.csv")
    parser.add_argument("--cv-audio-dir", required=True,
                        help="Path to directory of audio clips")
    parser.add_argument("--tag", default="e_only",
                        help="Tag for output filenames, e.g. 'frozen' or 'lora'")
    parser.add_argument("--output-dir", default="eval_results_e_only",
                        help="Directory for output CSV + plots")
    parser.add_argument("--test-seed", type=int, default=42, dest="test_seed",
                        help="Seed for speaker-independent split (must match training)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit test set size (for smoke tests)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = _get_device()
    print(f"Device: {device}")
    print(f"Tag:    {args.tag}")
    print(f"Ckpt:   {args.pe_checkpoint}")

    # ── Load test split ────────────────────────────────────────────────────────
    print("\nLoading dataset + computing test split...")
    df = load_commonvoice_df(args.cv_csv, args.cv_audio_dir)
    print(f"  {len(df)} valid samples total")
    _, _, test_idx = speaker_independent_split(df, seed=args.test_seed)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    if args.limit:
        test_df = test_df.iloc[: args.limit].reset_index(drop=True)
    print(f"  Test set: {len(test_df)} samples")

    # ── Load Pipeline E ────────────────────────────────────────────────────────
    print("\nLoading Pipeline E...")
    from pipeline_e.inference import PipelineE
    ckpt_pt = os.path.join(args.pe_checkpoint, "best_model.pt")
    if not os.path.exists(ckpt_pt):
        print(f"  [FATAL] {ckpt_pt} not found")
        return
    pipe_e = PipelineE(args.pe_checkpoint, device=device)
    print("  loaded.")

    # ── Inference loop ─────────────────────────────────────────────────────────
    print(f"\nRunning Pipeline E on {len(test_df)} test samples...")
    y_true, y_pred, y_proba = [], [], []
    gender_true, gender_pred = [], []
    errors = 0
    t0 = time.time()

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        try:
            y, _ = librosa.load(row["audio_path"], sr=TARGET_SR, mono=True)
            result = pipe_e.predict(y, sr=TARGET_SR, use_mc_dropout=False)
            y_true.append(int(row["age_label"]))
            y_pred.append(int(result["age_pred"]))
            y_proba.append(np.array(result["age_proba"], dtype=float))
            gender_true.append(int(row["gender_label"]) if pd.notna(row.get("gender_label")) else np.nan)
            gender_pred.append(int(result["gender_pred"]))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [warn] error on {row.get('audio_path','?')}: {e}")

    dt = time.time() - t0
    print(f"\nInference done in {dt/60:.1f} min ({dt/len(test_df):.2f} s/sample)")
    if errors:
        print(f"  {errors} samples failed")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    gender_true_arr = np.array(gender_true, dtype=float)
    gender_pred_arr = np.array(gender_pred, dtype=float)

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = compute_all_metrics(y_true, y_pred, y_proba, gender_true_arr, gender_pred_arr)

    f1_lo, f1_hi = bootstrap_ci(
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        y_true, y_pred,
    )
    acc_lo, acc_hi = bootstrap_ci(accuracy_score, y_true, y_pred)
    metrics["macro_f1_ci"] = (round(f1_lo, 4), round(f1_hi, 4))
    metrics["exact_acc_ci"] = (round(acc_lo, 4), round(acc_hi, 4))

    # ── Print headline ─────────────────────────────────────────────────────────
    print("\n=== PIPELINE E RESULTS ===")
    print(f"  Tag:           {args.tag}")
    print(f"  N test:        {len(y_true)}")
    print(f"  Exact acc:     {metrics['exact_acc']:.4f}  CI {metrics['exact_acc_ci']}")
    print(f"  Balanced acc:  {metrics['balanced_acc']:.4f}")
    print(f"  Macro F1:      {metrics['macro_f1']:.4f}  CI {metrics['macro_f1_ci']}")
    print(f"  Adjacent acc:  {metrics['adjacent_acc']:.4f}")
    print(f"  ECE:           {metrics['ece']:.4f}")
    print(f"  Brier:         {metrics['brier_score']:.4f}")

    if "subgroup" in metrics:
        for grp, gm in metrics["subgroup"].items():
            print(f"  [{grp}] acc={gm['exact_acc']:.3f}  f1={gm['macro_f1']:.3f}  n={gm['n']}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    out_csv = os.path.join(args.output_dir, f"pipeline_e_{args.tag}_metrics.csv")
    row = {
        "tag": args.tag,
        "checkpoint": args.pe_checkpoint,
        "n_test": len(y_true),
        "exact_acc": round(metrics["exact_acc"], 4),
        "exact_acc_ci_lo": metrics["exact_acc_ci"][0],
        "exact_acc_ci_hi": metrics["exact_acc_ci"][1],
        "balanced_acc": round(metrics["balanced_acc"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "macro_f1_ci_lo": metrics["macro_f1_ci"][0],
        "macro_f1_ci_hi": metrics["macro_f1_ci"][1],
        "adjacent_acc": round(metrics["adjacent_acc"], 4),
        "ece": round(metrics["ece"], 4),
        "brier_score": round(metrics["brier_score"], 4),
    }
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ── Confusion matrix plot ──────────────────────────────────────────────────
    cm = np.array(metrics["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_title(f"Pipeline E ({args.tag}) — confusion matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(AGE_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(AGE_LABELS, fontsize=8)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm_norm[i,j] > 0.5 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, f"confusion_matrix_{args.tag}.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"Saved: {cm_path}")

    # ── Save raw predictions for further analysis ──────────────────────────────
    np.savez(
        os.path.join(args.output_dir, f"predictions_{args.tag}.npz"),
        y_true=y_true, y_pred=y_pred, y_proba=y_proba,
        gender_true=gender_true_arr, gender_pred=gender_pred_arr,
    )
    print(f"Saved: {args.output_dir}/predictions_{args.tag}.npz")

    print("\nDone.")


if __name__ == "__main__":
    main()
