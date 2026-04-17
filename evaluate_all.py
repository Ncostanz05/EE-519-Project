#!/usr/bin/env python3
"""
Unified 5-pipeline benchmark evaluation on the full CV-AU test set.

Usage:
    python evaluate_all.py
    python evaluate_all.py --pe-checkpoint pipeline_e_checkpoints/seed_0/lora
    python evaluate_all.py --limit 500   # quick run on first N test samples
"""

import argparse
import os
import sys
import warnings

import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from shared.constants import AGE_LABELS, GENDER_LABELS, TARGET_SR, MAX_SAMPLES
from shared.data_utils import load_commonvoice_df, speaker_independent_split
from shared.metrics import (
    bootstrap_ci, compute_all_metrics, compute_ece, adjacent_accuracy,
)

RESULTS_DIR = "eval_results"


# ── Pipeline loader helpers ────────────────────────────────────────────────────

def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pipeline_a():
    from features import extract_features
    rf_path = os.path.join(PROJECT_DIR, "age_model.pkl")
    if not os.path.exists(rf_path):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = joblib.load(rf_path)
    return {
        "type": "rf_handcrafted",
        "estimator": pipeline.named_steps["model"],
        "scaler": pipeline.named_steps.get("scaler", None),
        "extract": extract_features,
    }


def load_pipeline_b(device):
    from cnn_model import AgeAudioCNN
    import torchaudio.transforms as T
    cnn_path = os.path.join(PROJECT_DIR, "cnn_age_model.pth")
    if not os.path.exists(cnn_path):
        return None
    model = AgeAudioCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(cnn_path, map_location=device))
    model.eval()
    mel = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64)
    amp2db = T.AmplitudeToDB()
    return {"type": "cnn", "model": model, "mel": mel, "amp2db": amp2db, "device": device}


def load_pipeline_c_d(device):
    from transformers import AutoFeatureExtractor, Wav2Vec2Model
    emb_age_path = os.path.join(PROJECT_DIR, "embeddings_age_model.pkl")
    emb_scaler_path = os.path.join(PROJECT_DIR, "embeddings_scaler.pkl")
    gender_path = os.path.join(PROJECT_DIR, "gender_model.pkl")
    male_path = os.path.join(PROJECT_DIR, "age_model_male.pkl")
    female_path = os.path.join(PROJECT_DIR, "age_model_female.pkl")

    if not (os.path.exists(emb_age_path) and os.path.exists(emb_scaler_path)):
        return None, None

    emb_age = joblib.load(emb_age_path)
    scaler = joblib.load(emb_scaler_path)

    gender_model = joblib.load(gender_path) if os.path.exists(gender_path) else None
    male_model = joblib.load(male_path) if os.path.exists(male_path) else None
    female_model = joblib.load(female_path) if os.path.exists(female_path) else None

    try:
        processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
        w2v.eval()
    except Exception as e:
        print(f"  Could not load Wav2Vec2: {e}")
        return None, None

    common = {"processor": processor, "w2v": w2v, "scaler": scaler, "device": device}
    c = {"type": "emb_age", "emb_age": emb_age, **common}
    d = {
        "type": "gender_conditioned",
        "gender_model": gender_model,
        "male_model": male_model,
        "female_model": female_model,
        **common,
    } if (gender_model and male_model and female_model) else None
    return c, d


def load_pipeline_e(checkpoint_dir, device):
    from pipeline_e.inference import PipelineE
    if not checkpoint_dir or not os.path.exists(os.path.join(checkpoint_dir, "best_model.pt")):
        return None
    try:
        return PipelineE(checkpoint_dir, device=device)
    except Exception as e:
        print(f"  Could not load Pipeline E from {checkpoint_dir}: {e}")
        return None


# ── Per-sample inference functions ────────────────────────────────────────────

def infer_a(pipe, y, sr):
    from sklearn.impute import SimpleImputer
    feats = pipe["extract"](y, sr)
    X = pd.DataFrame([feats])
    X = SimpleImputer(strategy="mean").fit_transform(X)
    if pipe["scaler"] is not None:
        X = pipe["scaler"].transform(X)
    pred = int(pipe["estimator"].predict(X)[0])
    try:
        proba = pipe["estimator"].predict_proba(X)[0]
    except Exception:
        proba = np.zeros(7); proba[pred] = 1.0
    return pred, proba


def infer_b(pipe, y):
    import torchaudio
    wt = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    if wt.shape[1] > MAX_SAMPLES:
        wt = wt[:, :MAX_SAMPLES]
    elif wt.shape[1] < MAX_SAMPLES:
        wt = F.pad(wt, (0, MAX_SAMPLES - wt.shape[1]))
    mel = pipe["mel"](wt)
    mel_db = pipe["amp2db"](mel)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    inp = mel_db.unsqueeze(0).to(pipe["device"])
    with torch.no_grad():
        logits = pipe["model"](inp)
        proba = F.softmax(logits, dim=1).cpu().numpy()[0]
    return int(np.argmax(proba)), proba


def _get_w2v_embedding(pipe, y):
    target_len = int(TARGET_SR * 3.0)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    inputs = pipe["processor"](y, sampling_rate=TARGET_SR, return_tensors="pt").to(pipe["device"])
    with torch.no_grad():
        out = pipe["w2v"](**inputs)
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def infer_c(pipe, y):
    emb = _get_w2v_embedding(pipe, y)
    X = pipe["scaler"].transform(emb.reshape(1, -1))
    pred = int(pipe["emb_age"].predict(X)[0])
    try:
        proba = pipe["emb_age"].predict_proba(X)[0]
    except Exception:
        proba = np.zeros(7); proba[pred] = 1.0
    return pred, proba


def infer_d(pipe, y):
    emb = _get_w2v_embedding(pipe, y)
    X = pipe["scaler"].transform(emb.reshape(1, -1))
    gender = int(pipe["gender_model"].predict(X)[0])
    age_model = pipe["male_model"] if gender == 0 else pipe["female_model"]
    pred = int(age_model.predict(X)[0])
    try:
        proba = age_model.predict_proba(X)[0]
    except Exception:
        proba = np.zeros(7); proba[pred] = 1.0
    gender_proba = pipe["gender_model"].predict_proba(X)[0] if hasattr(pipe["gender_model"], "predict_proba") else None
    return pred, proba, gender, gender_proba


def infer_e(pipe_e, audio_path):
    y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    result = pipe_e.predict(y, sr=TARGET_SR, use_mc_dropout=False)
    proba = np.array(result["age_proba"])
    return result["age_pred"], proba, result["gender_pred"], result["gender_proba"]


# ── Evaluation loop ────────────────────────────────────────────────────────────

def run_pipeline(name, infer_fn, test_df, limit=None):
    rows = test_df.iterrows() if limit is None else list(test_df.iterrows())[:limit]
    y_true, y_pred, y_proba = [], [], []
    gender_true_list, gender_pred_list = [], []
    errors = 0

    for _, row in rows:
        try:
            y_true.append(int(row["age_label"]))
            gender_true_list.append(int(row["gender_label"]) if pd.notna(row.get("gender_label")) else np.nan)
            result = infer_fn(row)
            if result is None:
                raise ValueError("None result")
            if len(result) == 4:
                pred, proba, g_pred, _ = result
                gender_pred_list.append(int(g_pred) if g_pred is not None else -1)
            else:
                pred, proba = result
                gender_pred_list.append(-1)
            y_pred.append(int(pred))
            y_proba.append(np.array(proba, dtype=float))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    [warn] {name}: error on {row.get('audio_path','?')}: {e}")
            y_pred.append(0)
            y_proba.append(np.zeros(7)); y_proba[-1][0] = 1.0
            gender_pred_list.append(-1)

    if errors > 0:
        print(f"  [{name}] {errors} errors out of {len(y_true)}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    gender_true = np.array(gender_true_list, dtype=float)
    gender_pred = np.where(np.array(gender_pred_list) >= 0, np.array(gender_pred_list), np.nan)

    metrics = compute_all_metrics(y_true, y_pred, y_proba, gender_true, gender_pred)

    # Bootstrap CI on macro-F1 and exact-acc
    from sklearn.metrics import f1_score, accuracy_score
    f1_lo, f1_hi = bootstrap_ci(
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        y_true, y_pred
    )
    acc_lo, acc_hi = bootstrap_ci(accuracy_score, y_true, y_pred)
    metrics["macro_f1_ci"] = (round(f1_lo, 4), round(f1_hi, 4))
    metrics["exact_acc_ci"] = (round(acc_lo, 4), round(acc_hi, 4))

    return metrics, y_true, y_pred, y_proba


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_confusion_matrices(results_by_name, save_dir):
    names = [n for n, r in results_by_name.items() if r is not None]
    n = len(names)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        cm = np.array(results_by_name[name]["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels(AGE_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(AGE_LABELS, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrices.png"), dpi=120)
    plt.close()


def plot_calibration_curves(results_by_name, true_by_name, proba_by_name, save_dir):
    names = [n for n in results_by_name if results_by_name[n] is not None]
    n = len(names)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    n_bins = 10
    for ax, name in zip(axes, names):
        y_true = true_by_name[name]
        y_proba = proba_by_name[name]
        conf = y_proba.max(axis=1)
        pred = y_proba.argmax(axis=1)
        correct = (pred == y_true).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_acc, bin_conf = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf >= lo) & (conf < hi)
            if mask.sum() > 0:
                bin_acc.append(correct[mask].mean())
                bin_conf.append(conf[mask].mean())
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        if bin_conf:
            ax.plot(bin_conf, bin_acc, "o-", linewidth=1.5, label=name)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{name} calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "calibration_curves.png"), dpi=120)
    plt.close()


def try_umap(pipe_e, test_df, save_dir, limit=500):
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed, skipping UMAP plot")
        return

    rows = list(test_df.iterrows())[:limit]
    embeddings, age_labels, gender_labels = [], [], []
    for _, row in rows:
        try:
            y, _ = librosa.load(row["audio_path"], sr=TARGET_SR, mono=True)
            res = pipe_e.predict(y)
            embeddings.append(res["embedding"])
            age_labels.append(int(row["age_label"]))
            gender_labels.append(int(row.get("gender_label", 0)))
        except Exception:
            pass

    if len(embeddings) < 10:
        return

    emb_arr = np.array(embeddings)
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb_2d = reducer.fit_transform(emb_arr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    scatter1 = ax1.scatter(emb_2d[:, 0], emb_2d[:, 1], c=age_labels, cmap="tab10", s=8, alpha=0.7)
    plt.colorbar(scatter1, ax=ax1, label="Age bin")
    ax1.set_title("Pipeline E embeddings — age")

    scatter2 = ax2.scatter(emb_2d[:, 0], emb_2d[:, 1], c=gender_labels, cmap="bwr", s=8, alpha=0.7)
    plt.colorbar(scatter2, ax=ax2, label="Gender (0=M, 1=F)")
    ax2.set_title("Pipeline E embeddings — gender")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap_embeddings.png"), dpi=120)
    plt.close()
    print(f"  UMAP saved ({len(embeddings)} samples)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pe-checkpoint", default=None, dest="pe_checkpoint",
                        help="Path to Pipeline E best_model.pt directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit test set size (for quick runs)")
    parser.add_argument("--test-seed", type=int, default=42, dest="test_seed")
    parser.add_argument("--cv-csv", default="commonvoice-v24_en-AU/commonvoice-v24_en-AU.csv")
    parser.add_argument("--cv-audio-dir", default="commonvoice-v24_en-AU/audio_files")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = _get_device()
    print(f"Device: {device}")

    # ── Load test set ──────────────────────────────────────────────────────────
    print("\nLoading test split from full CV-AU dataset...")
    df = load_commonvoice_df(args.cv_csv, args.cv_audio_dir)
    print(f"  {len(df)} valid samples total")
    _, _, test_idx = speaker_independent_split(df, seed=args.test_seed)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    if args.limit:
        test_df = test_df.iloc[:args.limit].reset_index(drop=True)
    print(f"  Test set: {len(test_df)} samples")

    # ── Load pipelines ─────────────────────────────────────────────────────────
    print("\nLoading pipelines...")
    pipe_a = load_pipeline_a()
    pipe_b = load_pipeline_b(device)
    pipe_c, pipe_d = load_pipeline_c_d(device)
    pipe_e = load_pipeline_e(args.pe_checkpoint, device)

    pipeline_names = {
        "A (RF+handcrafted)": pipe_a,
        "B (CNN+mel)": pipe_b,
        "C (w2v+RF)": pipe_c,
        "D (gender-cond RF)": pipe_d,
        "E (WavLM MoE)": pipe_e,
    }
    for name, p in pipeline_names.items():
        status = "loaded" if p is not None else "NOT FOUND (skipped)"
        print(f"  [{name}]: {status}")

    # ── Run evaluations ─────────────────────────────────────────────────────────
    all_metrics = {}
    true_by_name = {}
    proba_by_name = {}

    if pipe_a is not None:
        print("\nRunning Pipeline A...")
        def infer_a_row(row):
            y, _ = librosa.load(row["audio_path"], sr=TARGET_SR, mono=True)
            return infer_a(pipe_a, y, TARGET_SR)
        metrics, yt, yp, yproba = run_pipeline("A", infer_a_row, test_df, args.limit)
        all_metrics["A (RF+handcrafted)"] = metrics
        true_by_name["A (RF+handcrafted)"] = yt
        proba_by_name["A (RF+handcrafted)"] = yproba

    if pipe_b is not None:
        print("\nRunning Pipeline B...")
        def infer_b_row(row):
            y, _ = librosa.load(row["audio_path"], sr=TARGET_SR, mono=True)
            return infer_b(pipe_b, y)
        metrics, yt, yp, yproba = run_pipeline("B", infer_b_row, test_df, args.limit)
        all_metrics["B (CNN+mel)"] = metrics
        true_by_name["B (CNN+mel)"] = yt
        proba_by_name["B (CNN+mel)"] = yproba

    if pipe_c is not None:
        print("\nRunning Pipeline C...")
        def infer_c_row(row):
            y, _ = librosa.load(row["audio_path"], sr=TARGET_SR, mono=True)
            return infer_c(pipe_c, y)
        metrics, yt, yp, yproba = run_pipeline("C", infer_c_row, test_df, args.limit)
        all_metrics["C (w2v+RF)"] = metrics
        true_by_name["C (w2v+RF)"] = yt
        proba_by_name["C (w2v+RF)"] = yproba

    if pipe_d is not None:
        print("\nRunning Pipeline D...")
        def infer_d_row(row):
            y, _ = librosa.load(row["audio_path"], sr=TARGET_SR, mono=True)
            return infer_d(pipe_d, y)
        metrics, yt, yp, yproba = run_pipeline("D", infer_d_row, test_df, args.limit)
        all_metrics["D (gender-cond RF)"] = metrics
        true_by_name["D (gender-cond RF)"] = yt
        proba_by_name["D (gender-cond RF)"] = yproba

    if pipe_e is not None:
        print("\nRunning Pipeline E...")
        def infer_e_row(row):
            return infer_e(pipe_e, row["audio_path"])
        metrics, yt, yp, yproba = run_pipeline("E", infer_e_row, test_df, args.limit)
        all_metrics["E (WavLM MoE)"] = metrics
        true_by_name["E (WavLM MoE)"] = yt
        proba_by_name["E (WavLM MoE)"] = yproba

    if not all_metrics:
        print("\nNo pipelines could be evaluated. Check that model files exist.")
        return

    # ── Benchmark table ────────────────────────────────────────────────────────
    print("\n=== BENCHMARK RESULTS ===")
    rows = []
    for name, m in all_metrics.items():
        row = {
            "Pipeline": name,
            "Exact Acc": f"{m['exact_acc']:.4f}",
            "Balanced Acc": f"{m['balanced_acc']:.4f}",
            "Macro F1": f"{m['macro_f1']:.4f}",
            "Macro F1 CI": str(m.get("macro_f1_ci", "")),
            "Adjacent Acc": f"{m['adjacent_acc']:.4f}",
            "ECE": f"{m['ece']:.4f}",
            "Brier": f"{m['brier_score']:.4f}",
        }
        rows.append(row)
        print(
            f"  {name:25s}  acc={m['exact_acc']:.3f}  bal={m['balanced_acc']:.3f}  "
            f"f1={m['macro_f1']:.3f}  adj={m['adjacent_acc']:.3f}  "
            f"ece={m['ece']:.3f}  brier={m['brier_score']:.3f}"
        )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, "benchmark_table.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/benchmark_table.csv")

    # ── Subgroup breakdown ─────────────────────────────────────────────────────
    for name, m in all_metrics.items():
        if "subgroup" in m:
            for grp, gm in m["subgroup"].items():
                print(f"  [{name}] {grp}: acc={gm['exact_acc']:.3f}  f1={gm['macro_f1']:.3f}  n={gm['n']}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_confusion_matrices(all_metrics, RESULTS_DIR)
    plot_calibration_curves(all_metrics, true_by_name, proba_by_name, RESULTS_DIR)
    print(f"Plots saved to {RESULTS_DIR}/")

    if pipe_e is not None:
        print("\nGenerating UMAP visualization (Pipeline E)...")
        try:
            test_for_umap = test_df.iloc[:500] if len(test_df) > 500 else test_df
            try_umap(pipe_e, test_for_umap, RESULTS_DIR)
        except Exception as e:
            print(f"  UMAP failed: {e}")

    # ── Cross-domain (SPS) ─────────────────────────────────────────────────────
    from shared.data_utils import load_sps_df
    sps_df = load_sps_df()
    if len(sps_df) > 0:
        print(f"\nCross-domain SPS evaluation ({len(sps_df)} samples)...")
        for pname, pipe, infer_row in [
            ("D", pipe_d, lambda row: infer_d(pipe_d, librosa.load(row["audio_path"], sr=TARGET_SR, mono=True)[0]) if pipe_d else None),
            ("E", pipe_e, lambda row: infer_e(pipe_e, row["audio_path"]) if pipe_e else None),
        ]:
            if pipe is None:
                continue
            sps_preds, sps_true = [], []
            for _, row in sps_df.iterrows():
                try:
                    res = infer_row(row)
                    if res is not None:
                        sps_true.append(int(row["age_label"]))
                        sps_preds.append(int(res[0]))
                except Exception:
                    pass
            if sps_true:
                sps_adj = adjacent_accuracy(np.array(sps_true), np.array(sps_preds))
                sps_acc = np.mean(np.array(sps_true) == np.array(sps_preds))
                print(f"  [SPS Pipeline {pname}] acc={sps_acc:.3f}  adj_acc={sps_adj:.3f}  n={len(sps_true)}")


if __name__ == "__main__":
    main()
