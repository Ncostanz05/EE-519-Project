"""
Age Prediction Live Demo Server — Enhanced
Runs 4 prediction pipelines:
  A) Random Forest on handcrafted features (MFCCs, pitch, formants, energy)
  B) CNN on mel-spectrograms
  C) Wav2Vec2 embeddings → RF age classifier
  D) Gender-aware: Wav2Vec2 → gender prediction → gender-conditioned age RF
"""

import os, sys, tempfile, warnings
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import librosa
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from cnn_model import AgeAudioCNN
from features import extract_features

app = Flask(__name__, static_folder="demo_static")
CORS(app)

# ── Labels ──────────────────────────────────────────────────────────
AGE_LABELS = ["Teens", "Twenties", "Thirties", "Forties", "Fifties", "Sixties", "Seventies"]
GENDER_LABELS = ["Male", "Female"]

# ══════════════════════════════════════════════════════════════════════
# LOAD ALL MODELS AT STARTUP
# ══════════════════════════════════════════════════════════════════════
print("Loading models...")

# ── A) Handcrafted RF ───────────────────────────────────────────────
RF_ESTIMATOR = None
RF_SCALER = None
rf_path = os.path.join(PROJECT_DIR, "age_model.pkl")
if os.path.exists(rf_path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old_pipeline = joblib.load(rf_path)
        RF_ESTIMATOR = old_pipeline.named_steps["model"]
        try:
            RF_SCALER = old_pipeline.named_steps["scaler"]
            _ = RF_SCALER.mean_
        except Exception:
            RF_SCALER = None
        print("  ✓ [A] Handcrafted RF loaded")
    except Exception as e:
        print(f"  ✗ [A] Could not load age_model.pkl: {e}")
else:
    print("  ✗ [A] age_model.pkl not found")

# ── B) CNN ──────────────────────────────────────────────────────────
CNN_MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
cnn_path = os.path.join(PROJECT_DIR, "cnn_age_model.pth")
if os.path.exists(cnn_path):
    CNN_MODEL = AgeAudioCNN(num_classes=7).to(DEVICE)
    CNN_MODEL.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
    CNN_MODEL.eval()
    print(f"  ✓ [B] CNN loaded (device: {DEVICE})")
else:
    print("  ✗ [B] cnn_age_model.pth not found")

# ── C) Wav2Vec2 Embeddings → Age RF ────────────────────────────────
EMB_AGE_MODEL = None
EMB_SCALER = None
WAV2VEC_MODEL = None
WAV2VEC_PROCESSOR = None

emb_age_path = os.path.join(PROJECT_DIR, "embeddings_age_model.pkl")
emb_scaler_path = os.path.join(PROJECT_DIR, "embeddings_scaler.pkl")
if os.path.exists(emb_age_path) and os.path.exists(emb_scaler_path):
    EMB_AGE_MODEL = joblib.load(emb_age_path)
    EMB_SCALER = joblib.load(emb_scaler_path)
    print("  ✓ [C] Embeddings Age RF loaded")
else:
    print("  ✗ [C] embeddings_age_model.pkl not found")

# ── D) Gender Model + Gender-conditioned Age ───────────────────────
GENDER_MODEL = None
AGE_MODEL_MALE = None
AGE_MODEL_FEMALE = None

gender_path = os.path.join(PROJECT_DIR, "gender_model.pkl")
if os.path.exists(gender_path):
    GENDER_MODEL = joblib.load(gender_path)
    print("  ✓ [D] Gender classifier loaded")
else:
    print("  ✗ [D] gender_model.pkl not found")

male_path = os.path.join(PROJECT_DIR, "age_model_male.pkl")
female_path = os.path.join(PROJECT_DIR, "age_model_female.pkl")
if os.path.exists(male_path):
    AGE_MODEL_MALE = joblib.load(male_path)
    print("  ✓ [D] Male age model loaded")
if os.path.exists(female_path):
    AGE_MODEL_FEMALE = joblib.load(female_path)
    print("  ✓ [D] Female age model loaded")

# ── Load Wav2Vec2 if any embedding-based model exists ───────────────
if EMB_AGE_MODEL is not None or GENDER_MODEL is not None:
    try:
        from transformers import AutoFeatureExtractor, Wav2Vec2Model
        print("  Loading Wav2Vec2 base model...")
        WAV2VEC_PROCESSOR = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        WAV2VEC_MODEL = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
        WAV2VEC_MODEL.eval()
        print(f"  ✓ Wav2Vec2 loaded (device: {DEVICE})")
    except Exception as e:
        print(f"  ✗ Could not load Wav2Vec2: {e}")
        EMB_AGE_MODEL = None
        GENDER_MODEL = None

# ── Audio transforms ────────────────────────────────────────────────
MEL_TRANSFORM = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64)
AMP_TO_DB = T.AmplitudeToDB()
MAX_SAMPLES = int(16000 * 3.0)


# ══════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def get_wav2vec_embedding(y, sr=16000):
    """Extract a 768-dim embedding from Wav2Vec2."""
    if WAV2VEC_MODEL is None or WAV2VEC_PROCESSOR is None:
        return None

    # Pad/truncate to 3 seconds
    target_len = int(sr * 3.0)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    inputs = WAV2VEC_PROCESSOR(y, sampling_rate=sr, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = WAV2VEC_MODEL(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


def predict_rf(y, sr):
    """Pipeline A: Handcrafted features → RF."""
    if RF_ESTIMATOR is None:
        return None, None
    import pandas as pd
    from sklearn.impute import SimpleImputer
    feats = extract_features(y, sr)
    X = pd.DataFrame([feats])
    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X)
    X_scaled = RF_SCALER.transform(X_imp) if RF_SCALER is not None else X_imp
    pred = RF_ESTIMATOR.predict(X_scaled)[0]
    try:
        probs = RF_ESTIMATOR.predict_proba(X_scaled)[0]
    except Exception:
        probs = np.zeros(7); probs[int(pred)] = 1.0
    return int(pred), probs.tolist()


def predict_cnn(y, sr):
    """Pipeline B: Mel-spectrogram → CNN."""
    if CNN_MODEL is None:
        return None, None
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    if waveform.shape[1] > MAX_SAMPLES:
        waveform = waveform[:, :MAX_SAMPLES]
    elif waveform.shape[1] < MAX_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, MAX_SAMPLES - waveform.shape[1]))
    mel_spec = MEL_TRANSFORM(waveform)
    mel_spec_db = AMP_TO_DB(mel_spec)
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
    mel_spec_db = mel_spec_db.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = CNN_MODEL(mel_spec_db)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
    return pred, probs.tolist()


def predict_embeddings_age(embedding):
    """Pipeline C: Wav2Vec2 embedding → Age RF."""
    if EMB_AGE_MODEL is None or embedding is None:
        return None, None
    X = EMB_SCALER.transform(embedding.reshape(1, -1))
    pred = EMB_AGE_MODEL.predict(X)[0]
    try:
        probs = EMB_AGE_MODEL.predict_proba(X)[0]
    except Exception:
        probs = np.zeros(7); probs[int(pred)] = 1.0
    return int(pred), probs.tolist()


def predict_gender(embedding):
    """Predict gender from Wav2Vec2 embedding."""
    if GENDER_MODEL is None or embedding is None:
        return None, None
    X = EMB_SCALER.transform(embedding.reshape(1, -1))
    pred = GENDER_MODEL.predict(X)[0]
    try:
        probs = GENDER_MODEL.predict_proba(X)[0]
    except Exception:
        probs = np.zeros(2); probs[int(pred)] = 1.0
    return int(pred), probs.tolist()


def predict_gender_conditioned_age(embedding, gender_pred):
    """Pipeline D: Gender-conditioned age classifier."""
    if embedding is None or gender_pred is None:
        return None, None
    model = AGE_MODEL_MALE if gender_pred == 0 else AGE_MODEL_FEMALE
    if model is None:
        return None, None
    X = EMB_SCALER.transform(embedding.reshape(1, -1))
    pred = model.predict(X)[0]
    try:
        probs = model.predict_proba(X)[0]
    except Exception:
        probs = np.zeros(7); probs[int(pred)] = 1.0
    return int(pred), probs.tolist()


# ══════════════════════════════════════════════════════════════════════
#  SHARED PREDICTION HELPER
# ══════════════════════════════════════════════════════════════════════

def run_all_predictions(y, sr):
    """Run all 4 pipelines + gender on preprocessed audio. Returns dict."""
    # Normalize & trim
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    y, _ = librosa.effects.trim(y, top_db=20)

    embedding = get_wav2vec_embedding(y, sr)

    rf_pred, rf_probs = predict_rf(y, sr)
    cnn_pred, cnn_probs = predict_cnn(y, sr)
    emb_pred, emb_probs = predict_embeddings_age(embedding)
    gender_pred, gender_probs = predict_gender(embedding)
    gc_pred, gc_probs = predict_gender_conditioned_age(embedding, gender_pred)

    def make_result(pred, probs, labels=AGE_LABELS):
        if pred is None:
            return None
        return {
            "predicted_class": pred,
            "predicted_label": labels[pred] if pred < len(labels) else str(pred),
            "probabilities": probs,
        }

    return {
        "age_labels": AGE_LABELS,
        "gender_labels": GENDER_LABELS,
        "rf": make_result(rf_pred, rf_probs),
        "cnn": make_result(cnn_pred, cnn_probs),
        "embeddings": make_result(emb_pred, emb_probs),
        "gender": make_result(gender_pred, gender_probs, GENDER_LABELS),
        "gender_conditioned": make_result(gc_pred, gc_probs),
        "audio_duration_sec": round(len(y) / sr, 2),
    }


# ══════════════════════════════════════════════════════════════════════
# DATASET SAMPLE CACHE (loaded once)
# ══════════════════════════════════════════════════════════════════════

DATASET_DF = None
AUDIO_DIR = os.path.join(PROJECT_DIR, "cv_au_subset", "audio")

def _load_dataset_df():
    global DATASET_DF
    if DATASET_DF is not None:
        return DATASET_DF
    import pandas as pd
    metadata_path = os.path.join(PROJECT_DIR, "cv_au_subset", "metadata.csv")
    AGE_MAP_INV = {
        "teens": 0, "twenties": 1, "thirties": 2, "fourties": 3,
        "fifties": 4, "sixties": 5, "seventies": 6,
    }
    GENDER_MAP_INV = {"male_masculine": "Male", "female_feminine": "Female"}

    df = pd.read_csv(metadata_path, engine="python")
    df = df[df["age"].notna() & df["age"].isin(AGE_MAP_INV.keys())]
    df["age_label"] = df["age"].map(AGE_MAP_INV)
    df["age_display"] = df["age"].str.capitalize()
    df["gender_display"] = df["gender"].map(GENDER_MAP_INV).fillna("Unknown")

    # Keep only files that exist
    df = df[df["path"].apply(lambda p: os.path.exists(os.path.join(AUDIO_DIR, p)))]
    df = df.reset_index(drop=True)
    DATASET_DF = df
    return df


# ══════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("demo_static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("demo_static", path)


@app.route("/api/samples")
def api_samples():
    """Return random samples from the dataset, optionally filtered by age/gender."""
    df = _load_dataset_df()
    age_filter = request.args.get("age", None)       # e.g. "twenties"
    gender_filter = request.args.get("gender", None)  # e.g. "Male"
    count = min(int(request.args.get("count", 10)), 50)

    subset = df.copy()
    if age_filter:
        subset = subset[subset["age"] == age_filter.lower()]
    if gender_filter:
        gmap = {"Male": "male_masculine", "Female": "female_feminine"}
        if gender_filter in gmap:
            subset = subset[subset["gender"] == gmap[gender_filter]]

    if len(subset) == 0:
        return jsonify({"samples": []})

    samples = subset.sample(n=min(count, len(subset)), random_state=None)
    result = []
    for _, row in samples.iterrows():
        result.append({
            "path": row["path"],
            "true_age": row["age_display"],
            "true_age_class": int(row["age_label"]),
            "true_gender": row["gender_display"],
        })
    return jsonify({"samples": result, "total_available": len(subset)})


@app.route("/api/audio/<path:filename>")
def serve_audio(filename):
    """Serve an audio file from the dataset."""
    # Security: only allow files from the audio dir
    safe_name = os.path.basename(filename)
    return send_from_directory(AUDIO_DIR, safe_name)


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)

    try:
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
    except Exception as e:
        os.unlink(tmp_path)
        return jsonify({"error": f"Could not read audio: {e}"}), 400
    os.unlink(tmp_path)

    rms = np.sqrt(np.mean(y**2))
    if rms < 0.005:
        return jsonify({"error": "Audio too quiet – please speak louder or closer to the mic."}), 400

    result = run_all_predictions(y, sr)
    return jsonify(result)


@app.route("/predict_file", methods=["POST"])
def predict_file():
    """Run prediction on a dataset audio file by path."""
    data = request.get_json()
    if not data or "path" not in data:
        return jsonify({"error": "Missing 'path' in request body"}), 400

    filename = os.path.basename(data["path"])
    file_path = os.path.join(AUDIO_DIR, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {filename}"}), 404

    try:
        y, sr = librosa.load(file_path, sr=16000, mono=True)
    except Exception as e:
        return jsonify({"error": f"Could not read audio: {e}"}), 400

    result = run_all_predictions(y, sr)
    return jsonify(result)


if __name__ == "__main__":
    # Pre-load dataset index at startup
    _load_dataset_df()
    print("\n🎙️  Age Prediction Demo ready at http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)

