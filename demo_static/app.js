const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const recStatus = document.getElementById("recStatus");
const recPlayback = document.getElementById("recPlayback");
const samplesStatus = document.getElementById("samplesStatus");
const sampleList = document.getElementById("sampleList");
const resultsSection = document.getElementById("resultsSection");
const audioMeta = document.getElementById("audioMeta");
const trueLabels = document.getElementById("trueLabels");

let mediaRecorder = null;
let chunks = [];
let lastBlob = null;   // retained for /api/analyze

// ── Recording ──────────────────────────────────────────────────────
recordBtn.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : "";
    mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
    chunks = [];
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      if (!chunks.length) {
        recStatus.innerHTML = `<span class="error">No audio captured — try again.</span>`;
        return;
      }
      const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
      lastBlob = blob;
      recPlayback.hidden = true;
      recPlayback.src = "";
      trueLabels.hidden = true;
      await uploadAndPredict(blob);
    };
    mediaRecorder.start(100);
    recordBtn.disabled = true;
    recordBtn.classList.add("recording");
    stopBtn.disabled = false;
    recStatus.textContent = "Recording…";
    document.querySelectorAll(".sample.active").forEach((n) => n.classList.remove("active"));
  } catch (err) {
    recStatus.innerHTML = `<span class="error">Mic access denied: ${err.message}</span>`;
  }
});

stopBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  recordBtn.disabled = false;
  recordBtn.classList.remove("recording");
  stopBtn.disabled = true;
  recStatus.textContent = "";
});

async function uploadAndPredict(blob) {
  recStatus.innerHTML = `<span class="spinner"></span> Predicting…`;
  const fd = new FormData();
  fd.append("audio", blob, "recording.webm");
  try {
    const res = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    renderResults(data);
    recStatus.textContent = "Done.";
  } catch (err) {
    recStatus.innerHTML = `<span class="error">${err.message}</span>`;
  }
}

// ── Samples ────────────────────────────────────────────────────────
async function loadSamples() {
  samplesStatus.innerHTML = `<span class="spinner"></span> Loading…`;
  sampleList.innerHTML = "";
  try {
    const res = await fetch(`/api/samples?count=1000`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to load samples");
    if (!data.samples.length) {
      samplesStatus.textContent = "No samples available.";
      return;
    }
    samplesStatus.textContent = `${data.samples.length} of ${data.total_available} samples`;
    for (const s of data.samples) {
      const el = document.createElement("div");
      el.className = "sample";
      el.innerHTML = `<div class="sample-main"><strong>${s.true_age}</strong> · ${s.true_gender}</div>
                      <div class="meta">${s.path}</div>`;
      el.addEventListener("click", () => selectSample(el, s));
      sampleList.appendChild(el);
    }
  } catch (err) {
    samplesStatus.innerHTML = `<span class="error">${err.message}</span>`;
  }
}

async function selectSample(el, sample) {
  document.querySelectorAll(".sample.active").forEach((n) => n.classList.remove("active"));
  el.classList.add("active");
  lastBlob = null;

  recPlayback.src = `/api/audio/${encodeURIComponent(sample.path)}`;
  recPlayback.hidden = false;

  samplesStatus.innerHTML = `<span class="spinner"></span> Predicting on ${sample.path}…`;
  try {
    const res = await fetch("/predict_file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: sample.path }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    renderResults(data, sample);
    samplesStatus.textContent = `${sampleList.children.length} samples — last: ${sample.path}`;
  } catch (err) {
    samplesStatus.innerHTML = `<span class="error">${err.message}</span>`;
  }
}

// ── Rendering ──────────────────────────────────────────────────────
function renderResults(data, sample = null) {
  resultsSection.hidden = false;
  audioMeta.textContent = `Audio duration: ${data.audio_duration_sec}s`;

  // Pipeline E flagship panel
  renderPipelineE(data.pipeline_e, data.age_labels);

  // Baselines A–D
  const keys = ["rf", "cnn", "embeddings", "gender_conditioned", "gender"];
  for (const key of keys) {
    const node = document.querySelector(`.result[data-key="${key}"]`);
    const predEl = node.querySelector(".prediction");
    const barsEl = node.querySelector(".bars");
    const result = data[key];
    const labels = key === "gender" ? data.gender_labels : data.age_labels;

    if (!result) {
      predEl.textContent = "Model not loaded";
      predEl.classList.add("empty");
      barsEl.innerHTML = "";
      continue;
    }
    predEl.classList.remove("empty");
    predEl.textContent = result.predicted_label;

    const probs = result.probabilities || [];
    const topIdx = probs.indexOf(Math.max(...probs));
    barsEl.innerHTML = probs
      .map((p, i) => {
        const pct = (p * 100).toFixed(1);
        const top = i === topIdx ? " top" : "";
        return `<div class="bar-row${top}">
          <span class="label">${labels[i] || i}</span>
          <div class="track"><div class="fill" style="width:${pct}%"></div></div>
          <span class="pct">${pct}%</span>
        </div>`;
      })
      .join("");
  }

  if (sample) {
    trueLabels.hidden = false;
    trueLabels.innerHTML = `Ground truth: <strong>${sample.true_age}</strong> · <strong>${sample.true_gender}</strong>`;
  } else {
    trueLabels.hidden = true;
  }
}

function renderPipelineE(result, ageLabels) {
  const panel = document.getElementById("pipelineEPanel");
  const predEl = document.getElementById("ePrediction");
  const genderEl = document.getElementById("eGenderSub");
  const top3El = document.getElementById("eTop3");
  const barsEl = document.getElementById("eBars");
  const badge = document.getElementById("uncertaintyBadge");
  const analysisBtn = document.getElementById("analysisToggle");
  const analysisPanel = document.getElementById("analysisPanel");

  panel.hidden = false;
  analysisPanel.hidden = true;

  if (!result || result.error) {
    predEl.textContent = result?.error ? `Error: ${result.error}` : "Pipeline E not loaded";
    predEl.classList.add("empty");
    genderEl.textContent = "";
    top3El.innerHTML = "";
    barsEl.innerHTML = "";
    badge.hidden = true;
    analysisBtn.hidden = true;
    return;
  }

  predEl.classList.remove("empty");
  predEl.textContent = result.predicted_label;

  // Gender sub-line — backend returns P(female) as gender_proba,
  // so flip it when the prediction is male to show confidence in the chosen class.
  const pFemale = result.gender_proba;
  const pPredicted = result.gender_pred === 1 ? pFemale : 1 - pFemale;
  const gPct = (pPredicted * 100).toFixed(0);
  genderEl.textContent = `${result.gender_label} (${gPct}% confidence)`;

  // Uncertainty badge
  if (result.uncertainty != null) {
    badge.hidden = false;
    const u = result.uncertainty;
    badge.textContent = `uncertainty: ${u.toFixed(4)}`;
    badge.className = "uncertainty-badge " + (u < 0.01 ? "low" : u < 0.05 ? "mid" : "high");
  } else {
    badge.hidden = true;
  }

  // Top-3 table
  if (result.top3 && result.top3.length) {
    top3El.innerHTML = result.top3
      .map((t, i) => `<div class="top3-row rank-${i + 1}">
        <span class="top3-rank">#${i + 1}</span>
        <span class="top3-label">${t.label}</span>
        <span class="top3-prob">${(t.prob * 100).toFixed(1)}%</span>
      </div>`)
      .join("");
  }

  // Probability bars
  const probs = result.probabilities || [];
  const labels = ageLabels || ["Teens","Twenties","Thirties","Forties","Fifties","Sixties","Seventies"];
  const topIdx = probs.indexOf(Math.max(...probs));
  barsEl.innerHTML = probs
    .map((p, i) => {
      const pct = (p * 100).toFixed(1);
      const top = i === topIdx ? " top" : "";
      return `<div class="bar-row${top}">
        <span class="label">${labels[i] || i}</span>
        <div class="track"><div class="fill" style="width:${pct}%"></div></div>
        <span class="pct">${pct}%</span>
      </div>`;
    })
    .join("");

  // Analysis button — only show when we have a live recording blob
  if (lastBlob) {
    analysisBtn.hidden = false;
    analysisBtn.onclick = () => fetchAnalysis(lastBlob);
  } else {
    analysisBtn.hidden = true;
  }
}

// ── Analysis (saliency) ────────────────────────────────────────────
async function fetchAnalysis(blob) {
  const panel = document.getElementById("analysisPanel");
  const btn = document.getElementById("analysisToggle");
  btn.disabled = true;
  btn.textContent = "⚡ Analyzing…";
  panel.hidden = false;

  const fd = new FormData();
  fd.append("audio", blob, "recording.webm");
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Analysis failed");
    drawSaliency(data.saliency);

    if (data.uncertainty != null) {
      document.getElementById("analysisUncertainty").hidden = false;
      document.getElementById("uncertaintyValue").textContent =
        `Variance sum: ${data.uncertainty.toFixed(6)}`;
    }
  } catch (err) {
    panel.hidden = false;
    document.getElementById("saliencyCanvas").getContext("2d")
      .fillText(`Error: ${err.message}`, 10, 20);
  } finally {
    btn.disabled = false;
    btn.textContent = "⚡ Analysis";
  }
}

function drawSaliency(saliency) {
  const canvas = document.getElementById("saliencyCanvas");
  const W = canvas.parentElement.clientWidth || 600;
  const H = 60;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);

  const n = saliency.length;
  const bw = W / n;
  for (let i = 0; i < n; i++) {
    const v = saliency[i];   // 0–1
    const r = Math.round(239 * v + 34 * (1 - v));
    const g = Math.round(68  * v + 197 * (1 - v));
    const b = Math.round(68  * v + 94  * (1 - v));
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    const barH = Math.round(H * 0.2 + H * 0.8 * v);
    ctx.fillRect(i * bw, H - barH, Math.max(bw - 0.5, 1), barH);
  }
}

// Auto-load on page open
loadSamples();
