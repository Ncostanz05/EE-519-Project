const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const recStatus = document.getElementById("recStatus");
const recPlayback = document.getElementById("recPlayback");
const ageFilter = document.getElementById("ageFilter");
const genderFilter = document.getElementById("genderFilter");
const loadSamplesBtn = document.getElementById("loadSamplesBtn");
const samplesStatus = document.getElementById("samplesStatus");
const sampleList = document.getElementById("sampleList");
const resultsSection = document.getElementById("resultsSection");
const audioMeta = document.getElementById("audioMeta");
const trueLabels = document.getElementById("trueLabels");

let mediaRecorder = null;
let chunks = [];
let activeSample = null;

// ── Recording ──────────────────────────────────────────────────────
recordBtn.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    chunks = [];
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: "audio/webm" });
      recPlayback.src = URL.createObjectURL(blob);
      recPlayback.hidden = false;
      await uploadAndPredict(blob);
    };
    mediaRecorder.start();
    recordBtn.disabled = true;
    recordBtn.classList.add("recording");
    stopBtn.disabled = false;
    recStatus.textContent = "Recording…";
    activeSample = null;
    trueLabels.hidden = true;
  } catch (err) {
    recStatus.innerHTML = `<span class="error">Mic access denied: ${err.message}</span>`;
  }
});

stopBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
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
loadSamplesBtn.addEventListener("click", loadSamples);

async function loadSamples() {
  samplesStatus.innerHTML = `<span class="spinner"></span> Loading…`;
  sampleList.innerHTML = "";
  const params = new URLSearchParams({ count: 12 });
  if (ageFilter.value) params.set("age", ageFilter.value);
  if (genderFilter.value) params.set("gender", genderFilter.value);
  try {
    const res = await fetch(`/api/samples?${params}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to load samples");
    if (!data.samples.length) {
      samplesStatus.textContent = "No samples match these filters.";
      return;
    }
    samplesStatus.textContent = `${data.samples.length} shown (${data.total_available} available).`;
    for (const s of data.samples) {
      const el = document.createElement("div");
      el.className = "sample";
      el.innerHTML = `<div><strong>${s.true_age}</strong> · ${s.true_gender}</div>
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
  activeSample = sample;

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
    samplesStatus.textContent = "Done.";
  } catch (err) {
    samplesStatus.innerHTML = `<span class="error">${err.message}</span>`;
  }
}

// ── Rendering ──────────────────────────────────────────────────────
function renderResults(data, sample = null) {
  resultsSection.hidden = false;
  audioMeta.textContent = `Audio duration: ${data.audio_duration_sec}s`;

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
