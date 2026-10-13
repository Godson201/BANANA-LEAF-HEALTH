"use strict";
const byId = (id) => document.getElementById(id);
const input = byId("image-input");
const analyzeButton = byId("analyze-button");
const dropZone = byId("drop-zone");
let selectedFile = null;
let selectedSample = null;
let currentResult = null;
let previewUrl = null;
let modelReady = false;
let busy = false;
let diseaseCatalog = [];
let lastFieldCheck = null;
let modelMetadata = {};
let currentModelClasses = ["cordana", "healthy", "pestalotiopsis", "sigatoka"];
const classLabels = {fusarium_wilt: "Fusarium / Panama", moko_wilt: "Moko bacterial wilt"};
const titleCase = (value) => classLabels[value] || (value.charAt(0).toUpperCase() + value.slice(1).replaceAll("_", " "));
const percentage = (value) => `${(value * 100).toFixed(1)}%`;

function showError(message) {
  byId("error-message").textContent = message;
  byId("error-message").hidden = !message;
}
function updateControls() {
  analyzeButton.disabled = busy || !modelReady || (!selectedFile && !selectedSample);
  input.disabled = busy;
  dropZone.disabled = busy;
  byId("change-photo").disabled = busy;
  document.querySelectorAll(".sample-button").forEach((button) => {
    button.disabled = busy || !modelReady;
  });
  byId("analyze-label").textContent = busy ? "Analyzing…" : "Analyze leaf";
  byId("confirm-banana").disabled = busy;
  byId("model-choice").disabled = busy;
  byId("analyze-arrow").textContent = busy ? "…" : "↗";
  document.body.classList.toggle("busy", busy);
  byId("result-panel").setAttribute("aria-busy", String(busy));
}
function displayState(state) {
  if (state !== "result") byId("care-panel").hidden = true;
  byId("empty-result").hidden = state !== "empty";
  byId("loading-result").hidden = state !== "loading";
  byId("prediction-result").hidden = state !== "result";
  byId("unassessed-result").hidden = state !== "unassessed";
  byId("result-label").textContent = {empty: "Awaiting photo", loading: "Analyzing", result: "Analysis complete", unassessed: "Uncertain"}[state];
}
function clearPreviewUrl() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
}
function setPreview(url, name, detail) {
  byId("photo-preview").src = url;
  byId("file-name").textContent = name;
  byId("file-detail").textContent = detail;
  byId("preview-area").hidden = false;
  dropZone.hidden = true;
}
function selectFile(file) {
  if (!file || busy) return;
  showError("");
  if (file.size > 8 * 1024 * 1024) {
    showError("Please choose a photo smaller than 8 MB.");
    return;
  }
  const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
  if ((file.type && !allowedTypes.includes(file.type)) || (!file.type && !/\.(jpe?g|png|webp)$/i.test(file.name))) {
    showError("Please choose a JPG, PNG or WebP image.");
    return;
  }
  clearPreviewUrl();
  selectedFile = file;
  selectedSample = null;
  currentResult = null;
  previewUrl = URL.createObjectURL(file);
  const size = file.size > 1024 * 1024 ? `${(file.size / 1024 ** 2).toFixed(1)} MB` : `${Math.max(1, Math.round(file.size / 1024))} KB`;
  setPreview(previewUrl, file.name, size);
  document.querySelectorAll(".sample-button").forEach((button) => button.classList.remove("selected"));
  displayState("empty");
  updateControls();
}
function showPrediction(result) {
  currentResult = result;
  if (result.status === "rejected" || result.status === "uncertain" || !result.predicted_class) {
    byId("predicted-class").textContent = "";
    byId("probability-chart").replaceChildren();
    byId("unassessed-title").textContent = "Uncertain — banana leaf not confirmed";
    byId("unassessed-message").textContent = result.message;
    byId("confirm-banana").hidden = !result.can_confirm_banana;
    displayState("unassessed");
    return;
  }
  const healthy = result.predicted_class === "healthy";
  byId("predicted-class").textContent = titleCase(result.predicted_class);
  byId("class-type").textContent = healthy ? "Healthy class" : "Disease class";
  byId("class-type").classList.toggle("healthy", healthy);
  byId("result-description").textContent = healthy
    ? `This photo most closely matches the model's healthy leaf class. Compare all ${result.probabilities.length} class scores below.`
    : `This photo most closely matches the model's ${titleCase(result.predicted_class)} class. Compare all ${result.probabilities.length} scores below.`;
  if (result.tentative) byId("result-description").textContent += ` ${result.message}`;
  if (result.screening?.user_confirmed) byId("result-description").textContent += " Banana identity was confirmed by you, not verified automatically.";
  if (result.experimental) byId("result-description").textContent += " Research preview only: small new-class samples and no independent farm evaluation.";
  if (result.crop_agreement?.total > 1) byId("result-description").textContent += ` ${result.crop_agreement.matching} of ${result.crop_agreement.total} image crops agree on this class.`;
  byId("top-score").textContent = percentage(result.confidence);
  byId("confidence-fill").style.width = percentage(result.confidence);
  const chart = byId("probability-chart");
  chart.replaceChildren();
  result.probabilities.forEach(({class_name: name, probability: probability}) => {
    const row = document.createElement("div");
    row.className = `probability-row${name === result.predicted_class ? " winner" : ""}`;
    const label = document.createElement("span");
    label.textContent = titleCase(name);
    const track = document.createElement("div");
    track.className = "probability-track";
    const fill = document.createElement("div");
    fill.style.width = percentage(probability);
    track.append(fill);
    const value = document.createElement("span");
    value.textContent = percentage(probability);
    row.append(label, track, value);
    chart.append(row);
  });
  chart.setAttribute("aria-label", result.probabilities.map((item) => `${item.class_name}: ${percentage(item.probability)}`).join(", "));
  byId("sample-note").hidden = !result.true_class;
  byId("sample-note").textContent = result.true_class ? `Test sample · True label: ${titleCase(result.true_class)}. Saved prediction from the test set.` : "";
  if (result.image_size && selectedFile) {
    const size = selectedFile.size > 1024 * 1024 ? `${(selectedFile.size / 1024 ** 2).toFixed(1)} MB` : `${Math.max(1, Math.round(selectedFile.size / 1024))} KB`;
    byId("file-detail").textContent = `${size} · ${result.image_size.width} × ${result.image_size.height} pixels`;
  }
  displayState("result");
  renderCare(result.farmer_guidance);
  if (result.tentative) byId("result-label").textContent = "Tentative category";
}
async function analyze(bananaConfirmed = false) {
  if (busy || !modelReady || (!selectedFile && !selectedSample)) return;
  showError("");
  busy = true;
  updateControls();
  displayState("loading");
  try {
    let url, options;
    if (selectedSample) {
      url = `/api/sample/${encodeURIComponent(selectedSample)}?model=${byId("model-choice").value}`;
      options = {method: "POST"};
    } else {
      const form = new FormData();
      form.append("image", selectedFile);
      form.append("model", byId("model-choice").value);
      if (bananaConfirmed === true) form.append("banana_confirmed", "true");
      url = "/api/predict";
      options = {method: "POST", body: form};
    }
    const response = await fetch(url, options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction unavailable.");
    showPrediction(result);
  } catch (error) {
    showError(error.message === "Failed to fetch" ? "The local server is unavailable. Start the dashboard and try again." : error.message);
    displayState("empty");
  } finally {
    busy = false;
    updateControls();
  }
}
byId("upload-form").addEventListener("submit", (event) => { event.preventDefault(); analyze(); });
dropZone.addEventListener("click", () => { input.value = ""; input.click(); });
byId("change-photo").addEventListener("click", () => { input.value = ""; input.click(); });
input.addEventListener("change", () => selectFile(input.files[0]));
for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add("drag-over"); });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove("drag-over"); });
}
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
window.addEventListener("beforeunload", clearPreviewUrl);
document.querySelectorAll(".sample-button").forEach((button) => {
  button.addEventListener("click", () => {
    if (busy) return;
    clearPreviewUrl();
    selectedFile = null;
    selectedSample = button.dataset.sample;
    currentResult = null;
    setPreview(`assets/${selectedSample}.jpg`, `${titleCase(selectedSample)} · test sample`, "Labeled held-out dataset image");
    document.querySelectorAll(".sample-button").forEach((item) => item.classList.toggle("selected", item === button));
    analyze();
  });
});
byId("retry-photo").addEventListener("click", () => { input.value = ""; input.click(); });
byId("confirm-banana").addEventListener("click", () => analyze(true));
byId("reset-button").addEventListener("click", () => {
  clearPreviewUrl();
  selectedFile = null;
  selectedSample = null;
  currentResult = null;
  input.value = "";
  byId("photo-preview").removeAttribute("src");
  byId("preview-area").hidden = true;
  dropZone.hidden = false;
  document.querySelectorAll(".sample-button").forEach((button) => button.classList.remove("selected"));
  showError("");
  displayState("empty");
  updateControls();
  dropZone.focus();
});
byId("download-result").addEventListener("click", () => {
  if (!currentResult) return;
  const report = {image: selectedFile ? selectedFile.name : `${selectedSample} test sample`, ...currentResult,
    created_at: new Date().toISOString(), field_note: byId("farm-note").value.trim(), field_check: lastFieldCheck,
    note: "Model probabilities are uncalibrated. Unrecognized subjects remain uncertain; confirmed banana leaves receive a predicted category, not a confirmed diagnosis."};
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: "application/json"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = "leaflens-result.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
document.querySelectorAll(".nav-item").forEach((link) => {
  link.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item === link));
    document.querySelector(".breadcrumb strong").textContent = link.textContent.trim().replace(/^[^a-zA-Z]+/, "");
  });
});
updateControls();

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function renderCare(care) {
  byId("care-panel").hidden = !care;
  if (!care) return;
  byId("care-summary").textContent = care.summary;
  byId("care-actions").replaceChildren(...care.actions.map(action => element("li", action)));
  byId("care-review").textContent = care.review_note;
  byId("care-source").textContent = `${care.source.title} ↗`;
  byId("care-source").href = care.source.url;
}
function renderGuides() {
  const query = byId("disease-search").value.trim().toLowerCase();
  const filter = byId("disease-filter").value;
  const supports = item => currentModelClasses.includes(item.id);
  const matches = diseaseCatalog.filter(item =>
    (filter === "all" || (filter === "model") === supports(item)) &&
    `${item.name} ${item.summary} ${item.signs}`.toLowerCase().includes(query));
  byId("guide-count").textContent = `${matches.length} conditions shown · ${currentModelClasses.length} classes in the selected model; remaining conditions are guidance only.`;
  byId("disease-cards").replaceChildren(...matches.map(item => {
    const card = element("article", null, "panel disease-card");
    card.id = `guide-${item.id}`;
    const label = supports(item) ? (item.experimental_available ? "Research model only" : "Model-supported class") : "Guidance only · Not predicted";
    card.append(element("span", label, `disease-badge ${supports(item) ? "supported" : "guide-only"}`));
    card.append(element("h3", item.name), element("p", item.summary), element("p", item.signs, "disease-signs"));
    const details = element("details");
    details.append(element("summary", "Practical next steps"));
    const list = element("ol");
    list.append(...item.actions.map(action => element("li", action)));
    const link = element("a", `${item.source.title} ↗`);
    link.href = item.source.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    details.append(list, link);
    card.append(details);
    return card;
  }));
  if (!matches.length) byId("disease-cards").append(element("p", "No matching condition. Try a different name or clear the filter."));
}
byId("disease-search").addEventListener("input", renderGuides);
byId("disease-filter").addEventListener("change", renderGuides);
fetch("/api/diseases").then(async response => {
  if (!response.ok) throw new Error("Guide unavailable");
  diseaseCatalog = (await response.json()).diseases;
  renderGuides();
}).catch(() => {
  byId("disease-cards").replaceChildren(element("p", "The disease guide is unavailable. Refresh after starting the server."));
});
byId("field-form").addEventListener("change", () => { lastFieldCheck = null; });
byId("field-form").addEventListener("submit", async event => {
  event.preventDefault();
  const button = byId("field-button");
  button.disabled = true;
  const symptoms = [...document.querySelectorAll('input[name="symptom"]:checked')].map(input => input.value);
  try {
    const response = await fetch("/api/field-check", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({symptoms})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    lastFieldCheck = {...result, symptoms};
    const panel = byId("field-result");
    panel.classList.toggle("priority-review", result.priority === "extension_review");
    const list = element("ol");
    list.append(...result.actions.map(action => element("li", action)));
    panel.replaceChildren(element("span", "Based on the symptoms you selected", "empty-chip"), element("h3", result.title), element("p", result.message), list, element("p", result.basis, "guidance-note"));
    for (const guide of result.related_guides) {
      const item = diseaseCatalog.find(item => item.id === guide);
      const link = element("a", `Read about ${item?.name || guide} ↗`, "field-guide-link");
      link.href = `#guide-${guide}`;
      link.addEventListener("click", () => {
        byId("disease-search").value = "";
        byId("disease-filter").value = "all";
        renderGuides();
      });
      panel.append(link);
    }
  } catch (error) {
    byId("field-result").replaceChildren(element("p", error.message || "Field check unavailable. Try again."));
    lastFieldCheck = null;
  } finally { button.disabled = false; }
});
byId("print-result").addEventListener("click", () => window.print());
function applyModel() {
  const selected = byId("model-choice").value;
  const info = modelMetadata[selected];
  if (!info) return;
  currentModelClasses = info.class_names;
  byId("model-note").textContent = info.note;
  byId("model-note").classList.toggle("research-note", info.experimental);
  byId("metric-accuracy").textContent = `${(info.test_accuracy * 100).toFixed(2)}%`;
  byId("metric-classes").textContent = info.class_names.length;
  byId("metric-images").textContent = info.unique_images;
  byId("metric-correct").textContent = `${Math.round(info.test_accuracy * info.test_images)} / ${info.test_images} correct`;
  byId("metric-class-note").textContent = `${info.class_names.length - 1} disease classes · 1 healthy`;
  byId("class-comparison-title").textContent = `How the ${info.class_names.length} classes compare`;
  byId("performance-badge").textContent = `${info.experimental ? "Research preview · " : ""}${info.test_images} held-out images`;
  byId("performance-note").textContent = info.note + " These are held-out dataset results, not guaranteed accuracy on new farms.";
  byId("metric-f1").textContent = info.macro_f1.toFixed(4);
  byId("metric-loss").textContent = info.test_loss.toFixed(4);
  byId("metric-epochs").textContent = info.actual_epochs;
  byId("metric-best").textContent = info.best_epoch;
  byId("confusion-chart").src = info.confusion_chart;
  byId("accuracy-chart").src = info.accuracy_chart;
  byId("training-note").textContent = `Selected epoch ${info.best_epoch} using validation loss. Trained for ${info.actual_epochs} epochs. ${info.experimental ? "New-class tests include only 7 Fusarium and 9 Moko images." : ""}`;
  document.querySelectorAll(".experimental-sample").forEach(button => { button.hidden = selected !== "expanded"; });
  renderGuides();
}
byId("model-choice").addEventListener("change", () => {
  byId("reset-button").click();
  applyModel();
  updateControls();
});
fetch("/api/models").then(async response => {
  if (!response.ok) throw new Error("Model metadata unavailable");
  modelMetadata = (await response.json()).models;
  applyModel();
}).catch(() => { byId("model-note").textContent = "Model details unavailable. Refresh after starting the server."; });
fetch("/api/health").then(async response => {
  const result = await response.json();
  if (!response.ok) throw new Error(result.error);
  modelReady = true;
  byId("model-status").textContent = "Model ready · Local inference";
  updateControls();
}).catch(() => {
  byId("model-status").textContent = "Backend unavailable";
  showError("Start the backend with Start_Dashboard.cmd, then open http://127.0.0.1:5050.");
});
