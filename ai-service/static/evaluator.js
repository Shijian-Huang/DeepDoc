const HISTORY_KEY = "deepdoc.evaluator.history.v1";
const state = { config: null, busy: false, latest: null };
const $ = (id) => document.getElementById(id);

function setStatus(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function selectedValues(select) {
  return [...select.selectedOptions].map((option) => option.value);
}

function scenarioSettings() {
  const scenario = $("scenario").value;
  return {
    scenario,
    custom: scenario.startsWith("custom"),
    compare: scenario === "custom-compare",
    selectModels: scenario.startsWith("selected-") || scenario === "custom-compare",
    allModels: scenario.startsWith("all-"),
    allPapers: scenario.endsWith("-all"),
  };
}

function updateForm() {
  if (!state.config) return;
  const settings = scenarioSettings();
  $("custom-fields").classList.toggle("hidden", !settings.custom);
  $("paper-field").classList.toggle("hidden", settings.custom);
  $("model-field").classList.toggle("hidden", settings.custom && !settings.compare);

  const models = $("models");
  const multiple = settings.selectModels;
  if (models.multiple !== multiple) {
    models.multiple = multiple;
    [...models.options].forEach((option, index) => { option.selected = multiple ? index === 0 : index === 0; });
  }
  models.disabled = settings.allModels;
  $("model-help").textContent = settings.allModels
    ? "Every model in evaluation/models.py will be used."
    : multiple ? "Choose one or more models (Ctrl/Cmd-click to select)." : "Select the summary model to evaluate.";
  $("papers").disabled = settings.allPapers;
}

async function loadConfiguration() {
  try {
    const response = await fetch("/api/evaluator/config");
    if (!response.ok) throw new Error("Could not load evaluator configuration.");
    state.config = await response.json();
    $("models").innerHTML = state.config.models.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
    $("papers").innerHTML = state.config.papers.map((paper) => `<option value="${escapeHtml(paper.id)}">${escapeHtml(paper.label)} · ${escapeHtml(paper.id)}</option>`).join("");
    $("summary-modes").innerHTML = state.config.summary_modes.map((mode) => `
      <label><input type="radio" name="summary-mode" value="${escapeHtml(mode)}" ${mode === "standard" ? "checked" : ""}>${escapeHtml(mode.replace("_", " "))}</label>
    `).join("");
    setStatus(`${state.config.models.length} models · ${state.config.papers.length} papers`);
    updateForm();
  } catch (error) {
    setStatus(error.message, true);
  }
}

function summaryMode() {
  return document.querySelector('input[name="summary-mode"]:checked')?.value || "standard";
}

function chosenModels(settings) {
  if (settings.allModels) return [...state.config.models];
  return selectedValues($("models"));
}

function chosenPapers(settings) {
  return settings.allPapers ? state.config.papers.map((paper) => paper.id) : [$("papers").value];
}

async function runEvaluation(force = false) {
  if (state.busy || !state.config) return;
  const settings = scenarioSettings();
  const modelNames = chosenModels(settings);
  if ((!settings.custom || settings.compare) && !modelNames.length) {
    return setStatus("Select at least one model.", true);
  }

  let url = "/api/evaluator/evaluate";
  let options;
  if (settings.custom) {
    const file = $("reference-paper").files[0];
    const summary = $("custom-summary").value.trim();
    if (!file || !summary) return setStatus("Upload a reference paper and enter a custom summary.", true);
    const form = new FormData();
    form.append("paper", file);
    form.append("summary", summary);
    form.append("summary_mode", summaryMode());
    form.append("model_names", JSON.stringify(settings.compare ? modelNames : []));
    form.append("force", String(force));
    url = "/api/evaluator/custom";
    options = { method: "POST", body: form };
  } else {
    options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_names: modelNames,
        paper_ids: chosenPapers(settings),
        summary_mode: summaryMode(),
        force,
      }),
    };
  }

  setBusy(true);
  setStatus(force ? "Re-summarizing and evaluating…" : "Evaluating requested selection…");
  try {
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Evaluation failed.");
    state.latest = payload;
    renderResults(payload);
    saveHistory(payload, settings.scenario);
    setStatus(
      payload.partial
        ? `Evaluation completed with ${payload.error_count} model error(s).`
        : payload.packet_results?.every((row) => row.cached)
          ? "Loaded from local cache"
          : "Evaluation complete",
      payload.partial,
    );
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  state.busy = busy;
  $("evaluate").disabled = busy;
  $("force").disabled = busy;
}

const metrics = [
  ["avg_score", "Overall score"],
  ["key_ideas_score", "Key idea coverage"],
  ["contributions_score", "Contribution coverage"],
  ["hallucination_score", "Hallucination rate"],
];

const seriesColors = ["#165c4c", "#4b78a8", "#e2a440", "#8c62a8", "#ce5a4a", "#3f8f91"];

function renderResults(payload) {
  $("empty-state").classList.add("hidden");
  const container = $("results");
  container.classList.remove("hidden");
  const results = Array.isArray(payload?.results) ? payload.results : [];
  if (!results.length) {
    container.innerHTML = `<article class="result-card result-error"><p class="error-message">The evaluator returned no results.</p></article>`;
    return;
  }
  const successful = results.filter((row) => row.evaluation && typeof row.evaluation === "object");
  const failed = results.filter((row) => !row.evaluation || typeof row.evaluation !== "object");
  const modeLabel = String(payload.summary_mode || "standard").replace("_", " ");
  const paperLabel = payload.averaged
    ? `${Math.max(0, ...successful.map((row) => Number(row.paper_count) || 0))} papers averaged`
    : successful[0]?.paper_id || results[0]?.paper_id || "Custom reference";

  const chart = successful.length ? `<article class="result-card comparison-card">
    <div class="result-head">
      <div><h3>Model comparison</h3><p>${escapeHtml(paperLabel)} · ${escapeHtml(modeLabel)} mode</p></div>
    </div>
    <div class="chart-legend" aria-label="Models">
      ${successful.map((row, index) => `<span><i style="background:${seriesColors[index % seriesColors.length]}"></i>${escapeHtml(row.model)}${row.cached ? " · cached" : ""}</span>`).join("")}
    </div>
    ${successful.some((row) => row.warning) ? `<p class="result-warning">${escapeHtml(successful.map((row) => row.warning).filter(Boolean).join(" "))}</p>` : ""}
    <div class="comparison-chart" role="img" aria-label="Grouped evaluation scores for ${successful.length} model${successful.length === 1 ? "" : "s"}">
      ${metrics.map(([field, label]) => `<div class="metric-group">
        <div class="group-bars">
          ${successful.map((row, index) => {
            const value = Math.max(0, Math.min(1, Number(row.evaluation?.[field]) || 0));
            return `<div class="series-bar-wrap" style="--bar-height:${value * 100}%" title="${escapeHtml(row.model)}: ${(value * 100).toFixed(1)}%">
              <strong>${(value * 100).toFixed(1)}%</strong>
              <div class="series-bar" style="height:${value * 100}%;background:${seriesColors[index % seriesColors.length]}"></div>
            </div>`;
          }).join("")}
        </div>
        <span>${escapeHtml(label)}</span>
      </div>`).join("")}
    </div>
    <p class="chart-note">Higher is better, except hallucination rate.</p>
  </article>` : "";

  const failures = failed.map((row) => {
    const message = row.error || "This model did not return evaluation scores.";
    return `<article class="result-card result-error">
      <div class="result-head"><div><h3>${escapeHtml(row.model)}</h3><p>${escapeHtml(paperLabel)} · ${escapeHtml(modeLabel)} mode</p></div><span class="error-badge">FAILED</span></div>
      <p class="error-message">${escapeHtml(message)}</p>
    </article>`;
  }).join("");

  container.innerHTML = chart + failures;
}

function readHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); }
  catch (_) { return []; }
}

function saveHistory(payload, scenario) {
  const history = readHistory();
  history.unshift({ id: crypto.randomUUID?.() || String(Date.now()), at: new Date().toISOString(), scenario, payload });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 12)));
  renderHistory();
}

function renderHistory() {
  const history = readHistory();
  $("history-list").innerHTML = history.length ? history.map((item, index) => `
    <button class="history-item" data-history-index="${index}"><strong>${escapeHtml(item.scenario.replaceAll("-", " · "))}</strong><span>${new Date(item.at).toLocaleString()} · ${item.payload.results.length} result(s)</span></button>
  `).join("") : '<p class="history-empty">No locally saved comparisons yet.</p>';
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

$("scenario").addEventListener("change", updateForm);
$("evaluate").addEventListener("click", () => runEvaluation(false));
$("force").addEventListener("click", () => runEvaluation(true));
$("clear-history").addEventListener("click", () => { localStorage.removeItem(HISTORY_KEY); renderHistory(); });
$("history-list").addEventListener("click", (event) => {
  const button = event.target.closest("[data-history-index]");
  if (!button) return;
  const item = readHistory()[Number(button.dataset.historyIndex)];
  if (item) { state.latest = item.payload; renderResults(item.payload); setStatus("Loaded from browser history"); }
});

renderHistory();
loadConfiguration();
