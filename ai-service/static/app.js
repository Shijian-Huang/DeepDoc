const uploadForm = document.querySelector("#uploadForm");
const pdfFileInput = document.querySelector("#pdfFile");
const fileLabel = document.querySelector("#fileLabel");
const analyzeButton = document.querySelector("#analyzeButton");
const statusPanel = document.querySelector("#statusPanel");
const statusText = document.querySelector("#statusText");
const resultPanel = document.querySelector("#resultPanel");
const resultTemplate = document.querySelector("#resultTemplate");
const historyList = document.querySelector("#historyList");
const refreshHistoryButton = document.querySelector("#refreshHistoryButton");

pdfFileInput.addEventListener("change", () => {
  const file = pdfFileInput.files[0];
  fileLabel.textContent = file ? file.name : "Choose a PDF";
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = pdfFileInput.files[0];
  if (!file) {
    renderError("Choose a PDF before starting analysis.");
    return;
  }

  const formData = new FormData();
  const summaryMode = new FormData(uploadForm).get("summaryMode") || "standard";
  formData.append("file", file);
  formData.append("summary_mode", summaryMode);

  setBusy(true, "Analyzing paper...");

  try {
    const response = await fetch("/analyze-pdf", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Analysis failed with status ${response.status}`);
    }

    const result = await response.json();
    renderResult(result, file.name);
    await loadHistory();
  } catch (error) {
    renderError(error.message || "Analysis failed.");
  } finally {
    setBusy(false);
  }
});

refreshHistoryButton.addEventListener("click", loadHistory);

function setBusy(isBusy, message = "") {
  analyzeButton.disabled = isBusy;
  analyzeButton.textContent = isBusy ? "Analyzing..." : "Analyze PDF";
  statusText.textContent = message;
  statusPanel.hidden = !isBusy;
}

function renderError(message) {
  resultPanel.className = "result-panel error";
  resultPanel.innerHTML = `
    <div>
      <p class="eyebrow">Error</p>
      <h2>${escapeHtml(message)}</h2>
    </div>
  `;
}

function renderResult(result, fallbackFilename = "Analysis Result") {
  const node = resultTemplate.content.cloneNode(true);
  const summary = result.document_summary || {};
  const analysisId = result.analysis_id;

  node.querySelector('[data-field="mode"]').textContent = formatMode(result.summary_mode);
  node.querySelector('[data-field="title"]').textContent = fallbackFilename;
  node.querySelector('[data-field="meta"]').textContent = formatMeta(result);
  node.querySelector('[data-field="summary"]').textContent = summary.summary || "No summary returned.";

  const download = node.querySelector('[data-field="download"]');
  if (analysisId) {
    download.href = `/analyses/${analysisId}/download`;
  } else {
    download.removeAttribute("href");
  }

  renderList(node.querySelector('[data-field="keyIdeas"]'), summary.key_ideas || []);
  renderList(node.querySelector('[data-field="contributions"]'), summary.contributions || []);
  renderEvidence(node.querySelector('[data-field="evidence"]'), summary.evidence || []);
  renderSources(node.querySelector('[data-field="sources"]'), result.evidence_sources || []);
  renderReferences(node.querySelector('[data-field="references"]'), result.references || []);

  resultPanel.className = "result-panel";
  resultPanel.innerHTML = "";
  resultPanel.appendChild(node);
  bindTabs(resultPanel);
}

function renderList(container, items) {
  container.innerHTML = "";
  if (!items.length) {
    container.appendChild(emptyLine("No items returned."));
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    container.appendChild(li);
  });
}

function renderEvidence(container, evidence) {
  container.innerHTML = "";
  if (!evidence.length) {
    container.appendChild(emptyBlock("No evidence claims returned."));
    return;
  }

  evidence.forEach((item) => {
    const card = document.createElement("div");
    card.className = "evidence-card";
    card.innerHTML = `
      <div class="card-meta">${escapeHtml(item.section || "section")} · pages ${escapeHtml(formatPages(item.pages))}</div>
      <p>${escapeHtml(item.claim || "")}</p>
    `;
    container.appendChild(card);
  });
}

function renderSources(container, sources) {
  container.innerHTML = "";
  if (!sources.length) {
    container.appendChild(emptyBlock("No source sections returned."));
    return;
  }

  sources.forEach((source) => {
    const card = document.createElement("div");
    card.className = "source-card";
    card.innerHTML = `
      <div class="card-meta">${escapeHtml(source.section || "section")} · pages ${escapeHtml(formatPages(source.pages))}</div>
      <p>${escapeHtml(source.excerpt || "")}</p>
    `;
    container.appendChild(card);
  });
}

function renderReferences(container, references) {
  container.innerHTML = "";
  if (!references.length) {
    container.appendChild(emptyLine("No references extracted."));
    return;
  }

  references.forEach((reference) => {
    const li = document.createElement("li");
    li.textContent = stripReferenceNumber(reference);
    container.appendChild(li);
  });
}

function bindTabs(root) {
  const buttons = root.querySelectorAll(".tab-button");
  const panels = root.querySelectorAll(".tab-panel");

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;
      buttons.forEach((item) => item.classList.toggle("active", item === button));
      panels.forEach((panel) => {
        panel.classList.toggle("active", panel.dataset.panel === tab);
      });
    });
  });
}

async function loadHistory() {
  try {
    const response = await fetch("/analyses");
    if (!response.ok) {
      throw new Error("Could not load history.");
    }

    const payload = await response.json();
    renderHistory(payload.analyses || []);
  } catch (error) {
    historyList.innerHTML = `<p class="result-meta">${escapeHtml(error.message)}</p>`;
  }
}

function renderHistory(items) {
  historyList.innerHTML = "";
  if (!items.length) {
    historyList.innerHTML = '<p class="result-meta">No saved analyses yet.</p>';
    return;
  }

  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "history-item";
    card.innerHTML = `
      <strong>${escapeHtml(item.filename || "Untitled PDF")}</strong>
      <p>${escapeHtml(formatMode(item.summary_mode))} · ${escapeHtml(formatDate(item.created_at))} · ${escapeHtml(formatSeconds(item.processing_seconds))}</p>
      <p>${escapeHtml(trimText(item.summary || "", 145))}</p>
      <div class="history-actions">
        <button type="button" data-open="${escapeHtml(item.analysis_id || "")}">Open</button>
        <a href="/analyses/${encodeURIComponent(item.analysis_id || "")}/download">Download</a>
      </div>
    `;
    card.querySelector("button").addEventListener("click", () => openAnalysis(item.analysis_id));
    historyList.appendChild(card);
  });
}

async function openAnalysis(analysisId) {
  if (!analysisId) return;

  setBusy(true, "Loading saved analysis...");
  try {
    const response = await fetch(`/analyses/${encodeURIComponent(analysisId)}`);
    if (!response.ok) {
      throw new Error("Saved analysis was not found.");
    }

    const record = await response.json();
    renderResult(record.result || {}, record.filename || "Saved Analysis");
  } catch (error) {
    renderError(error.message);
  } finally {
    setBusy(false);
  }
}

function formatMode(mode) {
  const labels = {
    paragraph: "Paragraph summary",
    standard: "Standard summary",
    one_page: "One-page summary",
  };
  return labels[mode] || "Standard summary";
}

function formatMeta(result) {
  const parts = [];
  if (result.processing_seconds !== undefined) parts.push(formatSeconds(result.processing_seconds));
  if (result.summary_input_sections?.length) parts.push(`${result.summary_input_sections.length} sections`);
  if (result.references?.length) parts.push(`${result.references.length} references`);
  if (result.generated_at) parts.push(formatDate(result.generated_at));
  return parts.join(" · ");
}

function formatPages(pages) {
  if (!Array.isArray(pages) || pages.length === 0) return "unknown";
  return pages.join(", ");
}

function formatSeconds(seconds) {
  if (seconds === undefined || seconds === null) return "time unavailable";
  return `${Number(seconds).toFixed(2)}s`;
}

function formatDate(value) {
  if (!value) return "date unavailable";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function trimText(text, maxLength) {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength).trim()}...`;
}

function stripReferenceNumber(reference) {
  return reference.replace(/^\s*\[\d+\]\s*/, "");
}

function emptyLine(text) {
  const li = document.createElement("li");
  li.textContent = text;
  return li;
}

function emptyBlock(text) {
  const p = document.createElement("p");
  p.className = "result-meta";
  p.textContent = text;
  return p;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

loadHistory();
