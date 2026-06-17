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
resultPanel.addEventListener("click", handleResultPanelClick);


function handleResultPanelClick(event) {
  const scriptButton = event.target.closest('[data-field="generateVideoScript"]');
  const videoButton = event.target.closest('[data-field="generateVideo"]');
  if (!scriptButton && !videoButton) return;

  const analysisId = resultPanel.dataset.analysisId;
  if (!analysisId) return;

  const videoContainer = resultPanel.querySelector('[data-field="videoScript"]');
  const videoStatus = resultPanel.querySelector('[data-field="videoStatus"]');
  const downloadVideoLink = resultPanel.querySelector('[data-field="downloadVideo"]');

  if (scriptButton) {
    generateVideoScript(analysisId, videoContainer, scriptButton);
  }
  if (videoButton) {
    generateVideo(analysisId, videoStatus, downloadVideoLink, videoButton);
  }
}

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
  const paperTitle = result.paper_title || summary.title || result.video_script?.title || fallbackFilename;

  node.querySelector('[data-field="mode"]').textContent = formatMode(result.summary_mode);
  node.querySelector('[data-field="title"]').textContent = paperTitle;
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
  const videoContainer = node.querySelector('[data-field="videoScript"]');
  const videoStatus = node.querySelector('[data-field="videoStatus"]');
  const videoButton = node.querySelector('[data-field="generateVideoScript"]');
  const generateVideoButton = node.querySelector('[data-field="generateVideo"]');
  const downloadVideoLink = node.querySelector('[data-field="downloadVideo"]');
  renderVideoScript(videoContainer, result.video_script);
  renderVideoResult(videoStatus, downloadVideoLink, result.video);
  if (!analysisId) {
    videoButton.disabled = true;
    generateVideoButton.disabled = true;
    videoStatus.textContent = "This saved record is missing an analysis id. Reopen it from History or rerun analysis.";
  }

  resultPanel.className = "result-panel";
  resultPanel.dataset.analysisId = analysisId || "";
  resultPanel.innerHTML = "";
  resultPanel.appendChild(node);
  bindTabs(resultPanel);
}

async function generateVideo(analysisId, statusContainer, downloadLink, button) {
  button.disabled = true;
  button.textContent = "Generating...";
  statusContainer.textContent = "Generating MP4 locally...";
  downloadLink.hidden = true;

  try {
    const response = await fetch(`/analyses/${encodeURIComponent(analysisId)}/video`, {
      method: "POST",
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Video generation failed with status ${response.status}`);
    }

    renderVideoResult(statusContainer, downloadLink, payload.video);
    await loadHistory();
  } catch (error) {
    statusContainer.textContent = error.message || "Video generation failed.";
  } finally {
    button.disabled = false;
    button.textContent = "Generate Video";
  }
}

function renderVideoResult(statusContainer, downloadLink, video) {
  if (!video || !video.video_url) {
    statusContainer.textContent = "No MP4 generated yet.";
    downloadLink.hidden = true;
    return;
  }

  statusContainer.textContent = `Generated ${video.scene_count || ""} scenes at ${formatDate(video.generated_at)}.`;
  downloadLink.href = video.video_url;
  downloadLink.hidden = false;
}

async function generateVideoScript(analysisId, container, button) {
  const statusContainer = resultPanel.querySelector('[data-field="videoStatus"]');
  button.disabled = true;
  button.textContent = "Generating...";
  if (statusContainer) statusContainer.textContent = "Generating video script with Gemini...";
  container.innerHTML = '<p class="result-meta">Creating video script...</p>';

  try {
    const response = await fetch(`/analyses/${encodeURIComponent(analysisId)}/video-script`, {
      method: "POST",
    });

    if (!response.ok) {
      throw new Error(`Video script failed with status ${response.status}`);
    }

    const payload = await response.json();
    renderVideoScript(container, payload.video_script);
    if (statusContainer) statusContainer.textContent = "Video script generated.";
    await loadHistory();
  } catch (error) {
    const message = error.message || "Video script failed.";
    if (statusContainer) statusContainer.textContent = message;
    container.innerHTML = `<p class="result-meta">${escapeHtml(message)}</p>`;
  } finally {
    button.disabled = false;
    button.textContent = "Regenerate Script";
  }
}

function renderVideoScript(container, script) {
  container.innerHTML = "";
  if (!script || !Array.isArray(script.scenes) || script.scenes.length === 0) {
    container.innerHTML = '<p class="result-meta">No video script generated yet.</p>';
    return;
  }

  const header = document.createElement("div");
  header.className = "script-overview";
  header.innerHTML = `
    <strong>${escapeHtml(script.title || "Research explainer")}</strong>
    <span>${escapeHtml(String(script.duration_seconds || ""))} seconds · ${escapeHtml(script.audience || "general audience")}</span>
  `;
  container.appendChild(header);

  script.scenes.forEach((scene, index) => {
    const card = document.createElement("article");
    card.className = "scene-card";
    const bullets = Array.isArray(scene.bullets) ? scene.bullets : [];
    const evidence = scene.evidence && typeof scene.evidence === "object" ? scene.evidence : null;
    const evidenceLine = evidence?.claim
      ? `<p class="result-meta"><strong>Evidence:</strong> ${escapeHtml(evidence.section || "section")} · ${escapeHtml(formatPages(evidence.pages))}<br>${escapeHtml(evidence.claim || "")}</p>`
      : "";
    card.innerHTML = `
      <div class="card-meta">Scene ${escapeHtml(scene.scene_number || index + 1)} · ${escapeHtml(scene.role || "scene")}</div>
      <h4>${escapeHtml(scene.heading || "Untitled scene")}</h4>
      <ul>${bullets.map((bullet) => `<li>${escapeHtml(bullet)}</li>`).join("")}</ul>
      <p><strong>Voiceover:</strong> ${escapeHtml(scene.voiceover || "")}</p>
      ${evidenceLine}
      <p class="result-meta"><strong>Visual:</strong> ${escapeHtml(scene.visual_type || "template")} · ${escapeHtml(scene.visual_note || "")}</p>
    `;
    container.appendChild(card);
  });
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
    const displayTitle = item.paper_title || item.filename || "Untitled PDF";
    card.innerHTML = `
      <strong>${escapeHtml(displayTitle)}</strong>
      ${item.paper_title && item.filename ? `<p class="result-meta">${escapeHtml(item.filename)}</p>` : ""}
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
    const result = record.result || {};
    result.analysis_id = result.analysis_id || record.analysis_id;
    renderResult(result, record.filename || "Saved Analysis");
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
