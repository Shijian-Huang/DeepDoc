const uploadForm = document.querySelector("#uploadForm");
const pdfFileInput = document.querySelector("#pdfFile");
const fileLabel = document.querySelector("#fileLabel");
const analyzeButton = document.querySelector("#analyzeButton");
const statusPanel = document.querySelector("#statusPanel");
const statusText = document.querySelector("#statusText");
const resultPanel = document.querySelector("#resultPanel");
const resultTemplate = document.querySelector("#resultTemplate");
const brandHomeButton = document.querySelector("#brandHomeButton");
const historyPanel = document.querySelector("#historyPanel");
const openHistoryButton = document.querySelector("#openHistoryButton");
const closeHistoryButton = document.querySelector("#closeHistoryButton");
const historyOverlay = document.querySelector("#historyOverlay");
const historyList = document.querySelector("#historyList");
const refreshHistoryButton = document.querySelector("#refreshHistoryButton");
const historySearchInput = document.querySelector("#historySearch");
const historyModeFilter = document.querySelector("#historyModeFilter");
const historyDateFilter = document.querySelector("#historyDateFilter");
const sourceTabs = document.querySelectorAll("[data-source-tab]");
const sourcePanels = document.querySelectorAll("[data-source-panel]");
const arxivSearchForm = document.querySelector("#arxivSearchForm");
const arxivQueryInput = document.querySelector("#arxivQuery");
const arxivMaxResultsInput = document.querySelector("#arxivMaxResults");
const arxivSummaryModeInput = document.querySelector("#arxivSummaryMode");
const arxivSearchButton = document.querySelector("#arxivSearchButton");
const arxivSearchStatus = document.querySelector("#arxivSearchStatus");
const arxivResults = document.querySelector("#arxivResults");

let historyItems = [];
let arxivItems = [];

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
openHistoryButton.addEventListener("click", openHistoryDrawer);
closeHistoryButton.addEventListener("click", closeHistoryDrawer);
historyOverlay.addEventListener("click", closeHistoryDrawer);
brandHomeButton.addEventListener("click", returnHome);
brandHomeButton.addEventListener("keydown", handleBrandHomeKeydown);
resultPanel.addEventListener("click", handleResultPanelClick);
resultPanel.addEventListener("click", handleEvidenceNavigation);
resultPanel.addEventListener("change", handleResultPanelChange);
sourceTabs.forEach((tab) => tab.addEventListener("click", () => switchSourceTab(tab.dataset.sourceTab)));
arxivSearchForm.addEventListener("submit", searchArxiv);
arxivResults.addEventListener("click", handleArxivResultClick);
historySearchInput.addEventListener("input", renderFilteredHistory);
historyModeFilter.addEventListener("change", renderFilteredHistory);
historyDateFilter.addEventListener("change", renderFilteredHistory);


function openHistoryDrawer() {
  document.body.classList.add("is-history-open");
  historyPanel.setAttribute("aria-hidden", "false");
  historyOverlay.hidden = false;
  loadHistory();
}

function closeHistoryDrawer() {
  document.body.classList.remove("is-history-open");
  historyPanel.setAttribute("aria-hidden", "true");
  historyOverlay.hidden = true;
}

function handleBrandHomeKeydown(event) {
  if (event.key !== "Enter" && event.key !== " ") return;
  event.preventDefault();
  returnHome();
}

function returnHome() {
  if (!document.body.classList.contains("has-result")) return;
  document.body.classList.remove("has-result");
  document.body.classList.remove("source-collapsed");
  document.body.classList.remove("is-busy");
  resultPanel.className = "result-panel empty-state";
  resultPanel.dataset.analysisId = "";
  resultPanel.innerHTML = "";
  statusPanel.hidden = true;
}

function switchSourceTab(source) {
  sourceTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.sourceTab === source));
  sourcePanels.forEach((panel) => panel.classList.toggle("active", panel.dataset.sourcePanel === source));
}

async function searchArxiv(event) {
  event.preventDefault();
  const query = arxivQueryInput.value.trim();
  const maxResults = arxivMaxResultsInput.value || "10";
  if (!query) {
    arxivSearchStatus.textContent = "Enter a search query.";
    arxivResults.innerHTML = "";
    return;
  }

  arxivSearchButton.disabled = true;
  arxivSearchButton.textContent = "Searching";
  arxivSearchStatus.textContent = "Searching arXiv...";
  arxivResults.innerHTML = "";

  try {
    const response = await fetch(`/arxiv/search?q=${encodeURIComponent(query)}&max_results=${encodeURIComponent(maxResults)}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `arXiv search failed with status ${response.status}`);
    }
    arxivItems = payload.results || [];
    renderArxivResults(arxivItems);
    arxivSearchStatus.textContent = arxivItems.length ? `${arxivItems.length} papers found.` : "No arXiv papers found.";
  } catch (error) {
    arxivItems = [];
    arxivResults.innerHTML = "";
    arxivSearchStatus.textContent = error.message || "arXiv search failed.";
  } finally {
    arxivSearchButton.disabled = false;
    arxivSearchButton.textContent = "Search";
  }
}

function renderArxivResults(items) {
  arxivResults.innerHTML = "";
  if (!items.length) return;

  items.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "arxiv-card";
    const authors = Array.isArray(item.authors) ? item.authors.join(", ") : "";
    const categories = Array.isArray(item.categories) ? item.categories.slice(0, 4).join(" · ") : "";
    card.innerHTML = `
      <div>
        <p class="card-meta">${escapeHtml(item.published || "date unavailable")}${categories ? ` · ${escapeHtml(categories)}` : ""}</p>
        <h3>${escapeHtml(item.title || "Untitled arXiv paper")}</h3>
        <p class="result-meta">${escapeHtml(trimText(authors, 180))}</p>
        <p>${escapeHtml(trimText(item.summary || "", 360))}</p>
      </div>
      <div class="arxiv-card-actions">
        <a href="${escapeAttribute(item.abs_url || "#")}" target="_blank" rel="noreferrer">arXiv</a>
        <a href="${escapeAttribute(item.pdf_url || "#")}" target="_blank" rel="noreferrer">PDF</a>
        <button type="button" data-arxiv-index="${index}">Analyze</button>
      </div>
    `;
    arxivResults.appendChild(card);
  });
}

async function handleArxivResultClick(event) {
  const button = event.target.closest("[data-arxiv-index]");
  if (!button) return;
  const item = arxivItems[Number(button.dataset.arxivIndex)];
  if (!item) return;

  const summaryMode = arxivSummaryModeInput.value || "standard";
  setBusy(true, `Downloading ${item.arxiv_id || "arXiv paper"} and analyzing...`);
  button.disabled = true;
  button.textContent = "Analyzing";

  try {
    const response = await fetch("/arxiv/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        arxiv_id: item.arxiv_id,
        pdf_url: item.pdf_url,
        summary_mode: summaryMode,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `arXiv analysis failed with status ${response.status}`);
    }
    renderResult(payload, `arxiv-${item.arxiv_id}.pdf`);
    await loadHistory();
  } catch (error) {
    renderError(error.message || "arXiv analysis failed.");
  } finally {
    button.disabled = false;
    button.textContent = "Analyze";
    setBusy(false);
  }
}


function handleResultPanelClick(event) {
  const reanalyzeButton = event.target.closest('[data-field="reanalyze"]');
  const scriptButton = event.target.closest('[data-field="generateVideoScript"]');
  const videoButton = event.target.closest('[data-field="generateVideo"]');
  if (!reanalyzeButton && !scriptButton && !videoButton) return;

  const analysisId = resultPanel.dataset.analysisId;
  if (!analysisId) return;

  if (reanalyzeButton) {
    reanalyzeExistingAnalysis(analysisId, reanalyzeButton);
    return;
  }

  const videoContainer = resultPanel.querySelector('[data-field="videoScript"]');
  const videoStatus = resultPanel.querySelector('[data-field="videoStatus"]');
  const downloadVideoLink = resultPanel.querySelector('[data-field="downloadVideo"]');
  const downloadScriptLink = resultPanel.querySelector('[data-field="downloadScript"]');
  const downloadSlidesLink = resultPanel.querySelector('[data-field="downloadSlides"]');
  const downloadSlidesHtmlLink = resultPanel.querySelector('[data-field="downloadSlidesHtml"]');
  const slideCountControl = resultPanel.querySelector('[data-field="slideCount"]');

  if (scriptButton) {
    generateVideoScript(analysisId, videoContainer, scriptButton, slideCountControl, downloadScriptLink, downloadVideoLink, downloadSlidesLink, downloadSlidesHtmlLink);
  }
  if (videoButton) {
    generateVideo(analysisId, videoContainer, videoStatus, downloadVideoLink, videoButton, slideCountControl);
  }
}

async function reanalyzeExistingAnalysis(analysisId, button) {
  button.disabled = true;
  button.textContent = "Reanalyzing";
  setBusy(true, "Creating a new analysis version...");

  try {
    const response = await fetch(`/analyses/${encodeURIComponent(analysisId)}/reanalyze`, {
      method: "POST",
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Reanalysis failed with status ${response.status}`);
    }
    renderResult(payload, "Reanalyzed paper");
    await loadHistory();
  } catch (error) {
    renderError(error.message || "Reanalysis failed.");
  } finally {
    button.disabled = false;
    button.textContent = "Reanalyze as New";
    setBusy(false);
  }
}

function handleResultPanelChange(event) {
  if (!event.target.matches('[data-field="slideCount"]')) return;
  updateVideoArtifactAvailability(resultPanel);
}

function setBusy(isBusy, message = "") {
  document.body.classList.toggle("is-busy", isBusy);
  analyzeButton.disabled = isBusy;
  analyzeButton.classList.toggle("is-loading", isBusy);
  analyzeButton.setAttribute("aria-busy", String(isBusy));
  analyzeButton.textContent = "Analyze";
  statusText.textContent = message;
  statusPanel.hidden = !isBusy;
}

function renderError(message) {
  document.body.classList.remove("has-result");
  document.body.classList.remove("source-collapsed");
  document.body.classList.remove("is-busy");
  resultPanel.className = "result-panel error";
  resultPanel.innerHTML = `
    <div>
      <p class="eyebrow">Error</p>
      <h2>${escapeHtml(message)}</h2>
    </div>
  `;
}

function renderResult(result, fallbackFilename = "Analysis Result") {
  document.body.classList.add("has-result");
  document.body.classList.add("source-collapsed");
  const node = resultTemplate.content.cloneNode(true);
  const summary = result.document_summary || {};
  const analysisId = result.analysis_id;
  const paperTitle = result.paper_title || summary.title || result.video_script?.title || fallbackFilename;

  node.querySelector('[data-field="mode"]').textContent = formatMode(result.summary_mode);
  node.querySelector('[data-field="title"]').textContent = paperTitle;
  node.querySelector('[data-field="meta"]').textContent = formatMeta(result);
  node.querySelector('[data-field="summary"]').textContent = summary.summary || "No summary returned.";

  const download = node.querySelector('[data-field="download"]');
  const downloadMarkdown = node.querySelector('[data-field="downloadMarkdown"]');
  const reanalyzeButton = node.querySelector('[data-field="reanalyze"]');
  if (analysisId) {
    download.href = `/analyses/${analysisId}/download`;
    downloadMarkdown.href = `/analyses/${analysisId}/markdown/download`;
    downloadMarkdown.download = "";
    download.download = "";
  } else {
    download.removeAttribute("href");
    downloadMarkdown.removeAttribute("href");
    if (reanalyzeButton) reanalyzeButton.disabled = true;
  }
  renderPdfViewer(node, analysisId);

  renderList(node.querySelector('[data-field="keyIdeas"]'), summary.key_ideas || []);
  renderList(node.querySelector('[data-field="contributions"]'), summary.contributions || []);
  renderOptionalEvidenceSections(node.querySelector('[data-field="optionalEvidenceSections"]'), summary);
  renderEvidenceViewer(node.querySelector('[data-field="evidence"]'), summary.evidence || [], {
    emptyText: "No evidence claims returned.",
  });
  renderEvidenceViewer(node.querySelector('[data-field="sources"]'), result.evidence_sources || [], {
    emptyText: "No source sections returned.",
    sourceType: "source",
  });
  renderReferences(node.querySelector('[data-field="references"]'), result.references || []);
  const videoContainer = node.querySelector('[data-field="videoScript"]');
  const videoStatus = node.querySelector('[data-field="videoStatus"]');
  const videoButton = node.querySelector('[data-field="generateVideoScript"]');
  const generateVideoButton = node.querySelector('[data-field="generateVideo"]');
  const downloadScriptLink = node.querySelector('[data-field="downloadScript"]');
  const downloadSlidesLink = node.querySelector('[data-field="downloadSlides"]');
  const downloadSlidesHtmlLink = node.querySelector('[data-field="downloadSlidesHtml"]');
  const downloadVideoLink = node.querySelector('[data-field="downloadVideo"]');
  const slideCountControl = node.querySelector('[data-field="slideCount"]');
  renderVideoScript(videoContainer, result.video_script);
  renderVideoScriptDownload(downloadScriptLink, downloadSlidesLink, downloadSlidesHtmlLink, analysisId, result.video_script);
  renderVideoResult(videoStatus, downloadVideoLink, result.video);
  syncSlideCount(slideCountControl, result.video_script);
  updateVideoArtifactAvailability(node);
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

function renderPdfViewer(root, analysisId) {
  const panel = root.querySelector('[data-field="pdfPanel"]');
  const viewer = root.querySelector('[data-field="pdfViewer"]');
  const placeholder = root.querySelector('[data-field="pdfPlaceholder"]');
  const status = root.querySelector('[data-field="pdfStatus"]');
  const downloadPdf = root.querySelector('[data-field="downloadPdf"]');
  if (!panel || !viewer || !placeholder || !status) return;

  if (!analysisId) {
    panel.dataset.pdfBaseUrl = "";
    viewer.hidden = true;
    viewer.removeAttribute("src");
    placeholder.hidden = false;
    status.textContent = "Unavailable";
    if (downloadPdf) {
      downloadPdf.hidden = true;
      downloadPdf.removeAttribute("href");
    }
    return;
  }

  const pdfUrl = `/analyses/${encodeURIComponent(analysisId)}/pdf`;
  panel.dataset.pdfBaseUrl = pdfUrl;
  viewer.hidden = true;
  viewer.removeAttribute("src");
  placeholder.hidden = false;
  placeholder.textContent = "Checking original PDF...";
    status.textContent = "Loading";
  if (downloadPdf) {
    downloadPdf.hidden = true;
    downloadPdf.removeAttribute("href");
  }
  checkPdfAvailability(viewer, placeholder, status, pdfUrl, downloadPdf);
}

async function checkPdfAvailability(viewer, placeholder, status, pdfUrl, downloadPdf) {
  try {
    const response = await fetch(pdfUrl, {method: "HEAD"});
    if (!response.ok) throw new Error("PDF unavailable");
    viewer.hidden = false;
    viewer.src = pdfViewerUrl(pdfUrl, 1);
    placeholder.hidden = true;
    status.textContent = "Page 1";
    if (downloadPdf) {
      downloadPdf.href = pdfUrl;
      downloadPdf.hidden = false;
      downloadPdf.download = "";
    }
  } catch {
    viewer.hidden = true;
    viewer.removeAttribute("src");
    placeholder.hidden = false;
    placeholder.textContent = "Original PDF is not available for this analysis.";
    status.textContent = "Unavailable";
    if (downloadPdf) {
      downloadPdf.hidden = true;
      downloadPdf.removeAttribute("href");
    }
  }
}

function handleEvidenceNavigation(event) {
  const evidenceItem = event.target.closest(".evidence-viewer");
  if (!evidenceItem) return;
  const page = Number(evidenceItem?.dataset.page || 0);
  navigateToEvidencePage(page, evidenceItem);
}

function navigateToEvidencePage(page, activeEvidenceItem = null) {
  const panel = resultPanel.querySelector('[data-field="pdfPanel"]');
  const viewer = resultPanel.querySelector('[data-field="pdfViewer"]');
  const placeholder = resultPanel.querySelector('[data-field="pdfPlaceholder"]');
  const status = resultPanel.querySelector('[data-field="pdfStatus"]');
  const pdfUrl = panel?.dataset.pdfBaseUrl || "";

  if (!page) {
    if (status) status.textContent = "No page number";
    return;
  }
  if (!pdfUrl || !viewer) {
    if (status) status.textContent = "PDF unavailable";
    if (placeholder) {
      placeholder.hidden = false;
      placeholder.textContent = "PDF navigation is unavailable for this analysis.";
    }
    return;
  }

  // TODO(v2): map evidence snippets to text ranges or bounding boxes for true PDF highlighting.
  viewer.hidden = false;
  viewer.removeAttribute("src");
  requestAnimationFrame(() => {
    viewer.src = pdfViewerUrl(pdfUrl, page);
  });
  if (placeholder) placeholder.hidden = true;
  if (status) status.textContent = `Page ${page}`;
  markActiveEvidence(activeEvidenceItem);
}

function pdfViewerUrl(pdfUrl, page) {
  return `${pdfUrl}?view=${Date.now()}#page=${encodeURIComponent(page)}`;
}

function markActiveEvidence(activeEvidenceItem) {
  resultPanel.querySelectorAll(".evidence-viewer.active-evidence").forEach((item) => {
    item.classList.remove("active-evidence");
  });
  if (activeEvidenceItem) activeEvidenceItem.classList.add("active-evidence");
}

async function generateVideo(analysisId, scriptContainer, statusContainer, downloadLink, button, slideCountControl) {
  const selectedSlideCount = selectedSlides(slideCountControl);
  const currentScriptSlides = Number(scriptContainer?.dataset.slideCount || 0);
  if (!currentScriptSlides || currentScriptSlides !== selectedSlideCount) {
    statusContainer.textContent = `Generate a ${selectedSlideCount}-slide script before creating the MP4.`;
    clearDownloadLink(downloadLink);
    return;
  }

  button.disabled = true;
  button.textContent = "Generating...";
  statusContainer.textContent = `Generating ${selectedSlideCount}-slide MP4 locally...`;
  clearDownloadLink(downloadLink);

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
    button.textContent = "Create MP4";
  }
}

function renderVideoScriptDownload(downloadLink, slidesLink, slidesHtmlLink, analysisId, script) {
  const slideCount = scriptSlideCount(script);
  if (!analysisId || !slideCount) {
    clearDownloadLink(downloadLink);
    clearDownloadLink(slidesLink);
    clearDownloadLink(slidesHtmlLink);
    return;
  }

  if (slidesHtmlLink) {
    slidesHtmlLink.href = `/analyses/${encodeURIComponent(analysisId)}/slides-html/download`;
    slidesHtmlLink.dataset.slideCount = String(slideCount);
    slidesHtmlLink.textContent = `Slides HTML · ${slideCount} slides`;
    slidesHtmlLink.hidden = false;
  }

  if (slidesLink) {
    slidesLink.href = `/analyses/${encodeURIComponent(analysisId)}/slides/download`;
    slidesLink.dataset.slideCount = String(slideCount);
    slidesLink.textContent = `Slides Markdown · ${slideCount} slides`;
    slidesLink.hidden = false;
  }

  if (downloadLink) {
    downloadLink.href = `/analyses/${encodeURIComponent(analysisId)}/video-script/download`;
    downloadLink.dataset.slideCount = String(slideCount);
    downloadLink.textContent = `Slides JSON · ${slideCount} slides`;
    downloadLink.hidden = false;
  }
}

function renderVideoResult(statusContainer, downloadLink, video) {
  if (!video || !video.video_url) {
    statusContainer.textContent = "No MP4 generated yet.";
    clearDownloadLink(downloadLink);
    return;
  }

  const slideCount = Number(video.scene_count || 0);
  statusContainer.textContent = `Generated ${slideCount || ""}-slide video at ${formatDate(video.generated_at)}.`;
  downloadLink.href = video.video_url;
  downloadLink.dataset.slideCount = String(slideCount);
  downloadLink.dataset.generatedAt = video.generated_at || "";
  downloadLink.textContent = `MP4 Video · ${slideCount} slides`;
  downloadLink.hidden = false;
}

function clearDownloadLink(downloadLink) {
  if (!downloadLink) return;
  downloadLink.hidden = true;
  downloadLink.removeAttribute("href");
  delete downloadLink.dataset.slideCount;
  delete downloadLink.dataset.generatedAt;
}

function updateVideoArtifactAvailability(root) {
  const slideCountControl = root.querySelector('[data-field="slideCount"]');
  const selectedSlideCount = selectedSlides(slideCountControl);
  const scriptContainer = root.querySelector('[data-field="videoScript"]');
  const statusContainer = root.querySelector('[data-field="videoStatus"]');
  const downloadScriptLink = root.querySelector('[data-field="downloadScript"]');
  const downloadSlidesLink = root.querySelector('[data-field="downloadSlides"]');
  const downloadSlidesHtmlLink = root.querySelector('[data-field="downloadSlidesHtml"]');
  const downloadVideoLink = root.querySelector('[data-field="downloadVideo"]');
  const scriptSlides = Number(scriptContainer?.dataset.slideCount || 0);
  const videoSlides = Number(downloadVideoLink?.dataset.slideCount || 0);

  if (downloadScriptLink) {
    downloadScriptLink.hidden = !scriptSlides || scriptSlides !== selectedSlideCount;
  }
  if (downloadSlidesLink) {
    downloadSlidesLink.hidden = !scriptSlides || scriptSlides !== selectedSlideCount;
  }
  if (downloadSlidesHtmlLink) {
    downloadSlidesHtmlLink.hidden = !scriptSlides || scriptSlides !== selectedSlideCount;
  }
  if (downloadVideoLink) {
    downloadVideoLink.hidden = !videoSlides || videoSlides !== selectedSlideCount || scriptSlides !== selectedSlideCount;
  }

  if (statusContainer && scriptSlides && scriptSlides !== selectedSlideCount) {
    statusContainer.textContent = `Current script is ${scriptSlides} slides. Generate a ${selectedSlideCount}-slide script to update downloads.`;
  } else if (statusContainer && scriptSlides === selectedSlideCount && videoSlides === selectedSlideCount) {
    statusContainer.textContent = `Generated ${videoSlides}-slide video at ${formatDate(downloadVideoLink.dataset.generatedAt)}.`;
  } else if (statusContainer && scriptSlides === selectedSlideCount) {
    statusContainer.textContent = `Slides ready: ${scriptSlides} slides. Create an MP4 for this version when ready.`;
  } else if (statusContainer) {
    statusContainer.textContent = "Generate slides first, then download Markdown or optionally create an MP4.";
  }
}

async function generateVideoScript(analysisId, container, button, slideCountControl, downloadScriptLink, downloadVideoLink, downloadSlidesLink, downloadSlidesHtmlLink) {
  const statusContainer = resultPanel.querySelector('[data-field="videoStatus"]');
  const slideCount = selectedSlides(slideCountControl);
  button.disabled = true;
  button.textContent = "Generating";
  if (slideCountControl) slideCountControl.disabled = true;
  clearDownloadLink(downloadScriptLink);
  clearDownloadLink(downloadSlidesLink);
  clearDownloadLink(downloadSlidesHtmlLink);
  clearDownloadLink(downloadVideoLink);
  if (statusContainer) statusContainer.textContent = `Generating ${slideCount} slides with Gemini...`;
  container.dataset.slideCount = "";
  container.innerHTML = `<p class="result-meta">Creating ${escapeHtml(slideCount)} slides...</p>`;

  try {
    const response = await fetch(`/analyses/${encodeURIComponent(analysisId)}/video-script?slide_count=${encodeURIComponent(slideCount)}`, {
      method: "POST",
    });

    if (!response.ok) {
      throw new Error(`Video script failed with status ${response.status}`);
    }

    const payload = await response.json();
    renderVideoScript(container, payload.video_script);
    renderVideoScriptDownload(downloadScriptLink, downloadSlidesLink, downloadSlidesHtmlLink, analysisId, payload.video_script);
    syncSlideCount(slideCountControl, payload.video_script);
    if (statusContainer) statusContainer.textContent = `Slides generated: ${payload.video_script?.scenes?.length || slideCount} slides. Create MP4 for this version when ready.`;
    await loadHistory();
  } catch (error) {
    const message = error.message || "Video script failed.";
    if (statusContainer) statusContainer.textContent = message;
    container.innerHTML = `<p class="result-meta">${escapeHtml(message)}</p>`;
  } finally {
    button.disabled = false;
    if (slideCountControl) slideCountControl.disabled = false;
    button.textContent = "Regenerate Slides";
  }
}

function renderVideoScript(container, script) {
  container.innerHTML = "";
  if (!script || !Array.isArray(script.scenes) || script.scenes.length === 0) {
    container.dataset.slideCount = "";
    container.innerHTML = '<p class="result-meta">No video script generated yet.</p>';
    return;
  }

  const header = document.createElement("div");
  header.className = "script-overview";
  const sceneCount = scriptSlideCount(script);
  container.dataset.slideCount = String(sceneCount);
  header.innerHTML = `
    <strong>${escapeHtml(script.title || "Research explainer")}</strong>
    <span>${escapeHtml(String(sceneCount))} slides · ${escapeHtml(script.audience || "general audience")}</span>
  `;
  container.appendChild(header);

  script.scenes.forEach((scene, index) => {
    const card = document.createElement("article");
    card.className = "scene-card";
    const bullets = Array.isArray(scene.bullets) ? scene.bullets : [];
    card.innerHTML = `
      <div class="card-meta">Scene ${escapeHtml(scene.scene_number || index + 1)} · ${escapeHtml(scene.role || "scene")}</div>
      <h4>${escapeHtml(scene.heading || "Untitled scene")}</h4>
      <ul>${bullets.map((bullet) => `<li>${escapeHtml(bullet)}</li>`).join("")}</ul>
      <p><strong>Voiceover:</strong> ${escapeHtml(scene.voiceover || "")}</p>
      <p class="result-meta"><strong>Visual:</strong> ${escapeHtml(scene.visual_type || "template")} · ${escapeHtml(scene.visual_note || "")}</p>
    `;
    if (scene.evidence && typeof scene.evidence === "object") {
      card.appendChild(createEvidenceItem(scene.evidence, {compact: true}));
    } else {
      card.appendChild(createMissingEvidenceItem("No evidence attached to this scene."));
    }
    container.appendChild(card);
  });
}

function syncSlideCount(control, script) {
  if (!control || !script || !Array.isArray(script.scenes)) return;
  const count = String(scriptSlideCount(script));
  const hasOption = Array.from(control.options).some((option) => option.value === count);
  if (hasOption) control.value = count;
}

function scriptSlideCount(script) {
  if (!script || !Array.isArray(script.scenes)) return 0;
  return Number(script.slide_count || script.scenes.length || 0);
}

function selectedSlides(control) {
  return Number(control?.value || 10);
}

function renderList(container, items) {
  container.innerHTML = "";
  if (!items.length) {
    container.appendChild(emptyLine("No items returned."));
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");
    if (item && typeof item === "object") {
      li.appendChild(document.createTextNode(item.text || item.claim || item.title || item.summary || "Untitled item"));
      const evidenceItems = evidenceListFromItem(item);
      if (evidenceItems.length) {
        const nested = document.createElement("div");
        nested.className = "inline-evidence-list";
        renderEvidenceViewer(nested, evidenceItems, {emptyText: "", compact: true});
        li.appendChild(nested);
      } else if ("evidence" in item) {
        li.appendChild(createMissingEvidenceItem("No evidence attached."));
      }
    } else {
      li.textContent = item;
    }
    container.appendChild(li);
  });
}

function renderOptionalEvidenceSections(container, summary) {
  container.innerHTML = "";
  const sections = [
    ["Limitations", summary.limitations],
    ["Discussion Questions", summary.discussion_questions || summary.discussionQuestions],
    ["Reviewer Questions", summary.reviewer_questions || summary.reviewerQuestions],
  ].filter(([, items]) => Array.isArray(items) && items.length);

  if (!sections.length) {
    container.hidden = true;
    return;
  }

  container.hidden = false;
  sections.forEach(([title, items]) => {
    const article = document.createElement("article");
    article.innerHTML = `
      <h3>${escapeHtml(title)}</h3>
      <ol></ol>
    `;
    renderList(article.querySelector("ol"), items);
    container.appendChild(article);
  });
}

function renderEvidenceViewer(container, evidence, options = {}) {
  container.innerHTML = "";
  const items = Array.isArray(evidence) ? evidence.filter(Boolean) : evidence ? [evidence] : [];
  if (options.title && items.length) {
    const title = document.createElement("h4");
    title.className = "evidence-viewer-title";
    title.textContent = options.title;
    container.appendChild(title);
  }
  if (!items.length) {
    if (options.emptyText) container.appendChild(emptyBlock(options.emptyText));
    return;
  }

  items.forEach((item) => {
    container.appendChild(createEvidenceItem(item, options));
  });
}

function createEvidenceItem(item, options = {}) {
  if (!item || typeof item !== "object") {
    return createMissingEvidenceItem("Evidence is unavailable.");
  }

  const details = document.createElement("details");
  details.className = `evidence-viewer${options.compact ? " compact" : ""}`;
  const section = item.section || item.section_title || item.heading || "section unavailable";
  const pageValue = item.pages || item.page_numbers || item.page_number || item.page;
  const page = firstEvidencePage(pageValue);
  const pages = formatPages(pageValue);
  const snippet = evidenceSnippet(item);
  const label = item.claim || item.title || item.summary || item.excerpt || "Evidence snippet";
  if (page) details.dataset.page = String(page);
  details.innerHTML = `
    <summary>
      <span class="evidence-page">p. ${escapeHtml(pages)}</span>
      <span class="evidence-section">${escapeHtml(section)}</span>
      <span class="evidence-label">${escapeHtml(trimText(label, 132))}</span>
    </summary>
    <div class="evidence-body">
      <p>${escapeHtml(snippet || "No snippet available for this evidence item.")}</p>
    </div>
  `;
  return details;
}

function createMissingEvidenceItem(text) {
  const item = document.createElement("div");
  item.className = "missing-evidence";
  item.textContent = text;
  return item;
}

function evidenceSnippet(item) {
  return item.excerpt || item.snippet || item.quote || item.claim || item.text || "";
}

function firstEvidencePage(pages) {
  if (Array.isArray(pages)) {
    const value = pages.find((page) => Number(page) > 0);
    return Number(value || 0);
  }
  const match = String(pages || "").match(/\d+/);
  return match ? Number(match[0]) : 0;
}

function evidenceListFromItem(item) {
  if (Array.isArray(item.evidence)) return item.evidence;
  if (item.evidence && typeof item.evidence === "object") return [item.evidence];
  if (Array.isArray(item.evidence_items)) return item.evidence_items;
  if (Array.isArray(item.sources)) return item.sources;
  return [];
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
    historyItems = payload.analyses || [];
    renderFilteredHistory();
  } catch (error) {
    historyList.innerHTML = `<p class="result-meta">${escapeHtml(error.message)}</p>`;
  }
}

function renderFilteredHistory() {
  const query = normalizeSearch(historySearchInput.value);
  const mode = historyModeFilter.value;
  const date = historyDateFilter.value;
  const filteredItems = historyItems.filter((item) => {
    const modeMatches = mode === "all" || item.summary_mode === mode;
    const dateMatches = !date || historyDateKey(item.created_at) === date;
    const queryMatches = !query || historySearchText(item).includes(query);
    return modeMatches && dateMatches && queryMatches;
  });

  renderHistory(filteredItems, historyItems.length);
}

function renderHistory(items) {
  historyList.innerHTML = "";
  if (!historyItems.length) {
    historyList.innerHTML = '<p class="result-meta">No saved analyses yet.</p>';
    return;
  }

  if (!items.length) {
    historyList.innerHTML = '<p class="result-meta">No analyses match the current search or filter.</p>';
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
        <button class="danger-action" type="button" data-delete="${escapeHtml(item.analysis_id || "")}">Delete</button>
      </div>
    `;
    card.querySelector("[data-open]").addEventListener("click", () => openAnalysis(item.analysis_id));
    card.querySelector("[data-delete]").addEventListener("click", () => deleteHistoryItem(item));
    historyList.appendChild(card);
  });
}

function historySearchText(item) {
  return normalizeSearch([
    item.paper_title,
    item.filename,
    item.summary,
    formatMode(item.summary_mode),
    formatDate(item.created_at),
  ].join(" "));
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

async function deleteHistoryItem(item) {
  const analysisId = item?.analysis_id;
  if (!analysisId) return;

  const displayTitle = item.paper_title || item.filename || "this analysis";
  const confirmed = window.confirm(`Delete "${displayTitle}" from history?`);
  if (!confirmed) return;

  setBusy(true, "Deleting saved analysis...");
  try {
    const response = await fetch(`/analyses/${encodeURIComponent(analysisId)}`, {
      method: "DELETE",
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "Could not delete this analysis.");
    }

    historyItems = historyItems.filter((entry) => entry.analysis_id !== analysisId);
    renderFilteredHistory();
    if (resultPanel.dataset.analysisId === analysisId) {
      returnHome();
    }
  } catch (error) {
    renderError(error.message || "Could not delete this analysis.");
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
  if (pages === undefined || pages === null || pages === "") return "unknown";
  if (Array.isArray(pages)) {
    if (pages.length === 0) return "unknown";
    return pages.join(", ");
  }
  return String(pages);
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

function historyDateKey(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function trimText(text, maxLength) {
  const value = String(text || "");
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength).trim()}...`;
}

function normalizeSearch(value) {
  return String(value || "").trim().toLowerCase();
}

function stripReferenceNumber(reference) {
  return String(reference || "").replace(/^\s*(?:\[\d+\]|\d{1,3}[.)])\s*/, "");
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

function escapeAttribute(value) {
  const text = String(value || "");
  if (!/^https:\/\/(arxiv\.org|www\.arxiv\.org|export\.arxiv\.org)\//.test(text)) {
    return "#";
  }
  return escapeHtml(text);
}

loadHistory();
