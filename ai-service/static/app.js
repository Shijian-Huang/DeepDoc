const uploadForm = document.querySelector("#uploadForm");
const uploadDropZone = document.querySelector("#uploadDropZone");
const pdfFileInput = document.querySelector("#pdfFile");
const fileLabel = document.querySelector("#fileLabel");
const folderButton = document.querySelector("#folderButton");
const uploadEmptyState = document.querySelector("[data-upload-empty]");
const uploadSelectedState = document.querySelector("[data-upload-selected]");
const uploadFileList = document.querySelector("#uploadFileList");
const addFilesButton = document.querySelector("#addFilesButton");
const clearUploadButton = document.querySelector("#clearUploadButton");
const analyzeButton = document.querySelector("#analyzeButton");
const statusPanel = document.querySelector("#statusPanel");
const statusText = document.querySelector("#statusText");
const analysisProgress = document.querySelector("#analysisProgress");
const analysisProgressTrack = analysisProgress?.querySelector('[role="progressbar"]');
const analysisProgressBar = document.querySelector("#analysisProgressBar");
const analysisProgressStage = document.querySelector("#analysisProgressStage");
const analysisProgressTime = document.querySelector("#analysisProgressTime");
const cancelAnalysisButton = document.querySelector("#cancelAnalysisButton");
const resultPanel = document.querySelector("#resultPanel");
const resultTemplate = document.querySelector("#resultTemplate");
const brandHomeButton = document.querySelector("#brandHomeButton");
const modelStatusNames = document.querySelectorAll("[data-model-status-name]");
const modelStatusConnections = document.querySelectorAll("[data-model-status-connection]");
const modelSwitchers = document.querySelectorAll("[data-model-switcher]");
const modelSwitcherMenus = document.querySelectorAll("[data-model-switcher-menu]");
const historyPanel = document.querySelector("#historyPanel");
const openHistoryButton = document.querySelector("#openHistoryButton");
const closeHistoryButton = document.querySelector("#closeHistoryButton");
const historyOverlay = document.querySelector("#historyOverlay");
const historyList = document.querySelector("#historyList");
const refreshHistoryButton = document.querySelector("#refreshHistoryButton");
const authBox = document.querySelector("#authBox");
const authEntry = document.querySelector("#authEntry");
const authModal = document.querySelector("#authModal");
const authModalOverlay = document.querySelector("#authModalOverlay");
const authForm = document.querySelector("#authForm");
const authFormTitle = document.querySelector("#authFormTitle");
const authEmailInput = document.querySelector("#authEmail");
const authPasswordInput = document.querySelector("#authPassword");
const showSignInButton = document.querySelector("#showSignInButton");
const showSignUpButton = document.querySelector("#showSignUpButton");
const authSubmitButton = document.querySelector("#authSubmitButton");
const authCancelButton = document.querySelector("#authCancelButton");
const authModalCloseButton = document.querySelector("#authModalCloseButton");
const authModeToggleButton = document.querySelector("#authModeToggleButton");
const authAvatarButton = document.querySelector("#authAvatarButton");
const authAccountMenu = document.querySelector("#authAccountMenu");
const authSignOutButton = document.querySelector("#authSignOutButton");
const authSession = document.querySelector("#authSession");
const authUserEmail = document.querySelector("#authUserEmail");
const authStatus = document.querySelector("#authStatus");
const historySearchInput = document.querySelector("#historySearch");
const historyModeFilter = document.querySelector("#historyModeFilter");
const historyDateFilter = document.querySelector("#historyDateFilter");
const sourceTabs = document.querySelectorAll("[data-source-tab]");
const sourcePanels = document.querySelectorAll("[data-source-panel]");
const arxivSearchForm = document.querySelector("#arxivSearchForm");
const arxivDirectForm = document.querySelector("#arxivDirectForm");
const arxivDirectInput = document.querySelector("#arxivDirectInput");
const arxivDirectButton = document.querySelector("#arxivDirectButton");
const arxivQueryInput = document.querySelector("#arxivQuery");
const arxivSearchFieldInput = document.querySelector("#arxivSearchField");
const arxivCategoryInput = document.querySelector("#arxivCategory");
const arxivMaxResultsInput = document.querySelector("#arxivMaxResults");
const arxivSortModeInput = document.querySelector("#arxivSortMode");
const arxivSummaryModeInput = document.querySelector("#arxivSummaryMode");
const arxivSearchButton = document.querySelector("#arxivSearchButton");
const arxivSearchStatus = document.querySelector("#arxivSearchStatus");
const arxivResults = document.querySelector("#arxivResults");
const arxivPagination = document.querySelector("#arxivPagination");
const arxivPrevButton = document.querySelector("#arxivPrevButton");
const arxivNextButton = document.querySelector("#arxivNextButton");
const arxivPageInfo = document.querySelector("#arxivPageInfo");
const popularSearches = document.querySelector("#popularSearches");
const arxivPlaceholders = [
  "Search a paper you've always wanted to understand.",
  "Find the paper behind today's breakthrough.",
  "Explore ideas, not just titles.",
  "Start with a question.",
  "Search by title, author, or topic.",
];
const ANONYMOUS_HISTORY_KEY = "deepdoc.anonymousAnalyses.v1";
const ANONYMOUS_PROJECTS_KEY = "deepdoc.anonymousProjects.v1";
const ANONYMOUS_PDF_DB_NAME = "deepdoc.anonymousPdfs";
const ANONYMOUS_PDF_STORE = "pdfs";
const ANONYMOUS_HISTORY_LIMIT = 50;
const MAX_UPLOAD_FILES = 10;
const MODEL_PREFERENCE_KEY = "deepdoc.llmPreference.v2";

let historyItems = [];
let arxivItems = [];
let currentProject = null;
let projectLocalFiles = new Map();
let pendingAddProjectId = "";
let authState = {
  enabled: false,
  client: null,
  mode: "",
  ready: false,
  session: null,
};
let activeLocalPdfUrl = "";
let activeLocalDownloadUrls = [];
let uploadFiles = [];
let selectedLlm = readModelPreference();
let analysisProgressTimer = null;
let analysisProgressStartedAt = 0;
let activeAnalysisRequestId = "";
let activeAnalysisController = null;
let analysisCancellationRequested = false;
let arxivSearchState = {
  query: "",
  searchField: "all",
  category: "",
  maxResults: 10,
  sortBy: "relevance",
  sortOrder: "descending",
  start: 0,
  page: 1,
  totalPages: 0,
  totalResults: 0,
};

initAuth();
loadModelStatus();

async function loadModelStatus() {
  try {
    const params = new URLSearchParams();
    if (selectedLlm?.provider) params.set("provider", selectedLlm.provider);
    if (selectedLlm?.model) params.set("model", selectedLlm.model);
    const response = await fetch(`/llm/options?${params.toString()}`, {cache: "no-store"});
    if (!response.ok) throw new Error(`Model check failed with status ${response.status}`);
    const health = await response.json();
    const provider = String(health.provider || "").toLowerCase();
    const model = String(health.model || provider || "Unknown model");
    selectedLlm = {provider, model};
    saveModelPreference(selectedLlm);
    setModelStatusNames(formatModelName(model, provider));
    setModelConnectionState(health.connected ? "Connected" : "Unavailable", !health.connected);
    renderModelOptions(health.options || []);
  } catch (_error) {
    setModelStatusNames("Model unavailable");
    setModelConnectionState("Backend offline", true);
  }
}

function readModelPreference() {
  try {
    const value = JSON.parse(window.localStorage.getItem(modelPreferenceStorageKey()) || "null");
    return value?.provider && value?.model ? value : null;
  } catch {
    return null;
  }
}

function saveModelPreference(selection) {
  window.localStorage.setItem(modelPreferenceStorageKey(), JSON.stringify(selection));
}

function modelPreferenceStorageKey() {
  const userId = authState?.session?.user?.id || "anonymous";
  return `${MODEL_PREFERENCE_KEY}.${userId}`;
}

function reloadUserModelPreference() {
  selectedLlm = readModelPreference();
  loadModelStatus();
}

function renderModelOptions(options) {
  modelSwitcherMenus.forEach((menu) => {
    menu.innerHTML = options.map((option) => `
      <button type="button" role="menuitemradio" aria-checked="${option.selected ? "true" : "false"}"
        data-model-provider="${escapeAttribute(option.provider)}" data-model-name="${escapeAttribute(option.model)}"
      >
        <span class="model-option-check">${option.selected ? "✓" : ""}</span>
        <span>${escapeHtml(option.label)}</span>
      </button>
    `).join("");
  });
}

modelSwitcherMenus.forEach((menu) => menu.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-model-provider]");
  if (!button || button.disabled) return;
  selectedLlm = {provider: button.dataset.modelProvider, model: button.dataset.modelName};
  saveModelPreference(selectedLlm);
  modelSwitchers.forEach((switcher) => { switcher.open = false; });
  await loadModelStatus();
}));

function appendSelectedLlm(formData) {
  if (!selectedLlm) return;
  formData.append("llm_provider", selectedLlm.provider);
  formData.append("llm_model", selectedLlm.model);
}

function setModelStatusNames(name) {
  modelStatusNames.forEach((element) => {
    element.textContent = name;
  });
}

function setModelConnectionState(label, offline) {
  modelStatusConnections.forEach((element) => {
    element.textContent = label;
    element.classList.remove("is-loading", "is-offline");
    if (offline) element.classList.add("is-offline");
  });
}

function formatModelName(model, provider) {
  if (provider === "ollama") {
    return model.replace(/^qwen/i, "Qwen").replace(/:(\d+)b$/i, ":$1B");
  }
  if (provider === "gemini") {
    return model.replace(/^gemini/i, "Gemini");
  }
  return model;
}

const SUPPORTED_EXTENSIONS = new Set([".pdf", ".md", ".markdown", ".txt", ".docx"]);
const SUPPORTED_MIME_TYPES = new Set([
  "application/pdf",
  "text/markdown",
  "text/plain",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]);

function isSupportedFile(file) {
  if (!file) return false;
  if (SUPPORTED_MIME_TYPES.has(file.type)) return true;
  const name = file.name.toLowerCase();
  return Array.from(SUPPORTED_EXTENSIONS).some((ext) => name.endsWith(ext));
}

function isPdfFile(file) {
  return isSupportedFile(file);
}

function fileExtension(filename) {
  const dot = String(filename || "").lastIndexOf(".");
  return dot >= 0 ? String(filename).slice(dot).toLowerCase() : "";
}

function uploadDisplayName(file) {
  return file.relativePath || file.webkitRelativePath || file.name;
}

function selectedUploadFiles() {
  return uploadFiles.filter(isSupportedFile);
}

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function uploadFileKey(file) {
  return [
    uploadDisplayName(file),
    file.size,
    file.lastModified,
  ].join("|");
}

function setUploadFiles(files, { append = false } = {}) {
  const nextFiles = append ? [...uploadFiles] : [];
  const seen = new Set(nextFiles.map(uploadFileKey));

  for (const file of files.filter(isSupportedFile)) {
    const key = uploadFileKey(file);
    if (seen.has(key)) continue;
    nextFiles.push(file);
    seen.add(key);
    if (nextFiles.length >= MAX_UPLOAD_FILES) break;
  }

  uploadFiles = nextFiles;
  pdfFileInput.value = "";
  renderUploadState();
}

function removeUploadFile(index) {
  uploadFiles = uploadFiles.filter((_, itemIndex) => itemIndex !== index);
  renderUploadState();
}

function clearUploadFiles() {
  uploadFiles = [];
  pdfFileInput.value = "";
  renderUploadState();
}

function renderUploadState() {
  const files = selectedUploadFiles();
  const hasFiles = files.length > 0;
  uploadEmptyState.hidden = hasFiles;
  uploadSelectedState.hidden = !hasFiles;
  analyzeButton.disabled = document.body.classList.contains("is-busy") || !hasFiles;

  if (!hasFiles) {
    uploadFileList.innerHTML = "";
    analyzeButton.textContent = "Analyze Paper";
    return;
  }

  fileLabel.textContent = `${files.length} file${files.length === 1 ? "" : "s"} ready`;
  analyzeButton.textContent = `Analyze ${files.length} file${files.length === 1 ? "" : "s"}`;
  uploadFileList.innerHTML = files.map((file, index) => `
    <div class="upload-file-row">
      <span class="upload-file-name">${escapeHtml(uploadDisplayName(file))}</span>
      <span class="upload-file-size">${escapeHtml(formatFileSize(file.size))}</span>
      <button class="upload-remove-button" type="button" data-remove-upload="${index}" aria-label="Remove ${escapeHtml(uploadDisplayName(file))}">×</button>
    </div>
  `).join("");
}

renderUploadState();

async function analyzeUploadFile(file, summaryMode) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("summary_mode", summaryMode);
  appendSelectedLlm(formData);

  const response = await analysisFetch("/analyze-pdf", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Analysis failed with status ${response.status}`);
  }

  return response.json();
}

async function analyzeProjectUploadFiles(files, summaryMode, projectId = "") {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file, uploadDisplayName(file)));
  formData.append("summary_mode", summaryMode);
  appendSelectedLlm(formData);

  const endpoint = projectId
    ? `/projects/${encodeURIComponent(projectId)}/papers`
    : "/projects/analyze-pdfs";
  const response = await analysisFetch(endpoint, {
    method: "POST",
    body: formData,
  });
  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.detail || `Analysis failed with status ${response.status}`);
  }

  return payload;
}

function renderUploadBatchStatus(successes, failures) {
  if (successes.length <= 1 && !failures.length) return;

  const notice = document.createElement("div");
  notice.className = failures.length ? "upload-batch-status warning" : "upload-batch-status";
  notice.textContent = failures.length
    ? `Analyzed ${successes.length} file${successes.length === 1 ? "" : "s"}; ${failures.length} failed. Showing the latest completed analysis.`
    : `Analyzed ${successes.length} files. Showing the latest completed analysis.`;
  resultPanel.prepend(notice);
}

function readFileEntry(entry, path = "") {
  return new Promise((resolve) => {
    entry.file((file) => {
      file.relativePath = `${path}${file.name}`;
      resolve(file);
    }, () => resolve(null));
  });
}

function readDirectoryEntry(entry, path = "") {
  const reader = entry.createReader();
  const entries = [];

  return new Promise((resolve) => {
    const readBatch = () => {
      reader.readEntries(async (batch) => {
        if (!batch.length) {
          const files = await Promise.all(entries.map((child) => readDroppedEntry(child, `${path}${entry.name}/`)));
          resolve(files.flat().filter(Boolean));
          return;
        }
        entries.push(...batch);
        readBatch();
      }, () => resolve([]));
    };

    readBatch();
  });
}

function readDroppedEntry(entry, path = "") {
  if (!entry) return Promise.resolve([]);
  if (entry.isFile) return readFileEntry(entry, path).then((file) => file ? [file] : []);
  if (entry.isDirectory) return readDirectoryEntry(entry, path);
  return Promise.resolve([]);
}

async function droppedFilesFromEvent(event) {
  const items = Array.from(event.dataTransfer?.items || []);
  const entries = items
    .map((item) => item.webkitGetAsEntry?.())
    .filter(Boolean);

  if (entries.length) {
    const files = await Promise.all(entries.map((entry) => readDroppedEntry(entry)));
    return files.flat().filter(isSupportedFile);
  }

  return Array.from(event.dataTransfer?.files || []).filter(isSupportedFile);
}

pdfFileInput.addEventListener("change", () => {
  setUploadFiles(Array.from(pdfFileInput.files || []), { append: uploadFiles.length > 0 });
});

folderButton.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  pdfFileInput.click();
});

addFilesButton.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  pdfFileInput.click();
});

clearUploadButton.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  clearUploadFiles();
});

uploadFileList.addEventListener("click", (event) => {
  const removeButton = event.target.closest("[data-remove-upload]");
  if (!removeButton) return;
  event.preventDefault();
  event.stopPropagation();
  removeUploadFile(Number(removeButton.dataset.removeUpload));
});

uploadDropZone.addEventListener("click", (event) => {
  if (event.target.closest("button")) return;
  pdfFileInput.click();
});

uploadDropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadDropZone.classList.add("is-dragging");
});

uploadDropZone.addEventListener("dragleave", (event) => {
  if (!uploadDropZone.contains(event.relatedTarget)) {
    uploadDropZone.classList.remove("is-dragging");
  }
});

uploadDropZone.addEventListener("drop", async (event) => {
  event.preventDefault();
  uploadDropZone.classList.remove("is-dragging");
  if (document.body.classList.contains("is-busy")) return;

  const files = await droppedFilesFromEvent(event);
  setUploadFiles(files, { append: uploadFiles.length > 0 });
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const files = selectedUploadFiles();
  if (!files.length) {
    renderError("Choose a file before starting analysis.");
    return;
  }

  const summaryMode = new FormData(uploadForm).get("summaryMode") || "standard";
  const successes = [];
  const failures = [];

  try {
    if (!usesAnonymousHistory()) {
      setBusy(true, files.length > 1 ? `Analyzing 1 of ${files.length} papers...` : "Analyzing paper...");
      const payload = await analyzeProjectUploadFiles(files, summaryMode, pendingAddProjectId);
      const project = payload.project;
      projectLocalFiles = new Map();
      (payload.completed || []).forEach((paper) => {
        if (paper.analysis_id) {
          const matchingFile = files.find((file) => uploadDisplayName(file) === paper.filename || file.name === paper.filename);
          if (matchingFile) projectLocalFiles.set(paper.analysis_id, matchingFile);
        }
      });
      pendingAddProjectId = "";
      clearUploadFiles();
      if (project) {
        renderProject(project, project.active_analysis_id);
      }
      await loadHistory();
      return;
    }

    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      const displayName = uploadDisplayName(file);
      const progress = files.length > 1 ? `${index + 1} of ${files.length}: ` : "";
      setBusy(true, `Analyzing ${progress}${displayName}`);
      try {
        const result = await analyzeUploadFile(file, summaryMode);
        result.analysis_id = result.analysis_id || `local-${window.crypto?.randomUUID?.() || `${Date.now()}-${index}`}`;
        successes.push({ file, result, displayName });
      } catch (error) {
        if (isAnalysisCancellation(error)) throw error;
        failures.push({ file, displayName, message: error.message || "Analysis failed." });
      }
    }

    if (!successes.length) {
      throw new Error(failures[0]?.message || "Analysis failed.");
    }

    await Promise.allSettled(
      successes.map((item) => saveAnonymousPdfFile(item.result.analysis_id, item.file)),
    );
    const latestSuccess = successes[successes.length - 1];
    const project = pendingAddProjectId
      ? addAnonymousProjectPapers(pendingAddProjectId, successes, failures)
      : saveAnonymousProject(successes, failures);
    successes.forEach((item) => {
      projectLocalFiles.set(item.result.analysis_id, item.file);
    });
    pendingAddProjectId = "";
    clearUploadFiles();
    renderProject(project, latestSuccess.result.analysis_id);
    renderUploadBatchStatus(successes, failures);
    await loadHistory();
  } catch (error) {
    if (!isAnalysisCancellation(error)) {
      renderError(error.message || "Analysis failed.");
    }
  } finally {
    setBusy(false);
  }
});

refreshHistoryButton.addEventListener("click", loadHistory);
authForm.addEventListener("submit", submitAuthForm);
showSignInButton.addEventListener("click", () => openAuthForm("sign_in"));
showSignUpButton.addEventListener("click", () => openAuthForm("sign_up"));
authCancelButton.addEventListener("click", closeAuthForm);
authModalCloseButton.addEventListener("click", closeAuthForm);
authModalOverlay.addEventListener("click", closeAuthForm);
authModeToggleButton.addEventListener("click", toggleAuthMode);
authAvatarButton.addEventListener("click", toggleAccountMenu);
authSignOutButton.addEventListener("click", signOut);
openHistoryButton.addEventListener("click", openHistoryDrawer);
closeHistoryButton.addEventListener("click", closeHistoryDrawer);
historyOverlay.addEventListener("click", closeHistoryDrawer);
brandHomeButton.addEventListener("click", returnHome);
brandHomeButton.addEventListener("keydown", handleBrandHomeKeydown);
resultPanel.addEventListener("click", handleResultPanelClick);
resultPanel.addEventListener("click", handleEvidenceNavigation);
resultPanel.addEventListener("click", handleCopyClick);
resultPanel.addEventListener("change", handleResultPanelChange);
sourceTabs.forEach((tab) => tab.addEventListener("click", () => switchSourceTab(tab.dataset.sourceTab)));
arxivSearchForm.addEventListener("submit", searchArxiv);
arxivDirectForm.addEventListener("submit", analyzeDirectArxiv);
arxivResults.addEventListener("click", handleArxivResultClick);
arxivPrevButton.addEventListener("click", () => changeArxivPage(-1));
arxivNextButton.addEventListener("click", () => changeArxivPage(1));
popularSearches.addEventListener("click", handlePopularSearchClick);
historySearchInput.addEventListener("input", renderFilteredHistory);
historyModeFilter.addEventListener("change", renderFilteredHistory);
historyDateFilter.addEventListener("change", renderFilteredHistory);
startArxivPlaceholderRotation();
document.addEventListener("click", closeDownloadMenusOnOutsideClick);
document.addEventListener("click", closeProjectMenusOnOutsideClick);
document.addEventListener("click", closeAccountMenuOnOutsideClick);
document.addEventListener("click", closeModelSwitchersOnOutsideClick);
document.addEventListener("keydown", handleAuthModalKeydown);


async function initAuth() {
  try {
    const response = await fetch("/auth/config");
    const config = await response.json();
    authState.enabled = Boolean(config.enabled);
    authBox.hidden = !authState.enabled;
    if (!authState.enabled) {
      authState.ready = true;
      restoreProjectFromUrl();
      return;
    }
    if (!window.supabase?.createClient) {
      setAuthStatus("Auth library did not load.");
      restoreProjectFromUrl();
      return;
    }

    const publishableKey = config.publishable_key || config.anon_key;
    authState.client = window.supabase.createClient(config.url, publishableKey);
    const {data} = await authState.client.auth.getSession();
    authState.session = data?.session || null;
    authState.ready = true;
    renderAuthState();
    reloadUserModelPreference();
    restoreProjectFromUrl();
    authState.client.auth.onAuthStateChange((_event, session) => {
      authState.session = session;
      renderAuthState();
      reloadUserModelPreference();
      loadHistory();
      restoreProjectFromUrl();
    });
  } catch (error) {
    authBox.hidden = false;
    authState.ready = false;
    setAuthStatus(error.message || "Auth could not initialize.");
  }
}

function restoreProjectFromUrl() {
  const url = new URL(window.location.href);
  const projectId = url.searchParams.get("project") || "";
  const analysisId = url.searchParams.get("paper") || "";
  if (!projectId) return false;

  const localProject = getAnonymousProject(projectId);
  if (localProject) {
    renderProject(localProject, analysisId || localProject.active_analysis_id);
    return true;
  }

  if (!authState.session?.access_token) {
    renderError("Sign in to reopen this saved project.");
    return true;
  }

  openProject(projectId, analysisId);
  return true;
}

function isSignedIn() {
  return !authState.enabled || Boolean(authState.session?.access_token);
}

function requireAuthMessage() {
  if (isSignedIn()) return "";
  return "Sign in before saving or opening analyses.";
}

function usesAnonymousHistory() {
  return authState.enabled && !authState.session?.access_token;
}

function readAnonymousRecords() {
  try {
    const records = JSON.parse(window.localStorage.getItem(ANONYMOUS_HISTORY_KEY) || "[]");
    return Array.isArray(records) ? records.filter((record) => record && typeof record === "object") : [];
  } catch {
    return [];
  }
}

function readAnonymousProjects() {
  try {
    const projects = JSON.parse(window.localStorage.getItem(ANONYMOUS_PROJECTS_KEY) || "[]");
    return Array.isArray(projects) ? projects.filter((project) => project && typeof project === "object") : [];
  } catch {
    return [];
  }
}

function writeAnonymousProjects(projects) {
  window.localStorage.setItem(
    ANONYMOUS_PROJECTS_KEY,
    JSON.stringify(projects.slice(0, ANONYMOUS_HISTORY_LIMIT)),
  );
}

function writeAnonymousRecords(records) {
  window.localStorage.setItem(
    ANONYMOUS_HISTORY_KEY,
    JSON.stringify(records.slice(0, ANONYMOUS_HISTORY_LIMIT)),
  );
}

function openAnonymousPdfDb() {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error("Browser PDF storage is not available."));
      return;
    }

    const request = window.indexedDB.open(ANONYMOUS_PDF_DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(ANONYMOUS_PDF_STORE)) {
        db.createObjectStore(ANONYMOUS_PDF_STORE, {keyPath: "analysis_id"});
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Could not open browser PDF storage."));
  });
}

async function withAnonymousPdfStore(mode, callback) {
  const db = await openAnonymousPdfDb();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(ANONYMOUS_PDF_STORE, mode);
    const store = transaction.objectStore(ANONYMOUS_PDF_STORE);
    let callbackResult;
    transaction.oncomplete = () => {
      db.close();
      resolve(callbackResult);
    };
    transaction.onerror = () => {
      db.close();
      reject(transaction.error || new Error("Browser PDF storage failed."));
    };
    try {
      callbackResult = callback(store);
    } catch (error) {
      transaction.abort();
      reject(error);
    }
  });
}

async function saveAnonymousPdfFile(analysisId, file) {
  if (!analysisId || !file) return;
  await withAnonymousPdfStore("readwrite", (store) => {
    store.put({
      analysis_id: analysisId,
      file,
      filename: uploadDisplayName(file),
      type: file.type || "application/octet-stream",
      last_modified: file.lastModified || Date.now(),
      saved_at: new Date().toISOString(),
    });
  });
}

async function getAnonymousPdfFile(analysisId, fallbackFilename = "paper.pdf") {
  if (!analysisId) return null;
  return withAnonymousPdfStore("readonly", (store) => new Promise((resolve) => {
    const request = store.get(analysisId);
    request.onsuccess = () => {
      const record = request.result;
      if (!record?.file) {
        resolve(null);
        return;
      }
      if (record.file instanceof File) {
        resolve(record.file);
        return;
      }
      resolve(new File([record.file], record.filename || fallbackFilename, {
        type: record.type || "application/octet-stream",
        lastModified: record.last_modified || Date.now(),
      }));
    };
    request.onerror = () => resolve(null);
  }));
}

async function deleteAnonymousPdfFile(analysisId) {
  if (!analysisId) return;
  await withAnonymousPdfStore("readwrite", (store) => {
    store.delete(analysisId);
  });
}

async function deleteAnonymousProjectPdfFiles(project) {
  const papers = Array.isArray(project?.papers) ? project.papers : [];
  await Promise.allSettled(papers.map((paper) => deleteAnonymousPdfFile(paper.analysis_id)));
}

function anonymousRecordId() {
  const randomPart = window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `local-${randomPart}`;
}

function saveAnonymousAnalysis(result, filename) {
  const record = {
    local_id: anonymousRecordId(),
    filename,
    created_at: new Date().toISOString(),
    result,
  };
  writeAnonymousRecords([record, ...readAnonymousRecords()]);
  return record;
}

function saveAnonymousProject(successes, failures = []) {
  const now = new Date().toISOString();
  const projectId = anonymousRecordId();
  const papers = successes.map((item) => ({
    local: true,
    analysis_id: item.result.analysis_id,
    project_id: projectId,
    filename: item.displayName,
    paper_title: paperTitleFromResult(item.result, item.displayName),
    created_at: now,
    updated_at: now,
    status: "completed",
    record: {
      local: true,
      analysis_id: item.result.analysis_id,
      project_id: projectId,
      filename: item.displayName,
      created_at: now,
      updated_at: now,
      status: "completed",
      result: item.result,
    },
  }));
  const failedPapers = failures.map((item, index) => ({
    local: true,
    analysis_id: `${projectId}-failed-${index}`,
    project_id: projectId,
    filename: item.displayName,
    paper_title: item.displayName,
    created_at: now,
    updated_at: now,
    status: "failed",
    error: item.message,
  }));
  const firstPaper = papers[0] || failedPapers[0] || {};
  const project = {
    local: true,
    project_id: projectId,
    name: "Untitled Project",
    created_at: now,
    updated_at: now,
    paper_count: papers.length + failedPapers.length,
    completed_count: papers.length,
    failed_count: failedPapers.length,
    active_analysis_id: firstPaper.analysis_id,
    papers: [...papers, ...failedPapers],
  };
  writeAnonymousProjects([project, ...readAnonymousProjects()]);
  return project;
}

function addAnonymousProjectPapers(projectId, successes, failures = []) {
  const projects = readAnonymousProjects();
  const project = projects.find((item) => item.project_id === projectId);
  if (!project) {
    return saveAnonymousProject(successes, failures);
  }

  const now = new Date().toISOString();
  const papers = Array.isArray(project.papers) ? project.papers : [];
  const newPapers = successes.map((item) => ({
    local: true,
    analysis_id: item.result.analysis_id,
    project_id: projectId,
    filename: item.displayName,
    paper_title: paperTitleFromResult(item.result, item.displayName),
    created_at: now,
    updated_at: now,
    status: "completed",
    record: {
      local: true,
      analysis_id: item.result.analysis_id,
      project_id: projectId,
      filename: item.displayName,
      created_at: now,
      updated_at: now,
      status: "completed",
      result: item.result,
    },
  }));
  const failedPapers = failures.map((item, index) => ({
    local: true,
    analysis_id: `${projectId}-failed-${Date.now()}-${index}`,
    project_id: projectId,
    filename: item.displayName,
    paper_title: item.displayName,
    created_at: now,
    updated_at: now,
    status: "failed",
    error: item.message,
  }));
  const updatedProject = {
    ...project,
    updated_at: now,
    papers: [...papers, ...newPapers, ...failedPapers],
  };
  updatedProject.paper_count = updatedProject.papers.length;
  updatedProject.completed_count = updatedProject.papers.filter((paper) => paper.status === "completed").length;
  updatedProject.failed_count = updatedProject.papers.filter((paper) => paper.status === "failed").length;
  writeAnonymousProjects(projects.map((item) => (
    item.project_id === projectId ? updatedProject : item
  )));
  return updatedProject;
}

function anonymousHistorySummary(record) {
  const result = record.result || {};
  const summary = result.document_summary || {};
  const source = result.source_metadata || {};
  return {
    local: true,
    local_id: record.local_id,
    filename: record.filename,
    paper_title: source.title || result.paper_title || summary.title,
    created_at: record.created_at || result.generated_at || result.submitted_at,
    summary_mode: result.summary_mode,
    processing_seconds: result.processing_seconds,
    summary: summary.summary || "",
  };
}

function anonymousProjectSummary(project) {
  const papers = Array.isArray(project.papers) ? project.papers : [];
  return {
    ...project,
    local: true,
    paper_count: papers.length,
    completed_count: papers.filter((paper) => paper.status === "completed").length,
    failed_count: papers.filter((paper) => paper.status === "failed").length,
    summary: papers.map((paper) => paper.summary || paper.record?.result?.document_summary?.summary || "").find(Boolean) || "",
  };
}

function getAnonymousRecord(localId) {
  return readAnonymousRecords().find((record) => record.local_id === localId) || null;
}

function getAnonymousProject(projectId) {
  return readAnonymousProjects().find((project) => project.project_id === projectId) || null;
}

function deleteAnonymousRecord(localId) {
  writeAnonymousRecords(readAnonymousRecords().filter((record) => record.local_id !== localId));
}

function deleteAnonymousProject(projectId) {
  writeAnonymousProjects(readAnonymousProjects().filter((project) => project.project_id !== projectId));
}

function setActiveLocalPdf(file) {
  if (activeLocalPdfUrl) {
    URL.revokeObjectURL(activeLocalPdfUrl);
  }
  activeLocalPdfUrl = file ? URL.createObjectURL(file) : "";
}

function authHeaders(headers = {}) {
  if (!authState.enabled || !authState.session?.access_token) return headers;
  return {
    ...headers,
    Authorization: `Bearer ${authState.session.access_token}`,
  };
}

function withAccessToken(url) {
  if (!authState.enabled || !authState.session?.access_token) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}access_token=${encodeURIComponent(authState.session.access_token)}`;
}

async function apiFetch(url, options = {}) {
  return fetch(url, {
    ...options,
    headers: authHeaders(options.headers || {}),
  });
}

function beginAnalysisRequest() {
  if (activeAnalysisController) return;
  activeAnalysisRequestId = window.crypto?.randomUUID?.() || `analysis-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  activeAnalysisController = new AbortController();
  analysisCancellationRequested = false;
  if (cancelAnalysisButton) {
    cancelAnalysisButton.disabled = false;
    cancelAnalysisButton.textContent = "Cancel analysis";
  }
}

function finishAnalysisRequest() {
  activeAnalysisRequestId = "";
  activeAnalysisController = null;
  if (cancelAnalysisButton) {
    cancelAnalysisButton.disabled = false;
    cancelAnalysisButton.textContent = "Cancel analysis";
  }
}

function analysisFetch(url, options = {}) {
  beginAnalysisRequest();
  return apiFetch(url, {
    ...options,
    signal: activeAnalysisController?.signal,
    headers: {
      ...(options.headers || {}),
      "X-Analysis-Request-ID": activeAnalysisRequestId,
    },
  });
}

function isAnalysisCancellation(error) {
  return analysisCancellationRequested || error?.name === "AbortError";
}

async function cancelCurrentAnalysis() {
  if (!activeAnalysisController || !activeAnalysisRequestId || analysisCancellationRequested) return;
  const requestId = activeAnalysisRequestId;
  const controller = activeAnalysisController;
  analysisCancellationRequested = true;
  if (cancelAnalysisButton) {
    cancelAnalysisButton.disabled = true;
    cancelAnalysisButton.textContent = "Cancelling...";
  }
  statusText.textContent = "Cancelling analysis...";
  const cancelRequest = apiFetch(`/analysis-requests/${encodeURIComponent(requestId)}/cancel`, {
    method: "POST",
  }).catch(() => null);
  controller.abort();
  await cancelRequest;
  setBusy(false);
}

cancelAnalysisButton?.addEventListener("click", cancelCurrentAnalysis);

function renderAuthState() {
  if (!authState.enabled) return;
  const email = authState.session?.user?.email || "";
  authEntry.hidden = Boolean(email);
  authSession.hidden = !email;
  authUserEmail.textContent = email;
  authAvatarButton.textContent = accountInitial(email);
  closeAccountMenu();
  if (email) {
    authState.mode = "";
    authModal.hidden = true;
    document.body.classList.remove("is-auth-open");
    setAuthStatus("Signed in.");
  } else if (!authState.mode) {
    authModal.hidden = true;
    document.body.classList.remove("is-auth-open");
    setAuthStatus("");
  }
}

function setAuthStatus(message) {
  authStatus.textContent = message || "";
}

function accountInitial(email) {
  const firstCharacter = String(email || "").trim().charAt(0);
  return firstCharacter ? firstCharacter.toUpperCase() : "A";
}

function openAuthForm(mode) {
  authState.mode = mode;
  closeAccountMenu();
  authModal.hidden = false;
  document.body.classList.add("is-auth-open");
  authFormTitle.textContent = mode === "sign_up" ? "Create your DeepDoc account" : "Sign in to DeepDoc";
  authSubmitButton.textContent = "Continue";
  authModeToggleButton.textContent = mode === "sign_up" ? "Sign in instead" : "Sign up instead";
  authPasswordInput.autocomplete = mode === "sign_up" ? "new-password" : "current-password";
  authEmailInput.value = "";
  authPasswordInput.value = "";
  authEmailInput.focus();
  setAuthStatus("");
}

function closeAuthForm() {
  authState.mode = "";
  authModal.hidden = true;
  document.body.classList.remove("is-auth-open");
  setAuthStatus("");
}

function toggleAuthMode() {
  openAuthForm(authState.mode === "sign_up" ? "sign_in" : "sign_up");
}

function handleAuthModalKeydown(event) {
  if (event.key !== "Escape") return;
  if (!authModal.hidden) {
    closeAuthForm();
  }
  closeAccountMenu();
}

function toggleAccountMenu(event) {
  event.stopPropagation();
  const shouldOpen = authAccountMenu.hidden;
  authAccountMenu.hidden = !shouldOpen;
  authAvatarButton.setAttribute("aria-expanded", shouldOpen ? "true" : "false");
}

function closeAccountMenu() {
  authAccountMenu.hidden = true;
  authAvatarButton.setAttribute("aria-expanded", "false");
}

function closeAccountMenuOnOutsideClick(event) {
  if (authAccountMenu.hidden || event.target.closest("#authSession")) return;
  closeAccountMenu();
}

async function submitAuthForm(event) {
  event.preventDefault();
  if (!authState.client || !authState.ready) {
    setAuthStatus("Auth is still loading. Try again in a moment.");
    return;
  }
  const email = authEmailInput.value.trim();
  const password = authPasswordInput.value;
  if (!email || !password) {
    setAuthStatus("Enter both email and password.");
    return;
  }
  setAuthControlsDisabled(true);
  setAuthStatus(authState.mode === "sign_up" ? "Creating account..." : "Signing in...");
  try {
    const {error} = authState.mode === "sign_up"
      ? await authState.client.auth.signUp({email, password})
      : await authState.client.auth.signInWithPassword({email, password});
    if (error) {
      setAuthStatus(error.message);
    } else if (authState.mode === "sign_up") {
      setAuthStatus("Check your email if confirmation is enabled.");
    } else {
      closeAuthForm();
      setAuthStatus("Signed in.");
    }
  } catch (error) {
    setAuthStatus(error.message || "Auth failed.");
  } finally {
    setAuthControlsDisabled(false);
  }
}

async function signOut() {
  if (!authState.client || !authState.ready) {
    setAuthStatus("Auth is still loading. Try again in a moment.");
    return;
  }
  setAuthControlsDisabled(true);
  closeAccountMenu();
  await authState.client.auth.signOut();
  setAuthControlsDisabled(false);
  historyItems = [];
  renderFilteredHistory();
}

function setAuthControlsDisabled(disabled) {
  authForm.querySelectorAll("input, button").forEach((control) => {
    control.disabled = disabled;
  });
  showSignInButton.disabled = disabled;
  authAvatarButton.disabled = disabled;
  authSignOutButton.disabled = disabled;
}


function openHistoryDrawer() {
  closeAccountMenu();
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

function closeDownloadMenusOnOutsideClick(event) {
  const activeMenu = event.target.closest(".download-menu");
  document.querySelectorAll(".download-menu[open]").forEach((menu) => {
    if (menu !== activeMenu || event.target.closest(".download-menu-list a")) {
      menu.removeAttribute("open");
    }
  });
}

function closeModelSwitchersOnOutsideClick(event) {
  const activeSwitcher = event.target.closest("[data-model-switcher]");
  modelSwitchers.forEach((switcher) => {
    if (switcher !== activeSwitcher) switcher.open = false;
  });
}

function closeProjectMenusOnOutsideClick(event) {
  const activeProjectMenu = event.target.closest(".project-more-menu");
  resultPanel.querySelectorAll(".project-more-menu[open]").forEach((menu) => {
    if (menu !== activeProjectMenu) {
      menu.removeAttribute("open");
    }
  });
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
  stopAnalysisProgress();
  resultPanel.className = "result-panel empty-state";
  resultPanel.dataset.analysisId = "";
  resultPanel.innerHTML = "";
  statusPanel.hidden = true;
  clearProjectUrl();
}

function switchSourceTab(source) {
  sourceTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.sourceTab === source));
  sourcePanels.forEach((panel) => panel.classList.toggle("active", panel.dataset.sourcePanel === source));
}

async function searchArxiv(event) {
  if (event) event.preventDefault();
  const query = arxivQueryInput.value.trim();
  const maxResults = Number(arxivMaxResultsInput.value || 10);
  const searchField = arxivSearchFieldInput.value || "all";
  const category = arxivCategoryInput.value || "";
  const {sortBy, sortOrder} = parseArxivSortMode(arxivSortModeInput.value);
  if (!query) {
    arxivSearchStatus.textContent = "Enter a search query.";
    arxivResults.innerHTML = "";
    resetArxivPagination();
    return;
  }

  arxivSearchState = {
    query,
    searchField,
    category,
    maxResults,
    sortBy,
    sortOrder,
    start: 0,
    page: 1,
    totalPages: 0,
    totalResults: 0,
  };
  popularSearches.hidden = true;
  await fetchArxivPage();
}

async function fetchArxivPage() {
  arxivSearchButton.disabled = true;
  arxivPrevButton.disabled = true;
  arxivNextButton.disabled = true;
  arxivSearchButton.textContent = "Searching";
  arxivSearchStatus.textContent = "Searching arXiv...";
  arxivResults.innerHTML = "";

  try {
    const params = new URLSearchParams({
      q: arxivSearchState.query,
      max_results: String(arxivSearchState.maxResults),
      start: String(arxivSearchState.start),
      search_field: arxivSearchState.searchField,
      category: arxivSearchState.category,
      sort_by: arxivSearchState.sortBy,
      sort_order: arxivSearchState.sortOrder,
    });
    const response = await fetch(`/arxiv/search?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `arXiv search failed with status ${response.status}`);
    }
    arxivItems = payload.results || [];
    arxivSearchState = {
      ...arxivSearchState,
      start: Number(payload.start || arxivSearchState.start),
      searchField: payload.search_field || arxivSearchState.searchField,
      category: payload.category || arxivSearchState.category,
      maxResults: Number(payload.max_results || arxivSearchState.maxResults),
      sortBy: payload.sort_by || arxivSearchState.sortBy,
      sortOrder: payload.sort_order || arxivSearchState.sortOrder,
      page: Number(payload.page || 1),
      totalPages: Number(payload.total_pages || 0),
      totalResults: Number(payload.total_results || 0),
    };
    renderArxivResults(arxivItems);
    renderArxivPagination();
    const filterLabel = formatArxivFilterLabel(arxivSearchState);
    arxivSearchStatus.textContent = arxivItems.length
      ? `${arxivItems.length} papers shown · ${formatNumber(arxivSearchState.totalResults)} total results · ${filterLabel}`
      : "No arXiv papers found.";
  } catch (error) {
    arxivItems = [];
    arxivResults.innerHTML = "";
    resetArxivPagination();
    const message = error.message || "arXiv search failed.";
    arxivSearchStatus.textContent = `${message} Paste an arXiv ID or PDF URL to analyze directly.`;
  } finally {
    arxivSearchButton.disabled = false;
    arxivSearchButton.textContent = "Search";
  }
}

function normalizeDirectArxivInput(value) {
  const raw = String(value || "").trim();
  if (!raw) return null;

  const idPattern = /^([0-9]{4}\.[0-9]{4,5}(v[0-9]+)?|[a-z-]+(\.[A-Z]{2})?\/[0-9]{7}(v[0-9]+)?)$/;
  if (idPattern.test(raw)) {
    return {
      arxivId: raw,
      pdfUrl: `https://arxiv.org/pdf/${raw}`,
    };
  }

  try {
    const url = new URL(raw);
    const host = url.hostname.toLowerCase();
    const parts = url.pathname.split("/").filter(Boolean);
    if (!["arxiv.org", "www.arxiv.org", "export.arxiv.org"].includes(host) || parts.length < 2) {
      return null;
    }
    if (parts[0] !== "pdf" && parts[0] !== "abs") {
      return null;
    }
    const arxivId = parts.slice(1).join("/").replace(/\.pdf$/i, "");
    if (!idPattern.test(arxivId)) {
      return null;
    }
    return {
      arxivId,
      pdfUrl: `https://arxiv.org/pdf/${arxivId}`,
    };
  } catch {
    return null;
  }
}

async function analyzeDirectArxiv(event) {
  event.preventDefault();
  const directPaper = normalizeDirectArxivInput(arxivDirectInput.value);
  if (!directPaper) {
    arxivSearchStatus.textContent = "Enter a valid arXiv ID, abs URL, or PDF URL.";
    return;
  }

  const summaryMode = arxivSummaryModeInput.value || "standard";
  const item = {
    arxiv_id: directPaper.arxivId,
    pdf_url: directPaper.pdfUrl,
    title: "",
    authors: [],
    published: "",
    updated: "",
    categories: [],
  };

  await analyzeArxivItem(item, summaryMode, arxivDirectButton);
}

async function changeArxivPage(direction) {
  if (!arxivSearchState.query) return;
  const nextStart = Math.max(0, arxivSearchState.start + direction * arxivSearchState.maxResults);
  if (nextStart === arxivSearchState.start) return;
  if (arxivSearchState.totalResults && nextStart >= arxivSearchState.totalResults) return;
  arxivSearchState.start = nextStart;
  await fetchArxivPage();
}

function renderArxivPagination() {
  if (!arxivSearchState.query || (!arxivItems.length && !arxivSearchState.totalResults)) {
    resetArxivPagination();
    return;
  }

  arxivPagination.hidden = false;
  const totalPages = arxivSearchState.totalPages || Math.max(1, arxivSearchState.page);
  arxivPageInfo.textContent = `Page ${arxivSearchState.page}${totalPages ? ` of ${totalPages}` : ""}`;
  arxivPrevButton.disabled = arxivSearchState.start <= 0;
  const nextStart = arxivSearchState.start + arxivSearchState.maxResults;
  arxivNextButton.disabled = arxivSearchState.totalResults
    ? nextStart >= arxivSearchState.totalResults
    : arxivItems.length < arxivSearchState.maxResults;
}

function resetArxivPagination() {
  arxivPagination.hidden = true;
  arxivPageInfo.textContent = "Page 1";
  arxivPrevButton.disabled = true;
  arxivNextButton.disabled = true;
}

function handlePopularSearchClick(event) {
  const button = event.target.closest("[data-popular-query]");
  if (!button) return;
  arxivQueryInput.value = button.dataset.popularQuery || "";
  searchArxiv();
}

function startArxivPlaceholderRotation() {
  let index = 0;
  arxivQueryInput.placeholder = arxivPlaceholders[index];
  window.setInterval(() => {
    if (document.activeElement === arxivQueryInput || arxivQueryInput.value.trim()) return;
    index = (index + 1) % arxivPlaceholders.length;
    arxivQueryInput.placeholder = arxivPlaceholders[index];
  }, 3600);
}

function formatArxivFilterLabel(state) {
  const fieldLabels = {
    all: "all fields",
    title: "title",
    author: "author",
    abstract: "abstract",
  };
  const sortLabel = formatArxivSortLabel(state.sortBy, state.sortOrder);
  const category = state.category ? ` · ${state.category}` : "";
  return `${fieldLabels[state.searchField] || "all fields"}${category} · ${sortLabel}`;
}

function parseArxivSortMode(value) {
  const sortModes = {
    "relevance-desc": {sortBy: "relevance", sortOrder: "descending"},
    "submittedDate-desc": {sortBy: "submittedDate", sortOrder: "descending"},
    "submittedDate-asc": {sortBy: "submittedDate", sortOrder: "ascending"},
    "lastUpdatedDate-desc": {sortBy: "lastUpdatedDate", sortOrder: "descending"},
    "lastUpdatedDate-asc": {sortBy: "lastUpdatedDate", sortOrder: "ascending"},
  };
  return sortModes[value] || sortModes["relevance-desc"];
}

function formatArxivSortLabel(sortBy, sortOrder) {
  if (sortBy === "submittedDate") {
    return sortOrder === "ascending" ? "oldest submitted" : "newest submitted";
  }
  if (sortBy === "lastUpdatedDate") {
    return sortOrder === "ascending" ? "oldest updated" : "recently updated";
  }
  return "best match";
}

function renderArxivResults(items) {
  arxivResults.innerHTML = "";
  if (!items.length) return;

  items.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "arxiv-card";
    const authors = Array.isArray(item.authors) ? item.authors.join(", ") : "";
    const categories = Array.isArray(item.categories) ? item.categories.slice(0, 3).join(" · ") : "";
    const metadata = [categories, item.published || "date unavailable"].filter(Boolean).join(" • ");
    card.innerHTML = `
      <div>
        <p class="card-meta">${escapeHtml(metadata)}</p>
        <h3>${escapeHtml(item.title || "Untitled arXiv paper")}</h3>
        <p class="result-meta arxiv-authors">${escapeHtml(trimText(authors, 220))}</p>
        <p class="arxiv-abstract">${escapeHtml(item.summary || "")}</p>
      </div>
      <div class="arxiv-card-actions">
        <button class="arxiv-analyze" type="button" data-arxiv-index="${index}">Analyze</button>
        <a href="${safeArxivUrl(item.pdf_url || "#")}" target="_blank" rel="noreferrer">PDF</a>
        <a href="${safeArxivUrl(item.abs_url || "#")}" target="_blank" rel="noreferrer">arXiv</a>
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

  await analyzeArxivItem(item, arxivSummaryModeInput.value || "standard", button);
}

async function analyzeArxivItem(item, summaryMode, button = null) {
  setBusy(true, `Downloading ${item.arxiv_id || "arXiv paper"} and analyzing...`);
  if (button) {
    button.disabled = true;
    button.textContent = "Analyzing";
  }

  try {
    const response = await analysisFetch("/arxiv/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        arxiv_id: item.arxiv_id,
        pdf_url: item.pdf_url,
        summary_mode: summaryMode,
        title: item.title || "",
        authors: Array.isArray(item.authors) ? item.authors : [],
        published: item.published || "",
        updated: item.updated || "",
        categories: Array.isArray(item.categories) ? item.categories : [],
        llm_provider: selectedLlm?.provider || null,
        llm_model: selectedLlm?.model || null,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `arXiv analysis failed with status ${response.status}`);
    }
    if (usesAnonymousHistory()) {
      setActiveLocalPdf(null);
      payload.analysis_id = payload.analysis_id || anonymousRecordId();
      const project = saveAnonymousProject([{
        result: payload,
        displayName: `arxiv-${item.arxiv_id}.pdf`,
      }]);
      renderProject(project, payload.analysis_id);
      await loadHistory();
      return;
    }
    if (payload.project_id) {
      const projectResponse = await apiFetch(`/projects/${encodeURIComponent(payload.project_id)}`);
      if (projectResponse.ok) {
        const project = await projectResponse.json();
        renderProject(project, payload.analysis_id);
      } else {
        renderResult(payload, `arxiv-${item.arxiv_id}.pdf`);
      }
    } else {
      renderResult(payload, `arxiv-${item.arxiv_id}.pdf`);
    }
    await loadHistory();
  } catch (error) {
    if (!isAnalysisCancellation(error)) {
      renderError(error.message || "arXiv analysis failed.");
    }
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "Analyze";
    }
    setBusy(false);
  }
}


function handleResultPanelClick(event) {
  if (handleProjectActionClick(event)) return;

  const reanalyzeButton = event.target.closest('[data-field="reanalyze"]');
  const scriptButton = event.target.closest('[data-field="generateVideoScript"]');
  const videoButton = event.target.closest('[data-field="generateVideo"]');
  const overviewToggle = event.target.closest('[data-field="toggleOverview"]');
  const listToggle = event.target.closest("[data-list-toggle]");
  const referenceToggle = event.target.closest("[data-reference-toggle]");

  if (overviewToggle) {
    toggleOverview(overviewToggle);
    return;
  }

  if (listToggle) {
    toggleList(listToggle);
    return;
  }

  if (referenceToggle) {
    toggleReference(referenceToggle);
    return;
  }

  if (!reanalyzeButton && !scriptButton && !videoButton) return;

  const analysisId = resultPanel.dataset.analysisId;
  if (!analysisId) return;

  if (reanalyzeButton) {
    confirmAndReanalyzeAnalysis(analysisId, reanalyzeButton, "standard");
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

function handleProjectActionClick(event) {
  const projectNameButton = event.target.closest("[data-project-name-start]");
  const paperButton = event.target.closest("[data-project-paper]");
  const prevButton = event.target.closest("[data-project-prev]");
  const nextButton = event.target.closest("[data-project-next]");
  const addButton = event.target.closest("[data-project-add-papers]");
  const renameButton = event.target.closest("[data-project-rename]");
  const reanalyzePaperButton = event.target.closest("[data-paper-reanalyze]");
  const reanalyzeModeButton = event.target.closest("[data-paper-reanalyze-mode]");
  const deletePaperButton = event.target.closest("[data-paper-delete]");
  const deleteButton = event.target.closest("[data-project-delete]");

  if (projectNameButton) {
    startProjectNameEdit();
    return true;
  }

  if (paperButton) {
    renderProject(currentProject, paperButton.dataset.projectPaper);
    return true;
  }

  if ((prevButton || nextButton) && currentProject) {
    const papers = currentProject.papers || [];
    const currentIndex = papers.findIndex((paper) => paper.analysis_id === resultPanel.dataset.analysisId);
    const nextIndex = currentIndex + (nextButton ? 1 : -1);
    if (papers[nextIndex]) renderProject(currentProject, papers[nextIndex].analysis_id);
    return true;
  }

  if (addButton) {
    closeProjectMenus();
    pendingAddProjectId = addButton.dataset.projectAddPapers || "";
    switchSourceTab("upload");
    pdfFileInput.click();
    returnHome();
    return true;
  }

  if (renameButton) {
    closeProjectMenus();
    startProjectNameEdit();
    return true;
  }

  if (reanalyzePaperButton || reanalyzeModeButton) {
    closeProjectMenus();
    const trigger = reanalyzeModeButton || reanalyzePaperButton;
    const analysisId = trigger.dataset.paperReanalyzeMode || trigger.dataset.paperReanalyze || resultPanel.dataset.analysisId;
    const summaryMode = trigger.dataset.summaryMode || "standard";
    confirmAndReanalyzeAnalysis(analysisId, trigger, summaryMode);
    return true;
  }

  if (deletePaperButton) {
    closeProjectMenus();
    deleteCurrentPaper(deletePaperButton.dataset.paperDelete);
    return true;
  }

  if (deleteButton) {
    closeProjectMenus();
    deleteCurrentProject(deleteButton.dataset.projectDelete);
    return true;
  }

  return false;
}

function closeProjectMenus() {
  resultPanel.querySelectorAll(".project-more-menu[open]").forEach((menu) => {
    menu.removeAttribute("open");
  });
}

async function deleteCurrentPaper(analysisId) {
  if (!analysisId || !currentProject) return;
  const paper = (currentProject.papers || []).find((item) => item.analysis_id === analysisId);
  const displayName = paperTitleFromProjectPaper(paper) || "this paper";
  const confirmed = window.confirm(`Delete "${displayName}" from this project?`);
  if (!confirmed) return;

  if (currentProject.local) {
    currentProject.papers = (currentProject.papers || []).filter((item) => item.analysis_id !== analysisId);
    deleteAnonymousPdfFile(analysisId).catch(() => {});
    if (!currentProject.papers.length) {
      deleteAnonymousProject(currentProject.project_id);
      returnHome();
      await loadHistory();
      return;
    }
    currentProject.paper_count = currentProject.papers.length;
    currentProject.completed_count = currentProject.papers.filter((item) => item.status === "completed").length;
    currentProject.failed_count = currentProject.papers.filter((item) => item.status === "failed").length;
    writeAnonymousProjects(readAnonymousProjects().map((project) => (
      project.project_id === currentProject.project_id ? currentProject : project
    )));
    renderProject(currentProject, currentProject.papers[0].analysis_id);
    await loadHistory();
    return;
  }

  setBusy(true, "Deleting paper...");
  try {
    const response = await apiFetch(`/analyses/${encodeURIComponent(analysisId)}`, {method: "DELETE"});
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || "Could not delete paper.");
    const projectId = currentProject.project_id;
    await loadHistory();
    const refreshed = await apiFetch(`/projects/${encodeURIComponent(projectId)}`);
    if (!refreshed.ok) {
      currentProject = null;
      returnHome();
      return;
    }
    const project = await refreshed.json();
    renderProject(project, project.active_analysis_id);
  } catch (error) {
    renderError(error.message || "Could not delete paper.");
  } finally {
    setBusy(false);
  }
}

resultPanel.addEventListener("keydown", (event) => {
  const nameInput = event.target.closest("[data-project-name]");
  if (!nameInput) return;
  if (event.key === "Enter") {
    event.preventDefault();
    saveProjectNameEdit(nameInput);
  } else if (event.key === "Escape") {
    event.preventDefault();
    cancelProjectNameEdit(nameInput);
  }
});

resultPanel.addEventListener("focusout", (event) => {
  const nameInput = event.target.closest("[data-project-name]");
  if (!nameInput || nameInput.hidden || nameInput.dataset.cancelled === "true") return;
  saveProjectNameEdit(nameInput);
});

function startProjectNameEdit() {
  if (!currentProject || currentProject.legacy) return;
  const input = resultPanel.querySelector("[data-project-name]");
  const button = resultPanel.querySelector("[data-project-name-start]");
  if (!input || !button) return;
  input.value = currentProject.name || "Untitled Project";
  input.dataset.previousName = input.value;
  input.dataset.cancelled = "false";
  button.hidden = true;
  input.hidden = false;
  input.focus();
  input.select();
}

function cancelProjectNameEdit(input) {
  const button = resultPanel.querySelector("[data-project-name-start]");
  input.dataset.cancelled = "true";
  input.value = input.dataset.previousName || currentProject?.name || "Untitled Project";
  input.hidden = true;
  if (button) button.hidden = false;
}

async function saveProjectNameEdit(input) {
  if (!currentProject || currentProject.legacy || input.dataset.saving === "true") return;
  const previousName = input.dataset.previousName || currentProject.name || "Untitled Project";
  const name = input.value.trim();
  const button = resultPanel.querySelector("[data-project-name-start]");

  if (!name) {
    input.value = previousName;
    setAuthStatus("Project name cannot be empty.");
    return;
  }

  if (name === previousName) {
    input.hidden = true;
    if (button) button.hidden = false;
    return;
  }

  input.dataset.saving = "true";
  input.disabled = true;
  try {
    if (currentProject.local) {
      currentProject.name = name;
      currentProject.updated_at = new Date().toISOString();
      writeAnonymousProjects(readAnonymousProjects().map((project) => (
        project.project_id === currentProject.project_id ? currentProject : project
      )));
    } else {
      const response = await apiFetch(`/projects/${encodeURIComponent(currentProject.project_id)}`, {
        method: "PATCH",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({name}),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Could not rename project.");
      currentProject.name = payload.name || name;
    }
    if (button) {
      button.querySelector("span").textContent = currentProject.name;
      button.title = currentProject.name;
      button.hidden = false;
    }
    input.hidden = true;
    await loadHistory();
  } catch (error) {
    input.value = previousName;
    if (button) button.hidden = false;
    input.hidden = true;
    setAuthStatus(error.message || "Could not rename project.");
  } finally {
    input.disabled = false;
    delete input.dataset.saving;
  }
}

async function deleteCurrentProject(projectId) {
  if (!projectId || !currentProject) return;
  const displayName = currentProject.name || "this project";
  const confirmed = window.confirm(`Delete "${displayName}" and all papers in this project?`);
  if (!confirmed) return;

  if (currentProject.local) {
    deleteAnonymousProjectPdfFiles(currentProject).catch(() => {});
    deleteAnonymousProject(projectId);
    returnHome();
    await loadHistory();
    return;
  }

  setBusy(true, "Deleting project...");
  try {
    const response = await apiFetch(`/projects/${encodeURIComponent(projectId)}`, {method: "DELETE"});
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || "Could not delete project.");
    currentProject = null;
    returnHome();
    await loadHistory();
  } catch (error) {
    renderError(error.message || "Could not delete project.");
  } finally {
    setBusy(false);
  }
}

function toggleOverview(button) {
  const card = button.closest(".summary-block");
  const expanded = card?.classList.toggle("is-expanded");
  button.textContent = expanded ? "Collapse" : "Read more";
}

function toggleList(button) {
  const card = button.closest("article");
  const expanded = card?.classList.toggle("is-expanded");
  const list = card?.querySelector("ol, ul");
  const limit = 5;
  if (list) {
    [...list.children].forEach((item, index) => {
      item.hidden = !expanded && index >= limit;
    });
  }
  button.textContent = expanded ? "View less" : `View all ${button.dataset.hiddenCount || ""}`.trim();
}

async function reanalyzeExistingAnalysis(analysisId, button, summaryMode = "standard") {
  const authMessage = requireAuthMessage();
  if (authMessage) {
    renderError(authMessage);
    return;
  }
  const originalButtonText = button.textContent;
  button.disabled = true;
  button.textContent = "Reanalyzing";
  setBusy(true, `Creating a new ${formatMode(summaryMode).toLowerCase()} version...`);

  try {
    const params = new URLSearchParams({summary_mode: summaryMode || "standard"});
    const response = await analysisFetch(`/analyses/${encodeURIComponent(analysisId)}/reanalyze?${params.toString()}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        summary_mode: summaryMode || "standard",
        llm_provider: selectedLlm?.provider || null,
        llm_model: selectedLlm?.model || null,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Reanalysis failed with status ${response.status}`);
    }
    if (currentProject?.project_id && !currentProject.local) {
      const projectResponse = await apiFetch(`/projects/${encodeURIComponent(currentProject.project_id)}`);
      if (projectResponse.ok) {
        const project = await projectResponse.json();
        renderProject(project, payload.analysis_id);
      } else {
        renderResult(payload, "Reanalyzed paper");
      }
    } else {
      renderResult(payload, "Reanalyzed paper");
    }
    await loadHistory();
  } catch (error) {
    if (!isAnalysisCancellation(error)) {
      renderError(error.message || "Reanalysis failed.");
    }
  } finally {
    button.disabled = false;
    button.textContent = originalButtonText || "Reanalyze";
    setBusy(false);
  }
}

function confirmAndReanalyzeAnalysis(analysisId, button, summaryMode = "standard") {
  if (!analysisId) return;
  const authMessage = requireAuthMessage();
  if (authMessage) {
    renderError(authMessage);
    return;
  }
  const nextMode = normalizeSummaryMode(summaryMode);
  const currentMode = normalizeSummaryMode(currentAnalysisSummaryMode());
  const confirmed = window.confirm(
    `Reanalyze this paper in ${formatAnalysisModeName(nextMode)} mode? ` +
    `This will create a new analysis. The current ${formatAnalysisModeName(currentMode)} analysis will remain available in history.`,
  );
  if (!confirmed) return;
  reanalyzeExistingAnalysis(analysisId, button, nextMode);
}

function handleResultPanelChange(event) {
  if (!event.target.matches('[data-field="slideCount"]')) return;
  updateVideoArtifactAvailability(resultPanel);
}

function setBusy(isBusy, message = "") {
  document.body.classList.toggle("is-busy", isBusy);
  pdfFileInput.disabled = isBusy;
  folderButton.disabled = isBusy;
  addFilesButton.disabled = isBusy;
  clearUploadButton.disabled = isBusy;
  analyzeButton.disabled = isBusy;
  analyzeButton.classList.toggle("is-loading", isBusy);
  analyzeButton.setAttribute("aria-busy", String(isBusy));
  if (!isBusy) {
    stopAnalysisProgress();
    finishAnalysisRequest();
    renderUploadState();
  } else if (isAnalysisOperation(message)) {
    beginAnalysisRequest();
    startAnalysisProgress();
  } else {
    stopAnalysisProgress();
  }
  statusText.textContent = message;
  statusPanel.hidden = !isBusy;
}

function isAnalysisOperation(message) {
  return /analyz|creating a new .*version/i.test(String(message || ""));
}

function startAnalysisProgress() {
  stopAnalysisProgress();
  analysisProgressStartedAt = Date.now();
  if (analysisProgress) analysisProgress.hidden = false;
  updateAnalysisProgress();
  analysisProgressTimer = window.setInterval(updateAnalysisProgress, 500);
}

function stopAnalysisProgress() {
  if (analysisProgressTimer !== null) {
    window.clearInterval(analysisProgressTimer);
    analysisProgressTimer = null;
  }
  analysisProgressStartedAt = 0;
  if (analysisProgress) analysisProgress.hidden = true;
  if (analysisProgressBar) analysisProgressBar.style.width = "0%";
  analysisProgressTrack?.setAttribute("aria-valuenow", "0");
}

function updateAnalysisProgress() {
  if (!analysisProgressStartedAt) return;
  const elapsed = Math.max(0, (Date.now() - analysisProgressStartedAt) / 1000);
  let progress = 0;
  let stage = "Preparing document";
  if (elapsed < 5) {
    progress = 4 + (elapsed / 5) * 8;
  } else if (elapsed < 65) {
    progress = 12 + ((elapsed - 5) / 60) * 35;
    stage = "Extracting and verifying facts";
  } else if (elapsed < 150) {
    progress = 47 + ((elapsed - 65) / 85) * 34;
    stage = "Generating grounded analysis";
  } else {
    progress = 81 + 15 * (1 - Math.exp(-(elapsed - 150) / 120));
    stage = "Validating evidence and citations";
  }
  const roundedProgress = Math.min(96, Math.round(progress));
  const elapsedSeconds = Math.floor(elapsed);
  const minutes = Math.floor(elapsedSeconds / 60);
  const seconds = String(elapsedSeconds % 60).padStart(2, "0");
  if (analysisProgressBar) analysisProgressBar.style.width = `${roundedProgress}%`;
  if (analysisProgressStage) analysisProgressStage.textContent = stage;
  if (analysisProgressTime) {
    analysisProgressTime.textContent = `Estimated ${roundedProgress}% · ${minutes}:${seconds}`;
  }
  analysisProgressTrack?.setAttribute("aria-valuenow", String(roundedProgress));
}

function renderError(message) {
  setActiveLocalPdf(null);
  document.body.classList.remove("has-result");
  document.body.classList.remove("source-collapsed");
  document.body.classList.remove("is-busy");
  stopAnalysisProgress();
  resultPanel.className = "result-panel error";
  resultPanel.innerHTML = `
    <div>
      <p class="eyebrow">Error</p>
      <h2>${escapeHtml(message)}</h2>
    </div>
  `;
}

function paperTitleFromResult(result, fallbackFilename = "Untitled Paper") {
  const summary = result?.document_summary || {};
  const source = result?.source_metadata || {};
  return source.title || result?.paper_title || summary.title || fallbackFilename || "Untitled Paper";
}

function paperTitleFromProjectPaper(paper) {
  const result = paper?.record?.result || {};
  return paper.paper_title || paperTitleFromResult(result, paper.filename || "Untitled Paper");
}

function paperResultFromProjectPaper(paper) {
  const record = paper?.record || {};
  const result = record.result || paper.result || {};
  if (paper.analysis_id && !result.analysis_id) result.analysis_id = paper.analysis_id;
  return result;
}

function singlePaperProjectFromResult(result, fallbackFilename = "Analysis Result") {
  const analysisId = result?.analysis_id || "";
  const now = result?.generated_at || result?.submitted_at || new Date().toISOString();
  const paper = {
    legacy: true,
    analysis_id: analysisId,
    project_id: result?.project_id || `legacy-${analysisId || anonymousRecordId()}`,
    filename: fallbackFilename,
    paper_title: paperTitleFromResult(result, fallbackFilename),
    created_at: now,
    updated_at: now,
    status: "completed",
    record: {
      legacy: true,
      analysis_id: analysisId,
      project_id: result?.project_id || "",
      filename: fallbackFilename,
      created_at: now,
      updated_at: now,
      status: "completed",
      result,
    },
  };
  return {
    legacy: true,
    project_id: paper.project_id,
    name: "Untitled Project",
    created_at: now,
    updated_at: now,
    paper_count: 1,
    completed_count: 1,
    failed_count: 0,
    active_analysis_id: analysisId,
    papers: [paper],
  };
}

function renderProject(project, activeAnalysisId = "") {
  const papers = Array.isArray(project?.papers) ? project.papers : [];
  const completedPapers = papers.filter((paper) => (paper.status || "completed") === "completed");
  const activePaper = (
    papers.find((paper) => paper.analysis_id === activeAnalysisId)
    || completedPapers[0]
    || papers[0]
  );

  currentProject = project;
  if (!activePaper) {
    renderError("This project has no papers.");
    return;
  }

  const localFile = projectLocalFiles.get(activePaper.analysis_id) || null;
  setActiveLocalPdf(localFile || null);

  if ((activePaper.status || "completed") !== "completed") {
    renderProjectShell(project, activePaper);
    return;
  }

  renderResult(paperResultFromProjectPaper(activePaper), activePaper.filename || "Saved Analysis", {
    project,
    activeAnalysisId: activePaper.analysis_id,
    localPdfFile: localFile,
    localPdfMissing: Boolean(project.local && !localFile),
  });
  if (project.local && !localFile) {
    getAnonymousPdfFile(activePaper.analysis_id, activePaper.filename || "paper.pdf")
      .then((storedFile) => {
        if (currentProject?.project_id !== project.project_id) return;
        if (resultPanel.dataset.analysisId !== activePaper.analysis_id) return;
        if (!storedFile) {
          const placeholder = resultPanel.querySelector('[data-field="pdfPlaceholder"]');
          if (placeholder) {
            placeholder.textContent = "Original document was not found in this browser. Re-upload the file or sign in to save future papers across sessions.";
          }
          return;
        }
        projectLocalFiles.set(activePaper.analysis_id, storedFile);
        renderProject(project, activePaper.analysis_id);
      })
      .catch(() => {});
  }
  updateProjectUrl(project.project_id, activePaper.analysis_id);
}

function renderProjectShell(project, activePaper) {
  document.body.classList.add("has-result");
  document.body.classList.add("source-collapsed");
  resultPanel.className = "result-panel";
  resultPanel.dataset.analysisId = activePaper.analysis_id || "";
  resultPanel.innerHTML = "";
  resultPanel.appendChild(createProjectNavigator(project, activePaper.analysis_id));
  const message = document.createElement("div");
  message.className = "project-empty-state";
  message.innerHTML = `
    <p class="eyebrow">${escapeHtml(activePaper.status || "Queued")}</p>
    <h2>${escapeHtml(paperTitleFromProjectPaper(activePaper))}</h2>
    <p class="result-meta">${escapeHtml(activePaper.error || "This paper is not completed yet.")}</p>
    <button class="ghost-button compact-action" type="button" data-project-add-papers="${escapeHtml(project.project_id)}">Retry / Add papers</button>
  `;
  resultPanel.appendChild(message);
  updateProjectUrl(project.project_id, activePaper.analysis_id);
}

function createProjectNavigator(project, activeAnalysisId) {
  const papers = Array.isArray(project.papers) ? project.papers : [];
  const activeIndex = Math.max(0, papers.findIndex((paper) => paper.analysis_id === activeAnalysisId));
  const activePaper = papers[activeIndex] || papers[0] || {};
  const updatedAt = project.updated_at || project.created_at || "";
  const projectName = project.name || "Untitled Project";
  const activePaperTitle = paperTitleFromProjectPaper(activePaper);
  const processingSeconds = Number(
    activePaper.record?.result?.processing_seconds ?? activePaper.result?.processing_seconds,
  );
  const analysisTime = Number.isFinite(processingSeconds) && processingSeconds >= 0
    ? `<span>Analyzed in ${escapeHtml(processingSeconds.toFixed(2))}s</span>`
    : "";
  const wrapper = document.createElement("section");
  wrapper.className = `project-nav ${papers.length <= 1 ? "single-paper" : ""}`;
  wrapper.innerHTML = `
    <div class="project-header-row">
      <div class="project-title-block">
        <button class="project-name-button" type="button" data-project-name-start title="${escapeAttribute(projectName)}" ${project.legacy ? "disabled" : ""}>
          <span>${escapeHtml(projectName)}</span>
          <span class="project-name-edit-icon" aria-hidden="true">Edit</span>
        </button>
        <input class="project-name-input" value="${escapeHtml(projectName)}" data-project-name aria-label="Project name" hidden />
        <div class="project-status-summary">
          <span>${escapeHtml(String(papers.length))} paper${papers.length === 1 ? "" : "s"}</span>
          <span>Updated ${escapeHtml(formatDate(updatedAt))}</span>
          ${analysisTime}
        </div>
      </div>
      <div class="project-active-title" title="${escapeAttribute(activePaperTitle)}">${escapeHtml(activePaperTitle)}</div>
      <div class="project-actions">
        <span class="analysis-mode-badge" data-field="analysisModeBadge">${escapeHtml(formatAnalysisModeBadge(activePaper.record?.result?.summary_mode || activePaper.summary_mode))}</span>
        <button class="primary-button compact-action" type="button" data-project-add-papers="${escapeHtml(project.project_id)}" ${project.legacy ? "disabled" : ""}>Add Papers</button>
        <details class="project-more-menu">
          <summary class="ghost-button icon-button" aria-label="More project actions">⋯</summary>
          <div class="project-more-list">
            <button type="button" data-project-rename ${project.legacy ? "disabled" : ""}>Rename Project</button>
            <button type="button" data-project-add-papers="${escapeHtml(project.project_id)}" ${project.legacy ? "disabled" : ""}>Add Papers</button>
            <details class="workspace-reanalyze-menu">
              <summary>Reanalyze</summary>
              <div class="workspace-submenu-list">
                <button type="button" data-paper-reanalyze-mode="${escapeHtml(activePaper.analysis_id || "")}" data-summary-mode="paragraph">Quick</button>
                <button type="button" data-paper-reanalyze-mode="${escapeHtml(activePaper.analysis_id || "")}" data-summary-mode="standard">Standard</button>
                <button type="button" data-paper-reanalyze-mode="${escapeHtml(activePaper.analysis_id || "")}" data-summary-mode="one_page">Detailed</button>
              </div>
            </details>
            <details class="download-menu workspace-download-menu">
              <summary>Download</summary>
              <div class="download-menu-list">
                <p>Analysis</p>
                <a data-field="downloadMarkdown" href="#" download>Markdown</a>
                <a data-field="download" href="#" download>JSON</a>
                <p>Slides</p>
                <a data-field="downloadSlidesHtml" href="#" download hidden>HTML</a>
                <a data-field="downloadSlides" href="#" download hidden>Markdown</a>
                <a data-field="downloadScript" href="#" download hidden>JSON</a>
                <a data-field="downloadVideo" href="#" download hidden>Narrated MP4</a>
                <p>Document</p>
                <a data-field="downloadPdf" href="#" download hidden>Original PDF</a>
              </div>
            </details>
            <button type="button" disabled>Duplicate</button>
            <button type="button" disabled>Compare Papers</button>
            <button class="danger-action" type="button" data-paper-delete="${escapeHtml(activePaper.analysis_id || "")}">Delete Paper</button>
            <button class="danger-action" type="button" data-project-delete="${escapeHtml(project.project_id)}">Delete Project</button>
          </div>
        </details>
      </div>
    </div>
    <div class="paper-nav-row" ${papers.length <= 1 ? "hidden" : ""}>
      <div class="paper-selector">
        <span class="project-section-label">Papers</span>
        <div class="project-paper-list">
          ${papers.map((paper) => `
            <button class="project-paper-item ${paper.analysis_id === activeAnalysisId ? "active" : ""}" type="button" data-project-paper="${escapeHtml(paper.analysis_id || "")}" title="${escapeAttribute(paperTitleFromProjectPaper(paper))}">
              <span>${escapeHtml(paperTitleFromProjectPaper(paper))}</span>
              ${paperStatusBadge(paper)}
            </button>
          `).join("")}
          ${papers.length > 1 ? '<button class="compare-tab" type="button" disabled>Compare</button>' : ""}
        </div>
      </div>
      <div class="paper-stepper">
        <button class="ghost-button compact-action" type="button" data-project-prev ${activeIndex <= 0 ? "disabled" : ""}>‹ Previous</button>
        <div class="paper-navigator-current">
          <span>Paper ${escapeHtml(String(activeIndex + 1))} of ${escapeHtml(String(papers.length))}</span>
        </div>
        <button class="ghost-button compact-action" type="button" data-project-next ${activeIndex >= papers.length - 1 ? "disabled" : ""}>Next ›</button>
      </div>
    </div>
  `;
  return wrapper;
}

function formatPaperStatus(status) {
  const normalized = String(status || "completed").toLowerCase();
  if (normalized === "completed") return "Completed";
  if (normalized === "failed") return "Failed";
  if (normalized === "analyzing") return "Analyzing";
  return "Processing";
}

function releaseLocalAnalysisDownloads() {
  activeLocalDownloadUrls.forEach((url) => URL.revokeObjectURL(url));
  activeLocalDownloadUrls = [];
}

function configureLocalAnalysisDownloads({download, downloadMarkdown, result, filename, analysisId, projectId}) {
  const stem = downloadFilenameStem(result, filename);
  const record = {
    analysis_id: analysisId || "",
    project_id: projectId || "",
    filename,
    created_at: result.generated_at || result.submitted_at || new Date().toISOString(),
    status: "completed",
    result,
  };
  setBlobDownload(download, JSON.stringify(record, null, 2), "application/json;charset=utf-8", `${stem}-analysis.json`);
  setBlobDownload(
    downloadMarkdown,
    buildLocalAnalysisMarkdown(result, filename),
    "text/markdown;charset=utf-8",
    `${stem}-analysis.md`,
  );
}

function setBlobDownload(link, content, type, filename) {
  const url = URL.createObjectURL(new Blob([content], {type}));
  activeLocalDownloadUrls.push(url);
  link.href = url;
  link.download = filename;
}

function downloadFilenameStem(result, fallbackFilename) {
  const title = paperTitleFromResult(result, fallbackFilename).replace(/\.[^.]+$/, "");
  const safe = title
    .normalize("NFKD")
    .replace(/[^\p{L}\p{N}._-]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 90);
  return safe || "deepdoc";
}

function buildLocalAnalysisMarkdown(result, fallbackFilename) {
  const summary = result.document_summary || {};
  const title = paperTitleFromResult(result, fallbackFilename);
  const lines = [
    `# ${title}`,
    "",
    `- Summary mode: ${formatMode(result.summary_mode)}`,
    `- Generated: ${result.generated_at || result.submitted_at || "Unavailable"}`,
    "",
    "## Overview",
    "",
    summary.summary || "No summary returned.",
    "",
  ];
  appendMarkdownList(lines, "Key Ideas", summary.key_ideas);
  appendMarkdownList(lines, "Contributions", summary.contributions);
  appendMarkdownEvidence(lines, summary.evidence);
  appendMarkdownList(lines, "References", result.references);
  return `${lines.join("\n").trim()}\n`;
}

function appendMarkdownList(lines, heading, items) {
  lines.push(`## ${heading}`, "");
  const values = Array.isArray(items) ? items : [];
  if (!values.length) {
    lines.push("None.", "");
    return;
  }
  values.forEach((item) => {
    const text = typeof item === "object"
      ? (item.text || item.title || item.claim || JSON.stringify(item))
      : item;
    lines.push(`- ${String(text || "").trim()}`);
  });
  lines.push("");
}

function appendMarkdownEvidence(lines, items) {
  lines.push("## Evidence", "");
  const values = Array.isArray(items) ? items : [];
  if (!values.length) {
    lines.push("None.", "");
    return;
  }
  values.forEach((item) => {
    const claim = item?.claim || item?.summary || item?.text || "";
    const pages = Array.isArray(item?.pages) && item.pages.length ? `, pages ${item.pages.join(", ")}` : "";
    const source = item?.section ? ` (${item.section}${pages})` : (pages ? ` (${pages.slice(2)})` : "");
    lines.push(`- ${String(claim).trim()}${source}`);
  });
  lines.push("");
}

function paperStatusBadge(paper) {
  const status = String(paper?.status || "completed").toLowerCase();
  if (status === "completed") return "";
  return `<small class="${status === "failed" ? "failed" : ""}">${escapeHtml(formatPaperStatus(status))}</small>`;
}

function updateProjectUrl(projectId, analysisId) {
  if (!projectId || !analysisId) return;
  const url = new URL(window.location.href);
  url.searchParams.set("project", projectId);
  url.searchParams.set("paper", analysisId);
  window.history.replaceState({}, "", url);
}

function clearProjectUrl() {
  const url = new URL(window.location.href);
  if (!url.searchParams.has("project") && !url.searchParams.has("paper")) return;
  url.searchParams.delete("project");
  url.searchParams.delete("paper");
  window.history.replaceState({}, "", url);
}

function renderResult(result, fallbackFilename = "Analysis Result", options = {}) {
  document.body.classList.add("has-result");
  document.body.classList.add("source-collapsed");
  const node = resultTemplate.content.cloneNode(true);
  const workspaceProject = options.project || singlePaperProjectFromResult(result, fallbackFilename);
  currentProject = workspaceProject;
  const summary = result.document_summary || {};
  const source = result.source_metadata || {};
  const analysisId = result.analysis_id;
  const paperTitle = source.title || result.paper_title || summary.title || result.video_script?.title || fallbackFilename;
  const summaryText = summary.summary || "No summary returned.";

  node.querySelector('[data-field="mode"]').textContent = formatMode(result.summary_mode);
  const titleElement = node.querySelector('[data-field="title"]');
  titleElement.textContent = paperTitle;
  titleElement.title = paperTitle;
  node.querySelector('[data-field="meta"]').textContent = formatMeta(result);
  renderSummaryText(node.querySelector('[data-field="summary"]'), summaryText);
  node.querySelector('[data-field="overviewMeta"]').textContent = formatOverviewMeta(result, summaryText);
  renderPaperInfo(node.querySelector('[data-field="paperInfo"]'), result, fallbackFilename);

  renderList(node.querySelector('[data-field="keyIdeas"]'), summary.key_ideas || [], {variant: "ideas", limit: 5});
  renderList(node.querySelector('[data-field="contributions"]'), summary.contributions || [], {variant: "contributions", limit: 5});
  renderEvidenceViewer(node.querySelector('[data-field="evidence"]'), summary.evidence || [], {
    emptyText: "No evidence claims returned.",
  });
  renderEvidenceViewer(node.querySelector('[data-field="sources"]'), result.evidence_sources || [], {
    emptyText: "No source sections returned.",
    sourceType: "source",
  });
  renderReferences(node.querySelector('[data-field="references"]'), result.references || []);
  renderReferenceStatus(node.querySelector('[data-field="referenceStatus"]'), result);
  const videoContainer = node.querySelector('[data-field="videoScript"]');
  const videoStatus = node.querySelector('[data-field="videoStatus"]');
  const videoButton = node.querySelector('[data-field="generateVideoScript"]');
  const generateVideoButton = node.querySelector('[data-field="generateVideo"]');
  const slideCountControl = node.querySelector('[data-field="slideCount"]');
  renderVideoScript(videoContainer, result.video_script);
  syncSlideCount(slideCountControl, result.video_script);
  if (!analysisId) {
    videoButton.disabled = true;
    generateVideoButton.disabled = true;
    videoStatus.textContent = "This saved record is missing an analysis id. Reopen it from History or rerun analysis.";
  }

  resultPanel.className = "result-panel";
  resultPanel.dataset.analysisId = analysisId || "";
  resultPanel.innerHTML = "";
  resultPanel.appendChild(node);
  resultPanel.classList.add("project-workspace");
  resultPanel.prepend(createProjectNavigator(workspaceProject, options.activeAnalysisId || analysisId));
  const download = resultPanel.querySelector('[data-field="download"]');
  const downloadMarkdown = resultPanel.querySelector('[data-field="downloadMarkdown"]');
  const downloadScriptLink = resultPanel.querySelector('[data-field="downloadScript"]');
  const downloadSlidesLink = resultPanel.querySelector('[data-field="downloadSlides"]');
  const downloadSlidesHtmlLink = resultPanel.querySelector('[data-field="downloadSlidesHtml"]');
  const downloadVideoLink = resultPanel.querySelector('[data-field="downloadVideo"]');
  const modeBadge = resultPanel.querySelector('[data-field="analysisModeBadge"]');
  if (modeBadge) modeBadge.textContent = formatAnalysisModeBadge(result.summary_mode);
  releaseLocalAnalysisDownloads();
  if (workspaceProject.local && download && downloadMarkdown) {
    configureLocalAnalysisDownloads({
      download,
      downloadMarkdown,
      result,
      filename: fallbackFilename,
      analysisId,
      projectId: workspaceProject.project_id,
    });
  } else if (analysisId && download && downloadMarkdown) {
    download.href = withAccessToken(`/analyses/${analysisId}/download`);
    downloadMarkdown.href = withAccessToken(`/analyses/${analysisId}/markdown/download`);
    downloadMarkdown.download = "";
    download.download = "";
  } else {
    if (download) download.removeAttribute("href");
    if (downloadMarkdown) downloadMarkdown.removeAttribute("href");
  }
  renderVideoScriptDownload(downloadScriptLink, downloadSlidesLink, downloadSlidesHtmlLink, analysisId, result.video_script);
  renderVideoResult(videoStatus, downloadVideoLink, result.video);
  const sourceExt = (result.source_file_extension || "").toLowerCase() || fileExtension(fallbackFilename);
  renderPdfViewer(resultPanel, analysisId, {
    preferLocalPdf: Boolean(options.localPdfFile && activeLocalPdfUrl),
    fallbackPdfUrl: source.pdf_url || "",
    localPdfMissing: Boolean(options.localPdfMissing),
    sourceFileExtension: sourceExt,
  });
  updateVideoArtifactAvailability(resultPanel);
  bindTabs(resultPanel);
}

function isSourcePdf(options) {
  const ext = (options.sourceFileExtension || ".pdf").toLowerCase();
  return ext === ".pdf";
}

function renderPdfViewer(root, analysisId, options = {}) {
  const panel = root.querySelector('[data-field="pdfPanel"]');
  const viewer = root.querySelector('[data-field="pdfViewer"]');
  const placeholder = root.querySelector('[data-field="pdfPlaceholder"]');
  const downloadPdf = root.querySelector('[data-field="downloadPdf"]');
  if (!panel || !viewer || !placeholder) return;

  if (!isSourcePdf(options)) {
    panel.dataset.pdfBaseUrl = "";
    panel.dataset.sourceIsPdf = "false";
    viewer.hidden = true;
    viewer.removeAttribute("src");
    placeholder.hidden = false;
    placeholder.textContent = "Document preview is available for PDF files only. Use the Download button to view the original.";
    if (downloadPdf && analysisId) {
      downloadPdf.href = withAccessToken(`/analyses/${encodeURIComponent(analysisId)}/pdf`);
      downloadPdf.hidden = false;
      downloadPdf.download = "";
    }
    return;
  }

  panel.dataset.sourceIsPdf = "true";

  if (options.preferLocalPdf && activeLocalPdfUrl) {
    panel.dataset.pdfBaseUrl = activeLocalPdfUrl;
    viewer.hidden = false;
    viewer.src = pdfViewerUrl(activeLocalPdfUrl, 1);
    placeholder.hidden = true;
    if (downloadPdf) {
      downloadPdf.hidden = false;
      downloadPdf.href = activeLocalPdfUrl;
      downloadPdf.download = "";
    }
    return;
  }

  if (options.fallbackPdfUrl) {
    panel.dataset.pdfBaseUrl = options.fallbackPdfUrl;
    viewer.hidden = false;
    viewer.src = pdfViewerUrl(options.fallbackPdfUrl, 1);
    placeholder.hidden = true;
    if (downloadPdf) {
      downloadPdf.hidden = false;
      downloadPdf.href = options.fallbackPdfUrl;
      downloadPdf.download = "";
    }
    return;
  }

  if (options.localPdfMissing) {
    panel.dataset.pdfBaseUrl = "";
    viewer.hidden = true;
    viewer.removeAttribute("src");
    placeholder.hidden = false;
    placeholder.textContent = "Restoring the browser-only PDF from this device...";
    if (downloadPdf) {
      downloadPdf.hidden = true;
      downloadPdf.removeAttribute("href");
    }
    return;
  }

  if (!analysisId) {
    panel.dataset.pdfBaseUrl = "";
    viewer.hidden = true;
    viewer.removeAttribute("src");
    placeholder.hidden = false;
    placeholder.textContent = "Original document is available only during the upload session.";
    if (downloadPdf) {
      downloadPdf.hidden = true;
      downloadPdf.removeAttribute("href");
    }
    return;
  }

  const pdfUrl = withAccessToken(`/analyses/${encodeURIComponent(analysisId)}/pdf`);
  panel.dataset.pdfBaseUrl = pdfUrl;
  viewer.hidden = true;
  viewer.removeAttribute("src");
  placeholder.hidden = false;
  placeholder.textContent = "Checking original document...";
  if (downloadPdf) {
    downloadPdf.hidden = true;
    downloadPdf.removeAttribute("href");
  }
  checkPdfAvailability(viewer, placeholder, pdfUrl, downloadPdf);
}

async function checkPdfAvailability(viewer, placeholder, pdfUrl, downloadPdf) {
  try {
    const response = await apiFetch(pdfUrl, {method: "HEAD"});
    if (!response.ok) throw new Error("Document unavailable");
    viewer.hidden = false;
    viewer.src = pdfViewerUrl(pdfUrl, 1);
    placeholder.hidden = true;
    if (downloadPdf) {
      downloadPdf.href = pdfUrl;
      downloadPdf.hidden = false;
      downloadPdf.download = "";
    }
  } catch {
    viewer.hidden = true;
    viewer.removeAttribute("src");
    placeholder.hidden = false;
    placeholder.textContent = "Original document is not available for this analysis.";
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
  const pdfUrl = panel?.dataset.pdfBaseUrl || "";

  if (!page) {
    return;
  }

  if (panel?.dataset.sourceIsPdf === "false") {
    if (placeholder) {
      placeholder.hidden = false;
      placeholder.textContent = "Page navigation is available for PDF files only.";
    }
    markActiveEvidence(activeEvidenceItem);
    return;
  }

  if (!pdfUrl || !viewer) {
    if (placeholder) {
      placeholder.hidden = false;
      placeholder.textContent = "PDF navigation is unavailable for this analysis.";
    }
    return;
  }

  viewer.hidden = false;
  viewer.removeAttribute("src");
  requestAnimationFrame(() => {
    viewer.src = pdfViewerUrl(pdfUrl, page);
  });
  if (placeholder) placeholder.hidden = true;
  markActiveEvidence(activeEvidenceItem);
}

function pdfViewerUrl(pdfUrl, page) {
  if (String(pdfUrl || "").startsWith("blob:")) {
    return `${pdfUrl}#page=${encodeURIComponent(page)}`;
  }
  const separator = pdfUrl.includes("?") ? "&" : "?";
  return `${pdfUrl}${separator}view=${Date.now()}#page=${encodeURIComponent(page)}`;
}

function markActiveEvidence(activeEvidenceItem) {
  resultPanel.querySelectorAll(".evidence-viewer.active-evidence").forEach((item) => {
    item.classList.remove("active-evidence");
  });
  if (activeEvidenceItem) activeEvidenceItem.classList.add("active-evidence");
}

async function generateVideo(analysisId, scriptContainer, statusContainer, downloadLink, button, slideCountControl) {
  const authMessage = requireAuthMessage();
  if (authMessage) {
    statusContainer.textContent = authMessage;
    return;
  }
  const selectedSlideCount = selectedSlides(slideCountControl);
  const currentScriptSlides = Number(scriptContainer?.dataset.slideCount || 0);
  if (!currentScriptSlides || currentScriptSlides !== selectedSlideCount) {
    statusContainer.textContent = `Generate a ${selectedSlideCount}-slide script before creating the MP4.`;
    clearDownloadLink(downloadLink);
    return;
  }

  button.disabled = true;
  button.dataset.busy = "true";
  button.textContent = "Generating...";
  statusContainer.textContent = `Generating ${selectedSlideCount}-slide MP4 locally...`;
  clearDownloadLink(downloadLink);
  let videoGenerated = false;

  try {
    const response = await apiFetch(`/analyses/${encodeURIComponent(analysisId)}/video`, {
      method: "POST",
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Video generation failed with status ${response.status}`);
    }

    renderVideoResult(statusContainer, downloadLink, payload.video);
    videoGenerated = true;
    await loadHistory();
  } catch (error) {
    statusContainer.textContent = error.message || "Video generation failed.";
  } finally {
    delete button.dataset.busy;
    button.disabled = false;
    button.textContent = "Generate MP4";
    if (videoGenerated) updateVideoArtifactAvailability(resultPanel);
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
    slidesHtmlLink.href = withAccessToken(`/analyses/${encodeURIComponent(analysisId)}/slides-html/download`);
    slidesHtmlLink.dataset.slideCount = String(slideCount);
    slidesHtmlLink.textContent = `Slides HTML · ${slideCount} slides`;
    slidesHtmlLink.hidden = false;
  }

  if (slidesLink) {
    slidesLink.href = withAccessToken(`/analyses/${encodeURIComponent(analysisId)}/slides/download`);
    slidesLink.dataset.slideCount = String(slideCount);
    slidesLink.textContent = `Slides Markdown · ${slideCount} slides`;
    slidesLink.hidden = false;
  }

  if (downloadLink) {
    downloadLink.href = withAccessToken(`/analyses/${encodeURIComponent(analysisId)}/video-script/download`);
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
  downloadLink.href = withAccessToken(video.video_url);
  downloadLink.dataset.slideCount = String(slideCount);
  downloadLink.dataset.generatedAt = video.generated_at || "";
  downloadLink.textContent = `MP4 Video · ${slideCount} slides`;
  downloadLink.hidden = false;
}

function setArtifactState(element, text, state = "idle") {
  if (!element) return;
  element.textContent = text;
  element.dataset.state = state;
}

function clearDownloadLink(downloadLink) {
  if (!downloadLink) return;
  downloadLink.hidden = true;
  downloadLink.removeAttribute("href");
  delete downloadLink.dataset.slideCount;
  delete downloadLink.dataset.generatedAt;
}

function updateVideoArtifactAvailability(root, options = {}) {
  const slideCountControl = root.querySelector('[data-field="slideCount"]');
  const selectedSlideCount = selectedSlides(slideCountControl);
  const scriptContainer = root.querySelector('[data-field="videoScript"]');
  const statusContainer = root.querySelector('[data-field="videoStatus"]');
  const downloadScriptLink = root.querySelector('[data-field="downloadScript"]');
  const downloadSlidesLink = root.querySelector('[data-field="downloadSlides"]');
  const downloadSlidesHtmlLink = root.querySelector('[data-field="downloadSlidesHtml"]');
  const downloadVideoLink = root.querySelector('[data-field="downloadVideo"]');
  const slidesState = root.querySelector('[data-field="slidesState"]');
  const mp4State = root.querySelector('[data-field="mp4State"]');
  const scriptButton = root.querySelector('[data-field="generateVideoScript"]');
  const videoButton = root.querySelector('[data-field="generateVideo"]');
  const scriptSlides = Number(scriptContainer?.dataset.slideCount || 0);
  const videoSlides = Number(downloadVideoLink?.dataset.slideCount || 0);
  const selectedScriptReady = scriptSlides && scriptSlides === selectedSlideCount;
  const selectedVideoReady = videoSlides && videoSlides === selectedSlideCount && selectedScriptReady;

  if (downloadScriptLink) {
    downloadScriptLink.hidden = !selectedScriptReady;
  }
  if (downloadSlidesLink) {
    downloadSlidesLink.hidden = !selectedScriptReady;
  }
  if (downloadSlidesHtmlLink) {
    downloadSlidesHtmlLink.hidden = !selectedScriptReady;
  }
  if (downloadVideoLink) {
    downloadVideoLink.hidden = !selectedVideoReady;
  }

  if (selectedScriptReady) {
    setArtifactState(slidesState, `Slides ready · ${scriptSlides}`, "ready");
  } else if (scriptSlides) {
    setArtifactState(slidesState, `Slides ready · ${scriptSlides}, need ${selectedSlideCount}`, "stale");
  } else {
    setArtifactState(slidesState, `Need ${selectedSlideCount}-slide slides`, "idle");
  }

  if (selectedVideoReady) {
    setArtifactState(mp4State, `MP4 ready · ${videoSlides}`, "ready");
  } else if (videoSlides) {
    setArtifactState(mp4State, `MP4 ready · ${videoSlides}, need ${selectedSlideCount}`, "stale");
  } else if (selectedScriptReady) {
    setArtifactState(mp4State, "MP4 not generated", "idle");
  } else {
    setArtifactState(mp4State, "Create slides first", "blocked");
  }

  if (scriptButton && scriptButton.dataset.busy !== "true") {
    scriptButton.textContent = selectedScriptReady ? "Regenerate Slides" : `Generate ${selectedSlideCount} Slides`;
  }
  if (videoButton && videoButton.dataset.busy !== "true") {
    videoButton.textContent = selectedVideoReady ? "Regenerate Narrated MP4" : "Generate Narrated MP4";
  }

  if (options.preserveStatus) {
    return;
  }

  if (statusContainer && scriptSlides && scriptSlides !== selectedSlideCount) {
    statusContainer.textContent = `Current script is ${scriptSlides} slides. Generate a ${selectedSlideCount}-slide script to update downloads.`;
  } else if (statusContainer && selectedVideoReady) {
    statusContainer.textContent = `Generated ${videoSlides}-slide video at ${formatDate(downloadVideoLink.dataset.generatedAt)}.`;
  } else if (statusContainer && selectedScriptReady) {
    statusContainer.textContent = `Slides ready: ${scriptSlides} slides. Create an MP4 for this version when ready.`;
  } else if (statusContainer) {
    statusContainer.textContent = "Generate slides first, then download Markdown or optionally create an MP4.";
  }
}

async function generateVideoScript(analysisId, container, button, slideCountControl, downloadScriptLink, downloadVideoLink, downloadSlidesLink, downloadSlidesHtmlLink) {
  const authMessage = requireAuthMessage();
  const statusContainer = resultPanel.querySelector('[data-field="videoStatus"]');
  if (authMessage) {
    if (statusContainer) statusContainer.textContent = authMessage;
    return;
  }
  const slideCount = selectedSlides(slideCountControl);
  button.disabled = true;
  button.dataset.busy = "true";
  button.textContent = "Generating";
  if (slideCountControl) slideCountControl.disabled = true;
  clearDownloadLink(downloadScriptLink);
  clearDownloadLink(downloadSlidesLink);
  clearDownloadLink(downloadSlidesHtmlLink);
  clearDownloadLink(downloadVideoLink);
  if (statusContainer) statusContainer.textContent = `Generating ${slideCount} slides with Gemini...`;
  container.dataset.slideCount = "";
  container.innerHTML = `<p class="result-meta">Creating ${escapeHtml(slideCount)} slides...</p>`;
  let slidesGenerated = false;

  try {
    const response = await apiFetch(`/analyses/${encodeURIComponent(analysisId)}/video-script?slide_count=${encodeURIComponent(slideCount)}`, {
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
    slidesGenerated = true;
    await loadHistory();
  } catch (error) {
    const message = error.message || "Video script failed.";
    if (statusContainer) statusContainer.textContent = message;
    container.innerHTML = `<p class="result-meta">${escapeHtml(message)}</p>`;
  } finally {
    delete button.dataset.busy;
    button.disabled = false;
    if (slideCountControl) slideCountControl.disabled = false;
    button.textContent = "Regenerate Slides";
    updateVideoArtifactAvailability(resultPanel, {preserveStatus: !slidesGenerated});
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
    <div>
      <span class="stage-label">Latest Generated Outline</span>
      <strong>${escapeHtml(script.title || "Research explainer")}</strong>
    </div>
    <span>${escapeHtml(String(sceneCount))} slides · ${escapeHtml(script.audience || "general audience")}</span>
  `;
  container.appendChild(header);

  script.scenes.forEach((scene, index) => {
    const card = document.createElement("details");
    card.className = "scene-card slide-preview-card";
    const bullets = Array.isArray(scene.bullets) ? scene.bullets : [];
    card.innerHTML = `
      <summary>
        <span class="slide-number">Slide ${escapeHtml(scene.scene_number || index + 1)}</span>
        <span class="slide-title">${escapeHtml(scene.heading || "Untitled slide")}</span>
        <span class="slide-action">Preview</span>
      </summary>
      <div class="slide-preview-body">
        <ul>${bullets.slice(0, 4).map((bullet) => `<li>${escapeHtml(bullet)}</li>`).join("")}</ul>
        <div class="speaker-notes">
          <strong>Speaker Notes</strong>
          <p>${escapeHtml(scene.voiceover || "No speaker notes generated.")}</p>
        </div>
        <p class="result-meta"><strong>Visual:</strong> ${escapeHtml(scene.visual_type || "template")} · ${escapeHtml(scene.visual_note || "")}</p>
      </div>
    `;
    if (scene.evidence && typeof scene.evidence === "object") {
      card.querySelector(".slide-preview-body").appendChild(createEvidenceItem(scene.evidence, {compact: true}));
    } else {
      card.querySelector(".slide-preview-body").appendChild(createMissingEvidenceItem("No evidence attached to this slide."));
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

function renderSummaryText(container, text) {
  if (!container) return;
  container.innerHTML = "";
  const paragraphs = String(text || "No summary returned.")
    .split(/\n\s*\n+/)
    .map((paragraph) => paragraph.replace(/\s+/g, " ").trim())
    .filter(Boolean);

  (paragraphs.length ? paragraphs : ["No summary returned."]).forEach((paragraph) => {
    const item = document.createElement("p");
    item.textContent = paragraph;
    container.appendChild(item);
  });
}

function selectedSlides(control) {
  return Number(control?.value || 10);
}

function renderList(container, items, options = {}) {
  container.innerHTML = "";
  if (!items.length) {
    container.appendChild(emptyLine("No items returned."));
    return;
  }

  items.forEach((item, index) => {
    const li = document.createElement("li");
    if (options.limit && index >= options.limit) li.hidden = true;
    if (options.variant) li.dataset.variant = options.variant;
    if (options.variant === "ideas" || options.variant === "contributions") {
      li.dataset.index = String(index + 1);
    }
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

  if (options.limit && items.length > options.limit) {
    const button = document.createElement("button");
    button.className = "view-all-button";
    button.type = "button";
    button.dataset.listToggle = options.variant || "items";
    button.dataset.hiddenCount = String(items.length - options.limit);
    button.textContent = `View all ${items.length - options.limit}`;
    container.after(button);
  }
}

function handleCopyClick(event) {
  const referenceButton = event.target.closest("[data-copy-reference]");
  if (referenceButton) {
    const text = referenceButton.closest(".reference-entry")?.querySelector(".reference-text")?.textContent?.trim() || "";
    copyButtonText(referenceButton, text);
    return;
  }

  const button = event.target.closest("[data-copy-field]");
  if (!button) return;
  const field = button.dataset.copyField;
  const text = getCopyText(field);
  if (!text) return;

  copyButtonText(button, text);
}

function copyButtonText(button, text) {
  navigator.clipboard.writeText(text).then(() => {
    const previous = button.textContent;
    button.textContent = "Copied";
    window.setTimeout(() => {
      button.textContent = previous;
    }, 1100);
  }).catch(() => {
    button.textContent = "Copy failed";
  });
}

function getCopyText(field) {
  if (field === "summary") {
    return resultPanel.querySelector('[data-field="summary"]')?.textContent?.trim() || "";
  }
  if (field === "keyIdeas") {
    return listCopyText(resultPanel.querySelector('[data-field="keyIdeas"]'));
  }
  if (field === "contributions") {
    return listCopyText(resultPanel.querySelector('[data-field="contributions"]'));
  }
  return "";
}

function listCopyText(list) {
  if (!list) return "";
  return [...list.querySelectorAll(":scope > li")]
    .map((item, index) => `${index + 1}. ${item.childNodes[0]?.textContent?.trim() || item.textContent.trim()}`)
    .join("\n");
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

  references.forEach((reference, index) => {
    const li = document.createElement("li");
    const text = stripReferenceNumber(reference);
    const isLong = text.length > 260;
    li.className = "reference-entry";
    if (isLong) li.dataset.collapsed = "true";
    li.innerHTML = `
      <span class="reference-number">[${index + 1}]</span>
      <span class="reference-text">${escapeHtml(text)}</span>
      ${isLong ? '<button class="reference-more" type="button" data-reference-toggle>More</button>' : ""}
      <button class="copy-button reference-copy" type="button" data-copy-reference="${index}">Copy</button>
    `;
    container.appendChild(li);
  });
}

function renderReferenceStatus(container, result) {
  if (!container) return;
  const references = Array.isArray(result.references) ? result.references : [];
  const parts = [];
  const hasExpectedMismatch = result.references_expected_count && result.references_expected_count !== references.length;
  if (references.length) parts.push(`${references.length} extracted`);
  if (hasExpectedMismatch) {
    parts.push(`expected around ${result.references_expected_count}`);
  }
  if (result.references_repaired_from_pdf) {
    parts.push("repaired from PDF text");
  }
  if (result.references_low_confidence) {
    parts.push("may be incomplete");
  }
  if (result.references_extraction_method && result.references_extraction_method !== "local") {
    parts.push(result.references_extraction_method.replaceAll("_", " "));
  }

  container.textContent = parts.join(" · ");
  const shouldShow = Boolean(
    result.references_repaired_from_pdf
    || result.references_low_confidence
    || hasExpectedMismatch
    || (result.references_extraction_method && result.references_extraction_method !== "local")
  );
  container.hidden = !shouldShow || parts.length === 0;
}

function toggleReference(button) {
  const entry = button.closest(".reference-entry");
  if (!entry) return;
  const collapsed = entry.dataset.collapsed !== "false";
  entry.dataset.collapsed = collapsed ? "false" : "true";
  button.textContent = collapsed ? "Less" : "More";
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
  if (usesAnonymousHistory()) {
    const projectItems = readAnonymousProjects().map(anonymousProjectSummary);
    const legacyItems = readAnonymousRecords().map((record) => {
      const summary = anonymousHistorySummary(record);
      return {
        ...summary,
        project_id: `legacy-${summary.local_id}`,
        name: summary.paper_title || summary.filename || "Untitled Paper",
        paper_count: 1,
        completed_count: 1,
        failed_count: 0,
        active_analysis_id: summary.local_id,
        legacy: true,
      };
    });
    historyItems = [...projectItems, ...legacyItems];
    renderFilteredHistory();
    return;
  }
  try {
    const response = await apiFetch("/projects");
    if (!response.ok) {
      throw new Error("Could not load history.");
    }

    const payload = await response.json();
    historyItems = payload.projects || [];
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
    const papers = Array.isArray(item.papers) ? item.papers : [];
    const modeMatches = mode === "all" || papers.some((paper) => paper.summary_mode === mode) || item.summary_mode === mode;
    const dateMatches = !date || historyDateKey(item.updated_at || item.created_at) === date;
    const queryMatches = !query || historySearchText(item).includes(query);
    return modeMatches && dateMatches && queryMatches;
  });

  renderHistory(filteredItems, historyItems.length);
}

function renderHistory(items) {
  historyList.innerHTML = "";
  if (!historyItems.length) {
    historyList.innerHTML = '<p class="result-meta">No saved projects yet.</p>';
    return;
  }

  if (!items.length) {
    historyList.innerHTML = '<p class="result-meta">No projects match the current search or filter.</p>';
    return;
  }

  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "history-item";
    const displayTitle = item.name || item.paper_title || item.filename || "Untitled Project";
    const itemId = item.project_id || item.analysis_id || item.local_id;
    const paperCount = Number(item.paper_count || item.papers?.length || 1);
    const completedCount = Number(item.completed_count || 0);
    const failedCount = Number(item.failed_count || 0);
    const statusLabel = failedCount ? `${completedCount} completed · ${failedCount} failed` : "Completed";
    const statusClass = failedCount ? "has-failures" : "is-complete";
    const storageBadge = item.local
      ? '<span class="history-storage-badge">◎ <span>Browser only</span></span>'
      : '<span class="history-storage-badge is-cloud">☁ <span>Cloud saved</span></span>';
    const iconClass = paperCount > 1 ? "is-project" : "is-paper";
    card.innerHTML = `
      <div class="history-card-header">
        <span class="history-project-icon ${iconClass}" aria-hidden="true"></span>
        <div class="history-card-heading">
          <strong>${escapeHtml(displayTitle)}</strong>
          <div class="history-card-meta">
            <span class="history-paper-count">${escapeHtml(String(paperCount))} paper${paperCount === 1 ? "" : "s"}</span>
            <span class="history-meta-separator" aria-hidden="true"></span>
            <span class="history-status ${statusClass}">${escapeHtml(statusLabel)}</span>
            <span class="history-meta-separator" aria-hidden="true"></span>
            <time>${escapeHtml(formatDate(item.updated_at || item.created_at))}</time>
          </div>
        </div>
        <span class="history-card-more" aria-hidden="true">⋮</span>
      </div>
      <p class="history-card-summary">${escapeHtml(trimText(item.summary || "No summary available.", 210))}</p>
      <div class="history-card-footer">
        ${storageBadge}
        <div class="history-actions">
          <button class="history-open-action" type="button" data-open="${escapeHtml(itemId || "")}">Open</button>
          <button class="history-rename-action" type="button" data-rename-project="${escapeHtml(itemId || "")}" ${item.legacy ? "disabled" : ""}>Rename</button>
          <button class="danger-action history-delete-action" type="button" data-delete="${escapeHtml(itemId || "")}">Delete</button>
        </div>
      </div>
    `;
    card.querySelector("[data-open]").addEventListener("click", () => openAnalysisItem(item));
    card.querySelector("[data-delete]").addEventListener("click", () => deleteHistoryItem(item));
    card.querySelector("[data-rename-project]")?.addEventListener("click", () => renameHistoryProject(item));
    historyList.appendChild(card);
  });
}

function historySearchText(item) {
  const paperText = Array.isArray(item.papers)
    ? item.papers.map((paper) => [paper.paper_title, paper.filename, paper.summary].join(" ")).join(" ")
    : "";
  return normalizeSearch([
    item.name,
    item.paper_title,
    item.filename,
    item.summary,
    paperText,
    formatMode(item.summary_mode),
    formatDate(item.updated_at || item.created_at),
  ].join(" "));
}

async function openAnalysisItem(item) {
  if (item?.local && item?.papers) {
    const project = getAnonymousProject(item.project_id);
    if (!project) {
      renderError("This browser-only project was not found.");
      return;
    }
    renderProject(project, project.active_analysis_id);
    return;
  }
  if (item?.legacy && item?.local) {
    const localId = String(item.project_id || "").replace("legacy-", "");
    const record = getAnonymousRecord(item.local_id);
    const fallbackRecord = getAnonymousRecord(localId);
    const selectedRecord = record || fallbackRecord;
    if (!selectedRecord) {
      renderError("This browser-only analysis was not found.");
      return;
    }
    setActiveLocalPdf(null);
    renderResult(selectedRecord.result || {}, selectedRecord.filename || "Browser Analysis");
    return;
  }
  return openProject(item?.project_id, item?.active_analysis_id);
}

async function openProject(projectId, analysisId = "") {
  if (!projectId) return;
  const authMessage = requireAuthMessage();
  if (authMessage) {
    renderError(authMessage);
    return;
  }

  setBusy(true, "Loading project...");
  try {
    const response = await apiFetch(`/projects/${encodeURIComponent(projectId)}`);
    if (!response.ok) {
      throw new Error("Project was not found.");
    }

    const project = await response.json();
    renderProject(project, analysisId || project.active_analysis_id);
  } catch (error) {
    renderError(error.message);
  } finally {
    setBusy(false);
  }
}

async function openAnalysis(analysisId) {
  if (!analysisId) return;
  const authMessage = requireAuthMessage();
  if (authMessage) {
    renderError(authMessage);
    return;
  }

  setBusy(true, "Loading saved analysis...");
  try {
    const response = await apiFetch(`/analyses/${encodeURIComponent(analysisId)}`);
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
  const projectId = item?.project_id;
  const analysisId = item?.analysis_id;
  const localId = item?.local_id;
  if (!projectId && !analysisId && !localId) return;
  if (item?.local && item?.papers) {
    const displayTitle = item.name || "this project";
    const confirmed = window.confirm(`Delete "${displayTitle}" from this browser?`);
    if (!confirmed) return;
    deleteAnonymousProjectPdfFiles(getAnonymousProject(projectId)).catch(() => {});
    deleteAnonymousProject(projectId);
    await loadHistory();
    return;
  }
  const authMessage = requireAuthMessage();
  if (authMessage) {
    renderError(authMessage);
    return;
  }

  const displayTitle = item.name || item.paper_title || item.filename || "this project";
  const confirmed = window.confirm(`Delete "${displayTitle}" and its stored PDFs / generated files from this server?`);
  if (!confirmed) return;

  setBusy(true, "Deleting project...");
  try {
    const endpoint = projectId ? `/projects/${encodeURIComponent(projectId)}` : `/analyses/${encodeURIComponent(analysisId)}`;
    const response = await apiFetch(endpoint, {
      method: "DELETE",
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "Could not delete this project.");
    }

    historyItems = historyItems.filter((entry) => entry.project_id !== projectId);
    renderFilteredHistory();
    if (currentProject?.project_id === projectId || resultPanel.dataset.analysisId === analysisId) {
      returnHome();
    }
  } catch (error) {
    renderError(error.message || "Could not delete this project.");
  } finally {
    setBusy(false);
  }
}

async function renameHistoryProject(item) {
  if (!item?.project_id || item.legacy) return;
  const nextName = window.prompt("Project name", item.name || "Untitled Project");
  if (!nextName || nextName.trim() === item.name) return;
  if (item.local) {
    const projects = readAnonymousProjects();
    const updatedProjects = projects.map((project) => (
      project.project_id === item.project_id
        ? {...project, name: nextName.trim(), updated_at: new Date().toISOString()}
        : project
    ));
    writeAnonymousProjects(updatedProjects);
    if (currentProject?.project_id === item.project_id) {
      currentProject = {...currentProject, name: nextName.trim()};
    }
    await loadHistory();
    return;
  }
  try {
    const response = await apiFetch(`/projects/${encodeURIComponent(item.project_id)}`, {
      method: "PATCH",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name: nextName.trim()}),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || "Could not rename project.");
    await loadHistory();
  } catch (error) {
    renderError(error.message || "Could not rename project.");
  }
}

function formatMode(mode) {
  const labels = {
    paragraph: "Quick summary",
    standard: "Standard summary",
    one_page: "Detailed summary",
  };
  return labels[mode] || "Standard summary";
}

function formatAnalysisModeName(mode) {
  const labels = {
    paragraph: "Quick",
    standard: "Standard",
    one_page: "Detailed",
  };
  return labels[normalizeSummaryMode(mode)] || "Standard";
}

function formatAnalysisModeBadge(mode) {
  return `${formatAnalysisModeName(mode)} analysis`;
}

function currentAnalysisSummaryMode() {
  const analysisId = resultPanel.dataset.analysisId || "";
  const paper = (currentProject?.papers || []).find((item) => item.analysis_id === analysisId);
  return paper?.record?.result?.summary_mode || paper?.summary_mode || "standard";
}

function normalizeSummaryMode(mode) {
  return ["paragraph", "standard", "one_page"].includes(mode) ? mode : "standard";
}

function formatOverviewMeta(result, text) {
  const parts = [formatMode(result.summary_mode)];
  const words = countWords(text);
  if (words) parts.push(`≈${words} words`);
  if (result.processing_seconds !== undefined) parts.push(`Generated in ${formatSeconds(result.processing_seconds)}`);
  return parts.join(" · ");
}

function renderPaperInfo(container, result, fallbackFilename) {
  if (!container) return;
  container.innerHTML = "";
  const source = result.source_metadata || {};
  const summaryText = result.document_summary?.summary || "";
  const wordCount = countWords(summaryText);
  const authors = Array.isArray(source.authors) ? source.authors.filter(Boolean).join(", ") : "";
  const categories = Array.isArray(source.categories) ? source.categories.filter(Boolean).join(" · ") : "";
  const groups = [
    ["Document", [
      ["Title", source.title || result.paper_title || ""],
      ["Pages", result.page_count ? String(result.page_count) : ""],
      ["Sections", result.summary_input_sections?.length ? String(result.summary_input_sections.length) : ""],
      ["References", result.references?.length ? String(result.references.length) : ""],
      ["File", fallbackFilename],
    ]],
    ["Publication", [
      ["Authors", authors],
      ["Published", source.published || ""],
      ["Updated", source.updated || ""],
      ["Category", categories],
      ["arXiv", source.arxiv_id || ""],
      ["DOI", source.doi || ""],
    ]],
    ["Analysis", [
      ["Summary mode", formatMode(result.summary_mode)],
      ["Generated", result.generated_at ? formatDate(result.generated_at) : ""],
    ]],
    ["Reading", [
      ["Estimated time", wordCount ? `${Math.max(1, Math.ceil(wordCount / 220))} min` : ""],
      ["Word count", wordCount ? `≈${wordCount}` : ""],
    ]],
  ];

  groups.forEach(([heading, rows]) => {
    const visibleRows = rows.filter(([, value]) => value);
    if (!visibleRows.length) return;
    const group = document.createElement("div");
    group.className = "paper-info-group";
    group.innerHTML = `<h4>${escapeHtml(heading)}</h4><dl></dl>`;
    const list = group.querySelector("dl");
    visibleRows.forEach(([label, value]) => appendPaperInfoRow(list, label, value));
    container.appendChild(group);
  });

  if (source.abs_url || source.pdf_url) {
    const group = document.createElement("div");
    group.className = "paper-info-group";
    const links = document.createElement("div");
    links.className = "paper-info-links";
    group.innerHTML = "<h4>Open</h4>";
    if (source.abs_url) links.appendChild(paperInfoLink("View on arXiv", source.abs_url));
    if (source.pdf_url) links.appendChild(paperInfoLink("Open PDF", source.pdf_url));
    group.appendChild(links);
    container.appendChild(group);
  }
}

function appendPaperInfoRow(container, label, value) {
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = value;
  container.append(dt, dd);
}

function paperInfoLink(label, href) {
  const link = document.createElement("a");
  link.href = href;
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = label;
  return link;
}

function countWords(text) {
  return String(text || "").match(/\b[\w'-]+\b/g)?.length || 0;
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

function formatNumber(value) {
  return new Intl.NumberFormat(undefined).format(Number(value || 0));
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
  return escapeHtml(value);
}

function safeArxivUrl(value) {
  const text = String(value || "");
  if (!/^https:\/\/(arxiv\.org|www\.arxiv\.org|export\.arxiv\.org)\//.test(text)) {
    return "#";
  }
  return escapeHtml(text);
}

loadHistory();
