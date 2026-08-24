const form = document.querySelector("#agent-form");
const taskInput = document.querySelector("#task");
const weatherModelInput = document.querySelector("#weather-model");
const leadInput = document.querySelector("#lead-minutes");
const eventInput = document.querySelector("#event-id");
const thresholdInput = document.querySelector("#threshold");
// CasCast's published VIL evaluation gradient. These are encoded-VIL cut
// points, not rainfall/dBZ or official warning categories.
const cascastVilThresholds = [16, 74, 133, 160, 181, 219];
const selectedThreshold = () => cascastVilThresholds.includes(Number(thresholdInput.value))
  ? Number(thresholdInput.value) : 133;
const modeInput = document.querySelector("#mode");
const modeDescription = document.querySelector("#mode-description");
const questionInput = document.querySelector("#question");
const runButton = document.querySelector("#run-button");
const runState = document.querySelector("#run-state");
const uploadInput = document.querySelector("#vil-upload");
const uploadStatus = document.querySelector("#upload-status");
const uploadPreview = document.querySelector("#upload-preview");
const sequenceUploadStatus = document.querySelector("#sequence-upload-status");
const attachmentTray = document.querySelector("#attachment-tray");
const attachmentTitle = document.querySelector("#attachment-title");
const removeAttachmentButton = document.querySelector("#remove-attachment");
const dataSourceInput = document.querySelector("#data-source");
const dataSourceNote = document.querySelector("#data-source-note");
const predictionControls = document.querySelector("#prediction-controls");
const thresholdControl = document.querySelector("#threshold-control");
const fullVilFrame = document.querySelector("#full-vil-frame");
const selectedFrameTitle = document.querySelector("#selected-frame-title");
const frameDialog = document.querySelector("#frame-dialog");
const dialogFrameImage = document.querySelector("#dialog-frame-image");
const dialogFrameTitle = document.querySelector("#dialog-frame-title");
const weatherModelNote = document.querySelector("#weather-model-note");
const chatThread = document.querySelector("#chat-thread");
const deleteTurnDialog = document.querySelector("#delete-turn-dialog");
const confirmDeleteTurnButton = document.querySelector("#confirm-delete-turn");
const newChatButton = document.querySelector("#new-chat-button");
const conversationStatus = document.querySelector("#conversation-status");
const turnDetails = document.querySelector("#turn-details");
const turnDetailsSource = document.querySelector("#turn-details-source");
const analysisPlaceInput = document.querySelector("#analysis-place");
const analysisTimeInput = document.querySelector("#analysis-time-utc");
const analysisLatitudeInput = document.querySelector("#analysis-latitude");
const analysisLongitudeInput = document.querySelector("#analysis-longitude");
const conversationContentLayout = document.querySelector(".conversation-content-layout");
const conversationVisualPane = document.querySelector("#conversation-visual-pane");
const conversationVisualCount = document.querySelector("#conversation-visual-count");
const conversationVisualStage = document.querySelector("#conversation-visual-stage");
const conversationVisualImage = document.querySelector("#conversation-visual-image");
const conversationVisualTitle = document.querySelector("#conversation-visual-title");
const conversationVisualDescription = document.querySelector("#conversation-visual-description");
const conversationVisualThumbnails = document.querySelector("#conversation-visual-thumbnails");

const emptyState = document.querySelector("#empty-state");
const resultContent = document.querySelector("#result-content");
const errorBox = document.querySelector("#error-box");

let runtimeConfig = {};
let uploadedSampleId = "";
let uploadedSequenceId = "";
let requestInFlight = false;
let attachmentPendingForMessage = false;
let pendingAttachmentPreview = "";
let conversationVisualArtifacts = [];
let selectedConversationVisual = "";
const sessionStorageKey = "sevir-upload-chat-session-id";
const chatStoragePrefix = "sevir-agent-chat-history:";
const activeUploadPrefix = "sevir-active-upload:";
const conversationIndexKey = "sevir-agent-conversations";
const createSessionId = () => (
  window.crypto?.randomUUID?.() || `session-${Date.now()}-${Math.random().toString(16).slice(2)}`
);
const createTurnId = () => (
  window.crypto?.randomUUID?.().replaceAll("-", "")
  || `turn-${Date.now()}-${Math.random().toString(16).slice(2)}`
);
const createRequestId = () => (
  window.crypto?.randomUUID?.().replaceAll("-", "")
  || `request-${Date.now()}-${Math.random().toString(16).slice(2)}`
);
function readPersistentSessionId() {
  try {
    const stored = window.localStorage.getItem(sessionStorageKey);
    if (stored) return stored;
  } catch (_error) {}
  const match = document.cookie.match(new RegExp(`(?:^|; )${sessionStorageKey}=([^;]*)`));
  return match ? decodeURIComponent(match[1]) : "";
}
function writePersistentSessionId(value) {
  try { window.localStorage.setItem(sessionStorageKey, value); } catch (_error) {}
  document.cookie = `${sessionStorageKey}=${encodeURIComponent(value)}; Max-Age=31536000; Path=/; SameSite=Lax`;
}
let sessionId = (() => {
  const requested = new URLSearchParams(window.location.search).get("session_id")?.trim() || "";
  if (/^[A-Za-z0-9_-]{1,80}$/.test(requested)) {
    writePersistentSessionId(requested);
    return requested;
  }
  const existing = readPersistentSessionId();
  if (existing) return existing;
  const created = createSessionId();
  writePersistentSessionId(created);
  return created;
})();
let chatHistory = [];
let chatMessageSequence = 0;
let turnPendingDeletion = null;

function compactForDisplay(value, depth = 0) {
  if (value === null || typeof value !== "object") return value;
  if (depth >= 5) return "...[nested value omitted]";
  if (ArrayBuffer.isView(value)) {
    return { type: value.constructor?.name || "TypedArray", length: value.length, valuesOmitted: true };
  }
  if (Array.isArray(value)) {
    if (value.length > 64) {
      return {
        type: "array",
        length: value.length,
        head: value.slice(0, 3).map((item) => compactForDisplay(item, depth + 1)),
        valuesOmitted: true,
      };
    }
    return value.map((item) => compactForDisplay(item, depth + 1));
  }
  const keys = Object.keys(value);
  const output = {};
  keys.slice(0, 80).forEach((key) => {
    output[key] = compactForDisplay(value[key], depth + 1);
  });
  if (keys.length > 80) output.__keys_omitted__ = keys.length - 80;
  return output;
}

function pretty(value) {
  try {
    return JSON.stringify(compactForDisplay(value), null, 2);
  } catch (_error) {
    return "[结果过大，已隐藏原始数组；统计摘要仍可用]";
  }
}

function hasUploadedInput() {
  return Boolean(uploadedSampleId || uploadedSequenceId);
}

function isFoundationQuestionText(text = "") {
  return /(ERA5|Aurora|大尺度环境|环境背景|环境变化|环境趋势|未来环境|气压层|位势高度|风场|湿度|温度时间序列|环境时间序列)/i.test(text);
}

function isCatalogQuestionText(text = "") {
  const value = String(text || "");
  const asksForRecords = /历史|事件|目录|检索|查询|catalog/i.test(value);
  const identifiesSevirOrPlace = /SEVIR|休斯顿|Springfield|纬度|经度|坐标|附近|周围|公里|半径|州|县/i.test(value);
  return asksForRecords && identifiesSevirOrPlace;
}

function hasLiveContext() {
  const time = analysisTimeInput?.value.trim();
  const latitudeText = analysisLatitudeInput?.value.trim() || "";
  const longitudeText = analysisLongitudeInput?.value.trim() || "";
  const latitude = Number(latitudeText);
  const longitude = Number(longitudeText);
  const hasCoordinates = Boolean(latitudeText && longitudeText
    && Number.isFinite(latitude) && latitude >= -90 && latitude <= 90
    && Number.isFinite(longitude) && longitude >= -180 && longitude <= 180);
  return Boolean(time && (analysisPlaceInput?.value.trim() || hasCoordinates));
}

function canRunWithoutUpload() {
  return Boolean(questionInput.value.trim());
}

function syncComposerAvailability() {
  runButton.disabled = requestInFlight || !questionInput.value.trim();
  if (!hasUploadedInput() && hasLiveContext()) {
    questionInput.placeholder = "地点与时间已就绪，可请求开放数据、区域发现或局地追踪…";
  } else if (!hasUploadedInput()) {
    questionInput.placeholder = "询问系统能力，或输入地点、时间与天气问题…";
  } else if (uploadedSequenceId) {
    questionInput.placeholder = "询问这13张图的历史演变或未来60分钟预测…";
  } else {
    questionInput.placeholder = "询问这张图的强度、面积或空间结构…";
  }
}

[analysisPlaceInput, analysisTimeInput, analysisLatitudeInput, analysisLongitudeInput].forEach((input) => {
  input?.addEventListener("input", syncComposerAvailability);
});

function optionalCoordinate(input) {
  const value = input?.value.trim() || "";
  const parsed = Number(value);
  return value && Number.isFinite(parsed) ? parsed : null;
}

function normalizedAnalysisTimeUtc() {
  const value = analysisTimeInput?.value.trim() || "";
  if (!value) return "";
  if (/Z$|[+-]\d{2}:?\d{2}$/i.test(value)) {
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? value : parsed.toISOString();
  }
  return `${value.length === 16 ? `${value}:00` : value}Z`;
}

const requestStageLabels = {
  request_received: "已接收请求",
  asset_binding: "正在绑定问题所需数据",
  unified_agent: "Agent 正在执行、观察并继续分析",
  completed: "分析完成",
};

async function pollRequestProgress(requestId, state) {
  while (!state.done) {
    try {
      const response = await fetch(`/api/requests/${encodeURIComponent(requestId)}`, { cache: "no-store" });
      if (response.ok) {
        const snapshot = await response.json();
        if (state.done) break;
        const stage = String(snapshot.current_stage || "request_received");
        const progress = Number(snapshot.progress);
        const percent = Number.isFinite(progress) && progress > 0
          ? ` · ${Math.min(100, Math.max(1, Math.round(progress * 100)))}%`
          : "";
        const label = requestStageLabels[stage] || "Agent 正在分析";
        runState.textContent = `${label}${percent}`;
        conversationStatus.textContent = `${label}${percent}；完成后会保留工具证据与执行轨迹。`;
      }
    } catch (_error) {
      // The POST may not have registered the request yet. Keep polling quietly.
    }
    await new Promise((resolve) => setTimeout(resolve, 700));
  }
}

function activeUploadKey() {
  return `${activeUploadPrefix}${sessionId}`;
}

function persistActiveUpload() {
  try {
    window.localStorage.setItem(activeUploadKey(), JSON.stringify({
      uploadId: uploadedSampleId,
      sequenceUploadId: uploadedSequenceId,
      dataSource: uploadedSequenceId ? "upload_sequence" : "upload",
      title: attachmentTitle.textContent,
      preview: pendingAttachmentPreview || uploadPreview.src || "",
    }));
  } catch (_error) {
    // The server-side session remains the source of truth after a turn.
  }
}

function clearUploadedInput({ forget = true } = {}) {
  uploadedSampleId = "";
  uploadedSequenceId = "";
  attachmentPendingForMessage = false;
  pendingAttachmentPreview = "";
  uploadInput.value = "";
  uploadPreview.removeAttribute("src");
  uploadPreview.classList.add("hidden");
  sequenceUploadStatus.classList.add("hidden");
  sequenceUploadStatus.textContent = "";
  attachmentTray.classList.add("hidden");
  attachmentTitle.textContent = "正在读取图片…";
  uploadStatus.textContent = "请选择1张或13张384×384灰度PNG。";
  dataSourceInput.value = "upload";
  predictionControls.classList.add("hidden");
  if (forget) {
    try { window.localStorage.removeItem(activeUploadKey()); } catch (_error) { /* ignore */ }
  }
  syncComposerAvailability();
}

function restoreActiveUpload() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(activeUploadKey()) || "null");
    if (!saved || (!saved.uploadId && !saved.sequenceUploadId)) return;
    uploadedSampleId = saved.uploadId || "";
    uploadedSequenceId = saved.sequenceUploadId || "";
    dataSourceInput.value = uploadedSequenceId ? "upload_sequence" : "upload";
    pendingAttachmentPreview = saved.preview || "";
    attachmentTitle.textContent = saved.title || (uploadedSequenceId ? "已恢复13帧图片" : "已恢复图片");
    if (pendingAttachmentPreview) {
      uploadPreview.src = pendingAttachmentPreview;
      uploadPreview.classList.remove("hidden");
    }
    attachmentTray.classList.remove("hidden");
    uploadStatus.textContent = "已恢复上一轮图片；继续提问即可，上传新图片后会切换分析对象。";
    syncDataSource();
    syncComposerAvailability();
  } catch (_error) {
    // Ignore stale or unavailable browser storage.
  }
}

function chatHistoryKey() {
  return `${chatStoragePrefix}${sessionId}`;
}

function persistChatHistory() {
  const encoded = JSON.stringify(chatHistory.slice(-100));
  try {
    window.localStorage.setItem(chatHistoryKey(), encoded);
  } catch (_error) {
    // The server-side session remains the source of truth when browser storage
    // is unavailable or full.
  }
}

// Keep the metadata needed to rebuild forecast image URLs after a reload,
// without putting verbose traces and raw arrays into localStorage.
function persistedResult(data) {
  const runs = Array.isArray(data?.tool_runs) ? data.tool_runs : [];
  const displayResult = combinedDisplayResult(data || {}, runs);
  // Artifacts may be attached by the final Agent response or by the server's
  // deterministic annotation fallback, after the last tool result was built.
  // The live renderer reads these top-level fields, so persistence must do the
  // same or the image disappears on the first hard reload.
  if (Array.isArray(data?.artifacts) && data.artifacts.length > 0) {
    displayResult.artifacts = data.artifacts;
    displayResult.artifact_count = data.artifact_count ?? data.artifacts.length;
  }
  if (data?.request_id) displayResult.request_id = data.request_id;
  if (data?.fallback_reason) displayResult.fallback_reason = data.fallback_reason;
  if (data?.partial_reason) displayResult.partial_reason = data.partial_reason;
  // Keep only the metadata needed to rebuild image URLs after a reload. Raw
  // forecast/member grids can be several megabytes and previously made
  // localStorage fail silently, which removed the images on refresh.
  const persistedKeys = [
    "task", "status", "upload_id", "event_id", "model_name", "output_type",
    "ensemble_count", "ensemble_members_available", "member_summaries",
    "frame_summaries", "history_frame_summaries", "ground_truth_frame_summaries",
    "full_resolution_viewer", "presentation", "threshold", "data_source",
    "forecast_evidence_id", "ground_truth_available", "ground_truth_complete_for_horizon",
    "artifacts", "artifact_count", "request_id", "fallback_reason", "partial_reason",
  ];
  const compact = {};
  persistedKeys.forEach((key) => {
    if (displayResult[key] !== undefined) compact[key] = displayResult[key];
  });
  return {
    answer: data?.answer || "",
    tool_name: data?.tool_name || "",
    tool_result: compact,
    // Keep the live-response shape as well as the compact tool result. The
    // artifact renderer historically reads this top-level field.
    artifacts: compact.artifacts || [],
    artifact_count: compact.artifact_count || compact.artifacts?.length || 0,
    request_id: compact.request_id || data?.request_id || "",
    elapsed_ms: data?.elapsed_ms,
    status: data?.status || "",
    route_mode: data?.route_mode || "",
    termination_reason: data?.termination_reason || "",
    diagnostic: data?.diagnostic || null,
    request_trace: data?.request_trace || null,
    trace_file: data?.trace_file || null,
    event_resolution: data?.event_resolution || null,
    model_calls: Number(data?.model_calls || 0),
    code_execution_count: Number(data?.code_execution_count || 0),
    agent_trace: (Array.isArray(data?.agent_trace) ? data.agent_trace : []).slice(0, 20).map((item) => ({
      attempt: item.attempt,
      action: item.action,
      status: item.status,
      elapsed_ms: item.elapsed_ms,
      model_elapsed_ms: item.model_elapsed_ms,
      summary: item.summary || "",
      error: item.error || "",
      calls: Array.isArray(item.calls) ? item.calls : [],
    })),
    tool_runs: runs.map((run) => ({
      step: run.step,
      tool_name: run.tool_name || run.api,
      api: run.api || run.tool_name,
      arguments: run.arguments || {},
      status: run.status,
      result_status: run.result_status,
      result_summary: run.result_summary || "",
      error: run.error || "",
      elapsed_ms: run.elapsed_ms,
      cache_hit: run.cache_hit === true,
    })),
    presentation: compact.presentation || data?.presentation || {},
  };
}

function artifactReferencesFromText(text) {
  const artifacts = [];
  const seen = new Set();
  const markdownImage = /!\[([^\]]*)\]\((\/api\/artifacts\/([0-9a-f]{32})\.png)\)/gi;
  let match;
  while ((match = markdownImage.exec(String(text || ""))) !== null) {
    if (seen.has(match[3])) continue;
    seen.add(match[3]);
    artifacts.push({
      artifact_id: match[3],
      url: match[2],
      image_url: match[2],
      mime_type: "image/png",
      title: match[1] || "Agent生成图表",
      verified: true,
    });
  }
  return artifacts;
}

function visualArtifactsFromResult(result, answer = "") {
  if (!result || typeof result !== "object") return artifactReferencesFromText(answer);
  const toolResult = result.tool_result && typeof result.tool_result === "object"
    ? result.tool_result
    : {};
  const presentation = result.presentation && typeof result.presentation === "object"
    ? result.presentation
    : {};
  return [
    ...(Array.isArray(result.artifacts) ? result.artifacts : []),
    ...(Array.isArray(toolResult.artifacts) ? toolResult.artifacts : []),
    ...(Array.isArray(presentation.artifacts) ? presentation.artifacts : []),
    ...artifactReferencesFromText(answer || result.answer || ""),
  ];
}

function selectConversationVisual(identity) {
  const selected = conversationVisualArtifacts.find((artifact) => artifact.identity === identity)
    || conversationVisualArtifacts[0];
  if (!selected) return;
  selectedConversationVisual = selected.identity;
  conversationVisualImage.src = selected.source;
  conversationVisualImage.alt = selected.title;
  conversationVisualTitle.textContent = selected.title;
  conversationVisualDescription.textContent = selected.description;
  conversationVisualDescription.classList.toggle("hidden", !selected.description);
  conversationVisualThumbnails.querySelectorAll("button").forEach((button) => {
    const active = button.dataset.artifactIdentity === selected.identity;
    button.classList.toggle("active", active);
    button.setAttribute("aria-current", active ? "true" : "false");
  });
}

function renderConversationVisualPane(extraResult = null) {
  if (!conversationVisualPane) return;
  const candidates = [];
  chatHistory.forEach((item) => {
    if (item.role !== "assistant") return;
    candidates.push(...visualArtifactsFromResult(item.result, item.text));
  });
  if (extraResult) candidates.push(...visualArtifactsFromResult(extraResult, extraResult.answer));

  const seen = new Set();
  conversationVisualArtifacts = candidates.flatMap((artifact) => {
    if (!artifact || artifact.verified === false) return [];
    const source = artifact.url || artifact.image_url;
    if (!source) return [];
    const identity = artifact.artifact_id || source;
    if (seen.has(identity)) return [];
    seen.add(identity);
    return [{
      identity,
      source,
      thumbnail: artifact.thumbnail?.url || source,
      title: artifact.title || artifact.chart_type || "Agent生成图表",
      description: artifact.description || "",
    }];
  });

  conversationVisualPane.classList.toggle("hidden", conversationVisualArtifacts.length === 0);
  conversationContentLayout?.classList.toggle("has-visuals", conversationVisualArtifacts.length > 0);
  conversationVisualCount.textContent = `${conversationVisualArtifacts.length}张`;
  conversationVisualThumbnails.replaceChildren();
  if (!conversationVisualArtifacts.length) {
    selectedConversationVisual = "";
    conversationVisualImage.removeAttribute("src");
    return;
  }

  conversationVisualArtifacts.forEach((artifact, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.artifactIdentity = artifact.identity;
    button.setAttribute("role", "listitem");
    button.setAttribute("aria-label", `查看图片${index + 1}：${artifact.title}`);
    const image = document.createElement("img");
    image.src = artifact.thumbnail;
    image.alt = "";
    image.loading = "lazy";
    const number = document.createElement("span");
    number.textContent = String(index + 1).padStart(2, "0");
    button.append(image, number);
    button.addEventListener("click", () => selectConversationVisual(artifact.identity));
    conversationVisualThumbnails.appendChild(button);
  });
  selectConversationVisual(selectedConversationVisual);
}

conversationVisualStage?.addEventListener("click", () => {
  const selected = conversationVisualArtifacts.find((artifact) => artifact.identity === selectedConversationVisual);
  if (selected) openFrameDialog(selected.source, selected.title);
});

function repairPersistedArtifactReferences(item) {
  if (item?.role !== "assistant") return item;
  const result = item.result && typeof item.result === "object" ? item.result : {};
  const toolResult = result.tool_result && typeof result.tool_result === "object"
    ? result.tool_result
    : {};
  const existing = Array.isArray(result.artifacts) && result.artifacts.some((artifact) => artifact?.url || artifact?.image_url)
    ? result.artifacts
    : Array.isArray(toolResult.artifacts)
      ? toolResult.artifacts.filter((artifact) => artifact?.url || artifact?.image_url)
      : [];
  const artifacts = existing.length ? existing : artifactReferencesFromText(item.text);
  if (!artifacts.length) return item;
  return {
    ...item,
    result: {
      ...result,
      artifacts,
      artifact_count: artifacts.length,
      tool_result: { ...toolResult, artifacts, artifact_count: artifacts.length },
    },
  };
}

function readConversationIndex() {
  try {
    const value = JSON.parse(window.localStorage.getItem(conversationIndexKey) || "[]");
    return Array.isArray(value) ? value.filter((item) => item && item.id) : [];
  } catch (_error) {
    return [];
  }
}

function conversationTitle() {
  const firstQuestion = chatHistory.find((item) => item.role === "user" && String(item.text || "").trim());
  const title = String(firstQuestion?.text || "New conversation").replace(/\s+/g, " ").trim();
  return title.length > 34 ? `${title.slice(0, 34)}…` : title;
}

function conversationTestLabel(title = "", id = "") {
  if (/merged-q01-q02/i.test(id)) return "合并 Q01＋Q02";
  const text = `${title} ${id}`;
  const acceptance = text.match(/(?:验收|复测|acceptance)[^Q]*(?:Q)?(\d{1,2})/i);
  if (acceptance) return `Q${String(acceptance[1]).padStart(2, "0")}`;
  if (/多轮|followup|multiturn/i.test(text)) return "多轮上下文测试";
  if (/历史|休斯顿|坐标|Springfield|地区查询/i.test(text)) return "历史/地区查询测试";
  if (/预测明天|气温|空气质量|道路积水/i.test(text)) return "能力边界测试";
  if (/异常|数量|顺序|输入要求/i.test(text)) return "异常输入测试";
  return "普通分析对话";
}

function persistConversationIndex() {
  const existing = readConversationIndex();
  const previous = existing.find((item) => item.id === sessionId);
  const firstUserMessage = chatHistory.find((item) => item.role === "user" && item.timestamp);
  const firstQuestionTime = Number(firstUserMessage?.timestamp) || Date.now();
  const current = {
    id: sessionId,
    title: conversationTitle(),
    testLabel: conversationTestLabel(conversationTitle(), sessionId),
    // The list timestamp is the time of the first user question.  It must
    // remain unchanged when the answer finishes or the page is refreshed.
    createdAt: previous?.createdAt || firstQuestionTime,
    order: previous?.order ?? firstQuestionTime,
    updatedAt: previous?.updatedAt || firstQuestionTime,
  };
  const entries = existing.filter((item) => item.id !== sessionId);
  entries.push(current);
  try {
    window.localStorage.setItem(conversationIndexKey, JSON.stringify(entries));
  } catch (_error) {
    // The active conversation remains usable if browser storage is unavailable.
  }
}

async function requestDeleteTurn(turnId) {
  if (requestInFlight || !turnId) return;
  turnPendingDeletion = turnId;
  if (deleteTurnDialog && typeof deleteTurnDialog.showModal === "function") {
    deleteTurnDialog.showModal();
    return;
  }
  if (window.confirm("删除这轮问题和回答？")) {
    try {
      await deleteConversationTurn(turnId);
    } catch (error) {
      conversationStatus.textContent = error.message;
    } finally {
      turnPendingDeletion = null;
    }
  }
}

async function deleteConversationTurn(turnId) {
  const pair = chatHistory.filter((item) => item.turnId === turnId);
  const question = pair.find((item) => item.role === "user")?.text || "";
  const answer = pair.find((item) => item.role === "assistant")?.text || "";
  const response = await fetch("/api/session", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, turn_id: turnId, question, answer }),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result.error || "删除对话失败");

  chatHistory = chatHistory.filter((item) => item.turnId !== turnId);
  persistChatHistory();
  persistConversationIndex();
  renderConversationVisualPane();
  chatThread.querySelectorAll("[data-turn-id]").forEach((item) => {
    if (item.dataset.turnId === turnId) item.remove();
  });
  emptyState.classList.toggle("hidden", chatHistory.length > 0);
  turnDetails.classList.add("hidden");
  turnDetails.open = false;
  resultContent.classList.add("hidden");
  conversationStatus.textContent = chatHistory.length
    ? `已删除一轮对话 · 当前 ${Math.ceil(chatHistory.length / 2)} 轮`
    : "当前对话已清空，可以重新提问。";
  renderConversationList();
}

confirmDeleteTurnButton?.addEventListener("click", async (event) => {
  event.preventDefault();
  const turnId = turnPendingDeletion;
  if (!turnId) return;
  confirmDeleteTurnButton.disabled = true;
  try {
    await deleteConversationTurn(turnId);
    deleteTurnDialog.close();
  } catch (error) {
    conversationStatus.textContent = error.message;
  } finally {
    confirmDeleteTurnButton.disabled = false;
    turnPendingDeletion = null;
  }
});

async function renderConversationList() {
  const list = document.querySelector("#conversation-list");
  if (!list) return;
  let entries = readConversationIndex();
  try {
    let response = await fetch("/api/sessions");
    if (!response.ok) {
      // Compatibility with a server process started before the session index
      // endpoint was added: recover the server's current session directly.
      const latestResponse = await fetch("/api/session?latest=1");
      const latestState = latestResponse.ok ? await latestResponse.json() : {};
      if (latestState.session_id) {
        sessionId = latestState.session_id;
        writePersistentSessionId(sessionId);
      }
      response = null;
      entries = entries.filter((item) => item.id === sessionId);
      if (!entries.length && Array.isArray(latestState.turns) && latestState.turns.length) {
        entries.push({
          id: sessionId,
          title: String(latestState.turns[0].question || "Saved conversation").replace(/\s+/g, " ").slice(0, 34),
          testLabel: conversationTestLabel(latestState.turns[0].question || "", sessionId),
          createdAt: Date.parse(String(latestState.turns[0].created_at || "")) || Date.now(),
          order: Date.parse(String(latestState.turns[0].created_at || "")) || Date.now(),
          updatedAt: Date.parse(String(latestState.turns[0].created_at || "")) || Date.now(),
        });
      }
    }
    const data = response ? await response.json() : {};
    const activeIds = new Set(Array.isArray(data.session_ids) ? data.session_ids : []);
    const serverSessions = new Map(
      (Array.isArray(data.sessions) ? data.sessions : []).map((item) => [item.id, item]),
    );
    if (activeIds.size && !activeIds.has(sessionId)) {
      sessionId = [...activeIds][0];
      writePersistentSessionId(sessionId);
    }
    const localEntries = new Map(entries.map((item) => [item.id, item]));
    // Keep browser-only records as well as server records.  A temporary
    // session-root switch, a restart, or a brief empty response must never
    // erase the user's visible history from localStorage.
    const idsToRender = activeIds.size
      ? [...new Set([...entries.map((item) => item.id), ...activeIds])]
      : entries.map((item) => item.id);
    entries = idsToRender.map((id) => {
      const local = localEntries.get(id) || {};
      const server = serverSessions.get(id) || {};
      const firstQuestion = server.title || "";
      const serverCreatedAt = Date.parse(String(server.created_at || ""));
      const serverUpdatedAt = Date.parse(String(server.updated_at || server.created_at || ""));
      const staleTitle = !local.title || local.title === "Saved conversation" || local.title === "New conversation";
      let title = (!staleTitle ? local.title : "") || (firstQuestion
        ? firstQuestion.replace(/\s+/g, " ").slice(0, 34)
        : "未命名对话");
      const batch = id.match(/^batch(\d+)(?:fixed)?-(\d+)-/);
      if (staleTitle && batch) title = `第${batch[1]}轮测试 · 第${batch[2]}题`;
      if (staleTitle && id.startsWith("multiturn-")) title = "多轮追问测试";
      return {
        id,
        title,
        testLabel: local.testLabel || conversationTestLabel(firstQuestion || title, id),
        // Prefer the durable first-question timestamp from the server.  This
        // repairs old browser entries that were previously stamped on every
        // refresh.
        createdAt: Number.isFinite(serverCreatedAt) ? serverCreatedAt : (local.createdAt || 0),
        order: Number.isFinite(serverCreatedAt) ? serverCreatedAt : (local.order || 0),
        updatedAt: Number.isFinite(serverUpdatedAt) ? serverUpdatedAt : (local.updatedAt || local.createdAt || 0),
      };
    });
    const maxOrder = entries.reduce((max, item) => Math.max(max, Number(item.order) || 0), 0);
    entries.forEach((item, index) => { if (!item.order) item.order = maxOrder + index + 1; });
    if (entries.length) {
      window.localStorage.setItem(conversationIndexKey, JSON.stringify(entries));
    }
  } catch (_error) {
    // Keep the browser cache visible when the server is temporarily offline.
  }
  entries.sort((a, b) => Number(a.createdAt || a.order || 0) - Number(b.createdAt || b.order || 0));

  const renderItems = (target, items, emptyText, { showTime = true } = {}) => {
    if (!target) return;
    const previousScrollTop = target.scrollTop;
    target.replaceChildren();
    if (!items.length) {
      const empty = document.createElement("small");
      empty.textContent = emptyText;
      target.appendChild(empty);
      target.scrollTop = previousScrollTop;
      return;
    }
    items.forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.sessionId = item.id;
      button.className = `conversation-item${item.id === sessionId ? " active" : ""}`;
      const title = document.createElement("strong");
      title.textContent = item.title || "New conversation";
      const label = document.createElement("small");
      label.className = "conversation-test-label";
      label.textContent = String(item.testLabel || conversationTestLabel(item.title, item.id))
        .replace(/验收\s*/g, "");
      button.appendChild(title);
      button.appendChild(label);
      if (showTime) {
        const time = document.createElement("small");
        time.textContent = item.createdAt ? new Date(item.createdAt).toLocaleString("zh-CN", {
          month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit",
        }) : "";
        button.appendChild(time);
      }
      button.addEventListener("click", () => switchConversation(item.id));
      target.appendChild(button);
    });
    target.scrollTop = previousScrollTop;
  };

  renderItems(list, entries, "暂无已保存对话");
}

function resetConversationView() {
  chatHistory = [];
  chatThread.innerHTML = "";
  emptyState.classList.remove("hidden");
  resultContent.classList.add("hidden");
  turnDetails.classList.add("hidden");
  turnDetails.open = false;
  errorBox.classList.add("hidden");
  runState.textContent = "就绪";
  runState.className = "run-state";
  questionInput.value = "";
  clearUploadedInput({ forget: false });
  renderConversationVisualPane();
}

async function switchConversation(nextSessionId) {
  if (requestInFlight || !nextSessionId || nextSessionId === sessionId) return;
  sessionId = nextSessionId;
  writePersistentSessionId(sessionId);
  resetConversationView();
  conversationStatus.textContent = "正在恢复对话…";
  document.querySelectorAll(".conversation-item[data-session-id]").forEach((button) => {
    button.classList.toggle("active", button.dataset.sessionId === sessionId);
  });
  await restoreChatHistory();
  restoreActiveUpload();
  await renderConversationList();
}

function compactTurnMeta(data) {
  const runs = Array.isArray(data?.tool_runs) ? data.tool_runs : [];
  const attempts = Array.isArray(data?.agent_trace)
    ? data.agent_trace
    : Array.isArray(data?.progress_steps) ? data.progress_steps : [];
  return {
    source: resultSource(combinedDisplayResult(data || {}, runs)),
    // ``tool_name`` is the selected Agent route, not an external tool call.
    // Only actual tool_runs belong in the external-tool row.
    toolNames: runs.map((run) => displayToolName(run.tool_name)).filter(Boolean),
    route: data?.termination_reason || data?.route_mode || "completed",
    elapsedMs: data?.elapsed_ms,
    diagnostic: data?.diagnostic || null,
    requestTrace: data?.request_trace || null,
    traceFile: data?.trace_file || null,
    eventResolution: data?.event_resolution || null,
    sessionId: data?.session_id || sessionId,
    stepsUsed: data?.model_calls,
    maxSteps: null,
    codeExecutionCount: Number(data?.code_execution_count || 0),
    attempts: attempts.slice(0, 20).map((item) => ({
      attempt: item.attempt,
      action: item.action || "execute",
      status: item.status,
      error: item.error || "",
    })),
  };
}

function displayToolName(name) {
  const labels = {
    unified_agent: "统一 Agent",
    "sevir.event": "读取 SEVIR 事件元数据",
    "sevir.search": "检索 SEVIR 事件目录",
    "sevir.nearby": "按位置检索 SEVIR 事件",
    "sevir.observations": "读取连续 SEVIR 观测",
    "sevir.frame_histograms": "读取 SEVIR 帧直方图",
    "sevir.similar_events": "检索历史候选事件",
    "sevir.historical_observations": "读取历史事件观测",
    "sevir.match_upload": "匹配上传资料与事件",
    "geolocator.event": "读取事件地理范围",
    "geolocator.pixel": "转换像素坐标",
    "geolocator.place": "解析地点",
    "geolocator.administrative_area": "查询州县范围",
    "geolocator.region": "定位分析区域",
    "cascast.forecast": "读取 CasCast 事件预测",
    "cascast.forecast_upload": "运行或读取上传序列预测",
    "aurora.case": "读取缓存的 ERA5/Aurora 案例",
    "aurora.forecast": "运行或读取 Aurora 区域预测",
    "aurora.regional_fields": "读取 ERA5/Aurora 区域场",
    // Legacy trace labels remain so old saved conversations still render.
    "data.load_hdf5": "读取连续 SEVIR 观测",
    "data.match_uploaded_event": "匹配观测事件",
    "foundation.live_preflight": "检查 ERA5/Aurora 输入",
    "foundation.fetch_era5": "读取 ERA5 环境资料",
    "foundation.era5_snapshot": "读取 ERA5 环境快照",
    "foundation.forecast_aurora": "读取 Aurora 区域预测",
    "foundation.region_fields": "提取区域环境场",
    "forecaster.uploaded_sequence": "读取 CasCast 局地短临",
    "forecaster.rolling_raw": "读取 CasCast 滚动预测",
    "metrics.uploaded_raw_pair": "对齐预测与后续观测",
    "uncertainty.uploaded_raw": "分析 CasCast 成员分歧",
    "visualizer.plan_spec": "生成分析图表",
  };
  if (labels[name]) return labels[name];
  const prefixLabels = {
    "sevir.": "SEVIR 资料能力",
    "cascast.": "CasCast 短临能力",
    "aurora.": "Aurora 环境能力",
    "data.": "观测数据工具",
    "foundation.": "大尺度环境工具",
    "forecaster.": "CasCast 短临工具",
    "metrics.": "预报验证工具",
    "statistics.": "统计分析工具",
    "uncertainty.": "集合不确定性工具",
    "geolocator.": "地点定位工具",
    "visualizer.": "图表工具",
  };
  const prefix = Object.keys(prefixLabels).find((item) => String(name || "").startsWith(item));
  return prefix ? `${prefixLabels[prefix]}（${name}）` : name;
}

function appendChatMessage(role, text, {
  meta = null,
  pending = false,
  attachment = null,
  timestamp = Date.now(),
  turnId = "",
  scroll = true,
} = {}) {
  emptyState.classList.add("hidden");
  const message = document.createElement("article");
  message.className = `chat-message ${role}${pending ? " pending" : ""}`;
  chatMessageSequence += 1;
  message.dataset.turnMessageId = `${sessionId}-${chatMessageSequence}`;
  message.dataset.turnId = turnId;

  const avatar = document.createElement("div");
  avatar.className = "chat-avatar";
  avatar.textContent = role === "user" ? "你" : "S";

  const body = document.createElement("div");
  body.className = "chat-message-body";
  const heading = document.createElement("div");
  heading.className = "chat-message-heading";
  const author = document.createElement("strong");
  author.textContent = role === "user" ? "你" : "SEVIR Agent";
  const time = document.createElement("time");
  time.dateTime = new Date(timestamp).toISOString();
  time.textContent = new Date(timestamp).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  heading.append(author, time);
  if (role === "user" && turnId) {
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "delete-turn-button";
    deleteButton.setAttribute("aria-label", "删除这轮对话");
    deleteButton.title = "删除问题和回答";
    deleteButton.textContent = "删除";
    deleteButton.addEventListener("click", () => requestDeleteTurn(turnId));
    heading.appendChild(deleteButton);
  }

  const content = document.createElement("div");
  content.className = "chat-message-content answer-body";
  if (attachment) {
    const card = document.createElement("div");
    card.className = "chat-attachment";
    if (attachment.preview) {
      const image = document.createElement("img");
      image.src = attachment.preview;
      image.alt = attachment.title || "上传的VIL图片";
      if (attachment.fallback) {
        image.addEventListener("error", () => {
          if (image.dataset.fallbackTried) return;
          image.dataset.fallbackTried = "1";
          image.src = attachment.fallback;
        });
      }
      card.appendChild(image);
    }
    const label = document.createElement("div");
    const title = document.createElement("strong");
    const kind = document.createElement("small");
    title.textContent = attachment.title || "VIL图片";
    kind.textContent = attachment.kind === "sequence" ? "13帧VIL序列" : "单帧VIL图片";
    label.append(title, kind);
    card.appendChild(label);
    content.appendChild(card);
  }
  if (pending) {
    const status = document.createElement("div");
    status.className = "typing-indicator";
    status.innerHTML = "<span></span><span></span><span></span>";
    content.appendChild(status);
  } else {
    renderMarkdown(content, text);
  }
  body.append(heading, content);
  if (role === "assistant") {
    const presentation = document.createElement("div");
    presentation.className = "chat-turn-presentation";
    presentation.dataset.ownerMessageId = message.dataset.turnMessageId;
    body.appendChild(presentation);
  }
  message.append(avatar, body);
  chatThread.appendChild(message);
  if (scroll) message.scrollIntoView({ behavior: "smooth", block: "end" });
  return message;
}

function completePendingMessage(message, text, meta = null, isError = false) {
  message.classList.remove("pending");
  if (isError) message.classList.add("failed");
  const content = message.querySelector(".chat-message-content");
  content.innerHTML = "";
  renderMarkdown(content, text);
  const body = message.querySelector(".chat-message-body");
  let presentation = body.querySelector(":scope > .chat-turn-presentation");
  if (!presentation) {
    presentation = document.createElement("div");
    presentation.className = "chat-turn-presentation";
    presentation.dataset.ownerMessageId = message.dataset.turnMessageId || "legacy";
    body.appendChild(presentation);
  }
  presentation.replaceChildren();
  message.scrollIntoView({ behavior: "smooth", block: "end" });
}

async function restoreChatHistory() {
  let cachedHistory = [];
  try {
    const stored = JSON.parse(window.localStorage.getItem(chatHistoryKey()) || "[]");
    cachedHistory = Array.isArray(stored) ? stored : [];
  } catch (_error) {
    cachedHistory = [];
  }
  chatHistory = cachedHistory;
  try {
    let response = await fetch(`/api/session?session_id=${encodeURIComponent(sessionId)}`);
    let state = response.ok ? await response.json() : {};
    let restored = Array.isArray(state.turns) ? state.turns : [];
    if (restored.length === 0 && cachedHistory.length === 0) {
      response = await fetch("/api/session?latest=1");
      state = response.ok ? await response.json() : {};
      restored = Array.isArray(state.turns) ? state.turns : [];
      if (state.session_id) {
        sessionId = state.session_id;
        writePersistentSessionId(sessionId);
      }
    }
    if (response.ok && restored.length) {
      const cachedByMessage = new Map(
        cachedHistory.map((item) => [`${item.turnId || ""}:${item.role || ""}`, item])
      );
      const restoredTurnIds = new Set();
      const serverHistory = restored.flatMap((turn, index) => {
        const parsedTimestamp = turn.created_at ? Date.parse(turn.created_at) : NaN;
        const timestamp = Number.isFinite(parsedTimestamp)
          ? parsedTimestamp
          : Date.now() - (restored.length - index) * 1000;
        const turnId = turn.turn_id || createTurnId();
        restoredTurnIds.add(turnId);
        const toolNames = Array.isArray(turn.selected_tools)
          ? turn.selected_tools.filter((name) => name !== "ds")
          : [];
        return [
          { role: "user", text: turn.question || "", timestamp, turnId },
          {
            role: "assistant",
            text: turn.answer || "",
            result: {
              answer: turn.answer || "",
              tool_result: turn.presentation_result || {},
              presentation: turn.presentation_result?.presentation || {},
            },
            timestamp: timestamp + 1,
            turnId,
            meta: {
              toolNames,
              route: turn.termination_reason || "unknown",
              codeExecutionCount: Number(turn.code_execution_count || 0),
              visualContract: turn.presentation_result?.visual_contract || null,
            },
          },
        ].map((item) => {
          const cached = cachedByMessage.get(`${turnId}:${item.role}`);
          if (!cached) return item;
          const serverResult = item.result && typeof item.result === "object" ? item.result : {};
          const cachedResult = cached.result && typeof cached.result === "object" ? cached.result : {};
          const serverToolResult = serverResult.tool_result && typeof serverResult.tool_result === "object"
            ? serverResult.tool_result
            : {};
          const cachedToolResult = cachedResult.tool_result && typeof cachedResult.tool_result === "object"
            ? cachedResult.tool_result
            : {};
          const serverArtifacts = Array.isArray(serverResult.artifacts) && serverResult.artifacts.length
            ? serverResult.artifacts
            : (Array.isArray(serverToolResult.artifacts) ? serverToolResult.artifacts : []);
          const mergedResult = {
            ...serverResult,
            ...cachedResult,
            tool_result: { ...serverToolResult, ...cachedToolResult },
          };
          if (serverArtifacts.length) {
            mergedResult.artifacts = serverArtifacts;
            mergedResult.artifact_count = serverArtifacts.length;
            mergedResult.tool_result.artifacts = serverArtifacts;
            mergedResult.tool_result.artifact_count = serverArtifacts.length;
          }
          return {
            ...item,
            ...cached,
            result: mergedResult,
            text: item.text || cached.text || "",
            turnId,
          };
        });
      }).filter((item) => item.text);
      const localOnly = cachedHistory.filter((item) => !restoredTurnIds.has(item.turnId));
      chatHistory = [...serverHistory, ...localOnly];
      persistChatHistory();
    }
  } catch (_error) {
    // Cached history remains usable when the server is unavailable.
  }
  let activeTurnId = "";
  chatHistory = chatHistory.map((item) => {
    if (item.role === "user" || !activeTurnId) activeTurnId = item.turnId || createTurnId();
    return repairPersistedArtifactReferences({ ...item, turnId: item.turnId || activeTurnId });
  });
  persistChatHistory();
  chatThread.innerHTML = "";
  chatHistory.forEach((item) => {
    const message = appendChatMessage(item.role, item.text, {
      meta: item.meta || null,
      attachment: item.attachment || null,
      turnId: item.turnId,
      timestamp: item.timestamp || Date.now(),
      // Restoring a completed answer may rebuild many messages. Scrolling each
      // one makes the whole page oscillate between the sidebar and chat.
      scroll: false,
    });
    // Rebuild the visual presentation for completed turns after a hard reload.
    if (item.role === "assistant" && item.result) {
      renderResult(item.result, message);
    }
  });
  emptyState.classList.toggle("hidden", chatHistory.length > 0);
  renderConversationVisualPane();
  conversationStatus.textContent = chatHistory.length
    ? `已恢复当前对话 · ${Math.ceil(chatHistory.length / 2)}轮`
    : "同一对话中的追问会继承已验证的分析上下文。";
}


function selectedModelLabel() {
  if (weatherModelInput.value === "cascast_live") return "本地现场CasCast";
  return weatherModelInput.value === "cascast" ? "CasCast" : "persistence_baseline";
}

function syncWeatherModelNote() {
  const notes = {
    cascast_live: "根据用户上传的13张历史图，现场运行EarthFormer、CasCast扩散和VAE解码。",
    persistence_baseline: "把上传序列最后一张图复制到未来12个时效，仅作为基线。",
  };
  weatherModelNote.textContent = notes[weatherModelInput.value] || "";
}

function syncEventOptions() {
  const cascast = runtimeConfig.cascast || {};
  const live = weatherModelInput.value === "cascast_live";
  const eventIds = live ? (cascast.local_hdf5_event_ids || []) : (cascast.event_ids || []);
  const currentEvent = eventInput.value;
  eventInput.innerHTML = "";
  eventIds.forEach((eventId, index) => {
    const option = document.createElement("option");
    option.value = eventId;
    option.textContent = `事件 ${index + 1} · ${eventId}`;
    eventInput.appendChild(option);
  });
  if (eventIds.includes(currentEvent)) eventInput.value = currentEvent;
  if (!eventInput.value && eventInput.options.length) eventInput.selectedIndex = 0;
}

function syncMode() {
  modeDescription.textContent = "Unified Agent：一次能够完成就停止；只有真实信息不足或执行失败时才继续。";
}

function syncTaskControls() {
  const predictionContext = dataSourceInput.value === "upload_sequence";
  predictionControls.classList.toggle("hidden", !predictionContext);
  thresholdControl.classList.add("hidden");
  thresholdControl.setAttribute("aria-hidden", "true");
  eventInput.disabled = ["upload", "upload_sequence"].includes(dataSourceInput.value);
  weatherModelInput.disabled = false;
  syncWeatherModelNote();
  leadInput.disabled = !predictionContext;
  thresholdInput.disabled = false;
}

function syncDataSource() {
  const notes = {
    upload: "分析一张VIL图的当前强回波信号，不能预测未来演变或持续时间。",
    upload_sequence: "上传13帧连续VIL后，现场运行EarthFormer、CasCast扩散和VAE解码，生成未来12帧。",
  };
  dataSourceNote.textContent = notes[dataSourceInput.value];
  if (dataSourceInput.value === "upload_sequence") weatherModelInput.value = "cascast_live";
  taskInput.value = "auto";
  syncTaskControls();
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

async function uploadVilImage(file) {
  const converted = await imageFileToGray(file);
  const { canvas, gray, grayscaleSource } = converted;
  const response = await fetch("/api/upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: file.name,
      width: 384,
      height: 384,
      pixels_base64: bytesToBase64(gray),
      grayscale_source: grayscaleSource,
    }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "图片上传失败");
  uploadedSampleId = result.upload_id;
  uploadedSequenceId = "";
  attachmentPendingForMessage = true;
  uploadPreview.src = canvas.toDataURL("image/png");
  pendingAttachmentPreview = uploadPreview.src;
  uploadPreview.classList.remove("hidden");
  attachmentTray.classList.remove("hidden");
  attachmentTitle.textContent = file.name;
  const match = result.event_match || {};
  const matchText = match.status === "matched"
    ? `已识别事件 ${match.event_id}；可据此查找对应ERA5/Aurora case`
    : "未从文件名/元数据识别事件；仍可分析VIL，但ERA5/Aurora与位置问题必须补充SEVIR event ID";
  uploadStatus.textContent = `${grayscaleSource
    ? `已载入${file.name}；原始灰度值可用于VIL阈值分析`
    : `已载入${file.name}；彩色图已转灰度，只能作为图像强度分析`}`;
  uploadStatus.textContent += `。${matchText}`;
  dataSourceInput.value = "upload";
  taskInput.value = "auto";
  syncDataSource();
  persistActiveUpload();
  syncComposerAvailability();
}

async function imageFileToGray(file) {
  const bitmap = await createImageBitmap(file);
  const imageWidth = bitmap.width;
  const imageHeight = bitmap.height;
  if (imageWidth !== 384 || imageHeight !== 384) {
    bitmap.close();
    throw new Error(`图片尺寸为${imageWidth}×${imageHeight}；第一版要求384×384。`);
  }
  const canvas = document.createElement("canvas");
  canvas.width = 384;
  canvas.height = 384;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = context.getImageData(0, 0, 384, 384).data;
  const gray = new Uint8Array(384 * 384);
  let coloredPixels = 0;
  for (let index = 0; index < gray.length; index += 1) {
    const offset = index * 4;
    const red = rgba[offset];
    const green = rgba[offset + 1];
    const blue = rgba[offset + 2];
    if (Math.max(red, green, blue) - Math.min(red, green, blue) > 2) coloredPixels += 1;
    gray[index] = Math.round(0.299 * red + 0.587 * green + 0.114 * blue);
  }
  const grayscaleSource = coloredPixels / gray.length < 0.001;
  return { canvas, gray, grayscaleSource };
}

async function uploadVilSequence(files) {
  if (files.length !== 13) throw new Error(`当前选择了${files.length}张；必须恰好选择13张。`);
  const sortedFiles = [...files].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const frames = [];
  const previewCanvases = [];
  for (let index = 0; index < sortedFiles.length; index += 1) {
    sequenceUploadStatus.textContent = `正在读取第${index + 1}/13张：${sortedFiles[index].name}`;
    const converted = await imageFileToGray(sortedFiles[index]);
    previewCanvases.push(converted.canvas);
    frames.push({
      filename: sortedFiles[index].name,
      width: 384,
      height: 384,
      pixels_base64: bytesToBase64(converted.gray),
      grayscale_source: converted.grayscaleSource,
    });
  }
  const response = await fetch("/api/upload-sequence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frames }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "序列上传失败");
  uploadedSequenceId = result.upload_id;
  uploadedSampleId = "";
  attachmentPendingForMessage = true;
  const previewCanvas = document.createElement("canvas");
  previewCanvas.width = 780;
  previewCanvas.height = 112;
  const previewContext = previewCanvas.getContext("2d");
  previewContext.fillStyle = "#111827";
  previewContext.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewCanvases.forEach((canvas, index) => {
    const x = index * 60;
    previewContext.drawImage(canvas, x, 0, 58, 88);
    previewContext.fillStyle = "rgba(255,255,255,.86)";
    previewContext.font = "9px sans-serif";
    previewContext.fillText(String(index + 1).padStart(2, "0"), x + 3, 103);
  });
  pendingAttachmentPreview = previewCanvas.toDataURL("image/png");
  dataSourceInput.value = "upload_sequence";
  taskInput.value = "auto";
  weatherModelInput.value = "cascast_live";
  attachmentTray.classList.remove("hidden");
  uploadPreview.src = pendingAttachmentPreview;
  uploadPreview.classList.remove("hidden");
  attachmentTitle.textContent = `13张VIL图片 · ${sortedFiles[0].name} → ${sortedFiles[12].name}`;
  const match = result.event_match || {};
  const matchText = match.status === "matched"
    ? `已识别事件 ${match.event_id}；可据此查找对应ERA5/Aurora case`
    : "未识别事件；VIL仍可分析，但ERA5/Aurora问题必须补充event ID";
  uploadStatus.textContent = `13帧序列已载入；文件已按名称进行自然排序。${matchText}。`;
  sequenceUploadStatus.textContent = "如询问未来趋势，将现场执行本地CasCast；没有未来Ground Truth。";
  sequenceUploadStatus.classList.remove("hidden");
  syncDataSource();
  persistActiveUpload();
  syncComposerAvailability();
}

uploadInput.addEventListener("change", async () => {
  const files = uploadInput.files || [];
  if (!files.length) return;
  if (![1, 13].includes(files.length)) {
    attachmentTray.classList.remove("hidden");
    attachmentTitle.textContent = `选择了${files.length}张图片`;
    uploadStatus.textContent = "请选择1张图片进行当前帧分析，或恰好13张图片进行序列分析。";
    syncComposerAvailability();
    return;
  }
  uploadedSampleId = "";
  uploadedSequenceId = "";
  attachmentTray.classList.remove("hidden");
  attachmentTitle.textContent = "正在读取图片…";
  uploadStatus.textContent = "正在读取并检查图片…";
  try {
    if (files.length === 1) await uploadVilImage(files[0]);
    else await uploadVilSequence(files);
  } catch (error) {
    uploadPreview.classList.add("hidden");
    uploadStatus.textContent = error.message;
    syncComposerAvailability();
  }
});

removeAttachmentButton.addEventListener("click", clearUploadedInput);

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const data = await response.json();
    runtimeConfig = data.runtime || {};
    syncEventOptions();
  } catch (_error) {
    // Chat and uploads remain usable when optional runtime metadata is unavailable.
  }
}

function formatCoordinate(region) {
  const coordinate = region.approx_centroid_latlon;
  if (!Array.isArray(coordinate) || coordinate.length < 2) return "坐标未知";
  const [latitude, longitude] = coordinate.map(Number);
  const latitudeText = `${Math.abs(latitude).toFixed(4)}°${latitude >= 0 ? "N" : "S"}`;
  const longitudeText = `${Math.abs(longitude).toFixed(4)}°${longitude >= 0 ? "E" : "W"}`;
  return `${latitudeText}, ${longitudeText}`;
}

function formatLatitude(value) {
  const latitude = Number(value);
  return Number.isFinite(latitude) ? `${Math.abs(latitude).toFixed(6)}°${latitude >= 0 ? "N" : "S"}` : "—";
}

function formatLongitude(value) {
  const longitude = Number(value);
  return Number.isFinite(longitude) ? `${Math.abs(longitude).toFixed(6)}°${longitude >= 0 ? "E" : "W"}` : "—";
}

function frameImageUrl(result, series, index = 0, leadIndex = null) {
  const isUploadedSequence = result.task === "uploaded_sevir_vil_sequence_analysis"
    || Boolean(result.upload_id)
    || result.model_name === "cascast_live";
  if (isUploadedSequence && result.upload_id) {
    const sequenceParams = new URLSearchParams({
      upload_id: result.upload_id || "",
      series,
      index: String(index),
      model_name: result.model_name || "persistence_baseline",
    });
    if (leadIndex !== null) sequenceParams.set("lead", String(leadIndex));
    return `/api/uploaded-sequence-frame?${sequenceParams.toString()}`;
  }
  const params = new URLSearchParams({
    event_id: result.event_id,
    series,
    index: String(index),
    model_name: result.model_name || "cascast",
    threshold: String(result.threshold ?? selectedThreshold()),
  });
  if (leadIndex !== null) params.set("lead", String(leadIndex));
  return `/api/cascast-frame?${params.toString()}`;
}

function uploadedFrameUrl(result) {
  const params = new URLSearchParams({ upload_id: result.upload_id || "" });
  return `/api/uploaded-frame?${params.toString()}`;
}

function openFrameDialog(source, title) {
  dialogFrameImage.src = source;
  dialogFrameTitle.textContent = title;
  frameDialog.showModal();
}

function installImageFallback(image) {
  image.onload = () => {
    image.classList.remove("hidden");
    image.parentElement?.querySelector(".image-load-error")?.remove();
  };
  image.onerror = () => {
    image.classList.add("hidden");
    if (image.parentElement?.querySelector(".image-load-error")) return;
    const notice = document.createElement("div");
    notice.className = "image-load-error";
    notice.textContent = "完整图像接口不可用。请重启 Python 服务后按 Ctrl+F5 刷新页面。";
    image.insertAdjacentElement("afterend", notice);
  };
}

function selectFullFrame(result, series, index, title, button) {
  document.querySelectorAll(".history-frame, .forecast-frame, .ground-truth-frame").forEach((item) => item.classList.remove("selected"));
  if (button) button.classList.add("selected");
  const source = frameImageUrl(result, series, index);
  fullVilFrame.src = source;
  fullVilFrame.dataset.title = title;
  selectedFrameTitle.textContent = title;
}

function renderEvaluation(result) {
  const card = document.querySelector("#evaluation-card");
  const metrics = result.model_metrics || {};
  if (result.presentation?.show_evaluation === false || !metrics.ground_truth_available) {
    card.classList.add("hidden");
    return;
  }
  const formatScore = (value) => value === undefined ? "—" : Number(value).toFixed(3);
  document.querySelector("#evaluation-threshold").textContent = `VIL≥${metrics.metric_threshold}`;
  document.querySelector("#evaluation-csi").textContent = formatScore(metrics.csi_at_threshold);
  document.querySelector("#evaluation-pod").textContent = formatScore(metrics.pod_at_threshold);
  document.querySelector("#evaluation-far").textContent = formatScore(metrics.far_at_threshold);
  document.querySelector("#evaluation-mse").textContent = metrics.mse_raw_vil ?? "—";
  document.querySelector("#evaluation-mae").textContent = metrics.mae_raw_vil ?? "—";
  card.classList.remove("hidden");
}

function renderDatasetEvaluation(result) {
  const card = document.querySelector("#dataset-evaluation-card");
  if (result.task !== "cascast_dataset_evaluation") {
    card.classList.add("hidden");
    return;
  }
  const thresholdKey = String(Math.trunc(Number(result.primary_threshold)));
  const cascast = result.aggregate_metrics?.cascast_ensemble_mean?.[thresholdKey] || {};
  const baseline = result.aggregate_metrics?.persistence?.[thresholdKey] || {};
  const validation = result.ensemble_validation || {};
  const objects = result.storm_object_evaluation || {};
  document.querySelector("#dataset-evaluation-threshold").textContent = `VIL≥${result.primary_threshold}`;
  document.querySelector("#dataset-csi").textContent = Number(cascast.csi || 0).toFixed(3);
  document.querySelector("#baseline-csi").textContent = Number(baseline.csi || 0).toFixed(3);
  document.querySelector("#dataset-rmse").textContent = Number(cascast.rmse_raw_vil || 0).toFixed(2);
  document.querySelector("#spread-error-correlation").textContent = Number(validation.pixel_std_absolute_error_pearson || 0).toFixed(3);
  document.querySelector("#interval-coverage").textContent = `${(Number(validation.central_80_interval_coverage || 0) * 100).toFixed(1)}%`;
  document.querySelector("#object-csi").textContent = Number(objects.object_csi || 0).toFixed(3);

  const thresholdContainer = document.querySelector("#threshold-comparison");
  thresholdContainer.innerHTML = "";
  (result.thresholds || []).forEach((threshold) => {
    const key = String(Math.trunc(Number(threshold)));
    const modelScore = Number(result.aggregate_metrics?.cascast_ensemble_mean?.[key]?.csi || 0);
    const baselineScore = Number(result.aggregate_metrics?.persistence?.[key]?.csi || 0);
    const row = document.createElement("div");
    row.className = "comparison-row";
    row.innerHTML = `<b>${key}</b><div><i class="cascast-bar" style="width:${modelScore * 100}%"></i><i class="baseline-bar" style="width:${baselineScore * 100}%"></i></div><span>${modelScore.toFixed(3)} / ${baselineScore.toFixed(3)}</span>`;
    thresholdContainer.appendChild(row);
  });

  const leadContainer = document.querySelector("#lead-time-comparison");
  leadContainer.innerHTML = "";
  const cascastLead = result.lead_time_metrics?.cascast_ensemble_mean || [];
  const baselineLead = result.lead_time_metrics?.persistence || [];
  cascastLead.forEach((item, index) => {
    const baselineItem = baselineLead[index] || {};
    const column = document.createElement("div");
    column.className = "lead-column";
    column.innerHTML = `<div><i class="cascast-lead" style="height:${Math.max(2, Number(item.csi || 0) * 100)}%"></i><i class="baseline-lead" style="height:${Math.max(2, Number(baselineItem.csi || 0) * 100)}%"></i></div><b>+${item.lead_minutes}</b>`;
    leadContainer.appendChild(column);
  });
  card.classList.remove("hidden");
}

function renderHazardOverview(result) {
  if (!result.full_resolution_viewer) {
    document.querySelector("#visual-card").classList.add("hidden");
    return;
  }
  const card = document.querySelector("#visual-card");
  const overviewImage = document.querySelector("#hazard-overview-image");
  const probabilityImage = document.querySelector("#hazard-probability-image");
  const durationImage = document.querySelector("#hazard-duration-image");
  const mapGrid = overviewImage.closest(".hazard-map-grid");
  const probabilityPanel = probabilityImage.closest(".hazard-map-panel");
  const durationPanel = durationImage.closest(".hazard-map-panel");
  const requestedLead = Number(result.presentation?.requested_lead_minutes || result.lead_minutes || 60);
  const leadIndex = Math.max(0, Math.min(11, Math.ceil(requestedLead / 5) - 1));
  overviewImage.src = result.task === "uploaded_sevir_vil_analysis"
    ? uploadedFrameUrl(result)
    : frameImageUrl(result, "forecast_overview", 0, leadIndex);
  overviewImage.classList.remove("hidden");
  overviewImage.onclick = () => openFrameDialog(
    overviewImage.src,
    result.task === "uploaded_sevir_vil_analysis" ? "上传的SEVIR VIL图像" : `未来${requestedLead}分钟总体VIL场`,
  );
  const defaultShowEnsembleDetails = [
    "future_hazard_analysis",
    "ensemble_uncertainty_analysis",
  ].includes(result.task);
  const showEnsembleHazardMaps = result.presentation?.show_ensemble_details === false
    ? false
    : (result.presentation?.show_ensemble_details === true || defaultShowEnsembleDetails)
      && ["cascast", "cascast_live"].includes(result.model_name)
      && Number(result.ensemble_count || 0) > 1;
  mapGrid.classList.toggle("single-map", !showEnsembleHazardMaps);
  probabilityPanel.classList.toggle("hidden", !showEnsembleHazardMaps);
  durationPanel.classList.toggle("hidden", !showEnsembleHazardMaps);
  if (showEnsembleHazardMaps) {
    probabilityImage.src = frameImageUrl(result, "hazard_probability_any", 0, leadIndex);
    durationImage.src = frameImageUrl(result, "hazard_duration_mean", 0, leadIndex);
    probabilityImage.parentElement.onclick = () => openFrameDialog(probabilityImage.src, `未来${requestedLead}分钟超阈值成员支持率`);
    durationImage.parentElement.onclick = () => openFrameDialog(durationImage.src, `未来${requestedLead}分钟平均超阈值持续时间`);
  } else {
    probabilityImage.removeAttribute("src");
    durationImage.removeAttribute("src");
    probabilityImage.parentElement.onclick = null;
    durationImage.parentElement.onclick = null;
  }
  document.querySelector("#visual-resolution-label").textContent = "FULL 384×384";
  card.classList.remove("hidden");
}

function renderEnsembleUncertainty(result) {
  const card = document.querySelector("#uncertainty-card");
  const hasMembers = Array.isArray(result.member_summaries)
    && Number(result.ensemble_count || 0) > 1;
  const pixel = result.pixel_uncertainty || {};
  const objects = result.storm_object_uncertainty || {};
  const hasUncertaintyAnalysis = result.task === "ensemble_uncertainty_analysis"
    || (pixel.mean_std !== undefined && pixel.p95_std !== undefined)
    || (objects.members_with_object !== undefined && objects.centroid_spread_pixels !== undefined);
  if (!hasUncertaintyAnalysis) {
    card.classList.add("hidden");
    return;
  }
  const uncertaintyMemberCount = Number(result.ensemble_count || 0);
  const uncertaintyHeading = document.querySelector("#uncertainty-card .card-topline strong");
  if (uncertaintyHeading) uncertaintyHeading.textContent = `${uncertaintyMemberCount || "集合"}路CasCast预测分歧`;
  document.querySelector("#uncertainty-mean-std").textContent = pixel.mean_std ?? "—";
  document.querySelector("#uncertainty-p95-std").textContent = pixel.p95_std ?? "—";
  document.querySelector("#uncertainty-object-members").textContent = `${objects.members_with_object ?? 0}/${uncertaintyMemberCount || "—"}`;
  document.querySelector("#uncertainty-centroid-spread").textContent = `${objects.centroid_spread_pixels ?? 0} px`;
  const uncertaintyImages = {
    mean: {
      element: document.querySelector("#uncertainty-mean-map"),
      source: frameImageUrl(result, "uncertainty_mean", 0),
      title: "Ensemble mean · +60分钟",
    },
    std: {
      element: document.querySelector("#uncertainty-std-map"),
      source: frameImageUrl(result, "uncertainty_std", 0),
      title: "Pixel-wise标准差 · 60分钟内最大",
    },
    probability: {
      element: document.querySelector("#uncertainty-probability-map"),
      source: frameImageUrl(result, "uncertainty_probability", 0),
      title: `VIL≥${result.threshold}超阈值概率 · 60分钟内最大`,
    },
  };
  Object.entries(uncertaintyImages).forEach(([key, item]) => {
    installImageFallback(item.element);
    item.element.src = item.source;
    const button = document.querySelector(`[data-uncertainty-image="${key}"]`);
    button.onclick = () => openFrameDialog(item.source, item.title);
  });

  const bars = document.querySelector("#ensemble-member-bars");
  const memberImages = document.querySelector("#ensemble-member-images");
  bars.innerHTML = "";
  memberImages.innerHTML = "";
  const members = result.member_summaries || [];
  const memberCount = Number(result.ensemble_count || members.length || 0);
  const leadSlider = document.querySelector("#ensemble-lead-slider");
  const leadValue = document.querySelector("#ensemble-lead-value");
  const leadHeading = document.querySelector("#ensemble-member-heading");
  const updateMemberLead = (frameIndex) => {
    const index = Math.max(0, Math.min(11, Number(frameIndex) || 0));
    const leadMinutes = (index + 1) * 5;
    leadValue.textContent = `+${leadMinutes}分钟`;
    leadHeading.textContent = `${memberCount || "集合"}路Ensemble成员 · +${leadMinutes}分钟`;
    memberImages.querySelectorAll(".ensemble-member-image").forEach((button, memberIndex) => {
      const image = button.querySelector("img");
      const member = members[memberIndex];
      if (!image || !member) return;
      const sourceUrl = new URL(
        frameImageUrl(result, "ensemble_member", member.member - 1, index),
        window.location.origin,
      );
      sourceUrl.searchParams.set("display_lead", String(index));
      const source = `${sourceUrl.pathname}${sourceUrl.search}`;
      image.loading = "eager";
      image.dataset.lead = String(index);
      image.src = source;
      image.alt = `Ensemble成员M${member.member}的+${leadMinutes}分钟预测图`;
      button.querySelector("span").textContent = `M${member.member} · +${leadMinutes} min`;
      button.onclick = () => openFrameDialog(source, `Ensemble成员M${member.member} · +${leadMinutes}分钟`);
    });
  };
  const maxArea = Math.max(...members.map((item) => Number(item.peak_exceeding_area_ratio || 0)), 0.000001);
  members.forEach((member) => {
    const memberButton = document.createElement("button");
    memberButton.type = "button";
    memberButton.className = "ensemble-member-image";
    const memberSource = frameImageUrl(result, "ensemble_member", member.member - 1, 11);
    memberButton.innerHTML = `<img src="${memberSource}" alt="Ensemble成员M${member.member}的384×384预测图" loading="lazy" /><span>M${member.member} · +60 min</span>`;
    installImageFallback(memberButton.querySelector("img"));
    memberButton.onclick = () => openFrameDialog(memberSource, `Ensemble成员M${member.member} · +60分钟`);
    memberImages.appendChild(memberButton);
    const row = document.createElement("div");
    row.className = "ensemble-member-row";
    const area = Number(member.peak_exceeding_area_ratio || 0);
    row.innerHTML = `<b>M${member.member}</b><div><i style="width:${Math.max(2, area / maxArea * 100)}%"></i></div><span>${(area * 100).toFixed(2)}% · ${member.active_duration_minutes} min</span>`;
    bars.appendChild(row);
  });
  leadSlider.value = "11";
  leadSlider.oninput = (event) => updateMemberLead(event.currentTarget.value);
  leadSlider.onchange = (event) => updateMemberLead(event.currentTarget.value);
  updateMemberLead(leadSlider.value);
  card.classList.remove("hidden");
}

function renderHistoricalStatistics(result) {
  const card = document.querySelector("#statistics-card");
  const bars = document.querySelector("#statistics-bars");
  const topEvents = document.querySelector("#top-events");
  bars.innerHTML = "";
  topEvents.innerHTML = "";
  let distribution = null;
  if (result.task === "historical_sevir_statistics") {
    distribution = result.counts_by_event_type || result.counts_by_month;
    document.querySelector("#statistics-title").textContent = "Storm Type记录分布";
  } else if (result.task === "sevir_catalog_search") {
    const records = result.records || [];
    records.forEach((record) => {
      const item = document.createElement("div");
      item.className = "top-event";
      const label = document.createElement("span");
      label.textContent = `${record.id} · ${record.time_utc || "时间未知"}`;
      const value = document.createElement("b");
      value.textContent = `max ${record.data_max ?? "—"}`;
      item.append(label, value);
      topEvents.appendChild(item);
    });
    document.querySelector("#statistics-title").textContent = "检索结果";
    card.classList.toggle("hidden", records.length === 0);
    return;
  }
  if (!distribution) {
    card.classList.add("hidden");
    return;
  }
  const entries = Object.entries(distribution).slice(0, 12);
  const maximum = Math.max(...entries.map(([, value]) => Number(value)), 1);
  entries.forEach(([labelText, rawValue]) => {
    const valueNumber = Number(rawValue);
    const row = document.createElement("div");
    row.className = "stat-row";
    const label = document.createElement("span");
    label.textContent = labelText;
    label.title = labelText;
    const track = document.createElement("div");
    track.className = "stat-track";
    const fill = document.createElement("div");
    fill.className = "stat-fill";
    fill.style.width = `${(valueNumber / maximum) * 100}%`;
    track.appendChild(fill);
    const value = document.createElement("b");
    value.textContent = valueNumber.toLocaleString();
    row.append(label, track, value);
    bars.appendChild(row);
  });
  (result.top_events_by_catalog_data_max || []).slice(0, 5).forEach((record) => {
    const item = document.createElement("div");
    item.className = "top-event";
    const label = document.createElement("span");
    label.textContent = `${record.id} · ${record.event_type}`;
    const value = document.createElement("b");
    value.textContent = `max ${record.data_max}`;
    item.append(label, value);
    topEvents.appendChild(item);
  });
  card.classList.remove("hidden");
}

function executionStatusLabel(status) {
  const labels = {
    ok: "完成", completed: "完成", running: "进行中", queued: "等待中",
    partial: "部分完成", error: "失败", failed: "失败", unavailable: "不可用",
    cache_hit: "使用缓存", unknown: "状态未知",
  };
  return labels[String(status || "unknown").toLowerCase()] || String(status || "状态未知");
}

function executionStageLabel(name) {
  const value = String(name || "").toLowerCase();
  if (value.includes("asset") || value.includes("context") || value.includes("request")) return "绑定地点、时间和案例数据";
  if (value.includes("model_api") || value.includes("llm")) return "调用语言模型决定下一步";
  if (value.includes("unified")) return "执行代码、读取观察并继续分析";
  if (value.includes("foundation") || value.includes("aurora")) return "分析 ERA5/Aurora 大尺度环境";
  if (value.includes("cascast") || value.includes("forecast")) return "分析 CasCast 局地短临结果";
  if (value.includes("render") || value.includes("visual")) return "整理图表";
  if (value.includes("answer") || value.includes("synth")) return "整理居民可读结论";
  return String(name || "执行分析步骤").replaceAll("_", " ");
}

function routeDisplayLabel(route) {
  const value = String(route || "").toLowerCase();
  if (value.includes("unified") || value.includes("completed")) return "统一 Agent";
  return route ? "Agent 执行" : "—";
}

function toolDisplaySentence(run, index = 0) {
  const name = displayToolName(run?.tool_name || run?.api || "分析工具");
  const status = executionStatusLabel(run?.status);
  const elapsed = Number(run?.elapsed_ms);
  const cache = run?.cache_hit === true ? "，复用缓存" : "";
  const argumentsText = readableToolArguments(run?.arguments);
  const resultText = String(run?.result_summary || "").trim();
  return `${index + 1}. ${name}${argumentsText ? `（${argumentsText}）` : ""}：${status}${cache}${Number.isFinite(elapsed) ? `，耗时 ${(elapsed / 1000).toFixed(elapsed >= 1000 ? 1 : 2)} 秒` : ""}${resultText ? `；${resultText}` : ""}`;
}

function readableToolArguments(argumentsValue) {
  if (!argumentsValue || typeof argumentsValue !== "object" || Array.isArray(argumentsValue)) return "";
  const labels = {
    event_id: "事件", upload_id: "上传资料", threshold: "阈值",
    lead_minutes: "预报时效", start_frame: "起始帧", end_frame: "结束帧",
    place: "地点", latitude: "纬度", longitude: "经度",
  };
  const parts = [];
  Object.entries(argumentsValue).slice(0, 5).forEach(([key, value]) => {
    if (value == null || value === "" || typeof value === "object") return;
    const suffix = key === "lead_minutes" ? "分钟" : "";
    parts.push(`${labels[key] || key.replaceAll("_", " ")} ${value}${suffix}`);
  });
  return parts.join("，");
}

function renderRunDiagnostic(data = {}) {
  const panel = document.querySelector("#run-diagnostic");
  if (!panel) return;
  const statusNode = document.querySelector("#diagnostic-status");
  const grid = document.querySelector("#diagnostic-grid");
  const stagesNode = document.querySelector("#diagnostic-stages");
  const requestTrace = data.request_trace || null;
  const diagnostic = data.diagnostic || null;
  const eventResolution = data.event_resolution || data.trace?.event_resolution || null;
  const preflight = data.foundation_preflight || data.trace?.foundation_preflight || null;
  const runs = Array.isArray(data.tool_runs) ? data.tool_runs : [];
  const hasDetails = Boolean(
    requestTrace || diagnostic || preflight || runs.length || data.trace_file,
  );
  panel.classList.toggle("hidden", !hasDetails);
  if (!hasDetails) return;
  const status = String(data.status || requestTrace?.status || "ok");
  statusNode.textContent = status === "ok" ? "已完成" : (status === "partial" ? "部分完成" : "未完成");
  statusNode.dataset.status = status;
  grid.replaceChildren();
  const rows = [
    ["执行路径", routeDisplayLabel(data.route_mode || data.termination_reason)],
    ["状态", executionStatusLabel(status)],
    ["语言模型调用", `${data.model_calls ?? requestTrace?.model_calls ?? 0} 次`],
    ["工具调用", `${data.tool_calls ?? requestTrace?.tool_calls ?? runs.length} 次`],
    ["缓存命中", `${data.cache_hits ?? requestTrace?.cache_hits ?? 0} 次`],
    ["失败阶段", diagnostic?.phase || requestTrace?.failure_class ? executionStageLabel(diagnostic?.phase || "request") : "—"],
    ["事件绑定", eventResolution
      ? `${eventResolution.source || "unknown"} · ${eventResolution.event_id || "未匹配"}`
      : "—"],
    ["总耗时", data.elapsed_ms != null ? `${(Number(data.elapsed_ms) / 1000).toFixed(2)} 秒` : (requestTrace?.elapsed_ms != null ? `${(Number(requestTrace.elapsed_ms) / 1000).toFixed(2)} 秒` : "—")],
  ];
  rows.forEach(([label, value]) => {
    const item = document.createElement("div");
    const key = document.createElement("small");
    const text = document.createElement("span");
    key.textContent = label;
    text.textContent = String(value);
    item.append(key, text);
    grid.appendChild(item);
  });
  stagesNode.replaceChildren();
  const stages = Array.isArray(requestTrace?.stages) ? requestTrace.stages : [];
  const agentSteps = Array.isArray(data.agent_trace) ? data.agent_trace : [];
  stagesNode.classList.toggle("hidden", agentSteps.length > 0);
  if (agentSteps.length > 0) return;
  if (stages.length) {
    stages.forEach((stage) => {
      const item = document.createElement("li");
      item.className = `diagnostic-stage ${stage.status || "unknown"}`;
      const elapsed = Number(stage.elapsed_ms);
      const modelText = stage.model
        ? `，使用 ${stage.provider ? `${stage.provider} / ` : ""}${stage.model}`
        : "";
      item.textContent = `${executionStageLabel(stage.name)}：${executionStatusLabel(stage.status)}${Number.isFinite(elapsed) ? `，耗时 ${(elapsed / 1000).toFixed(elapsed >= 1000 ? 1 : 2)} 秒` : ""}${stage.route_mode ? `，选择 ${stage.route_mode} 模式` : ""}${modelText}`;
      stagesNode.appendChild(item);
    });
  } else if (runs.length) {
    runs.forEach((run) => {
      const item = document.createElement("li");
      item.className = `diagnostic-stage ${run.status || "unknown"}`;
      item.textContent = toolDisplaySentence(run, runs.indexOf(run));
      stagesNode.appendChild(item);
    });
  } else {
    const item = document.createElement("li");
    item.className = "diagnostic-stage unknown";
    item.textContent = diagnostic?.next_action || "没有进入代码或外部工具阶段；请先检查输入和事件绑定。";
    stagesNode.appendChild(item);
  }
}
function renderResult(data, anchorMessage = null) {
  const anchoredToChatMessage = Boolean(
    anchorMessage
      && anchorMessage.classList.contains("assistant")
      && chatThread.contains(anchorMessage),
  );
  emptyState.classList.add("hidden");
  errorBox.classList.add("hidden");
  resultContent.classList.remove("hidden");
  // `turnDetails` is a reusable rendering template.  During normal chat it
  // must never enter the visible conversation flow: only its clone belongs to
  // the exact assistant message supplied as `anchorMessage`.
  turnDetails.classList.toggle("hidden", anchoredToChatMessage);
  turnDetails.setAttribute("aria-hidden", anchoredToChatMessage ? "true" : "false");
  runState.textContent = "已完成";
  runState.className = "run-state done";

  const runs = Array.isArray(data.tool_runs) ? data.tool_runs : [];
  const displayResult = combinedDisplayResult(data, runs);
  displayResult.presentation = data.presentation || displayResult.presentation || {};
  renderAnswer(data.answer || "未返回文字结论");
  document.querySelector("#result-title").textContent = resultTitle(displayResult);
  document.querySelector("#result-source").textContent = resultSource(displayResult);
  turnDetailsSource.textContent = resultSource(displayResult);
  document.querySelector("#tool-name").textContent = runs.length > 1
    ? runs.map((run) => displayToolName(run.tool_name || run.api)).join(" → ")
    : displayToolName(runs[0]?.tool_name || runs[0]?.api || data.tool_name || "未调用外部工具");
  document.querySelector("#meta-time").textContent = data.elapsed_ms != null
    ? `${data.elapsed_ms} ms`
    : (data.request_trace?.elapsed_ms != null ? `${data.request_trace.elapsed_ms} ms` : "未记录");
  renderRunDiagnostic(data);
  renderAgentTrace(data);
  renderArtifacts({
    ...displayResult,
    artifacts: Array.isArray(data.artifacts) && data.artifacts.length
      ? data.artifacts
      : (displayResult.artifacts || []),
  });
  renderEvaluation(displayResult);
  renderDatasetEvaluation(displayResult);
  renderHistoricalStatistics(displayResult);
  renderForecastSequence(displayResult);
  renderEnsembleUncertainty(displayResult);
  renderHazardOverview(displayResult);
  renderConversationVisualPane(data);

  if (anchoredToChatMessage) attachTurnPresentation(anchorMessage, data, runs);
}

function cloneTurnNode(node) {
  const clone = node.cloneNode(true);
  clone.removeAttribute("id");
  clone.querySelectorAll("[id]").forEach((item) => item.removeAttribute("id"));
  clone.querySelectorAll("img").forEach((image) => {
    image.loading = "lazy";
    image.addEventListener("click", () => {
      if (image.src) window.open(image.src, "_blank", "noopener");
    });
  });
  return clone;
}

function attachTurnPresentation(anchorMessage, data, runs) {
  const body = anchorMessage.querySelector(".chat-message-body");
  if (!body) return;
  let presentation = body.querySelector(":scope > .chat-turn-presentation");
  if (!presentation) {
    presentation = document.createElement("div");
    presentation.className = "chat-turn-presentation";
    body.appendChild(presentation);
  }
  presentation.dataset.ownerMessageId = anchorMessage.dataset.turnMessageId || "legacy";
  presentation.querySelectorAll(":scope > .chat-turn-visuals, :scope > .chat-turn-technical").forEach((item) => item.remove());
  // Remove clones from the older layout if a persisted message is upgraded in
  // place. This prevents one turn from retaining two technical-detail blocks.
  body.querySelectorAll(":scope > .chat-turn-visuals, :scope > .chat-turn-technical").forEach((item) => item.remove());

  const visualIds = [
    "evaluation-card",
    "dataset-evaluation-card",
    "forecast-sequence-card",
    "uncertainty-card",
    "statistics-card",
    "visual-card",
    "agent-artifacts",
  ];
  const visibleVisuals = visualIds
    .map((id) => document.getElementById(id))
    .filter((node) => node && !node.classList.contains("hidden"));
  if (visibleVisuals.length) {
    const visuals = document.createElement("section");
    visuals.className = "chat-turn-visuals";
    visibleVisuals.forEach((node) => visuals.appendChild(cloneTurnNode(node)));
    presentation.appendChild(visuals);
  }

  const sourceTrace = turnDetails.querySelector(".trace-card");
  if (sourceTrace) {
    const technical = cloneTurnNode(sourceTrace);
    technical.classList.add("chat-turn-technical");
    technical.dataset.ownerMessageId = presentation.dataset.ownerMessageId;
    technical.open = false;
    const summaryLabel = technical.querySelector(":scope > summary span");
    if (summaryLabel) {
      const modelCalls = Number.isFinite(Number(data.model_calls)) ? Number(data.model_calls) : 0;
      const codeRuns = Number.isFinite(Number(data.code_execution_count)) ? Number(data.code_execution_count) : 0;
      summaryLabel.textContent = `本轮执行记录 · 模型 ${modelCalls} 次 · 代码 ${codeRuns} 次 · 工具 ${runs.length} 次`;
    }
    const content = technical.querySelector(".trace-content");
    const agentAnalysis = technical.querySelector("#agent-analysis, .agent-analysis");
    const artifactSection = technical.querySelector("#agent-artifacts, .agent-artifacts");
    if (artifactSection) artifactSection.remove();
    if (content && agentAnalysis && !agentAnalysis.classList.contains("hidden")) {
      agentAnalysis.classList.add("agent-analysis");
      content.insertBefore(agentAnalysis, content.firstChild);
    }
    presentation.appendChild(technical);
  }

  turnDetails.classList.add("hidden");
  turnDetails.setAttribute("aria-hidden", "true");
  turnDetails.open = false;
}

function combinedDisplayResult(data, runs) {
  const results = runs
    .map((run) => run?.tool_result)
    .filter((result) => result && typeof result === "object" && !Array.isArray(result));
  if (results.length === 0) return data.tool_result || {};

  // A later verification or chart call must not hide forecast imagery produced
  // by an earlier call. Components still decide whether to display from the
  // presence of their own fields, so metadata-only questions remain compact.
  const merged = Object.assign({}, ...results);
  const visualResult = [...results].reverse().find((result) => result.full_resolution_viewer);
  if (visualResult) {
    ["task", "event_id", "model_name", "data_source", "full_resolution_viewer"].forEach((key) => {
      if (visualResult[key] !== undefined) merged[key] = visualResult[key];
    });
  }
  return merged;
}

function renderAgentTrace(data) {
  const container = document.querySelector("#agent-analysis");
  const trace = document.querySelector("#agent-trace");
  const executionMode = document.querySelector("#agent-execution-mode");
  const executionFlow = document.querySelector("#agent-execution-flow");
  const sessionBudget = document.querySelector("#agent-session-budget");
  trace.innerHTML = "";
  const attempts = Array.isArray(data.agent_trace) ? data.agent_trace : [];
  container.classList.toggle("hidden", attempts.length === 0);
  if (executionMode) executionMode.textContent = "统一 Agent";
  if (executionFlow) {
    executionFlow.textContent = "生成代码 → 执行 → 观察 → 继续或回答";
  }
  if (sessionBudget) {
    const reason = data.termination_reason || data.route_mode || "unknown";
    sessionBudget.textContent = `${data.session_id || sessionId} · ${routeDisplayLabel(reason)} · 模型 ${data.model_calls ?? "—"} 次 · 执行 ${data.code_execution_count ?? "—"} 次`;
  }
  attempts.forEach((item) => {
    const block = document.createElement("details");
    block.className = "agent-attempt";
    const summary = document.createElement("summary");
    const actionLabels = {
      execute: "执行分析代码", finish: "整理最终回答", model: "调用语言模型",
      format_repair: "修正模型输出格式", stop: "停止本轮",
    };
    const elapsed = Number(item.elapsed_ms);
    const elapsedText = Number.isFinite(elapsed) ? ` · ${(elapsed / 1000).toFixed(elapsed >= 1000 ? 1 : 2)} 秒` : "";
    summary.textContent = `第 ${item.attempt} 步：${actionLabels[item.action] || "检查执行结果"} · ${executionStatusLabel(item.status)}${elapsedText}`;
    block.appendChild(summary);
    if (item.summary) {
      const activity = document.createElement("p");
      activity.className = "agent-decision";
      activity.textContent = item.summary;
      block.appendChild(activity);
    }
    if (item.decision) {
      const decision = document.createElement("p");
      decision.className = "agent-decision";
      decision.textContent = item.decision;
      block.appendChild(decision);
    }
    if (item.error) {
      const error = document.createElement("p");
      error.className = "agent-error";
      error.textContent = `这一步没有完成：${item.error}`;
      block.appendChild(error);
    }
    if (Array.isArray(item.calls) && item.calls.length) {
      const calls = document.createElement("ol");
      calls.className = "agent-calls";
      item.calls.forEach((call, index) => {
        const callNode = document.createElement("li");
        callNode.textContent = toolDisplaySentence(call, index);
        calls.appendChild(callNode);
      });
      block.appendChild(calls);
    }
    trace.appendChild(block);
  });
}

function renderArtifacts(data) {
  const container = document.querySelector("#agent-artifacts");
  const list = document.querySelector("#artifact-list");
  list.innerHTML = "";
  const artifactIds = new Set();
  const artifacts = (Array.isArray(data.artifacts) ? data.artifacts : []).filter((item) => {
    if (!item || !(item.url || item.image_url) || item.verified === false) return false;
    const identity = item.artifact_id || item.url || item.image_url;
    if (artifactIds.has(identity)) return false;
    artifactIds.add(identity);
    return true;
  });
  container.classList.toggle("hidden", artifacts.length === 0);
  artifacts.forEach((artifact) => {
    const figure = document.createElement("figure");
    figure.className = "agent-artifact";
    const image = document.createElement("img");
    image.src = artifact.thumbnail?.url || artifact.url || artifact.image_url;
    image.alt = artifact.title || artifact.chart_type || "Agent生成图表";
    image.loading = "lazy";
    const caption = document.createElement("figcaption");
    caption.textContent = artifact.description
      ? `${artifact.title || artifact.chart_type} — ${artifact.description}`
      : (artifact.title || artifact.chart_type || "Agent生成图表");
    image.addEventListener("error", () => {
      figure.classList.add("artifact-load-error");
      caption.textContent = `图片加载失败：${artifact.title || artifact.chart_type || artifact.artifact_id}`;
    });
    const link = document.createElement("a");
    link.href = artifact.url || artifact.image_url;
    link.target = "_blank";
    link.rel = "noopener";
    link.appendChild(image);
    figure.append(link, caption);
    list.appendChild(figure);
  });
}

function renderForecastSequence(result) {
  const card = document.querySelector("#forecast-sequence-card");
  const container = document.querySelector("#forecast-sequence");
  const historyContainer = document.querySelector("#history-sequence");
  const truthContainer = document.querySelector("#ground-truth-sequence");
  const truthHeading = document.querySelector("#ground-truth-heading");
  const requestedLead = Number(result.presentation?.requested_lead_minutes || result.lead_minutes || 60);
  const frames = Array.isArray(result.frame_summaries)
    ? result.frame_summaries.filter((frame) => Number(frame.lead_minutes) <= requestedLead)
    : [];
  container.innerHTML = "";
  historyContainer.innerHTML = "";
  truthContainer.innerHTML = "";
  truthContainer.classList.add("hidden");
  truthHeading.classList.add("hidden");
  if (!result.full_resolution_viewer || !Array.isArray(frames) || frames.length === 0) {
    card.classList.add("hidden");
    return;
  }
  const historyFrames = result.history_frame_summaries || [];
  historyFrames.forEach((frame, index) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "history-frame";
    const minute = Number(frame.relative_minutes);
    item.innerHTML = `<img src="${frameImageUrl(result, "history", index)}" alt="历史观测${minute}分钟" loading="lazy" /><strong>${minute === 0 ? "T0" : `${minute} min`}</strong><span>max ${Math.round(frame.max_vil)}</span>`;
    item.addEventListener("click", () => {
      selectFullFrame(result, "history", index, `历史观测 ${minute === 0 ? "当前时刻" : `${minute}分钟`}`, item);
    });
    historyContainer.appendChild(item);
  });
  const maximumArea = Math.max(...frames.map((frame) => Number(frame.exceeding_area_ratio ?? frame.strong_echo_area_ratio_at_vil40 ?? 0)), 0.000001);
  const areaThreshold = frames[0]?.area_threshold;
  document.querySelector("#sequence-threshold-label").textContent = `阈值以上面积占比（VIL≥${areaThreshold}）`;
  frames.forEach((frame, index) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `forecast-frame${index === frames.length - 1 ? " selected" : ""}`;
    const areaRatio = Number(frame.exceeding_area_ratio ?? frame.strong_echo_area_ratio_at_vil40 ?? 0);
    item.innerHTML = `<img src="${frameImageUrl(result, "forecast", index)}" alt="预测+${frame.lead_minutes}分钟VIL" loading="lazy" /><strong>+${frame.lead_minutes} min</strong><span>max ${Math.round(frame.max_vil)}</span><div><i style="height:${Math.max(5, areaRatio / maximumArea * 100)}%"></i></div><small>${(areaRatio * 100).toFixed(1)}%</small>`;
    item.addEventListener("click", () => {
      selectFullFrame(result, "forecast", index, `预测 +${frame.lead_minutes}分钟`, item);
    });
    container.appendChild(item);
  });
  const truthFrames = result.presentation?.show_ground_truth === true
    ? (result.ground_truth_frame_summaries || []).filter((frame) => Number(frame.lead_minutes) <= requestedLead)
    : [];
  truthFrames.forEach((frame, index) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "ground-truth-frame";
    item.innerHTML = `<img src="${frameImageUrl(result, "ground_truth", index)}" alt="真实未来+${frame.lead_minutes}分钟VIL" loading="lazy" /><strong>+${frame.lead_minutes} min</strong><span>max ${Math.round(frame.max_vil)}</span>`;
    item.addEventListener("click", () => {
      selectFullFrame(result, "ground_truth", index, `真实未来 +${frame.lead_minutes}分钟`, item);
    });
    truthContainer.appendChild(item);
  });
  if (truthFrames.length) {
    truthContainer.classList.remove("hidden");
    truthHeading.classList.remove("hidden");
  }
  const forecastHeading = document.querySelector("#forecast-sequence-heading");
  if (forecastHeading) {
    const outputLabel = result.output_type === "ensemble_mean"
      ? "集合平均预测"
      : result.output_type === "single_stochastic_member"
        ? "单成员预测"
        : "模型预测原图";
    forecastHeading.textContent = `${outputLabel} · ${frames.length}帧`;
  }
  const truthHeadingText = truthHeading.querySelector("strong");
  if (truthHeadingText) truthHeadingText.textContent = `真实未来 · ${truthFrames.length}帧`;
  const legendHorizon = document.querySelector("#sequence-horizon-label");
  if (legendHorizon) legendHorizon.textContent = `未来${requestedLead}分钟`;
  const lastIndex = frames.length - 1;
  selectFullFrame(
    result,
    "forecast",
    lastIndex,
    `预测 +${frames[lastIndex].lead_minutes}分钟`,
    container.lastElementChild,
  );
  card.classList.remove("hidden");
}

function resultTitle(result) {
  const titles = {
    future_hazard_analysis: "未来强回波与Hazard分析",
    nowcast: "CasCast未来回波预测",
    uploaded_sevir_vil_analysis: "上传VIL图像分析",
    uploaded_sevir_vil_sequence_analysis: "上传VIL序列演变分析",
    historical_sevir_statistics: "SEVIR历史统计",
    sevir_catalog_search: "SEVIR事件检索",
    sevir_event_geolocation: "事件空间范围",
    event_history_comparison: "事件历史强度对比",
    ensemble_uncertainty_analysis: "CasCast Ensemble不确定性",
    cascast_dataset_evaluation: "CasCast 10事件整体评价",
  };
  if (["cascast", "cascast_live"].includes(result.model_name) && result.forecast_frames) {
    const lead = Number(result.presentation?.requested_lead_minutes || result.lead_minutes || 60);
    return `CasCast未来${lead}分钟回波预测`;
  }
  return titles[result.task] || "分析结论";
}

function resultSource(result) {
  if (result.data_source?.kind === "cascast_local_live_inference") return "CASCAST · LOCAL LIVE INFERENCE";
  if (result.data_source?.kind === "cascast_official_test_prediction" || result.data_source === "cascast_saved_inference") return "CASCAST · FORECAST EVIDENCE";
  if (result.task === "uploaded_vil_analysis") return "USER INPUT · VIL";
  if (result.task === "uploaded_sevir_vil_sequence_analysis") return "USER INPUT · 13 VIL FRAMES";
  if (["historical_sevir_statistics", "sevir_catalog_search", "sevir_event_geolocation", "event_history_comparison"].includes(result.task)) return "SEVIR CATALOG";
  if (result.task === "cascast_dataset_evaluation") return "CASCAST · 10 EVENTS + GT";
  return "SEVIR TOOLBOX";
}

function renderAnswer(text) {
  const container = document.querySelector("#answer");
  renderMarkdown(container, text);
}

function renderMarkdown(container, text) {
  container.innerHTML = "";
  const rawLines = String(text ?? "").replace(/\r/g, "").split("\n");
  // Models often wrap ordinary Chinese prose at every sentence. Treat a
  // single newline inside prose as a transport wrap, not a new paragraph;
  // preserve blank lines and Markdown structures as real boundaries.
  const lines = [];
  let inFence = false;
  for (const rawLine of rawLines) {
    const current = rawLine.trim();
    if (current.startsWith("```")) {
      lines.push(rawLine);
      inFence = !inFence;
      continue;
    }
    if (inFence) {
      lines.push(rawLine);
      continue;
    }
    if (current === "|" || current === "｜") continue;
    const previous = lines.length ? lines[lines.length - 1] : "";
    const structural = /^(#{1,6}\s|[-*+]\s|\d+[.)]\s|\||```)/.test(current);
    const previousStructural = /^(#{1,6}\s|[-*+]\s|\d+[.)]\s|\||```)/.test(previous);
    if (current && previous && !structural && !previousStructural) {
      const separator = /[A-Za-z0-9]$/.test(previous) && /^[A-Za-z0-9]/.test(current) ? " " : "";
      lines[lines.length - 1] = previous + separator + current;
    } else {
      lines.push(rawLine);
    }
  }
  // Model responses and restored messages may carry transport newlines.  Drop
  // only blank lines at the two edges; blank lines inside Markdown still
  // delimit paragraphs and lists.
  while (lines.length && !lines[0].trim()) lines.shift();
  while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
  const clean = (value) => value.trim();
  const renderInline = (parent, value) => {
    const pattern = /(\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|__([^_]+)__|~~([^~]+)~~|\*([^*]+)\*|_([^_]+)_)/g;
    let cursor = 0;
    let match;
    const appendText = (text) => { if (text) parent.appendChild(document.createTextNode(text)); };
    while ((match = pattern.exec(String(value))) !== null) {
      appendText(String(value).slice(cursor, match.index));
      if (match[2]) {
        const link = document.createElement("a");
        link.href = match[3];
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = match[2];
        parent.appendChild(link);
      } else if (match[4]) {
        const code = document.createElement("code");
        code.textContent = match[4];
        parent.appendChild(code);
      } else {
        const element = document.createElement(match[5] || match[6] ? "strong" : match[7] ? "del" : "em");
        element.textContent = match[5] || match[6] || match[7] || match[8] || match[9];
        parent.appendChild(element);
      }
      cursor = match.index + match[0].length;
    }
    appendText(String(value).slice(cursor));
  };
  const tableCells = (line) => {
    let value = line.trim();
    if (value.startsWith("|")) value = value.slice(1);
    if (value.endsWith("|")) value = value.slice(0, -1);
    return value.split("|").map(clean);
  };
  const isTableDivider = (line) => {
    const cells = tableCells(line);
    return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s/g, "")));
  };
  let index = 0;
  let list = null;
  let codeBlock = null;
  const closeList = () => { list = null; };
  while (index < lines.length) {
    const raw = lines[index];
    const trimmed = raw.trim();
    if (trimmed.startsWith("```")) {
      closeList();
      if (!codeBlock) {
        codeBlock = document.createElement("pre");
        codeBlock.className = "answer-code";
        container.appendChild(codeBlock);
      } else {
        codeBlock = null;
      }
      index += 1;
      continue;
    }
    if (codeBlock) {
      codeBlock.textContent += `${raw}\n`;
      index += 1;
      continue;
    }
    if (!trimmed) {
      closeList();
      index += 1;
      continue;
    }
    if (trimmed.includes("|") && index + 1 < lines.length && isTableDivider(lines[index + 1])) {
      closeList();
      const wrapper = document.createElement("div");
      wrapper.className = "answer-table-wrap";
      const table = document.createElement("table");
      table.className = "answer-table";
      const head = document.createElement("thead");
      const headRow = document.createElement("tr");
      tableCells(trimmed).forEach((cell) => {
        const th = document.createElement("th");
        th.textContent = cell;
        headRow.appendChild(th);
      });
      head.appendChild(headRow);
      table.appendChild(head);
      const body = document.createElement("tbody");
      index += 2;
      const expectedColumns = tableCells(trimmed).length;
      while (
        index < lines.length
        && lines[index].includes("|")
        && tableCells(lines[index]).length === expectedColumns
      ) {
        const row = document.createElement("tr");
        tableCells(lines[index]).forEach((cell) => {
          const td = document.createElement("td");
          td.textContent = cell;
          row.appendChild(td);
        });
        body.appendChild(row);
        index += 1;
      }
      table.appendChild(body);
      wrapper.appendChild(table);
      container.appendChild(wrapper);
      continue;
    }
    if (/^>\s+/.test(trimmed)) {
      closeList();
      const quote = document.createElement("blockquote");
      renderInline(quote, trimmed.replace(/^>\s+/, ""));
      container.appendChild(quote);
      index += 1;
      continue;
    }
    if (/^#{1,6}\s+/.test(trimmed)) {
      closeList();
      const heading = document.createElement("h4");
      renderInline(heading, clean(trimmed.replace(/^#{1,6}\s+/, "")));
      container.appendChild(heading);
      index += 1;
      continue;
    }
    if (/^\d+[.)]\s+/.test(trimmed) || /^[-*+]\s+/.test(trimmed)) {
      const ordered = /^\d+[.)]\s+/.test(trimmed);
      if (!list) {
        list = document.createElement(ordered ? "ol" : "ul");
        container.appendChild(list);
      }
      const item = document.createElement("li");
      renderInline(item, clean(trimmed.replace(ordered ? /^\d+[.)]\s+/ : /^[-*+]\s+/, "")));
      list.appendChild(item);
      index += 1;
      continue;
    }
    closeList();
    const paragraph = document.createElement("p");
    renderInline(paragraph, clean(trimmed));
    container.appendChild(paragraph);
    index += 1;
  }
}

function renderError(message) {
  emptyState.classList.add("hidden");
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  runState.textContent = "出错";
  runState.className = "run-state";
}

taskInput.addEventListener("change", syncTaskControls);
dataSourceInput.addEventListener("change", syncDataSource);
[weatherModelInput, leadInput, eventInput, thresholdInput].forEach((element) => {
  element.addEventListener("change", syncTaskControls);
});
modeInput.addEventListener("change", syncMode);
weatherModelInput.addEventListener("change", () => {
  syncWeatherModelNote();
  syncEventOptions();
});
document.querySelector("#zoom-frame").addEventListener("click", () => {
  if (!fullVilFrame.src) return;
  dialogFrameImage.src = fullVilFrame.src;
  dialogFrameTitle.textContent = fullVilFrame.dataset.title || "VIL完整帧";
  frameDialog.showModal();
});
document.querySelector("#close-frame-dialog").addEventListener("click", () => frameDialog.close());
frameDialog.addEventListener("click", (event) => {
  if (event.target === frameDialog) frameDialog.close();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  const liveContextReady = hasLiveContext();
  const timestamp = Date.now();
  const turnId = createTurnId();
  const requestId = createRequestId();
  // Bind the current upload by ID as well as the pending flag.  The flag is
  // only UI state and can be cleared by a re-render; the upload IDs are the
  // authoritative proof that this question has an image attached.
  const shouldAttachUpload = attachmentPendingForMessage
    || Boolean(pendingAttachmentPreview)
    || Boolean(uploadPreview.src);
  const attachment = shouldAttachUpload ? {
    kind: uploadedSequenceId ? "sequence" : "single",
    title: attachmentTitle.textContent,
    preview: pendingAttachmentPreview || (uploadedSampleId
      ? `/api/uploaded-frame?upload_id=${encodeURIComponent(uploadedSampleId)}`
      : ""),
    fallback: uploadedSequenceId
      ? `/api/uploaded-sequence-frame?upload_id=${encodeURIComponent(uploadedSequenceId)}&series=history&index=12`
      : (uploadedSampleId ? `/api/uploaded-frame?upload_id=${encodeURIComponent(uploadedSampleId)}` : ""),
  } : null;
  appendChatMessage("user", question, { timestamp, attachment, turnId });
  chatHistory.push({ role: "user", text: question, attachment, timestamp, turnId });
  attachmentPendingForMessage = false;
  pendingAttachmentPreview = "";
  persistChatHistory();
  persistConversationIndex();
  renderConversationList();
  questionInput.value = "";
  questionInput.style.height = "";
  const pendingMessage = appendChatMessage("assistant", "", { pending: true, timestamp: Date.now(), turnId });

  requestInFlight = true;
  syncComposerAvailability();
  runButton.firstElementChild.textContent = "分析中";
  runState.textContent = "分析中";
  runState.className = "run-state running";
  errorBox.classList.add("hidden");
  conversationStatus.textContent = "Agent 正在结合当前对话与工具观察分析…";
  const progressState = { done: false };
  const progressPromise = pollRequestProgress(requestId, progressState);
  const requestController = new AbortController();
  // The server owns the 30-second hard budget.  Give it one second to return
  // the terminal trace, then close the HTTP request so cancellation propagates
  // instead of leaving an invisible background Agent running.
  const requestTimeout = window.setTimeout(
    () => requestController.abort("Agent request exceeded 31 seconds"),
    31_000,
  );

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      signal: requestController.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: modeInput.value,
        task: taskInput.value,
        data_source: uploadedSequenceId ? "upload_sequence" : (uploadedSampleId ? "upload" : (liveContextReady ? "live" : "none")),
        data_policy: liveContextReady ? "live" : "cached",
        analysis_time_utc: normalizedAnalysisTimeUtc(),
        place_name: analysisPlaceInput?.value.trim() || "",
        latitude: optionalCoordinate(analysisLatitudeInput),
        longitude: optionalCoordinate(analysisLongitudeInput),
        forecast_leads: [6, 12, 18, 24],
        weather_model: weatherModelInput.value,
        lead_minutes: Number(leadInput.value),
        event_id: eventInput?.value.trim() || "",
        upload_id: uploadedSampleId,
        sequence_upload_id: uploadedSequenceId,
        session_id: sessionId,
        turn_id: turnId,
        request_id: requestId,
        // The technical panel is collapsed by default, but every UI request
        // keeps an auditable operational trace for diagnosis and replay.
        trace: true,
        question,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.answer || data.error || "运行失败");
    const answer = data.answer || "本轮分析完成，但没有返回文字结论。";
    const meta = compactTurnMeta(data);
    completePendingMessage(pendingMessage, answer, meta, data.status === "error");
    chatHistory.push({
      role: "assistant",
      text: answer,
      meta,
      result: persistedResult(data),
      timestamp: Date.now(),
      turnId,
    });
    persistChatHistory();
    persistConversationIndex();
    // The Agent request is already complete at this point. Keep presentation
    // failures local so a DOM bug cannot be reported as an Agent/API failure.
    try {
      await renderConversationList();
      renderResult(data, pendingMessage);
      conversationStatus.textContent = `当前对话 · ${data.session_turn_count ?? Math.ceil(chatHistory.length / 2)}轮 · 可继续追问`;
    } catch (displayError) {
      console.error("Agent回答已返回，但页面展示失败：", displayError);
      renderError(`Agent回答已成功返回，但部分页面展示失败：${displayError.message}`);
      conversationStatus.textContent = "Agent回答已返回；部分执行记录或图表未能显示。";
    }
  } catch (error) {
    const aborted = error?.name === "AbortError" || requestController.signal.aborted;
    if (/upload_id|上传|图片|image/i.test(String(error.message || ""))) {
      clearUploadedInput({ forget: true });
      uploadStatus.textContent = "图片缓存已失效，请重新上传后再分析。";
      attachmentTitle.textContent = "需要重新上传图片";
    }
    const answer = aborted
      ? "本轮请求已在30秒预算后停止，系统不会继续后台分析。请查看进度记录定位耗时步骤。"
      : `本轮请求未完成：${error.message}`;
    completePendingMessage(pendingMessage, answer, { route: "request_failed", sessionId }, true);
    chatHistory.push({ role: "assistant", text: answer, meta: { route: "request_failed" }, timestamp: Date.now(), turnId });
    persistChatHistory();
    renderError(aborted ? "请求超过30秒总预算并已取消。" : error.message);
    conversationStatus.textContent = "请求失败；你可以修改问题后继续当前对话。";
  } finally {
    window.clearTimeout(requestTimeout);
    progressState.done = true;
    await progressPromise.catch(() => {});
    requestInFlight = false;
    syncComposerAvailability();
    runButton.firstElementChild.textContent = "发送";
    questionInput.focus();
  }
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 180)}px`;
  syncComposerAvailability();
});

newChatButton.addEventListener("click", () => {
  if (requestInFlight) return;
  persistConversationIndex();
  sessionId = createSessionId();
  writePersistentSessionId(sessionId);
  resetConversationView();
  persistChatHistory();
  persistConversationIndex();
  // Keep the reusable result subtree in the document while clearing the
  // conversation; renderResult() relies on document.querySelector() to fill
  // its fields on the next turn.
  if (turnDetails.parentNode === chatThread) {
    errorBox.parentNode.insertBefore(turnDetails, errorBox);
  }
  chatThread.innerHTML = "";
  emptyState.classList.remove("hidden");
  resultContent.classList.add("hidden");
  turnDetails.classList.add("hidden");
  turnDetails.open = false;
  errorBox.classList.add("hidden");
  runState.textContent = "就绪";
  runState.className = "run-state";
  conversationStatus.textContent = "新对话已建立；后续追问会在这个会话中保留上下文。";
  questionInput.value = "";
  clearUploadedInput();
  renderConversationList();
  questionInput.focus();
});

dataSourceInput.value = "upload";
weatherModelInput.value = "cascast_live";
modeInput.value = "unified";
syncTaskControls();
syncDataSource();
syncMode();
// Draw the server-side conversation list immediately. Restoring a malformed
// legacy turn must never hide the saved-session menu.
void renderConversationList();
void restoreChatHistory()
  .catch((error) => console.warn("会话恢复失败，保留会话列表：", error))
  .finally(() => {
    try { persistConversationIndex(); } catch (_error) {}
    void renderConversationList();
  });
loadConfig();
restoreActiveUpload();
