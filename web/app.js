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
const generateVisualInput = document.querySelector("#generate-visual");
const generateVisualState = document.querySelector("#generate-visual-state");
const runButton = document.querySelector("#run-button");
const cancelRequestButton = document.querySelector("#cancel-request");
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
const conversationLayerTabs = document.querySelector("#conversation-layer-tabs");
const conversationVisualKind = document.querySelector("#conversation-visual-kind");
const visualContextPlace = document.querySelector("#visual-context-place");
const visualContextTime = document.querySelector("#visual-context-time");
const visualContextStatus = document.querySelector("#visual-context-status");
const visualTimelineLabel = document.querySelector("#visual-timeline-label");
const visualTimeline = document.querySelector("#visual-timeline");
const visualTimelineTitle = document.querySelector("#visual-timeline-title");
const visualTimelineScale = document.querySelector("#visual-timeline-scale");
const visualTimelineNow = document.querySelector("#visual-timeline-now");
const visualTimelineMarker = document.querySelector("#visual-timeline-marker");
const visualTimelineSlider = document.querySelector("#visual-timeline-slider");
const visualEvidenceLabel = document.querySelector("#visual-evidence-label");

const emptyState = document.querySelector("#empty-state");
const resultContent = document.querySelector("#result-content");
const errorBox = document.querySelector("#error-box");

let runtimeConfig = {};
let uploadedSampleId = "";
let uploadedSequenceId = "";
let requestInFlight = false;
let activeRequestController = null;
let activeRequestId = "";
let activeRequestPhase = "idle";
let activeRequestTransport = "none";
let attachmentPendingForMessage = false;
let pendingAttachmentPreview = "";
let conversationVisualArtifacts = [];
let selectedConversationVisual = "";
let activeConversationLayer = "all";
let visualPaneResizeFrame = 0;
const sessionStorageKey = "sevir-upload-chat-session-id";
const chatStoragePrefix = "sevir-agent-chat-history:";
const activeUploadPrefix = "sevir-active-upload:";
const activeTaskPrefix = "sevir-active-task:";
const conversationIndexKey = "sevir-agent-conversations";
// The UI is sometimes opened directly as web/index.html during demos.  In
// that file:// mode relative /api URLs cannot reach the Python service, so
// use the batch runner's default port unless the URL explicitly overrides it.
const apiBase = (() => {
  const configured = new URLSearchParams(window.location.search).get("api_base")?.trim();
  if (configured) return configured.replace(/\/$/, "");
  return window.location.protocol === "file:" ? "http://127.0.0.1:8024" : "";
})();
const apiUrl = (path) => `${apiBase}${String(path || "")}`;
const assetUrl = (path) => {
  const value = String(path || "");
  return apiBase && value.startsWith("/api/") ? `${apiBase}${value}` : value;
};
const syncGenerateVisualState = () => {
  if (generateVisualState) {
    generateVisualState.value = generateVisualInput?.checked ? "ON" : "OFF";
    generateVisualState.textContent = generateVisualState.value;
  }
};
generateVisualInput?.addEventListener("change", syncGenerateVisualState);
syncGenerateVisualState();
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

function executionErrorText(error) {
  if (!error) return "";
  if (typeof error === "string") return error;
  return typeof error.message === "string" ? error.message : pretty(error);
}

function renderExecutionFeedback(container, item) {
  const feedback = item?.execution_feedback;
  let details = container.querySelector(":scope > .execution-feedback");
  if (!feedback) {
    details?.remove();
    return;
  }
  if (!details) {
    details = document.createElement("details");
    details.className = "execution-feedback";
    const title = document.createElement("summary");
    title.textContent = "本步返回值与数据复用";
    details.appendChild(title);
    details.appendChild(document.createElement("p"));
    details.appendChild(document.createElement("pre"));
    container.appendChild(details);
  }
  const repeated = Number(feedback.repeated_call_count || 0);
  const cached = Number(feedback.repeated_cached_count || 0);
  setTextIfChanged(details.querySelector("p"),
    `以下为程序返回值，执行成功不代表计算已验证。${repeated
      ? `本步重复调用已成功的工具 ${repeated} 次，其中 ${cached} 次命中缓存；后续计算可复用已有变量。` : ""}`);
  setTextIfChanged(details.querySelector("pre"), pretty(feedback.result_preview));
}

function hasUploadedInput() {
  return Boolean(uploadedSampleId || uploadedSequenceId);
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
  asset_binding: "正在校验结构化请求事实与输入资料",
  agent_queue: "正在等待 Agent 执行槽位",
  unified_agent: "Agent 正在执行、观察并继续分析",
  agent_model: "语言模型正在决定下一步",
  agent_execution: "正在执行代码并读取工具结果",
  agent_step_completed: "已取得一步结果，正在继续分析",
  verified_cache: "正在读取并校验已有结果",
  completed: "分析完成",
  failed: "本轮未完整完成",
};

function retrievalTimingText(call) {
  if (call?.cache_hit || !call?.retrieval?.timings) return "";
  const stages = call.retrieval.timings;
  const parts = [];
  if (Number.isFinite(stages.embedding_seconds)) parts.push(`向量化 ${stages.embedding_seconds.toFixed(2)} 秒`);
  if (Number.isFinite(stages.rerank_seconds)) parts.push(`重排 ${stages.rerank_seconds.toFixed(2)} 秒`);
  const runs = Object.values(call.retrieval.model_runs || {});
  if (runs.length) {
    const sum = (read) => runs.reduce((total, run) => {
      const value = read(run);
      return total + (Number.isFinite(value) ? value : 0);
    }, 0);
    const queue = sum((run) => run.host_timings?.gpu_queue_seconds);
    const load = sum((run) => run.timings?.dependency_import_seconds)
      + sum((run) => run.timings?.tokenizer_load_seconds) + sum((run) => run.timings?.model_load_seconds);
    const inference = sum((run) => run.timings?.inference_seconds);
    parts.push(`其中 GPU 排队 ${queue.toFixed(3)} 秒，依赖与模型加载 ${load.toFixed(2)} 秒，推理 ${inference.toFixed(2)} 秒`);
  }
  return parts.join("；");
}

const liveActionLabels = {
  request_received: "接收请求",
  asset_binding: "绑定数据与上下文",
  queue: "等待执行槽位",
  prepare: "准备 Agent",
  model: "调用语言模型",
  context: "整理模型上下文",
  context_budget: "检查上下文预算",
  select_tools: "选择工具并读取接口说明",
  read_context: "读取历史记录与接口说明",
  citation_repair: "补齐资料引用",
  knowledge_reuse: "复用已有资料",
  expand: "读取能力说明",
  execute: "执行代码与工具",
  cache_read: "读取验证缓存",
  solution: "形成回答",
  finish: "整理最终回答",
  solution_validation: "核对证据链",
  empty_action_repair: "修复空动作",
  format_repair: "检查模型动作格式",
  contract_required: "补充能力契约",
  execution_budget: "检查执行上限",
  stop: "停止本轮",
};

function executionStepNumber(steps, index) {
  const isPreparation = (item) => Number(item?.attempt) === 0
    && ["request_received", "asset_binding", "queue", "prepare", "cache_read"].includes(item?.action);
  if (isPreparation(steps[index])) return "准备";
  // Old saved traces used model-call counts for both fields, so count events
  // here as well. Context checks can fail without making another model call.
  return `第 ${steps.slice(0, index + 1).filter((item) => !isPreparation(item)).length} 步`;
}

function executionStepPresentation(steps, index) {
  const item = steps[index];
  const recovered = item?.action === "format_repair" && steps.slice(index + 1).some((later) =>
    ["select_tools", "execute", "read_context", "finish"].includes(later?.action));
  return {
    status: recovered ? "recovered" : String(item?.status || "unknown").toLowerCase(),
    summary: recovered ? "后续模型动作已通过格式检查，本次无效动作未执行。" : item?.summary || "",
  };
}

function executionFailurePhase(data) {
  const explicit = data?.diagnostic?.phase || data?.failure_phase || data?.request_trace?.failure_phase;
  if (explicit) return explicit;
  if (!(data?.failure_class || data?.request_trace?.failure_class)) return "";
  const steps = Array.isArray(data?.agent_trace) ? data.agent_trace : data?.request_trace?.live_steps || [];
  const last = steps[steps.length - 1];
  return last && ["error", "failed", "partial", "blocked", "incomplete"].includes(last.status)
    ? last.action : data?.failure_class || data?.request_trace?.failure_class;
}

function contextSummaryCallCount(data) {
  return Array.isArray(data?.llm_timings)
    ? data.llm_timings.filter((timing) => timing?.kind === "context_summary").length : 0;
}

function liveElapsedText(value) {
  const milliseconds = Number(value);
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return "";
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  const seconds = Math.round(milliseconds / 100) / 10;
  if (seconds < 60) return `${seconds} 秒`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} 分 ${(seconds % 60).toFixed(1).padStart(4, "0")} 秒`;
}

function stepTimingText(item) {
  const timingValue = (value) => (value == null || value === "" ? NaN : Number(value));
  const legacy = timingValue(item?.elapsed_ms);
  const model = timingValue(item?.model_elapsed_ms);
  const explicitExecution = timingValue(item?.execution_elapsed_ms);
  const execution = Number.isFinite(explicitExecution)
    ? explicitExecution
    : (item?.action === "execute" && Number.isFinite(legacy) ? legacy : NaN);
  const explicitTotal = timingValue(item?.total_elapsed_ms);
  const total = Number.isFinite(explicitTotal)
    ? explicitTotal
    : (item?.action === "execute" && [model, execution].some(Number.isFinite)
      ? [model, execution].filter(Number.isFinite).reduce((sum, value) => sum + value, 0)
      : legacy);
  if (!Number.isFinite(total)) return "";
  const parts = [`总计 ${liveElapsedText(total)}`];
  if (item?.action === "execute") {
    if (Number.isFinite(model)) parts.push(`模型 ${liveElapsedText(model)}`);
    if (Number.isFinite(execution)) parts.push(`执行 ${liveElapsedText(execution)}`);
  }
  return parts.join(" · ");
}

function setTextIfChanged(node, value) {
  if (!node) return;
  const text = String(value ?? "");
  if (node.textContent !== text) node.textContent = text;
}

function setClassIfChanged(node, value) {
  if (node && node.className !== value) node.className = value;
}

function syncLiveStepList(timeline, steps) {
  const existing = new Map(
    [...timeline.children].map((item) => [item.dataset.stepKey || "", item]),
  );
  if (!steps.length) {
    let waiting = existing.get("waiting");
    if (!waiting) {
      waiting = document.createElement("li");
      waiting.dataset.stepKey = "waiting";
    }
    setClassIfChanged(waiting, "live-step waiting");
    setTextIfChanged(waiting, "正在建立本轮执行记录…");
    if (timeline.firstElementChild !== waiting) timeline.insertBefore(waiting, timeline.firstElementChild);
    [...timeline.children].forEach((item) => { if (item !== waiting) item.remove(); });
    return;
  }

  const retained = new Set();
  steps.forEach((item, index) => {
    const presentation = executionStepPresentation(steps, index);
    const itemStatus = presentation.status;
    const key = String(item?.step_id || item?.id || `${item?.attempt || 0}:${item?.action || "step"}:${index}`);
    retained.add(key);
    let row = existing.get(key);
    if (!row) {
      row = document.createElement("li");
      row.dataset.stepKey = key;
      row.innerHTML = `
        <div data-step-role="heading"><span></span><strong></strong><small></small></div>
        <p data-step-role="summary" class="hidden"></p>
        <p data-step-role="error" class="live-step-error hidden"></p>
        <ul data-step-role="calls" class="hidden"></ul>
      `;
    }
    setClassIfChanged(row, `live-step ${itemStatus}`);
    const stepNumber = executionStepNumber(steps, index);
    const action = liveActionLabels[item?.action] || String(item?.action || "执行步骤").replaceAll("_", " ");
    const elapsedText = stepTimingText(item);
    setTextIfChanged(row.querySelector("[data-step-role='heading'] strong"), `${stepNumber} · ${action}`);
    setTextIfChanged(
      row.querySelector("[data-step-role='heading'] small"),
      `${executionStatusLabel(itemStatus)}${elapsedText ? ` · ${elapsedText}` : ""}`,
    );
    const summary = row.querySelector("[data-step-role='summary']");
    setTextIfChanged(summary, presentation.summary);
    summary.classList.toggle("hidden", !presentation.summary);
    const error = row.querySelector("[data-step-role='error']");
    setTextIfChanged(error, item?.error ? `原始错误：${executionErrorText(item.error)}` : "");
    error.classList.toggle("hidden", !item?.error || itemStatus === "recovered");
    const callList = row.querySelector("[data-step-role='calls']");
    const calls = Array.isArray(item?.calls) ? item.calls : [];
    calls.forEach((call, callIndex) => {
      let callNode = callList.children[callIndex];
      if (!callNode) {
        callNode = document.createElement("li");
        callList.appendChild(callNode);
      }
      const callStatus = executionStatusLabel(call?.status);
      const cache = call?.cache_hit ? " · 缓存命中" : "";
      const result = call?.result_summary ? ` · ${call.result_summary}` : "";
      const retrievalTiming = retrievalTimingText(call);
      const callError = call?.error ? ` · ${executionErrorText(call.error)}` : "";
      setTextIfChanged(
        callNode,
        `${displayToolName(call)}：${callStatus}${cache}${result}${retrievalTiming ? ` · ${retrievalTiming}` : ""}${callError}`,
      );
    });
    while (callList.children.length > calls.length) callList.lastElementChild.remove();
    callList.classList.toggle("hidden", !calls.length);
    renderExecutionFeedback(row, item);
    if (timeline.children[index] !== row) timeline.insertBefore(row, timeline.children[index] || null);
  });
  [...timeline.children].forEach((item) => {
    if (!retained.has(item.dataset.stepKey || "")) item.remove();
  });
}

function ensureLiveExecutionPanel(message) {
  const content = message?.querySelector(".chat-message-content");
  if (!content) return null;
  let panel = content.querySelector(":scope > .live-execution-panel");
  if (panel) return panel;
  content.querySelector(".typing-indicator")?.remove();
  panel = document.createElement("section");
  panel.className = "live-execution-panel";
  panel.setAttribute("aria-live", "polite");
  panel.innerHTML = `
    <div class="live-execution-heading">
      <div><span class="live-pulse" aria-hidden="true"></span><strong>Agent 执行路径</strong></div>
      <div class="live-heading-meta"><small data-live-role="task">任务尚未绑定</small><span data-live-role="elapsed">刚刚开始</span></div>
    </div>
    <div class="live-progress-track" role="progressbar" aria-label="Agent阶段进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="1">
      <i data-live-role="progress"></i>
    </div>
    <p class="live-current-activity" data-live-role="activity">正在连接执行服务…</p>
    <ol class="live-step-list" data-live-role="steps"></ol>
    <p class="live-progress-note" data-live-role="note">页面会持续更新；无需重复提交问题。</p>
  `;
  content.appendChild(panel);
  return panel;
}

function renderDurableTaskState(task, requestId, pendingMessage, { resumed = false } = {}) {
  const state = String(task?.state || "unknown");
  const labels = {
    queued: "队列等待",
    running: "后台执行",
    cancelling: "正在取消",
    succeeded: "已成功",
    failed: "未完整完成",
    cancelled: "已取消",
    interrupted: "服务重启中断",
  };
  const panel = ensureLiveExecutionPanel(pendingMessage);
  if (panel) {
    panel.dataset.taskState = state;
    const taskLabel = panel.querySelector("[data-live-role='task']");
    if (taskLabel) {
      setTextIfChanged(taskLabel, `${resumed ? "已恢复 · " : ""}${labels[state] || state} · ${String(requestId).slice(0, 12)}`);
      taskLabel.title = `持久任务 ID：${requestId}`;
    }
  }
  runState.dataset.taskState = state;
  runState.setAttribute(
    "aria-label",
    `${labels[state] || state}，持久任务 ${requestId}`,
  );
}

function renderLiveExecutionProgress(message, snapshot, state) {
  const panel = ensureLiveExecutionPanel(message);
  if (!panel) return;
  const now = Date.now();
  const stage = String(snapshot?.current_stage || "request_received");
  const activity = snapshot?.current_activity && typeof snapshot.current_activity === "object"
    ? snapshot.current_activity : {};
  const steps = Array.isArray(snapshot?.live_steps) ? snapshot.live_steps : [];
  const progressValue = Number(snapshot?.progress);
  const percent = Number.isFinite(progressValue)
    ? Math.min(100, Math.max(1, Math.round(progressValue * 100))) : 1;
  const status = String(snapshot?.status || "running").toLowerCase();
  const summary = String(activity.summary || requestStageLabels[stage] || "Agent 正在分析");
  const signature = JSON.stringify([
    stage, status, steps.length, activity.attempt, activity.action, activity.status, summary,
  ]);
  if (signature !== state.lastSignature) {
    state.lastSignature = signature;
    state.lastChangeAt = now;
  }
  state.lastSnapshot = snapshot;
  state.pollErrors = 0;

  panel.dataset.status = status;
  const elapsed = Number(snapshot?.elapsed_ms);
  setTextIfChanged(panel.querySelector("[data-live-role='elapsed']"), liveElapsedText(
    Number.isFinite(elapsed) ? elapsed : now - state.startedAt,
  ) || "正在执行");
  const track = panel.querySelector(".live-progress-track");
  track.setAttribute("aria-valuenow", String(percent));
  panel.querySelector("[data-live-role='progress']").style.width = `${percent}%`;
  const current = panel.querySelector("[data-live-role='activity']");
  const attemptPrefix = Number(activity.attempt) > 0 ? `第 ${activity.attempt} 步 · ` : "";
  setTextIfChanged(current, `${attemptPrefix}${summary}`);
  const timeline = panel.querySelector("[data-live-role='steps']");
  syncLiveStepList(timeline, steps);

  const note = panel.querySelector("[data-live-role='note']");
  const stalledSeconds = Math.max(0, Math.round((now - state.lastChangeAt) / 1000));
  if (snapshot?.failure_message || snapshot?.next_action) {
    note.className = "live-progress-note error";
    setTextIfChanged(note, `${snapshot.failure_message || "本轮未完整完成。"}${snapshot.next_action ? ` 下一步：${snapshot.next_action}` : ""}`);
  } else if (status === "cancelled") {
    note.className = "live-progress-note warning";
    setTextIfChanged(note, "正在停止剩余工作；已经完成的步骤会保留。");
  } else if (stalledSeconds >= 20) {
    note.className = "live-progress-note warning";
    setTextIfChanged(note, `当前步骤已持续 ${stalledSeconds} 秒，可能正在等待模型、下载数据或现场推理。任务仍在运行，你可以继续等待或点击“停止”。`);
  } else {
    note.className = "live-progress-note";
    setTextIfChanged(note, "页面会持续更新；无需重复提交问题，已完成步骤会保留在本轮记录中。");
  }
}

function renderProgressReconnectState(message, state) {
  const panel = ensureLiveExecutionPanel(message);
  if (!panel || state.pollErrors < 3) return;
  const note = panel.querySelector("[data-live-role='note']");
  note.className = "live-progress-note warning";
  setTextIfChanged(note, "暂时无法读取实时进度，正在自动重连。这不等于任务失败；你仍可点击“停止”结束等待。");
}

async function pollRequestProgress(requestId, state, pendingMessage) {
  while (!state.done) {
    const pollController = new AbortController();
    state.pollController = pollController;
    const pollTimeout = window.setTimeout(() => pollController.abort(), 4000);
    try {
      const response = await fetch(apiUrl(`/api/requests/${encodeURIComponent(requestId)}`), {
        cache: "no-store",
        signal: pollController.signal,
      });
      if (response.ok) {
        const snapshot = await response.json();
        if (state.done) break;
        const stage = String(snapshot.current_stage || "request_received");
        const progress = Number(snapshot.progress);
        const percent = Number.isFinite(progress) && progress > 0
          ? ` · ${Math.min(100, Math.max(1, Math.round(progress * 100)))}%`
          : "";
        const label = requestStageLabels[stage] || "Agent 正在分析";
        setTextIfChanged(runState, `${label}${percent}`);
        const elapsedText = liveElapsedText(snapshot.elapsed_ms);
        setTextIfChanged(
          conversationStatus,
          `${label}${percent}${elapsedText ? ` · 已运行 ${elapsedText}` : ""}；完成后会保留工具证据与执行轨迹。`,
        );
        renderLiveExecutionProgress(pendingMessage, snapshot, state);
      } else {
        state.pollErrors += 1;
        renderProgressReconnectState(pendingMessage, state);
      }
    } catch (_error) {
      state.pollErrors += 1;
      renderProgressReconnectState(pendingMessage, state);
    } finally {
      window.clearTimeout(pollTimeout);
      if (state.pollController === pollController) state.pollController = null;
    }
    await new Promise((resolve) => setTimeout(resolve, 700));
  }
}

async function fetchRequestSnapshotOnce(requestId, timeoutMilliseconds = 1800) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMilliseconds);
  try {
    const response = await fetch(apiUrl(`/api/requests/${encodeURIComponent(requestId)}`), {
      cache: "no-store",
      signal: controller.signal,
    });
    return response.ok ? await response.json() : null;
  } catch (_error) {
    return null;
  } finally {
    window.clearTimeout(timeout);
  }
}

async function requestTaskCancellation(requestId, timeoutMilliseconds = 2000) {
  if (!requestId) return null;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMilliseconds);
  try {
    const response = await fetch(apiUrl(`/api/tasks/${encodeURIComponent(requestId)}/cancel`), {
      method: "POST",
      signal: controller.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ requested_by: "web-ui" }),
    });
    return response.ok ? await response.json() : null;
  } catch (_error) {
    return null;
  } finally {
    window.clearTimeout(timeout);
  }
}

function activeTaskKey(targetSessionId = sessionId) {
  return `${activeTaskPrefix}${targetSessionId}`;
}

function persistActiveTask(record) {
  try {
    window.localStorage.setItem(activeTaskKey(record?.sessionId || sessionId), JSON.stringify({
      schemaVersion: "web-active-task-v1",
      requestId: String(record?.requestId || ""),
      sessionId: String(record?.sessionId || sessionId),
      turnId: String(record?.turnId || ""),
      question: String(record?.question || ""),
      submittedAt: Number(record?.submittedAt || Date.now()),
    }));
  } catch (_error) { /* the TaskStore remains authoritative */ }
}

function readActiveTask(targetSessionId = sessionId) {
  try {
    const value = JSON.parse(window.localStorage.getItem(activeTaskKey(targetSessionId)) || "null");
    return value && value.schemaVersion === "web-active-task-v1" && value.requestId ? value : null;
  } catch (_error) {
    return null;
  }
}

function clearActiveTask(targetSessionId = sessionId) {
  try { window.localStorage.removeItem(activeTaskKey(targetSessionId)); } catch (_error) { /* ignore */ }
}

function taskFallbackResult(task) {
  const state = String(task?.state || "unknown");
  const cancelled = state === "cancelled";
  return {
    status: cancelled ? "cancelled" : "unknown",
    answer: cancelled
      ? "本轮任务已取消。"
      : "任务已结束，但结果暂未取得。",
    request_id: task?.request_id || "",
    session_id: task?.session_id || sessionId,
    termination_reason: task?.termination_reason || state,
    failure_class: task?.failure_class || (cancelled ? "" : "task_result_unavailable"),
    model_calls: 0,
    tool_calls: 0,
  };
}

async function waitForDurableTask(requestId, state, pendingMessage) {
  let networkErrors = 0;
  let notFoundResponses = 0;
  while (true) {
    try {
      const response = await fetch(apiUrl(`/api/tasks/${encodeURIComponent(requestId)}`), {
        cache: "no-store",
      });
      if (response.status === 404) {
        notFoundResponses += 1;
        if (notFoundResponses >= 3) {
          state.done = true;
          return {
            status: "failed",
            answer: "持久任务记录在连续三次查询中都不存在；没有把它误报为分析成功。",
            request_id: requestId,
            session_id: sessionId,
            termination_reason: "durable_task_record_missing",
            failure_class: "task_state_missing",
            failure_message: "TaskStore 中没有对应记录。",
            next_action: "确认服务端是否切换了数据目录；如记录确实已清理，再重新提交问题。",
          };
        }
        throw new Error("持久任务暂未可见");
      }
      if (!response.ok) throw new Error(`任务状态接口返回 HTTP ${response.status}`);
      const body = await response.json();
      const task = body.task || {};
      networkErrors = 0;
      notFoundResponses = 0;
      const lifecycleChanged = state.lastTaskState !== task.state;
      state.lastTaskState = task.state;
      if (lifecycleChanged) renderDurableTaskState(task, requestId, pendingMessage);
      const lifecycleLabels = {
        queued: "已入队，等待执行",
        running: "后台正在执行",
        cancelling: "正在协作式取消",
        succeeded: "执行完成",
        failed: "执行未完整完成",
        cancelled: "已取消",
        interrupted: "服务重启时中断",
      };
      // The detailed request-progress poll owns the running labels. Without
      // this guard the 600 ms task poll and 700 ms progress poll alternated
      // different strings, producing the visible refresh/flicker loop.
      if (!state.lastSnapshot || task.state !== "running") {
        setTextIfChanged(runState, lifecycleLabels[task.state] || "读取任务状态");
        setTextIfChanged(
          conversationStatus,
          `${lifecycleLabels[task.state] || task.state} · 任务 ${String(requestId).slice(0, 12)}；刷新页面后仍可恢复。`,
        );
      }
      if (task.terminal) {
        state.done = true;
        const result = task.result && typeof task.result === "object" && Object.keys(task.result).length
          ? { ...task.result }
          : taskFallbackResult(task);
        result.request_id ||= task.request_id;
        result.session_id ||= task.session_id;
        result.termination_reason ||= task.termination_reason;
        result.failure_class ||= task.failure_class;
        return result;
      }
    } catch (_error) {
      networkErrors += 1;
      state.pollErrors = Math.max(state.pollErrors || 0, networkErrors);
      renderProgressReconnectState(pendingMessage, state);
      // A transient browser/server connection loss is not a scientific task
      // failure. Keep the durable task ID and reconnect until the page closes.
    }
    await new Promise((resolve) => setTimeout(resolve, networkErrors ? 1200 : 600));
  }
}

async function submitDurableTaskWithReconnect(requestPayload, requestId, state, pendingMessage) {
  let attempts = 0;
  while (true) {
    attempts += 1;
    try {
      const response = await fetch(apiUrl("/api/tasks"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": `web:${requestId}`,
        },
        body: JSON.stringify({
          ...requestPayload,
          idempotency_key: `web:${requestId}`,
        }),
      });
      const body = await response.json().catch(() => ({}));
      if (response.ok && body?.task) return body;
      const message = typeof body?.error === "object"
        ? body.error?.message || body.error?.type
        : body?.error;
      if (response.status >= 400 && response.status < 500) {
        const rejected = new Error(String(message || `任务提交被拒绝（HTTP ${response.status}）`));
        rejected.durableSubmissionRejected = true;
        throw rejected;
      }
      throw new Error(String(message || `任务提交接口返回 HTTP ${response.status}`));
    } catch (error) {
      if (error?.durableSubmissionRejected) throw error;
      state.pollErrors = Number(state.pollErrors || 0) + 1;
      renderProgressReconnectState(pendingMessage, state);
      runState.textContent = "重新连接任务队列";
      conversationStatus.textContent = `第 ${attempts} 次提交连接未确认；正在用同一幂等键重试，不会重复创建任务。`;
      await new Promise((resolve) => setTimeout(resolve, 1200));
    }
  }
}

async function readDurableTaskWithReconnect(requestId, state, pendingMessage) {
  let networkErrors = 0;
  let notFoundResponses = 0;
  while (true) {
    try {
      const response = await fetch(apiUrl(`/api/tasks/${encodeURIComponent(requestId)}`), {
        cache: "no-store",
      });
      if (response.status === 404) {
        notFoundResponses += 1;
        const activeResponse = await fetch(
          apiUrl(`/api/sessions/${encodeURIComponent(sessionId)}/active-tasks`),
          { cache: "no-store" },
        );
        if (activeResponse.ok) {
          const activeBody = await activeResponse.json();
          const discovered = (activeBody.active_tasks || []).find(
            (item) => item?.request_id === requestId,
          );
          if (discovered) return discovered;
        }
        if (notFoundResponses >= 4) return null;
        conversationStatus.textContent = `正在确认持久任务 ${String(requestId).slice(0, 12)} 是否已完成首次入队。`;
        await new Promise((resolve) => setTimeout(resolve, 800));
        continue;
      }
      if (!response.ok) throw new Error(`任务状态接口返回 HTTP ${response.status}`);
      const body = await response.json();
      return body.task || null;
    } catch (_error) {
      networkErrors += 1;
      state.pollErrors = Math.max(state.pollErrors || 0, networkErrors);
      renderProgressReconnectState(pendingMessage, state);
      conversationStatus.textContent = `正在重新连接持久任务 ${String(requestId).slice(0, 12)}；后台状态不会因页面断线而丢失。`;
      await new Promise((resolve) => setTimeout(resolve, 1200));
    }
  }
}

function resultPresentation(data) {
  const outcome = data?.presentation?.outcome;
  if (outcome && typeof outcome.incomplete === "boolean" && outcome.label) return outcome;
  // Compatibility boundary for persisted responses written before public outcomes.
  // Never infer completion from diagnostic prose, failure_class or missing fields.
  const status = String(data?.status || "unknown").toLowerCase();
  const labels = {
    ok: "已完成", success: "已完成", completed: "已完成", partial: "部分完成",
    cancelled: "已取消", canceled: "已取消", interrupted: "执行中断",
    error: "未完成", failed: "未完成", failure: "未完成", unavailable: "数据不可用",
    blocked: "未完成", clarification_needed: "等待补充信息", unknown: "结果待确认",
  };
  return { status, label: labels[status] || "结果待确认", incomplete: !["ok", "success", "completed"].includes(status) };
}

async function presentDurableTaskResult(data, pendingMessage, turnId) {
  const answer = data.answer || "本轮未返回文字结论。";
  const incompleteResult = resultPresentation(data).incomplete;
  const meta = compactTurnMeta(data);
  completePendingMessage(pendingMessage, answer, meta, incompleteResult);
  const existingIndex = chatHistory.findIndex(
    (item) => item.role === "assistant" && item.turnId === turnId,
  );
  const saved = {
    role: "assistant",
    text: answer,
    meta,
    result: persistedResult(data),
    timestamp: Date.now(),
    turnId,
  };
  if (existingIndex >= 0) chatHistory[existingIndex] = saved;
  else chatHistory.push(saved);
  persistChatHistory();
  persistConversationIndex();
  try {
    await renderConversationList();
    renderResult(data, pendingMessage);
    conversationStatus.textContent = incompleteResult
      ? `${resultPresentation(data).label}。${data.failure_message || ""}${data.next_action ? ` 下一步：${data.next_action}` : ""}`
      : `当前对话 · ${data.session_turn_count ?? Math.ceil(chatHistory.length / 2)}轮 · 可继续追问`;
  } catch (displayError) {
    console.error("Agent回答已返回，但页面展示失败：", displayError);
    renderError(`Agent回答已成功返回，但部分页面展示失败：${displayError.message}`);
    conversationStatus.textContent = "Agent回答已返回；部分执行记录或图表未能显示。";
  }
  return { answer, incompleteResult };
}

async function resumeActiveTaskIfPresent() {
  const active = readActiveTask();
  if (!active || requestInFlight) return;
  const turnId = active.turnId || createTurnId();
  const existingAssistantIndex = chatHistory.findIndex(
    (item) => item.role === "assistant" && item.turnId === turnId,
  );
  if (!chatHistory.some((item) => item.role === "user" && item.turnId === turnId)) {
    const restoredUser = {
      role: "user",
      text: active.question || "恢复中的持久任务",
      timestamp: active.submittedAt || Date.now(),
      turnId,
    };
    chatHistory.push(restoredUser);
    appendChatMessage("user", restoredUser.text, restoredUser);
    persistChatHistory();
  }

  // A prior browser-side transport message is not the terminal scientific
  // truth. Remove it before reconnecting to the authoritative TaskStore row.
  if (existingAssistantIndex >= 0) {
    chatHistory.splice(existingAssistantIndex, 1);
    [...chatThread.querySelectorAll(".chat-message.assistant")]
      .find((message) => message.dataset.turnId === turnId)?.remove();
    persistChatHistory();
  }
  const pendingMessage = appendChatMessage(
    "assistant", "", { pending: true, timestamp: Date.now(), turnId },
  );
  const progressState = {
    done: false,
    startedAt: active.submittedAt || Date.now(),
    lastChangeAt: Date.now(),
    lastSignature: "",
    lastSnapshot: null,
    pollErrors: 0,
  };
  let progressPromise = Promise.resolve();
  let durableTaskSettled = false;
  requestInFlight = true;
  activeRequestId = active.requestId;
  activeRequestPhase = "reconnecting";
  activeRequestTransport = "task";
  activeRequestController = new AbortController();
  syncComposerAvailability();
  runButton.firstElementChild.textContent = "恢复中";
  runState.textContent = "正在恢复";
  runState.className = "run-state running";
  conversationStatus.textContent = `正在接回持久任务 ${String(active.requestId).slice(0, 12)}。`;
  if (cancelRequestButton) {
    cancelRequestButton.classList.remove("hidden");
    cancelRequestButton.disabled = false;
    cancelRequestButton.querySelector("span").textContent = "停止";
  }

  try {
    let task = await readDurableTaskWithReconnect(
      active.requestId, progressState, pendingMessage,
    );
    if (!task) {
      const missing = {
        status: "failed",
        answer: "浏览器保存了任务编号，但服务端已找不到对应的持久任务记录。没有把它误报为分析成功。",
        request_id: active.requestId,
        session_id: active.sessionId || sessionId,
        termination_reason: "durable_task_record_missing",
        failure_class: "task_state_missing",
        failure_message: "TaskStore 中没有对应记录。",
        next_action: "确认是否切换了数据目录；如记录确实已清理，再重新提交问题。",
      };
      durableTaskSettled = true;
      await presentDurableTaskResult(missing, pendingMessage, turnId);
      return;
    }
    renderDurableTaskState(task, active.requestId, pendingMessage, { resumed: true });
    activeRequestPhase = "execution";
    if (cancelRequestButton) cancelRequestButton.querySelector("span").textContent = "停止";
    ensureLiveExecutionPanel(pendingMessage);
    progressPromise = pollRequestProgress(
      active.requestId, progressState, pendingMessage,
    );
    const data = await waitForDurableTask(
      active.requestId, progressState, pendingMessage,
    );
    durableTaskSettled = true;
    await presentDurableTaskResult(data, pendingMessage, turnId);
  } catch (error) {
    // Keep the active task pointer: a browser-side exception is not evidence
    // that the durable background task failed.
    renderError(`任务恢复界面发生错误：${error.message}`);
    conversationStatus.textContent = "后台任务编号仍已保留；刷新页面可继续恢复。";
  } finally {
    progressState.done = true;
    progressState.pollController?.abort();
    await progressPromise.catch(() => {});
    if (durableTaskSettled) clearActiveTask(active.sessionId || sessionId);
    activeRequestController = null;
    activeRequestId = "";
    activeRequestPhase = "idle";
    activeRequestTransport = "none";
    requestInFlight = false;
    if (cancelRequestButton) {
      cancelRequestButton.classList.add("hidden");
      cancelRequestButton.disabled = false;
      cancelRequestButton.querySelector("span").textContent = "停止";
    }
    syncComposerAvailability();
    runButton.firstElementChild.textContent = "发送";
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
  const persistedKeys = [
    "status", "artifacts", "artifact_count", "request_id", "fallback_reason", "partial_reason",
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
    failure_class: data?.failure_class || "",
    failure_phase: data?.failure_phase || "",
    failure_message: data?.failure_message || "",
    next_action: data?.next_action || "",
    diagnostic: data?.diagnostic || null,
    request_trace: data?.request_trace || null,
    event_resolution: data?.event_resolution || null,
    model_calls: Number(data?.model_calls || 0),
    llm_timings: Array.isArray(data?.llm_timings) ? data.llm_timings : [],
    code_execution_count: Number(data?.code_execution_count || 0),
    agent_trace: (Array.isArray(data?.agent_trace) ? data.agent_trace : []).slice(0, 20).map((item) => ({
      attempt: item.attempt,
      step: item.step,
      action: item.action,
      status: item.status,
      elapsed_ms: item.elapsed_ms,
      execution_elapsed_ms: item.execution_elapsed_ms,
      model_elapsed_ms: item.model_elapsed_ms,
      total_elapsed_ms: item.total_elapsed_ms,
      summary: item.summary || "",
      execution_feedback: item.execution_feedback,
      error: item.error || "",
      calls: Array.isArray(item.calls) ? item.calls : [],
    })),
    tool_runs: runs.map((run) => ({
      step: run.step,
      tool_name: run.tool_name || run.api,
      api: run.api || run.tool_name,
      arguments: run.arguments || {},
      status: run.status,
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
  const markdownImage = /!\[([^\]]*)\]\(\s*(\/api\/artifacts\/([0-9a-f]{32})\.png)\s*\)/gi;
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

const conversationLayerMeta = {
  all: { label: "全部", kind: "综合研判" },
  observation: { label: "观测", kind: "观测证据" },
  environment: { label: "环境", kind: "环境证据" },
  forecast: { label: "预测", kind: "预测证据" },
  verification: { label: "后续验证", kind: "实况检验" },
  analysis: { label: "分析图表", kind: "派生分析" },
};

function conversationArtifactLayer(artifact = {}) {
  if (conversationLayerMeta[artifact.layer]) return artifact.layer;
  return "analysis";
}

function updateConversationTimeline(selected) {
  if (!visualTimeline || !visualTimelineScale) return;
  const layer = selected?.layer || "analysis";
  const timelines = {
    observation: {
      title: "观测时间",
      label: "5分钟一帧 · 历史至当前",
      marks: ["−60 min", "−30 min", "当前"],
      position: 2,
    },
    forecast: {
      title: "预测时效",
      label: "5分钟一帧 · 当前至未来",
      marks: ["当前", "+60 min", "+120 min", "+180 min"],
      position: 0,
    },
    verification: {
      title: "预测与实况验证",
      label: "同一有效时间对照",
      marks: ["当前", "+60 min", "+120 min", "+180 min"],
      position: 0,
    },
    environment: {
      title: "环境数据时间",
      label: "小时级 · 大尺度背景",
      marks: ["−24 h", "−6 h", "当前", "+6 h", "+12 h", "+24 h"],
      position: 2,
    },
    analysis: {
      title: "分析图表时间范围",
      label: "由当前图表实际数据决定",
      marks: ["历史", "当前", "未来"],
      position: 1,
    },
  };
  const timeline = timelines[layer] || timelines.analysis;
  visualTimelineTitle.textContent = timeline.title;
  visualTimelineLabel.textContent = timeline.label;
  visualTimelineScale.replaceChildren();
  timeline.marks.forEach((mark) => {
    const span = document.createElement("span");
    span.textContent = mark;
    visualTimelineScale.appendChild(span);
  });
  let ratio = timeline.marks.length <= 1 ? 50 : (timeline.position / (timeline.marks.length - 1)) * 100;
  const leadMinutes = Number(selected?.leadMinutes);
  const offsetMinutes = Number(selected?.offsetMinutes);
  if ((layer === "forecast" || layer === "verification") && Number.isFinite(leadMinutes) && leadMinutes >= 0) {
    ratio = Math.max(0, Math.min(100, (leadMinutes / 180) * 100));
  } else if (layer === "observation" && Number.isFinite(offsetMinutes)) {
    ratio = Math.max(0, Math.min(100, ((offsetMinutes + 60) / 60) * 100));
  }
  visualTimelineNow.style.left = `${ratio}%`;
  visualTimelineMarker.style.left = `calc(${ratio}% - 5px)`;
  visualTimeline.dataset.layer = layer;
  const frames = Array.isArray(selected?.frames) ? selected.frames : [];
  const timelineEligible = ["observation", "forecast", "verification", "environment"].includes(layer)
    && frames.length > 1;
  visualTimeline.classList.toggle("hidden", !timelineEligible);
  if (!timelineEligible) return;
  const selectedIndex = Math.max(0, Math.min(frames.length - 1, Number(selected.frameIndex || 0)));
  const frameMarks = [...new Set([0, Math.floor((frames.length - 1) / 2), frames.length - 1])];
  visualTimelineScale.replaceChildren();
  frameMarks.forEach((index) => {
    const span = document.createElement("span");
    span.textContent = frames[index]?.title || `第${index + 1}帧`;
    visualTimelineScale.appendChild(span);
  });
  const frameRatio = (selectedIndex / (frames.length - 1)) * 100;
  visualTimelineMarker.style.left = `calc(${frameRatio}% - 5px)`;
  visualTimelineLabel.textContent = `${frames[selectedIndex]?.title || `第${selectedIndex + 1}帧`} · ${selectedIndex + 1}/${frames.length}`;
  if (visualTimelineSlider) {
    visualTimelineSlider.max = String(frames.length - 1);
    visualTimelineSlider.value = String(selectedIndex);
    visualTimelineSlider.disabled = false;
    visualTimelineSlider.setAttribute("aria-valuetext", frames[selectedIndex]?.title || selected?.title || "当前时刻");
  }
}

function renderConversationLayerTabs() {
  if (!conversationLayerTabs) return;
  conversationLayerTabs.replaceChildren();
  const counts = conversationVisualArtifacts.reduce((result, artifact) => {
    result[artifact.layer] = (result[artifact.layer] || 0) + 1;
    return result;
  }, {});
  const layers = ["all", "observation", "environment", "forecast", "verification", "analysis"]
    .filter((layer) => layer === "all" || counts[layer]);
  if (!layers.includes(activeConversationLayer)) activeConversationLayer = "all";
  layers.forEach((layer) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = layer === activeConversationLayer ? "active" : "";
    button.dataset.layer = layer;
    button.innerHTML = `<span>${conversationLayerMeta[layer].label}</span><b>${layer === "all" ? conversationVisualArtifacts.length : counts[layer]}</b>`;
    button.addEventListener("click", () => {
      activeConversationLayer = layer;
      renderConversationLayerTabs();
      renderConversationThumbnails();
      const first = conversationVisualArtifacts.find((artifact) => layer === "all" || artifact.layer === layer);
      if (first) selectConversationVisual(first.identity);
    });
    conversationLayerTabs.appendChild(button);
  });
}

function renderConversationThumbnails() {
  if (!conversationVisualThumbnails) return;
  conversationVisualThumbnails.replaceChildren();
  const visibleArtifacts = conversationVisualArtifacts.filter(
    (artifact) => activeConversationLayer === "all" || artifact.layer === activeConversationLayer,
  );
  visibleArtifacts.forEach((artifact, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.artifactIdentity = artifact.identity;
    button.setAttribute("role", "listitem");
    button.setAttribute("aria-label", `查看${conversationLayerMeta[artifact.layer].label}图片${index + 1}：${artifact.title}`);
    const image = document.createElement("img");
    image.src = assetUrl(artifact.thumbnail);
    image.alt = "";
    image.loading = "lazy";
    const number = document.createElement("span");
    number.textContent = String(index + 1).padStart(2, "0");
    const source = document.createElement("small");
    source.textContent = conversationLayerMeta[artifact.layer].label;
    button.append(image, source, number);
    button.addEventListener("click", () => selectConversationVisual(artifact.identity));
    image.addEventListener("error", () => button.remove());
    conversationVisualThumbnails.appendChild(button);
  });
}

// Desktop mouse wheels emit vertical deltas even though the evidence strip is
// horizontal. Translate that gesture locally so all right-side images remain
// reachable without moving the conversation page.
conversationVisualThumbnails?.addEventListener("wheel", (event) => {
  if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
  const canScroll = conversationVisualThumbnails.scrollWidth > conversationVisualThumbnails.clientWidth;
  if (!canScroll) return;
  conversationVisualThumbnails.scrollLeft += event.deltaY;
  event.preventDefault();
}, { passive: false });

function updateConversationVisualContext(result = {}) {
  const place = analysisPlaceInput?.value?.trim()
    || result?.presentation?.place
    || result?.tool_result?.place
    || result?.place
    || "当前事件";
  const initializedAt = analysisTimeInput?.value
    || result?.presentation?.initialization_time_utc
    || result?.initialization_time_utc
    || "最新可用";
  const runs = Array.isArray(result?.tool_runs) ? result.tool_runs : [];
  const cacheHit = runs.some((run) => run?.cache_hit === true || /cache_hit|cached/.test(String(run?.route_mode || "")));
  const liveRun = runs.some((run) => /live/.test(String(run?.route_mode || run?.data_policy || "")));
  if (visualContextPlace) visualContextPlace.textContent = place;
  if (visualContextTime) visualContextTime.textContent = initializedAt === "最新可用" ? initializedAt : String(initializedAt).replace("T", " ").replace(/Z$/, " UTC");
  if (visualContextStatus) visualContextStatus.textContent = cacheHit ? "缓存已命中" : (liveRun ? "实时计算完成" : "证据已就绪");
  if (visualEvidenceLabel) visualEvidenceLabel.textContent = `${conversationVisualArtifacts.length} 个可追溯视图`;
}

function selectConversationVisual(identity) {
  const selected = conversationVisualArtifacts.find((artifact) => artifact.identity === identity)
    || conversationVisualArtifacts[0];
  if (!selected) return;
  selectedConversationVisual = selected.identity;
  const frames = Array.isArray(selected.frames) ? selected.frames : [];
  const frameIndex = frames.length
    ? Math.max(0, Math.min(frames.length - 1, Number(selected.frameIndex || 0)))
    : 0;
  const currentFrame = frames[frameIndex];
  conversationVisualStage?.classList.remove("is-unavailable");
  conversationVisualImage?.classList.remove("hidden");
  conversationVisualImage.src = assetUrl(currentFrame?.source || selected.source);
  conversationVisualImage.alt = currentFrame ? `${selected.title} ${currentFrame.title}` : selected.title;
  conversationVisualTitle.textContent = currentFrame ? `${selected.title} · ${currentFrame.title}` : selected.title;
  conversationVisualDescription.textContent = selected.description;
  conversationVisualDescription.classList.toggle("hidden", !selected.description);
  const meta = conversationLayerMeta[selected.layer] || conversationLayerMeta.analysis;
  if (conversationVisualKind) conversationVisualKind.textContent = meta.kind;
  if (visualTimelineLabel) {
    visualTimelineLabel.textContent = selected.layer === "observation"
      ? "历史至当前"
      : selected.layer === "forecast"
        ? "当前至未来60分钟"
        : selected.layer === "verification"
          ? "预测与实况对照"
          : "事件有效时段";
  }
  updateConversationTimeline(selected);
  conversationVisualThumbnails.querySelectorAll("button").forEach((button) => {
    const active = button.dataset.artifactIdentity === selected.identity;
    button.classList.toggle("active", active);
    button.setAttribute("aria-current", active ? "true" : "false");
  });
}

function selectConversationFrame(index) {
  const selected = conversationVisualArtifacts.find((artifact) => artifact.identity === selectedConversationVisual);
  if (!selected || !Array.isArray(selected.frames) || selected.frames.length < 2) return;
  selected.frameIndex = Math.max(0, Math.min(selected.frames.length - 1, Number(index)));
  selectConversationVisual(selected.identity);
}

conversationVisualImage?.addEventListener("error", () => {
  conversationVisualStage?.classList.add("is-unavailable");
  conversationVisualImage?.classList.add("hidden");
  if (conversationVisualDescription) {
    conversationVisualDescription.textContent = "图表文件已失效或尚未生成，已从当前视图隐藏。";
    conversationVisualDescription.classList.remove("hidden");
  }
});

function renderConversationVisualPane(extraResult = null) {
  if (!conversationVisualPane) return;
  const candidates = [];
  chatHistory.forEach((item) => {
    if (item.role !== "assistant") return;
    candidates.push(...visualArtifactsFromResult(item.result, item.text));
  });
  if (extraResult) {
    candidates.push(...visualArtifactsFromResult(extraResult, extraResult.answer));
  }

  const seen = new Set();
  conversationVisualArtifacts = candidates.flatMap((artifact) => {
    if (!artifact || artifact.verified === false || artifact.quality === "empty_plot") return [];
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
      layer: conversationArtifactLayer(artifact),
      chartType: artifact.chart_type || "",
      leadMinutes: artifact.lead_minutes ?? artifact.forecast_relative_minutes ?? null,
      offsetMinutes: artifact.offset_minutes ?? artifact.relative_minutes ?? null,
      validTimeUtc: artifact.valid_time_utc || artifact.time_utc || "",
      dataSource: artifact.source || artifact.data_source || "",
      frames: Array.isArray(artifact.frames) ? artifact.frames : [],
      frameIndex: Array.isArray(artifact.frames) && artifact.frames.length ? artifact.frames.length - 1 : 0,
    }];
  });

  conversationVisualPane.classList.toggle("hidden", conversationVisualArtifacts.length === 0);
  conversationContentLayout?.classList.toggle("has-visuals", conversationVisualArtifacts.length > 0);
  fitConversationVisualPane();
  conversationVisualCount.textContent = `${conversationVisualArtifacts.length}张`;
  updateConversationVisualContext(extraResult || {});
  if (!conversationVisualArtifacts.length) {
    selectedConversationVisual = "";
    conversationVisualImage.removeAttribute("src");
    return;
  }

  renderConversationLayerTabs();
  renderConversationThumbnails();
  selectConversationVisual(selectedConversationVisual);
}

function fitConversationVisualPane() {
  if (!conversationVisualPane || window.innerWidth <= 980 || conversationVisualPane.classList.contains("hidden")) {
    conversationVisualPane?.style.removeProperty("--conversation-visual-height");
    return;
  }
  cancelAnimationFrame(visualPaneResizeFrame);
  visualPaneResizeFrame = requestAnimationFrame(() => {
    const top = Math.max(12, conversationVisualPane.getBoundingClientRect().top);
    const available = Math.max(420, window.innerHeight - top - 12);
    conversationVisualPane.style.setProperty("--conversation-visual-height", `${available}px`);
  });
}

window.addEventListener("resize", fitConversationVisualPane);
window.addEventListener("scroll", fitConversationVisualPane, { passive: true });

conversationVisualStage?.addEventListener("click", () => {
  const selected = conversationVisualArtifacts.find((artifact) => artifact.identity === selectedConversationVisual);
  if (selected) window.open(assetUrl(selected.source), "_blank", "noopener");
});

conversationVisualStage?.addEventListener("wheel", (event) => {
  const selected = conversationVisualArtifacts.find((artifact) => artifact.identity === selectedConversationVisual);
  if (!selected || !Array.isArray(selected.frames) || selected.frames.length < 2) return;
  const direction = event.deltaY > 0 || event.deltaX > 0 ? 1 : -1;
  const nextIndex = Math.max(0, Math.min(selected.frames.length - 1, Number(selected.frameIndex || 0) + direction));
  selectConversationFrame(nextIndex);
  event.preventDefault();
}, { passive: false });

visualTimelineSlider?.addEventListener("input", () => {
  selectConversationFrame(Number(visualTimelineSlider.value));
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

function persistConversationIndex() {
  const existing = readConversationIndex();
  const previous = existing.find((item) => item.id === sessionId);
  const firstUserMessage = chatHistory.find((item) => item.role === "user" && item.timestamp);
  const firstQuestionTime = Number(firstUserMessage?.timestamp) || Date.now();
  const current = {
    id: sessionId,
    title: conversationTitle(),
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
  const response = await fetch(apiUrl("/api/session"), {
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
    let response = await fetch(apiUrl("/api/sessions"));
    if (!response.ok) {
      // Compatibility with a server process started before the session index
      // endpoint was added: recover the server's current session directly.
      const latestResponse = await fetch(apiUrl("/api/session?latest=1"));
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
    // A new session is intentionally allowed to be absent from the server
    // index. Do not silently switch it to an older conversation merely because
    // other sessions exist; the user can choose those from the history menu.
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
      return {
        id,
        title,
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
      button.appendChild(title);
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
    source: "UNIFIED AGENT",
    // ``tool_name`` is the selected Agent route, not an external tool call.
    // Only actual tool_runs belong in the external-tool row.
    toolNames: runs.map((run) => displayToolName(run)).filter(Boolean),
    route: data?.termination_reason || data?.route_mode || "completed",
    elapsedMs: data?.elapsed_ms,
    diagnostic: data?.diagnostic || null,
    requestTrace: data?.request_trace || null,
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
  if (name && typeof name === "object") {
    if (name.display_name) return name.display_name;
    name = name.tool_name || name.api || "";
  }
  const capability = runtimeConfig.capabilities?.[String(name || "").split(".")[0]];
  const registered = capability?.operations?.[name];
  if (registered?.display_name) return registered.display_name;
  if (capability?.display_name) return capability.display_name;
  const labels = {
    unified_agent: "统一 Agent",
    sevir: "SEVIR 观测资料",
    geolocator: "地点与区域定位",
    cascast: "CasCast 局地短临",
    era5: "ERA5/ERA5T 再分析（非实时）",
    aurora: "Aurora 区域环境预测",
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
    "cascast.forecast": "运行或读取局地集合预测",
    "aurora.case": "读取缓存的 ERA5/Aurora 案例",
    "aurora.forecast": "运行或读取 Aurora 区域预测",
    "aurora.regional_fields": "读取 ERA5/Aurora 区域场",
    // Legacy trace labels remain so old saved conversations still render.
    "data.load_hdf5": "读取连续 SEVIR 观测",
    "data.match_uploaded_event": "匹配观测事件",
    "foundation.fetch_era5": "按需读取 ERA5/ERA5T 再分析",
    "foundation.era5_snapshot": "读取 ERA5/ERA5T 再分析快照",
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
  return prefix ? prefixLabels[prefix] : "执行分析";
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
      image.src = assetUrl(attachment.preview);
      image.alt = attachment.title || "上传的VIL图片";
      if (attachment.fallback) {
        image.addEventListener("error", () => {
          if (image.dataset.fallbackTried) return;
          image.dataset.fallbackTried = "1";
          image.src = assetUrl(attachment.fallback);
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
    let response = await fetch(apiUrl(`/api/session?session_id=${encodeURIComponent(sessionId)}`));
    let state = response.ok ? await response.json() : {};
    let restored = Array.isArray(state.turns) ? state.turns : [];
    // Batch runners may persist several retry records for one question.  The
    // test session should show one card per question, using the best completed
    // result while retaining genuine failures when no successful attempt exists.
    // A genuinely new browser session must open on the welcome screen.
    // Existing sessions still restore through their persisted session ID;
    // older conversations remain available from the conversation picker.
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
        const cachedResult = cachedByMessage.get(`${turnId}:assistant`)?.result || {};
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
              status: turn.status || cachedResult.status || "unknown",
              presentation: turn.presentation || cachedResult.presentation || null,
              termination_reason: turn.termination_reason || "",
              request_id: turn.request_id || "",
              tool_result: {
                artifacts: Array.isArray(turn.artifacts) ? turn.artifacts : [],
                artifact_count: Array.isArray(turn.artifacts) ? turn.artifacts.length : 0,
              },
              artifacts: Array.isArray(turn.artifacts) ? turn.artifacts : [],
              artifact_count: Array.isArray(turn.artifacts) ? turn.artifacts.length : 0,
            },
            timestamp: timestamp + 1,
            turnId,
            meta: {
              toolNames,
              route: turn.termination_reason || "unknown",
              codeExecutionCount: Number(turn.code_execution_count || 0),
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
  const cascast = { ...(runtimeConfig.cascast || {}), ...(runtimeConfig.capabilities?.cascast || {}) };
  const live = weatherModelInput.value === "cascast_live";
  const eventIds = live ? (cascast.local_hdf5_event_ids || []) : (cascast.event_ids || []);
  const currentEvent = eventInput.value;
  eventInput.innerHTML = "";
  const unboundOption = document.createElement("option");
  unboundOption.value = "";
  unboundOption.textContent = "不指定事件（由当前请求或 Agent 发现）";
  eventInput.appendChild(unboundOption);
  eventIds.forEach((eventId, index) => {
    const option = document.createElement("option");
    option.value = eventId;
    option.textContent = `事件 ${index + 1} · ${eventId}`;
    eventInput.appendChild(option);
  });
  if (eventIds.includes(currentEvent)) eventInput.value = currentEvent;
  else eventInput.value = "";
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
  const response = await fetch(apiUrl("/api/upload"), {
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
  if (files.length < 2) throw new Error("连续序列至少需要2张图片；13张才可用于CasCast。 ");
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
    const response = await fetch(apiUrl("/api/upload-sequence"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frames }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "序列上传失败");
  uploadedSequenceId = result.upload_id;
  uploadedSampleId = "";
  attachmentPendingForMessage = true;
  if (result.status !== "ok") {
    dataSourceInput.value = "upload_sequence";
    taskInput.value = "auto";
    weatherModelInput.value = "persistence_baseline";
    attachmentTitle.textContent = `${files.length}张图片 · 序列检查未通过`;
    uploadStatus.textContent = `已保留上传诊断：${result.reason || "序列结构不完整"}。可以继续询问具体缺帧或顺序问题。`;
    sequenceUploadStatus.textContent = "该序列不能用于CasCast；系统会先解释结构问题。";
    sequenceUploadStatus.classList.remove("hidden");
    syncDataSource();
    persistActiveUpload();
    syncComposerAvailability();
    return;
  }
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
  if (files.length > 1 && files.length < 13) {
    attachmentTray.classList.remove("hidden");
    attachmentTitle.textContent = `选择了${files.length}张图片`;
    uploadStatus.textContent = "当前序列数量不足13张，仍将上传以生成缺帧诊断。";
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
    const response = await fetch(apiUrl("/api/config"));
    const data = await response.json();
    runtimeConfig = data.runtime || {};
    runtimeConfig.cache_registry = data.cache_registry || {};
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

function executionStatusLabel(status) {
  const labels = {
    ok: "完成", completed: "完成", running: "进行中", queued: "等待中",
    partial: "部分完成", error: "失败", failed: "失败", unavailable: "不可用",
    blocked: "已阻止", incomplete: "未完成", cancelled: "已取消",
    cache_hit: "使用缓存", recovered: "已恢复", unknown: "状态未知",
  };
  return labels[String(status || "unknown").toLowerCase()] || String(status || "状态未知");
}

function executionStageLabel(name) {
  if (name && typeof name === "object") {
    if (name.display_name) return name.display_name;
    name = name.stage || name.name;
  }
  const value = String(name || "").toLowerCase();
  const stages = {
    data: "读取数据", discovery: "查询可用资料", prediction: "运行预测", analysis: "分析数据",
    context: "整理模型上下文", agent_context: "整理模型上下文",
    context_budget: "检查上下文预算", context_budget_exceeded: "检查上下文预算",
    model: "调用语言模型", execute: "执行代码与工具", finish: "整理最终回答",
    select_tools: "选择工具并读取接口说明", format_repair: "检查模型动作格式",
    read_context: "读取历史记录与接口说明", citation_repair: "补齐资料引用",
    knowledge_reuse: "复用已有资料",
  };
  if (stages[value]) return stages[value];
  if (value.includes("asset") || value.includes("context") || value.includes("request")) return "绑定地点、时间和案例数据";
  if (value.includes("model_api") || value.includes("llm") || value.includes("agent_model")) return "调用语言模型决定下一步";
  if (value.includes("queue")) return "等待 Agent 执行槽位";
  if (value.includes("agent_execution")) return "执行代码并读取工具结果";
  if (value.includes("unified")) return "执行代码、读取观察并继续分析";
  if (value.includes("foundation") || value.includes("aurora")) return "分析区域环境资料";
  if (value.includes("cascast") || value.includes("forecast")) return "运行预测并分析结果";
  if (value.includes("render") || value.includes("visual")) return "整理图表";
  if (value.includes("answer") || value.includes("synth")) return "整理居民可读结论";
  return "执行分析步骤";
}

function routeDisplayLabel(route) {
  const value = String(route || "").toLowerCase();
  if (value.includes("unified") || value.includes("completed")) return "统一 Agent";
  return route ? "Agent 执行" : "—";
}

function toolDisplaySentence(run) {
  const name = displayToolName(run);
  const status = executionStatusLabel(run?.status);
  const elapsed = run?.elapsed_ms == null ? NaN : Number(run.elapsed_ms);
  const cache = run?.cache_hit === true ? "，复用缓存" : "";
  const argumentsText = readableToolArguments(run?.arguments);
  const resultText = String(run?.result_summary || "").trim();
  const errorText = executionErrorText(run?.error);
  return `${name}${argumentsText ? `（${argumentsText}）` : ""}：${status}${cache}${Number.isFinite(elapsed) ? `，耗时 ${(elapsed / 1000).toFixed(elapsed >= 1000 ? 1 : 2)} 秒` : ""}${resultText ? `；${resultText}` : ""}${errorText ? `；${errorText}` : ""}`;
}

function readableToolArguments(argumentsValue) {
  if (!argumentsValue || typeof argumentsValue !== "object" || Array.isArray(argumentsValue)) return "";
  const labels = {
    subject_key: "精确数据", event_id: "事件", upload_id: "上传资料", threshold: "阈值",
    lead_minutes: "预报时效", minutes_before: "往前观测", minutes_after: "往后观测",
    horizon_hours: "预报时长", ensemble_members: "集合成员",
    place: "地点", latitude: "纬度", longitude: "经度",
    name: "名称", level: "层级", query: "检索词", limit: "上限",
    year: "年份", month: "月份", event_type: "事件类型",
    start_time_utc: "开始时间（UTC）", end_time_utc: "结束时间（UTC）",
    observation_time_utc: "观测时间（UTC）", time_match: "时间匹配",
  };
  const parts = [];
  Object.entries(argumentsValue)
    .filter(([, value]) => value != null && value !== "" && typeof value !== "object")
    .slice(0, 5).forEach(([key, value]) => {
      const suffix = ["lead_minutes", "minutes_before", "minutes_after"].includes(key)
        ? "分钟" : key === "horizon_hours" ? "小时" : "";
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
  const eventResolution = data.event_resolution || null;
  const preflight = data.foundation_preflight || null;
  const runs = Array.isArray(data.tool_runs) ? data.tool_runs : [];
  const hasDetails = Boolean(
    requestTrace || diagnostic || preflight || runs.length,
  );
  panel.classList.toggle("hidden", !hasDetails);
  if (!hasDetails) return;
  const outcome = resultPresentation(data);
  const status = outcome.status;
  const failureMessage = data.failure_message || requestTrace?.failure_message || diagnostic?.message || "";
  const nextAction = data.next_action || requestTrace?.next_action || diagnostic?.next_action || "";
  const failurePhase = executionFailurePhase(data);
  statusNode.textContent = outcome.label;
  statusNode.dataset.status = status;
  grid.replaceChildren();
  const rows = [
    ["执行路径", routeDisplayLabel(data.route_mode || data.termination_reason)],
    ["状态", executionStatusLabel(status)],
    ["Agent 模型调用", `${data.model_calls ?? requestTrace?.model_calls ?? 0} 次`],
    ...(contextSummaryCallCount(data) ? [["上下文摘要调用", `${contextSummaryCallCount(data)} 次`]] : []),
    ["工具调用", `${data.tool_calls ?? requestTrace?.tool_calls ?? runs.length} 次`],
    ["缓存命中", `${data.cache_hits ?? requestTrace?.cache_hits ?? 0} 次`],
    ["失败阶段", failurePhase ? executionStageLabel(failurePhase) : "—"],
    ["事件绑定", eventResolution
      ? `${eventResolution.source || "unknown"} · ${eventResolution.event_id || "未匹配"}`
      : "—"],
    ["总耗时", data.elapsed_ms != null ? `${(Number(data.elapsed_ms) / 1000).toFixed(2)} 秒` : (requestTrace?.elapsed_ms != null ? `${(Number(requestTrace.elapsed_ms) / 1000).toFixed(2)} 秒` : "—")],
    ...(failureMessage ? [["失败原因", failureMessage]] : []),
    ...(nextAction ? [["建议操作", nextAction]] : []),
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
  stagesNode.classList.toggle("hidden", stages.length === 0 && runs.length === 0);
  if (stages.length) {
    stages.forEach((stage) => {
      const item = document.createElement("li");
      item.className = `diagnostic-stage ${stage.status || "unknown"}`;
      const elapsed = Number(stage.elapsed_ms);
      const modelText = stage.model
        ? `，使用 ${stage.provider ? `${stage.provider} / ` : ""}${stage.model}`
        : "";
      item.textContent = `${executionStageLabel(stage)}：${executionStatusLabel(stage.status)}${Number.isFinite(elapsed) ? `，耗时 ${(elapsed / 1000).toFixed(elapsed >= 1000 ? 1 : 2)} 秒` : ""}${stage.route_mode ? `，选择 ${stage.route_mode} 模式` : ""}${modelText}`;
      stagesNode.appendChild(item);
    });
  } else if (runs.length) {
    runs.forEach((run) => {
      const item = document.createElement("li");
      item.className = `diagnostic-stage ${run.status || "unknown"}`;
      item.textContent = toolDisplaySentence(run);
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
  const outcome = resultPresentation(data);
  runState.textContent = outcome.label;
  runState.className = `run-state ${outcome.status === "partial" ? "partial" : outcome.incomplete ? "failed" : "done"}`;

  const runs = Array.isArray(data.tool_runs) ? data.tool_runs : [];
  const displayResult = combinedDisplayResult(data, runs);
  // Build the right-side artifact index before rendering Markdown so image
  // references in the answer become stable labels such as 分析图表02.
  renderConversationVisualPane(data);
  renderAnswer(data.answer || "未返回文字结论");
  document.querySelector("#result-title").textContent = "分析结论";
  document.querySelector("#result-source").textContent = "UNIFIED AGENT";
  turnDetailsSource.textContent = "UNIFIED AGENT";
  document.querySelector("#tool-name").textContent = runs.length > 1
    ? runs.map((run) => displayToolName(run)).join(" → ")
    : runs.length ? displayToolName(runs[0]) : "未调用外部工具";
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

  const sourceTrace = turnDetails.querySelector(".trace-card");
  if (sourceTrace) {
    const technical = cloneTurnNode(sourceTrace);
    technical.classList.add("chat-turn-technical");
    technical.dataset.ownerMessageId = presentation.dataset.ownerMessageId;
    // Diagnostics are optional even when a task is incomplete.
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

  return Object.assign({}, ...results);
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
    const defaultExecutionFlow = "理解问题 → 按需读取数据 → 计算/绘图 → 根据证据回答";
    const path = ["绑定数据与上下文"];
    attempts.forEach((item) => {
      const label = liveActionLabels[item?.action] || String(item?.action || "继续分析").replaceAll("_", " ");
      if (path[path.length - 1] !== label) path.push(label);
    });
    executionFlow.textContent = attempts.length ? path.join(" → ") : defaultExecutionFlow;
  }
  if (sessionBudget) {
    const summaryCalls = contextSummaryCallCount(data);
    sessionBudget.textContent = `${data.session_id || sessionId} · ${routeDisplayLabel(data.route_mode)} · 模型 ${data.model_calls ?? "—"} 次${summaryCalls ? ` · 摘要 ${summaryCalls} 次` : ""} · 执行 ${data.code_execution_count ?? "—"} 次`;
  }
  attempts.forEach((item, index) => {
    const presentation = executionStepPresentation(attempts, index);
    const block = document.createElement("details");
    block.className = "agent-attempt";
    const summary = document.createElement("summary");
    const timingText = stepTimingText(item);
    const elapsedText = timingText ? ` · ${timingText}` : "";
    const stepNumber = executionStepNumber(attempts, index);
    summary.textContent = `${stepNumber}：${liveActionLabels[item.action] || "执行步骤"} · ${executionStatusLabel(presentation.status)}${elapsedText}`;
    block.appendChild(summary);
    if (presentation.status !== "recovered" && (["error", "failed", "partial", "blocked", "incomplete"].includes(presentation.status) || item.error)) {
      block.open = true;
    }
    if (presentation.summary) {
      const activity = document.createElement("p");
      activity.className = "agent-decision";
      activity.textContent = presentation.summary;
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
      error.textContent = `原始错误：${executionErrorText(item.error)}`;
      block.appendChild(error);
    }
    const stepCalls = Array.isArray(item.calls) ? item.calls : (Array.isArray(item.tools) ? item.tools : []);
    if (stepCalls.length) {
      const calls = document.createElement("ol");
      calls.className = "agent-calls";
      stepCalls.forEach((call) => {
        const callNode = document.createElement("li");
        callNode.textContent = toolDisplaySentence(call);
        calls.appendChild(callNode);
      });
      block.appendChild(calls);
    }
    renderExecutionFeedback(block, item);
    trace.appendChild(block);
  });
}

function renderArtifacts(data) {
  const container = document.querySelector("#agent-artifacts");
  const list = document.querySelector("#artifact-list");
  list.innerHTML = "";
  const artifactIds = new Set();
  const artifacts = (Array.isArray(data.artifacts) ? data.artifacts : []).filter((item) => {
    if (!item || !(item.url || item.image_url) || item.verified === false || item.quality === "empty_plot") return false;
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
    image.src = assetUrl(artifact.thumbnail?.url || artifact.url || artifact.image_url);
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
    link.href = assetUrl(artifact.url || artifact.image_url);
    link.target = "_blank";
    link.rel = "noopener";
    link.appendChild(image);
    figure.append(link, caption);
    list.appendChild(figure);
  });
}

function renderAnswer(text) {
  const container = document.querySelector("#answer");
  renderMarkdown(container, text);
}

function renderMarkdown(container, text) {
  container.innerHTML = "";
  const artifactLabels = new Map();
  const normalizedText = String(text ?? "").replace(
    /!\[([^\]]*)\]\(\s*(\/api\/artifacts\/[0-9a-f]{32}\.png)\s*\)/gi,
    (_match, _alt, url) => {
      const knownIndex = conversationVisualArtifacts.findIndex((artifact) => artifact.source === url);
      const index = knownIndex >= 0 ? knownIndex + 1 : artifactLabels.size + 1;
      artifactLabels.set(url, index);
      const identity = knownIndex >= 0 ? conversationVisualArtifacts[knownIndex].identity : url;
      return `[分析图表${String(index).padStart(2, "0")}](#artifact:${encodeURIComponent(identity)})`;
    },
  );
  const rawLines = normalizedText.replace(/\r/g, "").split("\n");
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
    const pattern = /(\[([^\]]+)\]\(((?:https?:\/\/|#artifact:)[^\s)]+|\/api\/knowledge\/chunks\/\d+)\)|`([^`]+)`|\*\*([^*]+)\*\*|__([^_]+)__|~~([^~]+)~~|\*([^*]+)\*|_([^_]+)_)/g;
    let cursor = 0;
    let match;
    const appendText = (text) => { if (text) parent.appendChild(document.createTextNode(text)); };
    while ((match = pattern.exec(String(value))) !== null) {
      appendText(String(value).slice(cursor, match.index));
      if (match[2]) {
        const link = document.createElement("a");
        if (match[3].startsWith("#artifact:")) {
          const identity = decodeURIComponent(match[3].slice("#artifact:".length));
          link.href = "#";
          link.className = "artifact-text-link";
          link.addEventListener("click", (event) => {
            event.preventDefault();
            const artifact = conversationVisualArtifacts.find((item) => item.identity === identity || item.source === identity);
            if (artifact) {
              activeConversationLayer = artifact.layer;
              renderConversationLayerTabs();
              renderConversationThumbnails();
              selectConversationVisual(artifact.identity);
              conversationVisualPane?.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
          });
        } else {
          link.href = match[3].startsWith("/api/knowledge/") ? assetUrl(match[3]) : match[3];
          link.target = "_blank";
          link.rel = "noopener noreferrer";
        }
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
cancelRequestButton?.addEventListener("click", async () => {
  if (!requestInFlight) return;
  if (!activeRequestController) return;
  const controller = activeRequestController;
  const requestId = activeRequestId;
  cancelRequestButton.disabled = true;
  cancelRequestButton.querySelector("span").textContent = "停止中";
  runState.textContent = "正在停止";
  conversationStatus.textContent = "正在停止剩余执行；已完成的步骤会保留。";
  try {
    const cancellation = await requestTaskCancellation(requestId);
    if (!cancellation) {
      cancelRequestButton.disabled = false;
      cancelRequestButton.querySelector("span").textContent = "重试停止";
      runState.textContent = "停止尚未确认";
      conversationStatus.textContent = "停止请求尚未得到服务端确认；任务仍以后台状态为准，可再次点击停止。";
    }
  } catch (_error) {
    cancelRequestButton.disabled = false;
    cancelRequestButton.querySelector("span").textContent = "重试停止";
  } finally {
    if (activeRequestTransport !== "task") controller.abort();
  }
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
      ? apiUrl(`/api/uploaded-frame?upload_id=${encodeURIComponent(uploadedSampleId)}`)
      : ""),
    fallback: uploadedSequenceId
      ? apiUrl(`/api/uploaded-sequence-frame?upload_id=${encodeURIComponent(uploadedSequenceId)}&series=history&index=12`)
      : (uploadedSampleId ? apiUrl(`/api/uploaded-frame?upload_id=${encodeURIComponent(uploadedSampleId)}`) : ""),
  } : null;
  const requestPayload = {
    data_source: uploadedSequenceId ? "upload_sequence" : (uploadedSampleId ? "upload" : (liveContextReady ? "live" : "none")),
    data_policy: "live",
    requested_time_utc: normalizedAnalysisTimeUtc(),
    place_name: analysisPlaceInput?.value.trim() || "",
    latitude: optionalCoordinate(analysisLatitudeInput),
    longitude: optionalCoordinate(analysisLongitudeInput),
    event_id: (uploadedSequenceId || uploadedSampleId)
      ? ""
      : (eventInput?.value.trim() || ""),
    upload_id: uploadedSequenceId || uploadedSampleId,
    session_id: sessionId,
    turn_id: turnId,
    request_id: requestId,
    continue_study_case: chatHistory.some((item) => item.role === "assistant"),
    trace: true,
    generate_visual: Boolean(generateVisualInput?.checked),
    question,
  };
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
  const progressState = {
    done: false,
    startedAt: Date.now(),
    lastChangeAt: Date.now(),
    lastSignature: "",
    lastSnapshot: null,
    pollErrors: 0,
  };
  let progressPromise = Promise.resolve();
  let requestController = new AbortController();
  let durableTaskSettled = false;
  activeRequestController = requestController;
  activeRequestId = requestId;
  activeRequestPhase = "submission";
  activeRequestTransport = "none";
  if (cancelRequestButton) {
    cancelRequestButton.classList.remove("hidden");
    cancelRequestButton.disabled = false;
    cancelRequestButton.querySelector("span").textContent = "停止";
  }
  // Live requests may download data and run local inference. Do not impose
  // the old 90-second browser abort; the server/model pipeline owns its
  // configurable live safety ceiling and persists successful outputs.
  const requestBudgetSeconds = Math.max(
    1,
    Number(runtimeConfig.agent_request_budget_seconds || 30),
  );
  const requestTimeout = null;

  try {
    runState.textContent = "正在提交";
    conversationStatus.textContent = "正在绑定请求事实并启动统一 Agent。";
    activeRequestTransport = "task";
    persistActiveTask({ requestId, sessionId, turnId, question, submittedAt: timestamp });
    if (cancelRequestButton) {
      cancelRequestButton.disabled = true;
      cancelRequestButton.querySelector("span").textContent = "提交中";
    }
    const submitted = await submitDurableTaskWithReconnect(
      requestPayload, requestId, progressState, pendingMessage,
    );
    if (cancelRequestButton) {
      cancelRequestButton.disabled = false;
      cancelRequestButton.querySelector("span").textContent = "停止";
    }
    activeRequestPhase = "execution";
    runState.textContent = "准备执行";
    conversationStatus.textContent = "统一 Agent 正在直接处理，并持续更新进度。";
    if (cancelRequestButton) cancelRequestButton.querySelector("span").textContent = "停止";
    ensureLiveExecutionPanel(pendingMessage);
    renderLiveExecutionProgress(pendingMessage, {
      status: "running",
      terminal: false,
      current_stage: "unified_agent",
      progress: 0.13,
      current_activity: {
        action: "agent_started",
        status: "ok",
        summary: "已启动统一 Agent，等待执行槽位。",
      },
      live_steps: [],
    }, progressState);
    progressPromise = pollRequestProgress(requestId, progressState, pendingMessage);
    const data = await waitForDurableTask(requestId, progressState, pendingMessage);
    durableTaskSettled = true;
    await presentDurableTaskResult(data, pendingMessage, turnId);
  } catch (error) {
    if (error?.durableSubmissionRejected) durableTaskSettled = true;
    progressState.done = true;
    progressState.pollController?.abort();
    const lastSnapshot = await fetchRequestSnapshotOnce(requestId);
    const rejected = Boolean(error?.durableSubmissionRejected);
    const answer = rejected
      ? `任务未能提交：${error.message}`
      : "连接已中断，后台任务状态尚未确认。恢复连接后可继续查看任务结果。";
    const failureData = {
      status: rejected ? "failed" : "unknown",
      answer,
      route_mode: "request_failed",
      termination_reason: rejected ? "submission_rejected" : "client_connection_lost",
      failure_class: rejected ? "" : "transport_reset",
      failure_message: rejected ? "服务端未接受本轮任务。" : "连接中断，任务状态待确认。",
      next_action: rejected ? "请检查输入后重新提交。" : "恢复连接后继续查看本轮任务。",
      request_trace: lastSnapshot,
      agent_trace: Array.isArray(lastSnapshot?.live_steps) ? lastSnapshot.live_steps : [],
      model_calls: lastSnapshot?.model_calls ?? 0,
      tool_calls: lastSnapshot?.tool_calls ?? 0,
      request_id: requestId,
      session_id: sessionId,
    };
    const failureMeta = compactTurnMeta(failureData);
    completePendingMessage(pendingMessage, answer, failureMeta, true);
    chatHistory.push({
      role: "assistant",
      text: answer,
      meta: failureMeta,
      result: persistedResult(failureData),
      timestamp: Date.now(),
      turnId,
    });
    persistChatHistory();
    try { renderResult(failureData, pendingMessage); } catch (displayError) { console.error(displayError); }
    renderError(answer);
    conversationStatus.textContent = rejected ? "任务未提交。" : "连接中断，任务状态待确认。";
  } finally {
    window.clearTimeout(requestTimeout);
    progressState.done = true;
    progressState.pollController?.abort();
    await progressPromise.catch(() => {});
    activeRequestController = null;
    activeRequestId = "";
    activeRequestPhase = "idle";
    if (durableTaskSettled) clearActiveTask();
    activeRequestTransport = "none";
    if (cancelRequestButton) {
      cancelRequestButton.classList.add("hidden");
      cancelRequestButton.disabled = false;
      cancelRequestButton.querySelector("span").textContent = "停止";
    }
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
  .finally(async () => {
    try { persistConversationIndex(); } catch (_error) {}
    void renderConversationList();
    await resumeActiveTaskIfPresent();
  });
loadConfig();
restoreActiveUpload();
