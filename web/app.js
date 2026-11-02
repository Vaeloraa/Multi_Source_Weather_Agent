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
const sessionStorageKey = "weather-agent-session-id";
const chatStoragePrefix = "weather-agent-chat-history:";
const activeUploadPrefix = "weather-agent-active-upload:";
const activeTaskPrefix = "weather-agent-active-task:";
const conversationIndexKey = "weather-agent-conversations";
// The interface can also be opened directly as web/index.html.  In
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
    return "[Large array hidden. Summary statistics remain available.]";
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
    title.textContent = "Step results and reusable data";
    details.appendChild(title);
    details.appendChild(document.createElement("p"));
    details.appendChild(document.createElement("pre"));
    container.appendChild(details);
  }
  const repeated = Number(feedback.repeated_call_count || 0);
  const cached = Number(feedback.repeated_cached_count || 0);
  setTextIfChanged(details.querySelector("p"),
    `Returned results. Review the values and method before drawing conclusions.${repeated
      ? ` ${repeated} repeated calls, ${cached} served from cache. Existing variables can be reused.` : ""}`);
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
    questionInput.placeholder = "Location and time are set. Ask about observations or forecasts…";
  } else if (!hasUploadedInput()) {
    questionInput.placeholder = "Ask about weather, data or models. Include a place and time…";
  } else if (uploadedSequenceId) {
    questionInput.placeholder = "Ask about this sequence or its next-hour forecast…";
  } else {
    questionInput.placeholder = "Ask about intensity, coverage or spatial structure…";
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
  request_received: "Request received",
  asset_binding: "Validating inputs",
  agent_queue: "Waiting for an execution slot",
  unified_agent: "Running analysis and reviewing results",
  agent_model: "Choosing the next step",
  agent_execution: "Running code and reading tool results",
  agent_step_completed: "Step complete. Continuing analysis",
  verified_cache: "Loading and checking saved results",
  completed: "Analysis complete",
  failed: "Analysis incomplete",
};

function retrievalTimingText(call) {
  if (call?.cache_hit || !call?.retrieval?.timings) return "";
  const stages = call.retrieval.timings;
  const parts = [];
  if (Number.isFinite(stages.embedding_seconds)) parts.push(`Embedding ${stages.embedding_seconds.toFixed(2)} s`);
  if (Number.isFinite(stages.rerank_seconds)) parts.push(`Reranking ${stages.rerank_seconds.toFixed(2)} s`);
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
    parts.push(`GPU queue ${queue.toFixed(3)} s; model and dependency loading ${load.toFixed(2)} s; inference ${inference.toFixed(2)} s`);
  }
  return parts.join("; ");
}

const liveActionLabels = {
  request_received: "Receive request",
  asset_binding: "Bind data and context",
  queue: "Wait for an execution slot",
  prepare: "Prepare the agent",
  model: "Call the language model",
  context: "Prepare model context",
  context_budget: "Check context budget",
  select_tools: "Select tools and load instructions",
  read_context: "Read records and tool instructions",
  citation_repair: "Add source citations",
  knowledge_reuse: "Reuse retrieved documents",
  expand: "Read capability instructions",
  execute: "Run code and tools",
  cache_read: "Read verified cache",
  solution: "Compose answer",
  finish: "Finalize answer",
  solution_validation: "Check supporting results",
  empty_action_repair: "Retry empty action",
  format_repair: "Check action format",
  contract_required: "Load tool contracts",
  execution_budget: "Check execution budget",
  stop: "Stop analysis",
};

function executionStepNumber(steps, index) {
  const isPreparation = (item) => Number(item?.attempt) === 0
    && ["request_received", "asset_binding", "queue", "prepare", "cache_read"].includes(item?.action);
  if (isPreparation(steps[index])) return "Preparation";
  // Old saved traces used model-call counts for both fields, so count events
  // here as well. Context checks can fail without making another model call.
  return `Step ${steps.slice(0, index + 1).filter((item) => !isPreparation(item)).length}`;
}

function executionStepPresentation(steps, index) {
  const item = steps[index];
  const recovered = item?.action === "format_repair" && steps.slice(index + 1).some((later) =>
    ["select_tools", "execute", "read_context", "finish"].includes(later?.action));
  return {
    status: recovered ? "recovered" : String(item?.status || "unknown").toLowerCase(),
    summary: recovered ? "A valid replacement action was received. The invalid action was not executed." : item?.summary || "",
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
  if (seconds < 60) return `${seconds} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${(seconds % 60).toFixed(1).padStart(4, "0")} s`;
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
  const parts = [`Total ${liveElapsedText(total)}`];
  if (item?.action === "execute") {
    if (Number.isFinite(model)) parts.push(`Model ${liveElapsedText(model)}`);
    if (Number.isFinite(execution)) parts.push(`Execution ${liveElapsedText(execution)}`);
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
    setTextIfChanged(waiting, "Creating the execution record…");
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
    const action = liveActionLabels[item?.action] || String(item?.action || "Execution step").replaceAll("_", " ");
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
    setTextIfChanged(error, item?.error ? `Error details: ${executionErrorText(item.error)}` : "");
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
      const cache = call?.cache_hit ? " · Cache hit" : "";
      const result = call?.result_summary ? ` · ${call.result_summary}` : "";
      const retrievalTiming = retrievalTimingText(call);
      const callError = call?.error ? ` · ${executionErrorText(call.error)}` : "";
      setTextIfChanged(
        callNode,
        `${displayToolName(call)}: ${callStatus}${cache}${result}${retrievalTiming ? ` · ${retrievalTiming}` : ""}${callError}`,
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
      <div><span class="live-pulse" aria-hidden="true"></span><strong>Analysis progress</strong></div>
      <div class="live-heading-meta"><small data-live-role="task">Waiting for task</small><span data-live-role="elapsed">Just started</span></div>
    </div>
    <div class="live-progress-track" role="progressbar" aria-label="Analysis progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="1">
      <i data-live-role="progress"></i>
    </div>
    <p class="live-current-activity" data-live-role="activity">Connecting to the analysis service…</p>
    <ol class="live-step-list" data-live-role="steps"></ol>
    <p class="live-progress-note" data-live-role="note">Progress updates automatically.</p>
  `;
  content.appendChild(panel);
  return panel;
}

function renderDurableTaskState(task, requestId, pendingMessage, { resumed = false } = {}) {
  const state = String(task?.state || "unknown");
  const labels = {
    queued: "Queued",
    running: "Running",
    cancelling: "Cancelling",
    succeeded: "Succeeded",
    failed: "Incomplete",
    cancelled: "Cancelled",
    interrupted: "Interrupted by service restart",
  };
  const panel = ensureLiveExecutionPanel(pendingMessage);
  if (panel) {
    panel.dataset.taskState = state;
    const taskLabel = panel.querySelector("[data-live-role='task']");
    if (taskLabel) {
      setTextIfChanged(taskLabel, `${resumed ? "Resumed · " : ""}${labels[state] || state} · ${String(requestId).slice(0, 12)}`);
      taskLabel.title = `Task ID: ${requestId}`;
    }
  }
  runState.dataset.taskState = state;
  runState.setAttribute(
    "aria-label",
    `${labels[state] || state}; task ${requestId}`,
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
  const summary = String(activity.summary || requestStageLabels[stage] || "Analyzing");
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
  ) || "Running");
  const track = panel.querySelector(".live-progress-track");
  track.setAttribute("aria-valuenow", String(percent));
  panel.querySelector("[data-live-role='progress']").style.width = `${percent}%`;
  const current = panel.querySelector("[data-live-role='activity']");
  const attemptPrefix = Number(activity.attempt) > 0 ? `Step ${activity.attempt} · ` : "";
  setTextIfChanged(current, `${attemptPrefix}${summary}`);
  const timeline = panel.querySelector("[data-live-role='steps']");
  syncLiveStepList(timeline, steps);

  const note = panel.querySelector("[data-live-role='note']");
  const stalledSeconds = Math.max(0, Math.round((now - state.lastChangeAt) / 1000));
  if (snapshot?.failure_message || snapshot?.next_action) {
    note.className = "live-progress-note error";
    setTextIfChanged(note, `${snapshot.failure_message || "Analysis incomplete. "}${snapshot.next_action ? ` Next: ${snapshot.next_action}` : ""}`);
  } else if (status === "cancelled") {
    note.className = "live-progress-note warning";
    setTextIfChanged(note, "Stopping. Completed results will be retained.");
  } else if (stalledSeconds >= 20) {
    note.className = "live-progress-note warning";
    setTextIfChanged(note, `This step has been running for ${stalledSeconds} s. Data retrieval or inference may still be in progress. Wait or select Stop.`);
  } else {
    note.className = "live-progress-note";
    setTextIfChanged(note, "Progress updates automatically. Completed steps remain in the record.");
  }
}

function renderProgressReconnectState(message, state) {
  const panel = ensureLiveExecutionPanel(message);
  if (!panel || state.pollErrors < 3) return;
  const note = panel.querySelector("[data-live-role='note']");
  note.className = "live-progress-note warning";
  setTextIfChanged(note, "Reconnecting to live progress. You can select Stop to cancel the task.");
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
        const label = requestStageLabels[stage] || "Analyzing";
        setTextIfChanged(runState, `${label}${percent}`);
        const elapsedText = liveElapsedText(snapshot.elapsed_ms);
        setTextIfChanged(
          conversationStatus,
          `${label}${percent}${elapsedText ? ` · Elapsed ${elapsedText}` : ""}; tool results and execution records are saved on completion.`,
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
      ? "Task cancelled."
      : "Task ended. Results are not yet available.",
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
            answer: "The task record was not found after three checks.",
            request_id: requestId,
            session_id: sessionId,
            termination_reason: "durable_task_record_missing",
            failure_class: "task_state_missing",
            failure_message: "No matching task record was found.",
            next_action: "Check whether the server data directory changed. Resubmit if the task record was removed.",
          };
        }
        throw new Error("Task record not yet available");
      }
      if (!response.ok) throw new Error(`Task status request returned HTTP ${response.status}`);
      const body = await response.json();
      const task = body.task || {};
      networkErrors = 0;
      notFoundResponses = 0;
      const lifecycleChanged = state.lastTaskState !== task.state;
      state.lastTaskState = task.state;
      if (lifecycleChanged) renderDurableTaskState(task, requestId, pendingMessage);
      const lifecycleLabels = {
        queued: "Queued for execution",
        running: "Running in the background",
        cancelling: "Stopping safely",
        succeeded: "Execution complete",
        failed: "Execution incomplete",
        cancelled: "Cancelled",
        interrupted: "Interrupted by service restart",
      };
      // The detailed request-progress poll owns the running labels. Without
      // this guard the 600 ms task poll and 700 ms progress poll alternated
      // different strings, producing the visible refresh/flicker loop.
      if (!state.lastSnapshot || task.state !== "running") {
        setTextIfChanged(runState, lifecycleLabels[task.state] || "Reading task status");
        setTextIfChanged(
          conversationStatus,
          `${lifecycleLabels[task.state] || task.state} · Task ${String(requestId).slice(0, 12)}; progress can be restored after a refresh.`,
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
        const rejected = new Error(String(message || `Task rejected (HTTP ${response.status})`));
        rejected.durableSubmissionRejected = true;
        throw rejected;
      }
      throw new Error(String(message || `Task submission returned HTTP ${response.status}`));
    } catch (error) {
      if (error?.durableSubmissionRejected) throw error;
      state.pollErrors = Number(state.pollErrors || 0) + 1;
      renderProgressReconnectState(pendingMessage, state);
      runState.textContent = "Reconnecting to the task queue";
      conversationStatus.textContent = `Submission attempt ${attempts} is unconfirmed. Retrying with the same request key.`;
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
        conversationStatus.textContent = `Checking whether task ${String(requestId).slice(0, 12)} was queued.`;
        await new Promise((resolve) => setTimeout(resolve, 800));
        continue;
      }
      if (!response.ok) throw new Error(`Task status request returned HTTP ${response.status}`);
      const body = await response.json();
      return body.task || null;
    } catch (_error) {
      networkErrors += 1;
      state.pollErrors = Math.max(state.pollErrors || 0, networkErrors);
      renderProgressReconnectState(pendingMessage, state);
      conversationStatus.textContent = `Reconnecting to task ${String(requestId).slice(0, 12)}. Its background state is saved.`;
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
    ok: "Completed", success: "Completed", completed: "Completed", partial: "Partially completed",
    cancelled: "Cancelled", canceled: "Cancelled", interrupted: "Interrupted",
    error: "Incomplete", failed: "Incomplete", failure: "Incomplete", unavailable: "Data unavailable",
    blocked: "Incomplete", clarification_needed: "More information needed", unknown: "Result pending",
  };
  return { status, label: labels[status] || "Result pending", incomplete: !["ok", "success", "completed"].includes(status) };
}

async function presentDurableTaskResult(data, pendingMessage, turnId) {
  const answer = data.answer || "No text answer was returned.";
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
      ? `${resultPresentation(data).label}. ${data.failure_message || ""}${data.next_action ? ` Next: ${data.next_action}` : ""}`
      : `Current conversation · ${data.session_turn_count ?? Math.ceil(chatHistory.length / 2)} exchanges · Ask a follow-up`;
  } catch (displayError) {
    console.error("The answer was returned, but display failed: ", displayError);
    renderError(`The answer was returned, but part of the display failed: ${displayError.message}`);
    conversationStatus.textContent = "Answer received. Some records or charts could not be displayed.";
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
      text: active.question || "Restoring saved task",
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
  runButton.firstElementChild.textContent = "Restoring";
  runState.textContent = "Restoring";
  runState.className = "run-state running";
  conversationStatus.textContent = `Reconnecting to task ${String(active.requestId).slice(0, 12)}.`;
  if (cancelRequestButton) {
    cancelRequestButton.classList.remove("hidden");
    cancelRequestButton.disabled = false;
    cancelRequestButton.querySelector("span").textContent = "Stop";
  }

  try {
    let task = await readDurableTaskWithReconnect(
      active.requestId, progressState, pendingMessage,
    );
    if (!task) {
      const missing = {
        status: "failed",
        answer: "The saved task ID has no matching record on the server.",
        request_id: active.requestId,
        session_id: active.sessionId || sessionId,
        termination_reason: "durable_task_record_missing",
        failure_class: "task_state_missing",
        failure_message: "No matching task record was found.",
        next_action: "Check the server data directory. Resubmit if the task record was removed.",
      };
      durableTaskSettled = true;
      await presentDurableTaskResult(missing, pendingMessage, turnId);
      return;
    }
    renderDurableTaskState(task, active.requestId, pendingMessage, { resumed: true });
    activeRequestPhase = "execution";
    if (cancelRequestButton) cancelRequestButton.querySelector("span").textContent = "Stop";
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
    renderError(`Could not restore the task display: ${error.message}`);
    conversationStatus.textContent = "The task ID is saved. Refresh to reconnect.";
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
      cancelRequestButton.querySelector("span").textContent = "Stop";
    }
    syncComposerAvailability();
    runButton.firstElementChild.textContent = "Send";
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
  attachmentTitle.textContent = "Reading images…";
  uploadStatus.textContent = "Select one or 13 grayscale PNG images at 384 × 384 pixels.";
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
    attachmentTitle.textContent = saved.title || (uploadedSequenceId ? "13-frame sequence restored" : "Image restored");
    if (pendingAttachmentPreview) {
      uploadPreview.src = pendingAttachmentPreview;
      uploadPreview.classList.remove("hidden");
    }
    attachmentTray.classList.remove("hidden");
    uploadStatus.textContent = "Previous images restored. Ask a follow-up or upload new observations.";
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
      title: match[1] || "Analysis chart",
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
  all: { label: "All", kind: "Combined analysis" },
  observation: { label: "Observations", kind: "Observations" },
  environment: { label: "Environment", kind: "Regional conditions" },
  forecast: { label: "Forecast", kind: "Forecasts" },
  verification: { label: "Verification", kind: "Verification" },
  analysis: { label: "Analysis chart", kind: "Derived analysis" },
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
      title: "Observation time",
      label: "5-minute frames · Past to present",
      marks: ["−60 min", "−30 min", "Now"],
      position: 2,
    },
    forecast: {
      title: "Forecast lead time",
      label: "5-minute frames · Present to future",
      marks: ["Now", "+60 min", "+120 min", "+180 min"],
      position: 0,
    },
    verification: {
      title: "Forecast verification",
      label: "Matched valid times",
      marks: ["Now", "+60 min", "+120 min", "+180 min"],
      position: 0,
    },
    environment: {
      title: "Regional data time",
      label: "Hourly · Regional context",
      marks: ["−24 h", "−6 h", "Now", "+6 h", "+12 h", "+24 h"],
      position: 2,
    },
    analysis: {
      title: "Chart time range",
      label: "Based on the chart data",
      marks: ["Past", "Now", "Future"],
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
    span.textContent = frames[index]?.title || `Frame ${index + 1}`;
    visualTimelineScale.appendChild(span);
  });
  const frameRatio = (selectedIndex / (frames.length - 1)) * 100;
  visualTimelineMarker.style.left = `calc(${frameRatio}% - 5px)`;
  visualTimelineLabel.textContent = `${frames[selectedIndex]?.title || `Frame ${selectedIndex + 1}`} · ${selectedIndex + 1}/${frames.length}`;
  if (visualTimelineSlider) {
    visualTimelineSlider.max = String(frames.length - 1);
    visualTimelineSlider.value = String(selectedIndex);
    visualTimelineSlider.disabled = false;
    visualTimelineSlider.setAttribute("aria-valuetext", frames[selectedIndex]?.title || selected?.title || "Current time");
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
    button.setAttribute("aria-label", `View ${conversationLayerMeta[artifact.layer].label} chart ${index + 1}: ${artifact.title}`);
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
    || "Current event";
  const initializedAt = analysisTimeInput?.value
    || result?.presentation?.initialization_time_utc
    || result?.initialization_time_utc
    || "Latest available";
  const runs = Array.isArray(result?.tool_runs) ? result.tool_runs : [];
  const cacheHit = runs.some((run) => run?.cache_hit === true || /cache_hit|cached/.test(String(run?.route_mode || "")));
  const liveRun = runs.some((run) => /live/.test(String(run?.route_mode || run?.data_policy || "")));
  if (visualContextPlace) visualContextPlace.textContent = place;
  if (visualContextTime) visualContextTime.textContent = initializedAt === "Latest available" ? initializedAt : String(initializedAt).replace("T", " ").replace(/Z$/, " UTC");
  if (visualContextStatus) visualContextStatus.textContent = cacheHit ? "Loaded from cache" : (liveRun ? "Live analysis complete" : "Results ready");
  if (visualEvidenceLabel) visualEvidenceLabel.textContent = `${conversationVisualArtifacts.length} traceable views`;
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
      ? "Past to present"
      : selected.layer === "forecast"
        ? "Next 60 minutes"
        : selected.layer === "verification"
          ? "Forecast and observations"
          : "Event time range";
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
    conversationVisualDescription.textContent = "Chart unavailable. It has been hidden from this view.";
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
      title: artifact.title || artifact.chart_type || "Analysis chart",
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
  conversationVisualCount.textContent = `${conversationVisualArtifacts.length} charts`;
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
  if (window.confirm("Delete this question and answer?")) {
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
  if (!response.ok) throw new Error(result.error || "Could not delete the exchange");

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
    ? `Exchange deleted · ${Math.ceil(chatHistory.length / 2)} exchanges remaining`
    : "Conversation cleared. Ask a new question.";
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
        : "Untitled conversation");
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
        time.textContent = item.createdAt ? new Date(item.createdAt).toLocaleString("en-US", {
          month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit",
        }) : "";
        button.appendChild(time);
      }
      button.addEventListener("click", () => switchConversation(item.id));
      target.appendChild(button);
    });
    target.scrollTop = previousScrollTop;
  };

  renderItems(list, entries, "No saved conversations");
}

function resetConversationView() {
  chatHistory = [];
  chatThread.innerHTML = "";
  emptyState.classList.remove("hidden");
  resultContent.classList.add("hidden");
  turnDetails.classList.add("hidden");
  turnDetails.open = false;
  errorBox.classList.add("hidden");
  runState.textContent = "Ready";
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
  conversationStatus.textContent = "Restoring conversation…";
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
    source: "Multi-Source Weather Agent",
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
    unified_agent: "Multi-Source Weather Agent",
    knowledge: "Reference documents",
    sevir: "SEVIR observations",
    geospatial: "Places and regions",
    cascast: "CasCast local nowcasts",
    era5: "ERA5/ERA5T reanalysis",
    aurora: "Aurora regional forecasts",
    "knowledge.search": "Search source documents",
    "knowledge.read": "Read source passage",
    "knowledge.describe": "Inspect document index",
    "sevir.event": "Read SEVIR event metadata",
    "sevir.search": "Search the SEVIR catalog",
    "sevir.observations": "Read SEVIR observation sequence",
    "sevir.channel_observations": "Read channel observations",
    "sevir.multimodal_observations": "Read multi-channel observations",
    "sevir.observation_batch": "Read observations for multiple events",
    "sevir.find_analogues": "Find similar historical events",
    "sevir.match_upload": "Match uploads to events",
    "geospatial.resolve_place": "Resolve place",
    "geospatial.point_to_location": "Locate geographic coordinates",
    "geospatial.boundary": "Read administrative boundary",
    "geospatial.region_bounds": "Read region bounds",
    "geospatial.pixel_to_latlon": "Convert image coordinates to location",
    "geospatial.latlon_to_pixel": "Convert location to image coordinates",
    "geospatial.mask": "Build region mask",
    "geospatial.select_region": "Select regional data",
    "cascast.available_forecasts": "Check forecast availability",
    "cascast.forecast": "Run or read local ensemble forecasts",
    "era5.datasets": "Find CDS datasets",
    "era5.describe": "Read dataset parameters",
    "era5.retrieve": "Download ERA5 data",
    "era5.fields": "Read ERA5 fields",
    "era5.regional_fields": "Read regional ERA5 fields",
    "aurora.regional_fields": "Read ERA5/Aurora regional fields",
  };
  if (labels[name]) return labels[name];
  const prefixLabels = {
    "knowledge.": "Reference documents",
    "sevir.": "SEVIR observations",
    "geospatial.": "Geolocation",
    "cascast.": "CasCast nowcasting",
    "era5.": "ERA5 reanalysis",
    "aurora.": "Aurora regional forecasting",
    "analytics.": "Weather analysis",
  };
  const prefix = Object.keys(prefixLabels).find((item) => String(name || "").startsWith(item));
  return prefix ? prefixLabels[prefix] : "Run analysis";
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
  avatar.textContent = role === "user" ? "You" : "W";

  const body = document.createElement("div");
  body.className = "chat-message-body";
  const heading = document.createElement("div");
  heading.className = "chat-message-heading";
  const author = document.createElement("strong");
  author.textContent = role === "user" ? "You" : "Multi-Source Weather Agent";
  const time = document.createElement("time");
  time.dateTime = new Date(timestamp).toISOString();
  time.textContent = new Date(timestamp).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  heading.append(author, time);
  if (role === "user" && turnId) {
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "delete-turn-button";
    deleteButton.setAttribute("aria-label", "Delete this exchange");
    deleteButton.title = "Delete question and answer";
    deleteButton.textContent = "Delete";
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
      image.alt = attachment.title || "Uploaded VIL image";
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
    title.textContent = attachment.title || "VIL image";
    kind.textContent = attachment.kind === "sequence" ? "13-frame VIL sequence" : "Single VIL image";
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
    ? `Conversation restored · ${Math.ceil(chatHistory.length / 2)} exchanges`
    : "Follow-up questions use the existing analysis context.";
}


function selectedModelLabel() {
  if (weatherModelInput.value === "cascast_live") return "Local CasCast inference";
  return weatherModelInput.value === "cascast" ? "CasCast" : "persistence_baseline";
}

function syncWeatherModelNote() {
  const notes = {
    cascast_live: "Run EarthFormer, CasCast diffusion and VAE decoding on the uploaded 13-frame sequence.",
    persistence_baseline: "Repeat the last observed frame for 12 forecast steps as a persistence baseline.",
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
  unboundOption.textContent = "No event selected; use the request or search for one";
  eventInput.appendChild(unboundOption);
  eventIds.forEach((eventId, index) => {
    const option = document.createElement("option");
    option.value = eventId;
    option.textContent = `Event ${index + 1} · ${eventId}`;
    eventInput.appendChild(option);
  });
  if (eventIds.includes(currentEvent)) eventInput.value = currentEvent;
  else eventInput.value = "";
}

function syncMode() {
  modeDescription.textContent = "The agent continues only when more data or another execution step is needed.";
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
    upload: "Analyze intensity and spatial structure in a single VIL observation.",
    upload_sequence: "Use 13 consecutive VIL frames to generate 12 future frames with local CasCast inference.",
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
  if (!response.ok) throw new Error(result.error || "Image upload failed");
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
    ? `Matched event ${match.event_id}; regional ERA5/Aurora data can be linked`
    : "No event matched. Provide a SEVIR event ID to link regional data and location.";
  uploadStatus.textContent = `${grayscaleSource
    ? `Loaded ${file.name} with original grayscale values for VIL analysis`
    : `Loaded ${file.name}; color converted to grayscale for image-intensity analysis`}`;
  uploadStatus.textContent += `. ${matchText}`;
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
    throw new Error(`Image size is ${imageWidth} × ${imageHeight}; 384 × 384 is required.`);
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
  if (files.length < 2) throw new Error("A sequence needs at least two images; CasCast requires 13. ");
  const sortedFiles = [...files].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const frames = [];
  const previewCanvases = [];
  for (let index = 0; index < sortedFiles.length; index += 1) {
    sequenceUploadStatus.textContent = `Reading image ${index + 1}/13: ${sortedFiles[index].name}`;
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
  if (!response.ok) throw new Error(result.error || "Sequence upload failed");
  uploadedSequenceId = result.upload_id;
  uploadedSampleId = "";
  attachmentPendingForMessage = true;
  if (result.status !== "ok") {
    dataSourceInput.value = "upload_sequence";
    taskInput.value = "auto";
    weatherModelInput.value = "persistence_baseline";
    attachmentTitle.textContent = `${files.length} images · Sequence validation failed`;
    uploadStatus.textContent = `Upload diagnostics saved: ${result.reason || "Incomplete sequence"}. Ask about missing frames or ordering.`;
    sequenceUploadStatus.textContent = "Sequence validation is required before CasCast inference.";
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
  attachmentTitle.textContent = `13 VIL images · ${sortedFiles[0].name} → ${sortedFiles[12].name}`;
  const match = result.event_match || {};
  const matchText = match.status === "matched"
    ? `Matched event ${match.event_id}; regional ERA5/Aurora data can be linked`
    : "No event matched. Provide an event ID to link ERA5/Aurora data.";
  uploadStatus.textContent = `13-frame sequence loaded in natural filename order. ${matchText}. `;
  sequenceUploadStatus.textContent = "Forecast questions run local CasCast inference. Future observations are not included.";
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
    attachmentTitle.textContent = `Selected ${files.length} images`;
    uploadStatus.textContent = "The sequence has fewer than 13 frames. Uploading it for missing-frame diagnostics.";
  }
  uploadedSampleId = "";
  uploadedSequenceId = "";
  attachmentTray.classList.remove("hidden");
  attachmentTitle.textContent = "Reading images…";
  uploadStatus.textContent = "Reading and checking images…";
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
  if (!Array.isArray(coordinate) || coordinate.length < 2) return "Unknown coordinates";
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
    ok: "Complete", completed: "Complete", running: "Running", queued: "Waiting",
    partial: "Partially completed", error: "Failed", failed: "Failed", unavailable: "Unavailable",
    blocked: "Blocked", incomplete: "Incomplete", cancelled: "Cancelled",
    cache_hit: "Using cache", recovered: "Restored", unknown: "Unknown status",
  };
  return labels[String(status || "unknown").toLowerCase()] || String(status || "Unknown status");
}

function executionStageLabel(name) {
  if (name && typeof name === "object") {
    if (name.display_name) return name.display_name;
    name = name.stage || name.name;
  }
  const value = String(name || "").toLowerCase();
  const stages = {
    data: "Read data", discovery: "Find available data", prediction: "Run forecast", analysis: "Analyze data",
    context: "Prepare model context", agent_context: "Prepare model context",
    context_budget: "Check context budget", context_budget_exceeded: "Check context budget",
    model: "Call the language model", execute: "Run code and tools", finish: "Finalize answer",
    select_tools: "Select tools and load instructions", format_repair: "Check action format",
    read_context: "Read records and tool instructions", citation_repair: "Add source citations",
    knowledge_reuse: "Reuse retrieved documents",
  };
  if (stages[value]) return stages[value];
  if (value.includes("asset") || value.includes("context") || value.includes("request")) return "Bind location, time and event data";
  if (value.includes("model_api") || value.includes("llm") || value.includes("agent_model")) return "Choose the next step with the language model";
  if (value.includes("queue")) return "Wait for an execution slot";
  if (value.includes("agent_execution")) return "Run code and read tool results";
  if (value.includes("unified")) return "Run code, review results and continue analysis";
  if (value.includes("foundation") || value.includes("aurora")) return "Analyze regional weather";
  if (value.includes("cascast") || value.includes("forecast")) return "Run and analyze forecasts";
  if (value.includes("render") || value.includes("visual")) return "Prepare charts";
  if (value.includes("answer") || value.includes("synth")) return "Prepare a clear answer";
  return "Run analysis step";
}

function routeDisplayLabel(route) {
  const value = String(route || "").toLowerCase();
  if (value.includes("unified") || value.includes("completed")) return "Multi-Source Weather Agent";
  return route ? "Agent execution" : "—";
}

function toolDisplaySentence(run) {
  const name = displayToolName(run);
  const status = executionStatusLabel(run?.status);
  const elapsed = run?.elapsed_ms == null ? NaN : Number(run.elapsed_ms);
  const cache = run?.cache_hit === true ? "; cached result" : "";
  const argumentsText = readableToolArguments(run?.arguments);
  const resultText = String(run?.result_summary || "").trim();
  const errorText = executionErrorText(run?.error);
  return `${name}${argumentsText ? `(${argumentsText})` : ""}: ${status}${cache}${Number.isFinite(elapsed) ? `; elapsed ${(elapsed / 1000).toFixed(elapsed >= 1000 ? 1 : 2)} s` : ""}${resultText ? `; ${resultText}` : ""}${errorText ? `; ${errorText}` : ""}`;
}

function readableToolArguments(argumentsValue) {
  if (!argumentsValue || typeof argumentsValue !== "object" || Array.isArray(argumentsValue)) return "";
  const labels = {
    subject_key: "Data identity", event_id: "Event", upload_id: "Upload", threshold: "Threshold",
    lead_minutes: "Lead time", minutes_before: "History window", minutes_after: "Future window",
    horizon_hours: "Forecast horizon", ensemble_members: "Ensemble members",
    place: "Place", latitude: "Latitude", longitude: "Longitude",
    name: "Name", level: "Level", query: "Query", limit: "Limit",
    year: "Year", month: "Month", event_type: "Event type",
    start_time_utc: "Start time (UTC)", end_time_utc: "End time (UTC)",
    observation_time_utc: "Observation time (UTC)", time_match: "Time matching",
  };
  const parts = [];
  Object.entries(argumentsValue)
    .filter(([, value]) => value != null && value !== "" && typeof value !== "object")
    .slice(0, 5).forEach(([key, value]) => {
      const suffix = ["lead_minutes", "minutes_before", "minutes_after"].includes(key)
        ? " min" : key === "horizon_hours" ? " h" : "";
      parts.push(`${labels[key] || key.replaceAll("_", " ")} ${value}${suffix}`);
    });
  return parts.join(", ");
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
    ["Execution path", routeDisplayLabel(data.route_mode || data.termination_reason)],
    ["Status", executionStatusLabel(status)],
    ["Model calls", `${data.model_calls ?? requestTrace?.model_calls ?? 0}`],
    ...(contextSummaryCallCount(data) ? [["Summary calls", `${contextSummaryCallCount(data)}`]] : []),
    ["Tool calls", `${data.tool_calls ?? requestTrace?.tool_calls ?? runs.length}`],
    ["Cache hit", `${data.cache_hits ?? requestTrace?.cache_hits ?? 0}`],
    ["Failure stage", failurePhase ? executionStageLabel(failurePhase) : "—"],
    ["Event binding", eventResolution
      ? `${eventResolution.source || "unknown"} · ${eventResolution.event_id || "No match"}`
      : "—"],
    ["Total time", data.elapsed_ms != null ? `${(Number(data.elapsed_ms) / 1000).toFixed(2)} s` : (requestTrace?.elapsed_ms != null ? `${(Number(requestTrace.elapsed_ms) / 1000).toFixed(2)} s` : "—")],
    ...(failureMessage ? [["Failure reason", failureMessage]] : []),
    ...(nextAction ? [["Suggested action", nextAction]] : []),
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
        ? `; model ${stage.provider ? `${stage.provider} / ` : ""}${stage.model}`
        : "";
      item.textContent = `${executionStageLabel(stage)}: ${executionStatusLabel(stage.status)}${Number.isFinite(elapsed) ? `; elapsed ${(elapsed / 1000).toFixed(elapsed >= 1000 ? 1 : 2)} s` : ""}${stage.route_mode ? `; mode ${stage.route_mode}` : ""}${modelText}`;
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
    item.textContent = diagnostic?.next_action || "No code or external tool was run. Check the inputs and event binding.";
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
  // references in the answer become stable labels such as Analysis chart02.
  renderConversationVisualPane(data);
  renderAnswer(data.answer || "No text answer returned");
  document.querySelector("#result-title").textContent = "Analysis summary";
  document.querySelector("#result-source").textContent = "Multi-Source Weather Agent";
  turnDetailsSource.textContent = "Multi-Source Weather Agent";
  document.querySelector("#tool-name").textContent = runs.length > 1
    ? runs.map((run) => displayToolName(run)).join(" → ")
    : runs.length ? displayToolName(runs[0]) : "No external tools used";
  document.querySelector("#meta-time").textContent = data.elapsed_ms != null
    ? `${data.elapsed_ms} ms`
    : (data.request_trace?.elapsed_ms != null ? `${data.request_trace.elapsed_ms} ms` : "Not recorded");
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
      summaryLabel.textContent = `Execution record · ${modelCalls} model calls · ${codeRuns} code runs · ${runs.length} tool calls`;
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
  if (executionMode) executionMode.textContent = "Multi-Source Weather Agent";
  if (executionFlow) {
    const defaultExecutionFlow = "Understand the question → Retrieve data → Calculate and plot → Explain the results";
    const path = ["Bind data and context"];
    attempts.forEach((item) => {
      const label = liveActionLabels[item?.action] || String(item?.action || "Continue analysis").replaceAll("_", " ");
      if (path[path.length - 1] !== label) path.push(label);
    });
    executionFlow.textContent = attempts.length ? path.join(" → ") : defaultExecutionFlow;
  }
  if (sessionBudget) {
    const summaryCalls = contextSummaryCallCount(data);
    sessionBudget.textContent = `${data.session_id || sessionId} · ${routeDisplayLabel(data.route_mode)} · Model calls ${data.model_calls ?? "—"}${summaryCalls ? ` · Summary calls ${summaryCalls}` : ""} · Code runs ${data.code_execution_count ?? "—"}`;
  }
  attempts.forEach((item, index) => {
    const presentation = executionStepPresentation(attempts, index);
    const block = document.createElement("details");
    block.className = "agent-attempt";
    const summary = document.createElement("summary");
    const timingText = stepTimingText(item);
    const elapsedText = timingText ? ` · ${timingText}` : "";
    const stepNumber = executionStepNumber(attempts, index);
    summary.textContent = `${stepNumber}: ${liveActionLabels[item.action] || "Execution step"} · ${executionStatusLabel(presentation.status)}${elapsedText}`;
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
      error.textContent = `Error details: ${executionErrorText(item.error)}`;
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
    image.alt = artifact.title || artifact.chart_type || "Analysis chart";
    image.loading = "lazy";
    const caption = document.createElement("figcaption");
    caption.textContent = artifact.description
      ? `${artifact.title || artifact.chart_type} — ${artifact.description}`
      : (artifact.title || artifact.chart_type || "Analysis chart");
    image.addEventListener("error", () => {
      figure.classList.add("artifact-load-error");
      caption.textContent = `Chart could not be loaded: ${artifact.title || artifact.chart_type || artifact.artifact_id}`;
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
      return `[Analysis chart${String(index).padStart(2, "0")}](#artifact:${encodeURIComponent(identity)})`;
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
  runState.textContent = "Error";
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
  cancelRequestButton.querySelector("span").textContent = "Stopping";
  runState.textContent = "Stopping";
  conversationStatus.textContent = "Stopping execution. Completed results will be retained.";
  try {
    const cancellation = await requestTaskCancellation(requestId);
    if (!cancellation) {
      cancelRequestButton.disabled = false;
      cancelRequestButton.querySelector("span").textContent = "Retry stop";
      runState.textContent = "Stop unconfirmed";
      conversationStatus.textContent = "The server has not confirmed the stop request. Check task status or select Stop again.";
    }
  } catch (_error) {
    cancelRequestButton.disabled = false;
    cancelRequestButton.querySelector("span").textContent = "Retry stop";
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
  runButton.firstElementChild.textContent = "Analyzing";
  runState.textContent = "Analyzing";
  runState.className = "run-state running";
  errorBox.classList.add("hidden");
  conversationStatus.textContent = "Analyzing the conversation and tool results…";
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
    cancelRequestButton.querySelector("span").textContent = "Stop";
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
    runState.textContent = "Submitting";
    conversationStatus.textContent = "Binding inputs and starting the agent.";
    activeRequestTransport = "task";
    persistActiveTask({ requestId, sessionId, turnId, question, submittedAt: timestamp });
    if (cancelRequestButton) {
      cancelRequestButton.disabled = true;
      cancelRequestButton.querySelector("span").textContent = "Submitting";
    }
    const submitted = await submitDurableTaskWithReconnect(
      requestPayload, requestId, progressState, pendingMessage,
    );
    if (cancelRequestButton) {
      cancelRequestButton.disabled = false;
      cancelRequestButton.querySelector("span").textContent = "Stop";
    }
    activeRequestPhase = "execution";
    runState.textContent = "Preparing execution";
    conversationStatus.textContent = "Analysis is running. Progress updates automatically.";
    if (cancelRequestButton) cancelRequestButton.querySelector("span").textContent = "Stop";
    ensureLiveExecutionPanel(pendingMessage);
    renderLiveExecutionProgress(pendingMessage, {
      status: "running",
      terminal: false,
      current_stage: "unified_agent",
      progress: 0.13,
      current_activity: {
        action: "agent_started",
        status: "ok",
        summary: "Agent started. Waiting for an execution slot.",
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
      ? `Task could not be submitted: ${error.message}`
      : "Connection lost. Reconnect to check the task status and results.";
    const failureData = {
      status: rejected ? "failed" : "unknown",
      answer,
      route_mode: "request_failed",
      termination_reason: rejected ? "submission_rejected" : "client_connection_lost",
      failure_class: rejected ? "" : "transport_reset",
      failure_message: rejected ? "The server did not accept the task." : "Connection lost. Task status is unconfirmed.",
      next_action: rejected ? "Check the inputs and submit again." : "Reconnect to check this task.",
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
    conversationStatus.textContent = rejected ? "Task not submitted." : "Connection lost. Task status is unconfirmed.";
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
      cancelRequestButton.querySelector("span").textContent = "Stop";
    }
    requestInFlight = false;
    syncComposerAvailability();
    runButton.firstElementChild.textContent = "Send";
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
  runState.textContent = "Ready";
  runState.className = "run-state";
  conversationStatus.textContent = "New conversation started. Follow-up questions will use this context.";
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
  .catch((error) => console.warn("Could not restore the conversation. The conversation list was retained: ", error))
  .finally(async () => {
    try { persistConversationIndex(); } catch (_error) {}
    void renderConversationList();
    await resumeActiveTaskIfPresent();
  });
loadConfig();
restoreActiveUpload();
