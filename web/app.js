const form = document.querySelector("#agent-form");
const taskInput = document.querySelector("#task");
const weatherModelInput = document.querySelector("#weather-model");
const leadInput = document.querySelector("#lead-minutes");
const eventInput = document.querySelector("#event-id");
const thresholdInput = document.querySelector("#threshold");
const hazardThresholds = { noticeable: 74, strong: 133, very_strong: 160, extreme: 181, extreme_candidate: 219 };
const selectedThreshold = () => hazardThresholds[thresholdInput.value] ?? 133;
const modeInput = document.querySelector("#mode");
const providerInput = document.querySelector("#provider");
const modelInput = document.querySelector("#model");
const questionInput = document.querySelector("#question");
const providerState = document.querySelector("#provider-state");
const apiFields = document.querySelector("#api-fields");
const runButton = document.querySelector("#run-button");
const runState = document.querySelector("#run-state");
const uploadInput = document.querySelector("#vil-upload");
const uploadStatus = document.querySelector("#upload-status");
const uploadPreview = document.querySelector("#upload-preview");
const sequenceUploadInput = document.querySelector("#vil-sequence-upload");
const sequenceUploadStatus = document.querySelector("#sequence-upload-status");
const dataSourceInput = document.querySelector("#data-source");
const contextBadge = document.querySelector("#context-badge");
const dataSourceNote = document.querySelector("#data-source-note");
const uploadPanel = document.querySelector("#upload-panel");
const predictionControls = document.querySelector("#prediction-controls");
const thresholdControl = document.querySelector("#threshold-control");
const fullVilFrame = document.querySelector("#full-vil-frame");
const selectedFrameTitle = document.querySelector("#selected-frame-title");
const frameDialog = document.querySelector("#frame-dialog");
const dialogFrameImage = document.querySelector("#dialog-frame-image");
const dialogFrameTitle = document.querySelector("#dialog-frame-title");
const weatherModelNote = document.querySelector("#weather-model-note");

const emptyState = document.querySelector("#empty-state");
const resultContent = document.querySelector("#result-content");
const errorBox = document.querySelector("#error-box");

let providerConfig = {};
let runtimeConfig = {};
let uploadedSampleId = "";
let uploadedSequenceId = "";

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function selectedModelLabel() {
  return weatherModelInput.value === "cascast" ? "CasCast" : "persistence_baseline";
}

function syncWeatherModelNote() {
  weatherModelNote.textContent = weatherModelInput.value === "cascast"
    ? "读取同学提供的10路CasCast集合均值；不是现场重新推理。"
    : "把当前事件最后一帧历史观测复制到未来12个时效；没有GT时只作可视化基线。";
}

function buildQuestion() {
  const eventId = eventInput.value.trim() || "R19100105107994";
  const lead = leadInput.value;
  const threshold = selectedThreshold();
  const weatherModel = selectedModelLabel();
  const hours = Number(lead) / 60;
  const questions = {
    auto: questionInput.value,
    multi_hour_forecast: `请使用${weatherModel}根据事件${eventId}过去65分钟VIL，预测未来${hours}小时强回波，并报告不同时效的预测结果与误差限制。`,
    ensemble_uncertainty: `请分析事件${eventId}的10路CasCast ensemble，在VIL阈值${threshold}下比较逐像素不确定性、强回波概率、storm object位置、面积和持续时间。`,
    dataset_evaluation: `请评价全部10个本地SEVIR测试事件，在VIL阈值${threshold}及其他标准阈值下比较CasCast与Persistence，分析逐时效表现、ensemble可靠性和storm-object评价。`,
    hazard_locations: `请使用${weatherModel}预测事件${eventId}未来${hours}小时VIL超过${threshold}的区域，说明H1、H2、H3分别位于哪里、影响范围多大。`,
    hazard_duration: `请使用${weatherModel}预测事件${eventId}未来${hours}小时的强回波，分析VIL超过${threshold}的各个Hazard分别从何时开始、持续多久。`,
    hazard_evolution: `请使用${weatherModel}预测事件${eventId}未来${hours}小时的VIL序列，分析强回波区域的生成、增强、减弱和移动趋势。`,
    uploaded_image_analysis: `请分析我刚上传的SEVIR VIL图片。在阈值${threshold}下，说明图像强度统计、强回波面积以及主要候选Hazard区域，并明确这不是未来预测。`,
    catalog_search: `请在SEVIR CATALOG中查询事件${eventId}，返回时间、storm type、文件位置和catalog强度信息。`,
    historical_statistics: "请统计SEVIR VIL历史catalog中不同年份、月份和storm type的记录分布，给出data_max的中位数、95分位数和最强事件排序，并说明统计限制。",
    event_geolocation: `请查询SEVIR事件${eventId}的经纬度覆盖范围和近似中心，并说明定位精度限制。`,
    event_history_comparison: `请将SEVIR事件${eventId}的catalog data_max与历史同类记录比较，给出经验分位并说明它不是灾害概率。`,
  };
  if (taskInput.value !== "auto") questionInput.value = questions[taskInput.value];
}

function setStatus(elementId, available, text) {
  const detail = document.querySelector(`#${elementId}`);
  detail.textContent = text;
  const icon = detail.closest("article").querySelector(".status-icon");
  icon.className = `status-icon ${available ? "ready" : "pending"}`;
}

function syncProvider() {
  const config = providerConfig[providerInput.value];
  if (!config) return;
  providerState.textContent = config.configured
    ? `${config.label} API已配置`
    : `${config.label} API未配置，可使用本地工具直跑`;
}

function syncMode() {
  apiFields.classList.toggle("hidden", !["api", "reflective", "direct"].includes(modeInput.value));
  syncProvider();
}

function syncTaskControls() {
  const predictionTask = ["multi_hour_forecast", "hazard_locations", "ensemble_uncertainty"].includes(taskInput.value);
  const metadataTask = ["historical_statistics", "event_geolocation"].includes(taskInput.value);
  predictionControls.classList.toggle("hidden", !predictionTask);
  thresholdControl.classList.toggle("hidden", metadataTask || dataSourceInput.value === "catalog");
  eventInput.disabled = ["upload", "upload_sequence"].includes(dataSourceInput.value) || ["historical_statistics", "dataset_evaluation"].includes(taskInput.value);
  weatherModelInput.disabled = taskInput.value === "ensemble_uncertainty";
  if (taskInput.value === "ensemble_uncertainty") weatherModelInput.value = "cascast";
  syncWeatherModelNote();
  leadInput.disabled = !predictionTask;
  thresholdInput.disabled = metadataTask;
  if (taskInput.value !== "auto") buildQuestion();
}

function syncDataSource() {
  const labels = {
    cascast: "CASCAST SAVED EVENT",
    upload: uploadedSampleId ? "UPLOADED VIL READY" : "UPLOAD REQUIRED",
    upload_sequence: uploadedSequenceId ? "13-FRAME SEQUENCE READY" : "13 FRAMES REQUIRED",
    catalog: "SEVIR CATALOG",
  };
  const notes = {
    cascast: "读取已保存的13帧历史、10路CasCast预测和12帧ground truth；不是现场运行模型。",
    upload: "分析一张VIL图的当前强回波信号，不能预测未来演变或持续时间。",
    upload_sequence: "分析过去60分钟的13帧VIL演变，并可运行Persistence基线；当前不能现场运行CasCast。",
    catalog: "查询SEVIR事件时间、类型、文件和NOAA关联等目录字段，不读取预测图像。",
  };
  contextBadge.textContent = labels[dataSourceInput.value];
  dataSourceNote.textContent = notes[dataSourceInput.value];
  if (["upload", "upload_sequence"].includes(dataSourceInput.value)) {
    uploadPanel.open = true;
    taskInput.value = "auto";
  } else if (dataSourceInput.value === "catalog" && ["multi_hour_forecast", "hazard_locations", "ensemble_uncertainty"].includes(taskInput.value)) {
    taskInput.value = "auto";
  }
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
  uploadPreview.src = canvas.toDataURL("image/png");
  uploadPreview.classList.remove("hidden");
  uploadStatus.textContent = grayscaleSource
    ? `已载入${file.name}；原始灰度值可用于VIL阈值分析`
    : `已载入${file.name}；彩色图已转灰度，只能作为图像强度分析`;
  dataSourceInput.value = "upload";
  taskInput.value = "auto";
  contextBadge.textContent = "UPLOADED VIL READY";
  syncDataSource();
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
  for (let index = 0; index < sortedFiles.length; index += 1) {
    sequenceUploadStatus.textContent = `正在读取第${index + 1}/13张：${sortedFiles[index].name}`;
    const converted = await imageFileToGray(sortedFiles[index]);
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
  dataSourceInput.value = "upload_sequence";
  taskInput.value = "auto";
  sequenceUploadStatus.textContent = "13帧序列已载入；可以询问过去60分钟演变或运行Persistence基线。";
  syncDataSource();
}

async function restorePersistedUploads() {
  try {
    const response = await fetch("/api/uploads");
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "上传记录读取失败");
    const records = Array.isArray(result.records) ? result.records : [];
    if (!records.length) return;

    const latestSingle = records.find((item) => item.kind === "single_frame");
    const latestSequence = records.find((item) => item.kind === "sequence");
    if (latestSingle) {
      uploadedSampleId = latestSingle.upload_id;
      uploadPreview.src = `/api/uploaded-frame?upload_id=${encodeURIComponent(uploadedSampleId)}`;
      uploadPreview.classList.remove("hidden");
      uploadStatus.textContent = `已恢复上传记录：${latestSingle.filename || uploadedSampleId}`;
    }
    if (latestSequence) {
      uploadedSequenceId = latestSequence.upload_id;
      sequenceUploadStatus.textContent = `已恢复13帧序列：${uploadedSequenceId.slice(0, 8)}`;
    }

    const latest = records[0];
    dataSourceInput.value = latest.kind === "sequence" ? "upload_sequence" : "upload";
    syncDataSource();
  } catch (error) {
    uploadStatus.textContent = `上传记录恢复失败：${error.message}`;
  }
}

uploadInput.addEventListener("change", async () => {
  const file = uploadInput.files?.[0];
  if (!file) return;
  uploadedSampleId = "";
  uploadStatus.textContent = "正在读取并检查图片…";
  try {
    await uploadVilImage(file);
  } catch (error) {
    uploadPreview.classList.add("hidden");
    uploadStatus.textContent = error.message;
  }
});

sequenceUploadInput.addEventListener("change", async () => {
  uploadedSequenceId = "";
  try {
    await uploadVilSequence(sequenceUploadInput.files || []);
  } catch (error) {
    sequenceUploadStatus.textContent = error.message;
  }
});

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const data = await response.json();
    providerConfig = data.providers || {};
    runtimeConfig = data.runtime || {};
    const qwen = providerConfig.qwen || {};
    document.querySelector("#qwen-status").textContent = qwen.model || "Qwen API";
    setStatus("qwen-detail", Boolean(qwen.configured), qwen.configured ? "真实API已配置" : "API Key未配置");
    const sample = runtimeConfig.sevir_sample || {};
    setStatus("data-detail", Boolean(sample.available), sample.available ? `真实样本 ${sample.event_id}` : "样本缺失");
    const cascast = runtimeConfig.cascast || {};
    setStatus("cascast-detail", Boolean(cascast.available), cascast.available ? `${cascast.available_event_count || 0}事件 · ${cascast.ensemble_count || 0}路` : cascast.reason || "待接入");
    const currentEvent = eventInput.value;
    eventInput.innerHTML = "";
    (cascast.event_ids || []).forEach((eventId, index) => {
      const option = document.createElement("option");
      option.value = eventId;
      option.textContent = `事件 ${index + 1} · ${eventId}`;
      eventInput.appendChild(option);
    });
    if ((cascast.event_ids || []).includes(currentEvent)) eventInput.value = currentEvent;
    if (!eventInput.value && eventInput.options.length) eventInput.selectedIndex = 0;
    syncProvider();
  } catch (error) {
    providerState.textContent = `配置读取失败：${error.message}`;
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

function renderHazardRegions(regions, spatialShape = [384, 384]) {
  const overlay = document.querySelector("#hazard-overlay");
  const list = document.querySelector("#region-list");
  const note = document.querySelector("#region-note");
  overlay.innerHTML = "";
  list.innerHTML = "";

  if (!Array.isArray(regions) || regions.length === 0) {
    list.classList.add("hidden");
    note.classList.add("hidden");
    return;
  }

  const [height, width] = spatialShape;
  regions.forEach((region, index) => {
    const colorIndex = Math.min(index, 2) + 1;
    const box = Array.isArray(region.pixel_bbox) ? region.pixel_bbox.map(Number) : null;
    if (box?.length === 4) {
      const [minX, minY, maxX, maxY] = box;
      const marker = document.createElement("div");
      marker.className = `hazard-marker hazard-${colorIndex}`;
      marker.style.left = `${(minX / width) * 100}%`;
      marker.style.top = `${(minY / height) * 100}%`;
      marker.style.width = `${((maxX - minX + 1) / width) * 100}%`;
      marker.style.height = `${((maxY - minY + 1) / height) * 100}%`;
      marker.title = `${region.region_id}：${region.pixel_count}个像素，中心${formatCoordinate(region)}`;
      const label = document.createElement("b");
      label.textContent = region.region_id;
      marker.appendChild(label);
      overlay.appendChild(marker);
    }

    const item = document.createElement("article");
    item.className = `region-item hazard-${colorIndex}`;
    const areaPercent = (Number(region.area_ratio || 0) * 100).toFixed(2);
    const duration = region.duration_minutes
      ? ` · ${region.duration_is_lower_bound ? "至少" : ""}${region.duration_minutes}分钟`
      : "";
    item.innerHTML = `<strong>${region.region_id}</strong><span>${areaPercent}% · ${Number(region.pixel_count).toLocaleString()} 像素${duration}</span><small>${formatCoordinate(region)}</small>`;
    list.appendChild(item);
  });

  note.textContent = "候选区域按面积由大到小编号；框线是预测窗口内强回波范围，不代表行政边界。";
  note.classList.remove("hidden");
  list.classList.remove("hidden");
}

function frameImageUrl(result, series, index = 0, leadIndex = null) {
  if (result.task === "uploaded_sevir_vil_sequence_analysis") {
    const sequenceParams = new URLSearchParams({
      upload_id: result.upload_id || "",
      series,
      index: String(index),
    });
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
  if (!metrics.ground_truth_available) {
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
  const container = document.querySelector("#heatmap");
  const overviewImage = document.querySelector("#hazard-overview-image");
  const probabilityImage = document.querySelector("#hazard-probability-image");
  const durationImage = document.querySelector("#hazard-duration-image");
  container.classList.add("hidden");
  overviewImage.src = result.task === "uploaded_sevir_vil_analysis"
    ? uploadedFrameUrl(result)
    : frameImageUrl(result, "forecast_overview", 0);
  overviewImage.classList.remove("hidden");
  overviewImage.onclick = () => openFrameDialog(
    overviewImage.src,
    result.task === "uploaded_sevir_vil_analysis" ? "上传的SEVIR VIL图像" : "未来60分钟总体VIL场",
  );
  const showEnsembleHazardMaps = result.model_name === "cascast" && [
    "future_hazard_analysis",
    "ensemble_uncertainty_analysis",
  ].includes(result.task);
  probabilityImage.parentElement.classList.toggle("hidden", !showEnsembleHazardMaps);
  durationImage.parentElement.classList.toggle("hidden", !showEnsembleHazardMaps);
  if (showEnsembleHazardMaps) {
    probabilityImage.src = frameImageUrl(result, "hazard_probability_any", 0);
    durationImage.src = frameImageUrl(result, "hazard_duration_mean", 0);
    probabilityImage.parentElement.onclick = () => openFrameDialog(probabilityImage.src, "未来60分钟超阈值成员支持率");
    durationImage.parentElement.onclick = () => openFrameDialog(durationImage.src, "未来60分钟平均超阈值持续时间");
  }
  document.querySelector("#visual-resolution-label").textContent = "FULL 384×384";
  renderHazardRegions(result.hazard_regions || [], result.spatial_shape || [384, 384]);
  card.classList.remove("hidden");
}

function renderEnsembleUncertainty(result) {
  const card = document.querySelector("#uncertainty-card");
  if (result.task !== "ensemble_uncertainty_analysis") {
    card.classList.add("hidden");
    return;
  }
  const pixel = result.pixel_uncertainty || {};
  const objects = result.storm_object_uncertainty || {};
  document.querySelector("#uncertainty-mean-std").textContent = pixel.mean_std ?? "—";
  document.querySelector("#uncertainty-p95-std").textContent = pixel.p95_std ?? "—";
  document.querySelector("#uncertainty-object-members").textContent = `${objects.members_with_object ?? 0}/${result.ensemble_count ?? 0}`;
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
  const leadSlider = document.querySelector("#ensemble-lead-slider");
  const leadValue = document.querySelector("#ensemble-lead-value");
  const leadHeading = document.querySelector("#ensemble-member-heading");
  const updateMemberLead = (frameIndex) => {
    const index = Math.max(0, Math.min(11, Number(frameIndex) || 0));
    const leadMinutes = (index + 1) * 5;
    leadValue.textContent = `+${leadMinutes}分钟`;
    leadHeading.textContent = `10路Ensemble成员 · +${leadMinutes}分钟`;
    memberImages.querySelectorAll(".ensemble-member-image").forEach((button, memberIndex) => {
      const image = button.querySelector("img");
      const member = members[memberIndex];
      if (!image || !member) return;
      const source = frameImageUrl(result, "ensemble_member", member.member - 1, index);
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
  leadSlider.oninput = () => updateMemberLead(leadSlider.value);
  card.classList.remove("hidden");
}

function renderMetrics(result) {
  const labels = ["预测时效", "危险区域", "持续时间", "运行模型"];
  ["lead", "regions", "duration", "model"].forEach((key, index) => {
    document.querySelector(`#metric-label-${key}`).textContent = labels[index];
  });
  if (result.task === "ensemble_uncertainty_analysis") {
    document.querySelector("#metric-label-lead").textContent = "Ensemble成员";
    document.querySelector("#metric-label-regions").textContent = "对象成员";
    document.querySelector("#metric-label-duration").textContent = "持续时间范围";
    document.querySelector("#metric-label-model").textContent = "运行模型";
    document.querySelector("#metric-lead").textContent = result.ensemble_count;
    document.querySelector("#metric-regions").textContent = `${result.storm_object_uncertainty?.members_with_object ?? 0}/${result.ensemble_count}`;
    document.querySelector("#metric-duration").textContent = `${result.member_active_duration_minutes?.min ?? 0}–${result.member_active_duration_minutes?.max ?? 0} min`;
    document.querySelector("#metric-model").textContent = "CasCast";
    return;
  }
  if (result.task === "cascast_dataset_evaluation") {
    const thresholdKey = String(Math.trunc(Number(result.primary_threshold)));
    const cascast = result.aggregate_metrics?.cascast_ensemble_mean?.[thresholdKey] || {};
    document.querySelector("#metric-label-lead").textContent = "测试事件";
    document.querySelector("#metric-label-regions").textContent = "有效像素";
    document.querySelector("#metric-label-duration").textContent = "预测时效";
    document.querySelector("#metric-label-model").textContent = "评价模型";
    document.querySelector("#metric-lead").textContent = result.event_count;
    document.querySelector("#metric-regions").textContent = Number(cascast.valid_pixel_count || 0).toLocaleString();
    document.querySelector("#metric-duration").textContent = `${result.lead_minutes} min`;
    document.querySelector("#metric-model").textContent = "CasCast vs Persistence";
    return;
  }
  if (result.task === "historical_sevir_statistics") {
    document.querySelector("#metric-label-lead").textContent = "记录数";
    document.querySelector("#metric-label-regions").textContent = "年份数";
    document.querySelector("#metric-label-duration").textContent = "Storm Type";
    document.querySelector("#metric-label-model").textContent = "数据来源";
    document.querySelector("#metric-lead").textContent = Number(result.record_count).toLocaleString();
    document.querySelector("#metric-regions").textContent = Object.keys(result.counts_by_year || {}).length;
    document.querySelector("#metric-duration").textContent = Object.keys(result.counts_by_event_type || {}).length;
    document.querySelector("#metric-model").textContent = "CATALOG";
    return;
  }
  if (result.task === "sevir_catalog_search") {
    document.querySelector("#metric-label-lead").textContent = "匹配记录";
    document.querySelector("#metric-label-regions").textContent = "返回记录";
    document.querySelector("#metric-label-duration").textContent = "数据通道";
    document.querySelector("#metric-label-model").textContent = "数据来源";
    document.querySelector("#metric-lead").textContent = Number(result.total_matches).toLocaleString();
    document.querySelector("#metric-regions").textContent = result.returned_count;
    document.querySelector("#metric-duration").textContent = "VIL";
    document.querySelector("#metric-model").textContent = "CATALOG";
    return;
  }
  if (result.task === "event_history_comparison") {
    document.querySelector("#metric-label-lead").textContent = "历史分位";
    document.querySelector("#metric-label-regions").textContent = "data_max";
    document.querySelector("#metric-label-duration").textContent = "参考记录";
    document.querySelector("#metric-label-model").textContent = "参考组";
    document.querySelector("#metric-lead").textContent = `${result.data_max_percentile}%`;
    document.querySelector("#metric-regions").textContent = result.catalog_data_max;
    document.querySelector("#metric-duration").textContent = Number(result.reference_count).toLocaleString();
    document.querySelector("#metric-model").textContent = result.reference_group;
    return;
  }
  if (result.task === "sevir_event_geolocation") {
    document.querySelector("#metric-label-lead").textContent = "中心纬度";
    document.querySelector("#metric-label-regions").textContent = "中心经度";
    document.querySelector("#metric-label-duration").textContent = "Storm Type";
    document.querySelector("#metric-label-model").textContent = "数据通道";
    document.querySelector("#metric-lead").textContent = formatLatitude(result.approx_center_latlon?.[0]);
    document.querySelector("#metric-regions").textContent = formatLongitude(result.approx_center_latlon?.[1]);
    document.querySelector("#metric-duration").textContent = result.event_type || "Unlabeled";
    document.querySelector("#metric-model").textContent = "VIL";
    return;
  }
  const lead = result.lead_minutes || (result.forecast_frames ? result.forecast_frames * 5 : null);
  const duration = result.strong_echo_duration_minutes_within_horizon ?? result.longest_active_duration_minutes;
  document.querySelector("#metric-lead").textContent = lead ? `${lead} min` : "—";
  document.querySelector("#metric-regions").textContent = result.hazard_region_count !== undefined ? `${result.hazard_region_count}处` : "—";
  document.querySelector("#metric-duration").textContent = duration !== undefined ? `${duration} min` : "—";
  document.querySelector("#metric-model").textContent = result.model_name || (result.status?.startsWith("awaiting") ? "待接入" : "数据工具");
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

function renderResult(data) {
  emptyState.classList.add("hidden");
  errorBox.classList.add("hidden");
  resultContent.classList.remove("hidden");
  runState.textContent = "COMPLETE";
  runState.className = "run-state done";

  renderAnswer(data.answer || "未返回文字结论");
  document.querySelector("#result-title").textContent = resultTitle(data.tool_result || {});
  document.querySelector("#result-source").textContent = resultSource(data.tool_result || {});
  const runs = Array.isArray(data.tool_runs) ? data.tool_runs : [];
  document.querySelector("#tool-name").textContent = runs.length > 1
    ? runs.map((run) => run.tool_name).join(" → ")
    : data.tool_name;
  document.querySelector("#tool-arguments").textContent = pretty(
    runs.length > 1 ? runs.map(({ step, tool_name, tool_arguments }) => ({ step, tool_name, tool_arguments })) : data.tool_arguments,
  );
  document.querySelector("#tool-result").textContent = pretty(runs.length > 1 ? runs : data.tool_result);
  document.querySelector("#meta-time").textContent = `${data.elapsed_ms} ms`;
  renderDirectTrace(data);
  renderArtifacts(data);
  renderMetrics(data.tool_result || {});
  renderEvaluation(data.tool_result || {});
  renderDatasetEvaluation(data.tool_result || {});
  renderHistoricalStatistics(data.tool_result || {});
  renderForecastSequence(data.tool_result || {});
  renderEnsembleUncertainty(data.tool_result || {});
  renderLimit(data.tool_result || {});
  renderHazardOverview(data.tool_result || {});
}

function renderDirectTrace(data) {
  const container = document.querySelector("#direct-analysis");
  const trace = document.querySelector("#direct-trace");
  trace.innerHTML = "";
  const attempts = Array.isArray(data.direct_trace) ? data.direct_trace : [];
  container.classList.toggle("hidden", attempts.length === 0);
  attempts.forEach((item) => {
    const block = document.createElement("details");
    block.className = "direct-attempt";
    const summary = document.createElement("summary");
    summary.textContent = `第${item.attempt}轮 · ${item.status}${item.plan ? ` · ${item.plan}` : ""}`;
    block.appendChild(summary);
    if (item.code) {
      const code = document.createElement("pre");
      code.className = "direct-code";
      code.textContent = item.code;
      block.appendChild(code);
    }
    if (item.error) {
      const error = document.createElement("pre");
      error.className = "direct-error";
      error.textContent = item.error;
      block.appendChild(error);
    }
    trace.appendChild(block);
  });
}

function renderArtifacts(data) {
  const container = document.querySelector("#agent-artifacts");
  const list = document.querySelector("#artifact-list");
  list.innerHTML = "";
  const artifacts = (data.artifacts || []).filter((item) => item && item.image_url);
  container.classList.toggle("hidden", artifacts.length === 0);
  artifacts.forEach((artifact) => {
    const figure = document.createElement("figure");
    figure.className = "agent-artifact";
    const image = document.createElement("img");
    image.src = artifact.image_url;
    image.alt = artifact.chart_type || "Agent生成图表";
    const caption = document.createElement("figcaption");
    caption.textContent = artifact.chart_type || "Agent生成图表";
    figure.append(image, caption);
    list.appendChild(figure);
  });
}

function renderForecastSequence(result) {
  const card = document.querySelector("#forecast-sequence-card");
  const container = document.querySelector("#forecast-sequence");
  const historyContainer = document.querySelector("#history-sequence");
  const truthContainer = document.querySelector("#ground-truth-sequence");
  const truthHeading = document.querySelector("#ground-truth-heading");
  const frames = result.frame_summaries;
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
  const truthFrames = result.ground_truth_frame_summaries || [];
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
  if (result.model_name === "cascast" && result.forecast_frames) return "CasCast未来60分钟回波预测";
  return titles[result.task] || "分析结论";
}

function resultSource(result) {
  if (result.data_source?.kind === "cascast_official_test_prediction" || result.data_source === "cascast_saved_inference") return "CASCAST · SAVED PREDICTION";
  if (result.task === "uploaded_vil_analysis") return "USER INPUT · VIL";
  if (result.task === "uploaded_sevir_vil_sequence_analysis") return "USER INPUT · 13 VIL FRAMES";
  if (["historical_sevir_statistics", "sevir_catalog_search", "sevir_event_geolocation", "event_history_comparison"].includes(result.task)) return "SEVIR CATALOG";
  if (result.task === "cascast_dataset_evaluation") return "CASCAST · 10 EVENTS + GT";
  return "SEVIR TOOLBOX";
}

function renderAnswer(text) {
  const container = document.querySelector("#answer");
  container.innerHTML = "";
  const lines = String(text).replace(/\r/g, "").split("\n");
  const clean = (value) => value.trim().replace(/\*\*/g, "").replace(/`([^`]+)`/g, "$1");
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
      while (index < lines.length && lines[index].includes("|")) {
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
    if (/^#{1,6}\s+/.test(trimmed)) {
      closeList();
      const heading = document.createElement("h4");
      heading.textContent = clean(trimmed.replace(/^#{1,6}\s+/, ""));
      container.appendChild(heading);
      index += 1;
      continue;
    }
    if (/^[-*•]\s+/.test(trimmed)) {
      if (!list) {
        list = document.createElement("ul");
        container.appendChild(list);
      }
      const item = document.createElement("li");
      item.textContent = clean(trimmed.replace(/^[-*•]\s+/, ""));
      list.appendChild(item);
      index += 1;
      continue;
    }
    closeList();
    const paragraph = document.createElement("p");
    paragraph.textContent = clean(trimmed);
    container.appendChild(paragraph);
    index += 1;
  }
}

function renderLimit(result) {
  const card = document.querySelector("#limit-card");
  const messages = [];
  if (result.interpretation_limit) messages.push(result.interpretation_limit);
  if (result.ground_truth_available === false || result.model_metrics?.ground_truth_available === false) messages.push("当前样本没有ground truth，不能计算真实预测误差。");
  const ensembles = result.ensemble_count ?? result.model_metrics?.ensemble_count;
  if (ensembles === 1) messages.push("当前仅有1个ensemble成员，暂时不能形成可靠的不确定性分布。");
  const unique = [...new Set(messages)];
  document.querySelector("#limit-text").textContent = unique.join(" ");
  card.classList.toggle("hidden", unique.length === 0);
}

function renderError(message) {
  emptyState.classList.add("hidden");
  resultContent.classList.add("hidden");
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  runState.textContent = "ERROR";
  runState.className = "run-state";
}

taskInput.addEventListener("change", syncTaskControls);
dataSourceInput.addEventListener("change", syncDataSource);
[weatherModelInput, leadInput, eventInput, thresholdInput].forEach((element) => {
  element.addEventListener("change", () => {
    if (taskInput.value !== "auto") buildQuestion();
  });
});
modeInput.addEventListener("change", syncMode);
providerInput.addEventListener("change", syncProvider);
weatherModelInput.addEventListener("change", syncWeatherModelNote);
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
  runButton.disabled = true;
  runButton.firstElementChild.textContent = "正在分析…";
  runState.textContent = "RUNNING";
  runState.className = "run-state running";
  errorBox.classList.add("hidden");

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: modeInput.value,
        provider: providerInput.value,
        model: modelInput.value.trim(),
        task: taskInput.value,
        data_source: dataSourceInput.value,
        weather_model: weatherModelInput.value,
        lead_minutes: Number(leadInput.value),
        hazard_level: thresholdInput.value,
        event_id: eventInput.value.trim(),
        upload_id: dataSourceInput.value === "upload" ? uploadedSampleId : "",
        sequence_upload_id: dataSourceInput.value === "upload_sequence" ? uploadedSequenceId : "",
        question: questionInput.value.trim(),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "运行失败");
    renderResult(data);
  } catch (error) {
    renderError(error.message);
  } finally {
    runButton.disabled = false;
    runButton.firstElementChild.textContent = "让Agent分析";
  }
});

syncTaskControls();
syncDataSource();
syncMode();
loadConfig();
restorePersistedUploads();
