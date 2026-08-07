import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = ".";

const gapHeaders = ["部分", "Zephyrus", "当前Demo", "差距判断", "改进动作", "优先级"];
const gapRows = [
  ["数据读取", "WeatherBench 2、气候态和地理数据访问能力预先封装", "SEVIR目录、HDF5读取和CasCast缓存读取已封装", "基础方向一致；当前历史统计仍较依赖CATALOG元数据", "补充逐像素VIL统计、数据校验、统一数据句柄，避免工具重复加载大数组", "P1"],
  ["模型调用", "预报模型作为可调用能力，由Agent在任务中触发", "10个事件的CasCast结果预先运行并保存；提问时不现场推理", "关键差距", "实现predict(history_13_frames, ensemble_size)接口；保留离线缓存作为可复现实验模式", "P0"],
  ["统计分析代码", "Agent按问题现场生成Python并组合基础API", "核心统计由forecast_hazards等固定Python函数完成", "稳定但开放性不足", "采用混合方案：科学公式继续封装；向Agent开放只读、限时的组合分析环境", "P1"],
  ["工具组合", "Agent自主选择并组合数据、预报、气候态和地理工具", "最多连续调用3个工具，但三个预测家族工具互斥", "复杂问题拆解能力不足", "引入任务计划和结果句柄，允许预测、Hazard、不确定性、历史比较按需组合且不重复推理", "P0"],
  ["中间结果反思", "Reflective Agent观察中间输出、修正代码并继续分析", "有短工具循环和失败降级，但缺少显式计划、观察、校验与重试状态", "较弱", "增加plan-observe-check-revise循环、数值一致性检查和最大步骤/时间限制", "P1"],
  ["Hazard定义", "按变量、区域、阈值、面积/面积占比和时间窗口构造任务", "VIL阈值明确，但Hazard和对象统计仍受minimum_region_pixels=16影响", "必须重构", "从Hazard判据移除16像素；改用VIL等级、面积占比、连续时长和ensemble支持率；连通域仅用于定位展示", "P0"],
  ["评测体系", "ZephyrusBench分别评估数值、布尔、位置、时间、描述等任务", "已有10事件预测指标，但缺少Agent问题理解、工具选择和最终回答评测", "模型评测有基础，Agent评测缺失", "使用任务CSV生成测试问题，分别评估路由、参数、计算结果、证据一致性和限制说明", "P0"],
  ["最终解释", "LLM根据工具执行结果生成回答", "Qwen根据结构化JSON生成中文回答和页面图像", "方向一致", "为结论绑定工具字段和图像；区分模型分歧、真实误差、代理Hazard与官方预警", "P1"],
];

const taskHeaders = [
  "task_id", "task_name", "category", "priority", "current_status", "difficulty",
  "question_source_style", "user_question_example", "task_goal", "required_inputs",
  "agent_plan", "tool_sequence", "expected_outputs", "visualization",
  "zephyrus_reference", "eval_type", "acceptance_criteria", "scientific_constraints", "next_change"
];

const taskRows = [
  ["T01", "指定时效VIL预报", "基础预报", "P0", "已有", "Medium", "Human+Synthetic", "该事件在+20分钟和+60分钟的VIL强度与空间分布如何？", "查询指定未来时效的VIL统计和完整预测场。", "event_id; lead_minutes; model_name=cascast", "读取对应时效预测；计算最大值、均值和超等级面积占比；返回图像及数据来源。", "run_nowcast", "指定时效VIL统计; 完整预测帧; 数据来源", "指定时效预测图", "直接参考Zephyrus任务8", "numerical+image", "请求时效必须对应真实预测帧；数值与图像来自同一数组。", "VIL是0-255缩放像素值；当前结果来自离线CasCast缓存。", "为run_nowcast增加按lead_minutes聚焦的结构化输出。"],
  ["T02", "未来极值出现时间", "时效分析", "P0", "已有数据，回答需结构化", "Medium", "Human+Synthetic", "未来60分钟内，该事件的最大VIL预计在第几分钟出现？", "确定未来窗口中区域最大值或面积峰值出现的时效。", "event_id; horizon; extremum_metric=max_vil或area_ratio", "逐时效计算指定指标；寻找极值；处理并列极值；返回对应预测帧。", "run_nowcast -> extremum_time_analysis", "极值; 首次极值时效; 全部并列时效; 对应帧", "逐时效曲线; 极值帧", "直接参考Zephyrus任务9", "temporal+numerical", "必须明确极值指标；不能把最大VIL时效与最大超阈值面积时效混为一谈。", "极值可能受单像素噪声影响，需同时报告面积指标。", "新增extremum_metric和并列极值处理。"],
  ["T03", "区域内是否出现阈值超越", "阈值检测", "P0", "已有，但应与Hazard分开", "Medium", "Human+Synthetic", "未来60分钟内，事件范围任意位置是否会达到较强回波等级？", "判断指定区域内是否至少一次出现VIL阈值超越，只回答检测结果，不直接认定Hazard。", "event_id; hazard_level; region_scope; horizon", "后端将等级映射为VIL阈值；逐时效检查区域；返回首次与最后超越时效。", "run_nowcast -> threshold_exceedance_analysis", "是否超越; 首次/末次时效; 峰值; 超越像素数", "超阈值掩膜; 时效标记", "直接参考Zephyrus任务20", "boolean+temporal", "threshold_exceeded与hazard_detected使用不同字段；不得用16像素决定是否超越。", "任意像素超越对噪声敏感，所以只能称阈值检测。", "从forecast_hazards拆出纯阈值检测结果。"],
  ["T04", "超阈值面积占比", "空间范围", "P0", "已有逐帧面积，需成为核心任务", "Medium", "Human+Synthetic", "未来各时效中，达到较强回波等级的面积占事件范围多少？是否超过指定比例？", "计算每个时效超阈值面积占比，并判断是否达到研究配置中的面积比例。", "event_id; hazard_level; region_scope; area_ratio_rule; horizon", "构造区域掩膜；逐时效统计超阈值像素占有效区域比例；计算峰值和首次达到面积规则的时效。", "run_nowcast -> area_ratio_analysis", "逐时效面积占比; 峰值占比; 是否达到面积规则; 首次达到时效", "面积占比时序; 超阈值区域图", "直接参考Zephyrus任务22", "area+boolean+temporal", "分母必须是明确的有效区域；面积规则必须来自配置和实验，不能由16像素代替。", "像素占比尚非平方公里；不同事件图幅物理面积可能不同。", "增加region_scope和area_ratio_rule；从Hazard配置移除minimum_region_pixels。"],
  ["T05", "连续超阈值持续时间", "时间持续性", "P0", "部分已有，需改为连续规则", "Hard", "Human+Synthetic", "较强回波空间范围达到要求后，最长连续持续多少分钟？", "在面积条件成立的时效序列上计算最长连续持续时间、开始和结束时效。", "event_id; hazard_level; area_ratio_rule; minimum_consecutive_minutes; horizon", "先计算逐时效面积占比；生成面积规则布尔序列；寻找最长连续True区间；区分累计时长和连续时长。", "area_ratio_analysis -> duration_analysis", "最长连续时长; 累计时长; 起止时效; 是否覆盖窗口末端", "持续时间条带; 面积占比时序", "改造参考Zephyrus任务27", "temporal+boolean", "Hazard持续时间必须使用连续时长；不能使用12帧中活跃帧数量简单相加代替。", "5分钟时间分辨率限制了持续时间精度。", "重写成员active_duration为longest_consecutive_duration并保留累计值。"],
  ["T06", "Ensemble强回波中心分歧", "空间不确定性", "P1", "已有中心离散度，定义需改进", "Hard", "Human+Synthetic", "10路成员预测的主要强回波中心相差多远，位置是否有共识？", "比较ensemble成员的超阈值区域质心，描述位置分歧与可能的多峰情景。", "event_id; hazard_level; lead_minutes或time_window; centroid_region_rule", "按相同空间规则为每个成员计算质心；计算成员间距离和聚类；报告无有效区域成员。", "analyze_ensemble_uncertainty -> centroid_scenario_analysis", "成员质心; 离散度; 成员间距离; 情景簇; 有效成员数", "质心散点; 成员场; 情景簇", "改造参考Zephyrus任务32", "distance+coordinate+description", "所有成员必须使用相同区域和阈值规则；不能只取时间窗最大连通域后假装为同一对象。", "像素距离不是公里；成员离散度不等于真实位置误差。", "将质心定义与lead/time_window显式化；增加多峰识别。"],
  ["T07", "强回波位置移动", "对象与轨迹", "P1", "后端部分已有", "Hard", "Human+Synthetic", "未来60分钟主要强回波中心移动了多远，方向如何？", "跟踪主要超阈值区域质心，计算起止位移和逐时效路径。", "event_id; hazard_level; tracking_rule; horizon", "逐时效提取区域；执行对象关联；计算轨迹、起止质心和位移；标记分裂、合并和失配。", "run_nowcast -> storm_object_tracking -> geolocate_sevir_event", "轨迹; 起止位置; 位移; 方向; 分裂/合并标记", "轨迹叠加图; 逐时效对象图", "改造参考Zephyrus任务38", "distance+coordinate+temporal", "对象过滤配置只服务提取，不参与Hazard判定；轨迹失配必须显式报告。", "当前坐标为近似映射，不能作为官方路径预报。", "分离object_extraction_config；补充分裂合并和物理距离换算。"],
  ["T08", "相对历史气候态的异常持续", "历史异常", "P1", "暂不完整，需先升级历史数据", "Hard", "Human+Synthetic", "该事件较强VIL是否超过历史同类事件的高分位，并持续一段时间？", "使用历史逐像素或逐事件VIL基准定义异常，并计算异常持续时间。", "event_id; climatology_period; quantile; comparison_group; duration_rule", "构建历史基准；计算目标分位阈值；逐时效比较；计算连续异常时长并返回样本量。", "historical_pixel_statistics -> quantile_exceedance_duration", "历史阈值; 异常标记; 连续时长; 基准样本量", "历史分布; 分位阈值; 异常时序", "改进后参考Zephyrus任务10", "list+temporal+numerical", "必须使用真实历史VIL基准；CATALOG data_max不能冒充逐像素气候态。", "SEVIR事件样本并非均匀气候观测，基准集合选择会影响结论。", "先建立HDF5逐像素/逐事件历史统计，再开放此任务。"],
  ["T09", "当前/未来极端强回波综合识别", "综合Hazard", "P1", "概念已有，科学判据未完成", "Hard", "Human", "当前是否存在极端强回波事件？未来60分钟是否会发展为Hazard？", "让Agent组合阈值、面积、持续时间、ensemble支持和历史异常证据，生成综合研判。", "event_id; hazard_level; area_ratio_rule; duration_rule; ensemble_support_rule; historical_baseline", "先规划；调用预报、面积、持续时间、ensemble和历史工具；检查证据是否满足规则；区分观测、预测与不确定性。", "run_nowcast -> area_duration_analysis + ensemble_support_analysis + historical_anomaly", "当前/未来结论; 证据表; 成员支持; 历史异常; 限制", "综合证据面板; 关键时效图; 工具轨迹", "改进后参考Zephyrus任务11/12", "list+description+tool_trace", "没有完整规则或历史基准时只能报告候选强回波，不能宣称极端天气事件。", "VIL代理不能直接推出洪水、暴雨或官方灾害等级。", "完成T04/T05/T08和ensemble校准后再开放正式综合判定。"],
  ["T10", "超阈值物理面积", "物理空间量", "P2", "缺失", "Medium", "Synthetic", "未来+30分钟，达到较强回波等级的区域面积是多少平方公里？", "把超阈值像素转换为物理面积，支持跨事件比较。", "event_id; lead_minutes; hazard_level; projection; pixel_area", "读取投影和网格面积；构造超阈值掩膜；按像素面积求和；输出空间范围和误差说明。", "run_nowcast -> projected_area_analysis", "面积km2; 面积占比; 网格面积来源; 不确定性", "超阈值区域; 面积数字", "改进后参考Zephyrus任务31", "area+numerical", "必须使用经验证投影/网格面积；不能用像素数直接标成平方公里。", "当前只可靠报告像素占比，物理面积尚未校准。", "验证SEVIR投影和像素面积后实现。"],
];

const fieldHeaders = ["字段", "用途"];
const fieldRows = [
  ["task_id", "稳定任务编号，用于测试、日志和版本管理。"],
  ["question_source_style", "仅表示问题样式可来自人工或模板生成，不决定运行时工具。"],
  ["agent_plan", "Agent应完成的分析步骤；不是固定答案。"],
  ["tool_sequence", "推荐工具组合；运行时允许Agent根据问题裁剪，但必须满足科学约束。"],
  ["zephyrus_reference", "参考question_template_matching.csv中的任务编号或Zephyrus架构层，不表示直接调用其CSV。"],
  ["eval_type", "答案评测类型，可组合numerical、temporal、location、description、execution等。"],
  ["scientific_constraints", "必须在回答与实验中遵守的科学边界。"],
  ["next_change", "当前Demo实现该任务的下一项具体工作。"],
];

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(headers, rows) {
  return [headers, ...rows].map((row) => row.map(csvEscape).join(",")).join("\r\n") + "\r\n";
}

await fs.mkdir(outputDir, { recursive: true });
await fs.writeFile(`${outputDir}/cascast_task_design.csv`, "\uFEFF" + toCsv(taskHeaders, taskRows), "utf8");

const workbook = Workbook.create();
const gapSheet = workbook.worksheets.add("架构差距");
const taskSheet = workbook.worksheets.add("任务设计");
const fieldSheet = workbook.worksheets.add("字段说明");

gapSheet.showGridLines = false;
taskSheet.showGridLines = false;
fieldSheet.showGridLines = false;

gapSheet.getRange("A1:F1").merge();
gapSheet.getRange("A1").values = [["Zephyrus与SEVIR/CasCast Demo差距及改进"]];
gapSheet.getRange("A1:F1").format = { fill: "#17324D", font: { bold: true, color: "#FFFFFF", size: 16 }, verticalAlignment: "center" };
gapSheet.getRange("A1:F1").format.rowHeight = 34;
gapSheet.getRange("A3:F11").values = [gapHeaders, ...gapRows];
gapSheet.getRange("A3:F3").format = { fill: "#DCEAF4", font: { bold: true, color: "#17324D" }, wrapText: true, verticalAlignment: "center" };
gapSheet.getRange("A4:F11").format = { wrapText: true, verticalAlignment: "top", borders: { insideHorizontal: { style: "thin", color: "#D9E2EA" } } };
gapSheet.getRange("A3:F11").format.borders = { outside: { style: "thin", color: "#AAB7C4" }, insideHorizontal: { style: "thin", color: "#D9E2EA" } };
gapSheet.getRange("A4:A11").format.font = { bold: true, color: "#17324D" };
gapSheet.getRange("F4:F11").format = { horizontalAlignment: "center", verticalAlignment: "center", font: { bold: true } };
gapSheet.getRange("A:A").format.columnWidth = 18;
gapSheet.getRange("B:B").format.columnWidth = 34;
gapSheet.getRange("C:C").format.columnWidth = 38;
gapSheet.getRange("D:D").format.columnWidth = 22;
gapSheet.getRange("E:E").format.columnWidth = 48;
gapSheet.getRange("F:F").format.columnWidth = 10;
gapSheet.getRange("4:11").format.rowHeight = 70;
gapSheet.freezePanes.freezeRows(3);

taskSheet.getRange("A1:S1").merge();
taskSheet.getRange("A1").values = [["SEVIR / CasCast Agent任务设计（参考Zephyrus任务范式）"]];
taskSheet.getRange("A1:S1").format = { fill: "#17324D", font: { bold: true, color: "#FFFFFF", size: 16 }, verticalAlignment: "center" };
taskSheet.getRange("A1:S1").format.rowHeight = 34;
taskSheet.getRange("A2:S2").merge();
taskSheet.getRange("A2").values = [["说明：Zephyrus用于启发任务拆解与Agent工作流；VIL阈值、面积、持续时间和ensemble支持规则必须由本项目单独定义和验证。"]];
taskSheet.getRange("A2:S2").format = { fill: "#FFF4D6", font: { color: "#6D4C00" }, wrapText: true, verticalAlignment: "center" };
taskSheet.getRange("A2:S2").format.rowHeight = 38;
taskSheet.getRange(`A4:S${taskRows.length + 4}`).values = [taskHeaders, ...taskRows];
taskSheet.getRange("A4:S4").format = { fill: "#DCEAF4", font: { bold: true, color: "#17324D" }, wrapText: true, verticalAlignment: "center" };
taskSheet.getRange(`A5:S${taskRows.length + 4}`).format = { wrapText: true, verticalAlignment: "top", borders: { insideHorizontal: { style: "thin", color: "#D9E2EA" } } };
taskSheet.getRange(`A4:S${taskRows.length + 4}`).format.borders = { outside: { style: "thin", color: "#AAB7C4" }, insideHorizontal: { style: "thin", color: "#D9E2EA" } };
taskSheet.getRange(`A5:A${taskRows.length + 4}`).format.font = { bold: true, color: "#17324D" };
taskSheet.getRange(`D5:D${taskRows.length + 4}`).format = { horizontalAlignment: "center", verticalAlignment: "center", font: { bold: true } };
taskSheet.getRange(`A5:S${taskRows.length + 4}`).conditionalFormats.add("containsText", { text: "P0", format: { fill: "#FCE8E6", font: { color: "#9C1C13", bold: true } } });
const widths = [9, 24, 16, 9, 21, 11, 19, 42, 38, 34, 52, 48, 46, 34, 20, 24, 48, 48, 48];
for (let i = 0; i < widths.length; i += 1) taskSheet.getRangeByIndexes(0, i, taskRows.length + 4, 1).format.columnWidth = widths[i];
taskSheet.getRange(`5:${taskRows.length + 4}`).format.rowHeight = 104;
taskSheet.freezePanes.freezeRows(4);
taskSheet.freezePanes.freezeColumns(2);

fieldSheet.getRange("A1:B1").merge();
fieldSheet.getRange("A1").values = [["任务CSV字段说明"]];
fieldSheet.getRange("A1:B1").format = { fill: "#17324D", font: { bold: true, color: "#FFFFFF", size: 16 }, verticalAlignment: "center" };
fieldSheet.getRange("A1:B1").format.rowHeight = 34;
fieldSheet.getRange(`A3:B${fieldRows.length + 3}`).values = [fieldHeaders, ...fieldRows];
fieldSheet.getRange("A3:B3").format = { fill: "#DCEAF4", font: { bold: true, color: "#17324D" } };
fieldSheet.getRange(`A4:B${fieldRows.length + 3}`).format = { wrapText: true, verticalAlignment: "top", borders: { insideHorizontal: { style: "thin", color: "#D9E2EA" } } };
fieldSheet.getRange(`A3:B${fieldRows.length + 3}`).format.borders = { outside: { style: "thin", color: "#AAB7C4" }, insideHorizontal: { style: "thin", color: "#D9E2EA" } };
fieldSheet.getRange("A:A").format.columnWidth = 28;
fieldSheet.getRange("B:B").format.columnWidth = 86;
fieldSheet.getRange(`4:${fieldRows.length + 3}`).format.rowHeight = 42;
fieldSheet.freezePanes.freezeRows(3);

const gapTable = gapSheet.tables.add("A3:F11", true, "ArchitectureGapTable");
gapTable.style = "TableStyleMedium2";
const taskTable = taskSheet.tables.add(`A4:S${taskRows.length + 4}`, true, "CasCastTaskTable");
taskTable.style = "TableStyleMedium2";
const fieldTable = fieldSheet.tables.add(`A3:B${fieldRows.length + 3}`, true, "TaskFieldTable");
fieldTable.style = "TableStyleMedium2";

const inspect = await workbook.inspect({ kind: "table", range: `任务设计!A4:S${taskRows.length + 4}`, include: "values,formulas", tableMaxRows: 15, tableMaxCols: 19, maxChars: 12000 });
await fs.writeFile(`${outputDir}/verification.txt`, inspect.ndjson, "utf8");

for (const [sheetName, range, fileName, scale] of [
  ["架构差距", "A1:F11", "preview_architecture.png", 1.2],
  ["任务设计", "A1:H16", "preview_tasks.png", 0.8],
  ["字段说明", "A1:B11", "preview_fields.png", 1.2],
]) {
  const preview = await workbook.render({ sheetName, range, scale, format: "png" });
  await fs.writeFile(`${outputDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(`${outputDir}/cascast_task_design.xlsx`);

console.log(JSON.stringify({ csvRows: taskRows.length, sheets: 3, outputDir }, null, 2));
