# SEVIR Hazard Agent

## SEVIRWorld 与 Zephyrus-Direct

当前版本已加入可组合的 `sevir_world` 高层 API，包括数据索引、地理定位、
CasCast 缓存预测、历史统计、不确定性分析、多阈值验证和图表生成。网页的
`Direct` 模式会让 Qwen 生成只调用这些 API 的短 Python 程序，并在独立子进程
中执行；执行失败时最多自动修复 3 次，最终回答只根据真实执行结果生成。

代码执行边界禁止导入模块、文件访问、网络和系统命令，并限制 AST 大小、
运行时间、内存、标准输出和结果长度。每次执行会返回生成代码、工具调用链、
执行结果及错误记录，供网页展示和复核。完整记录以执行 ID 保存到
`data/executions/<execution_id>.json`。

行政区反向定位使用 `data/boundaries` 中的美国 Census 2025 州和县 1:20m
边界文件，完全离线运行。Agent 生成的 PNG 保存到 `data/artifacts`，并通过只允许
32 位十六进制产物 ID 的 `/api/artifacts/<artifact_id>.png` 路由读取。

运行回归测试：

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s tests -v
```

运行 6 类自然语言问题的固定 API 路由与 Direct 模式对比：

```powershell
python scripts\evaluate_agent_modes.py `
  --base-url http://127.0.0.1:8000 `
  --provider qwen `
  --modes api,direct
```

评测报告保存到 `data/agent_evaluations`，包含请求成功率、多工具组合成功率、
预期工具命中率、执行结果引用率、自动修复率、平均调用次数和响应时间。

现场上传的 13 帧序列目前仍只运行 Persistence 基线；任意上传序列的实时
CasCast 推理仍需要完整 EarthFormer 预处理和确定性预测链路，Direct 模式不会
把该结果表述为实时 CasCast。

这是一个受Zephyrus启发、面向SEVIR降雨临近预报的Agentic AI原型。

## 当前已经完成

- Qwen API真实接入，可自主选择气象工具并读取执行结果；
- 默认交互已改为自然语言自动路由，用户无需先选择固定任务类型；
- 已接入10个SEVIR测试事件；每个事件包含13帧历史观测、10路CasCast ensemble和未来12帧输出；
- 已严格对齐上述10个事件的12帧ground truth，可进行真实定量评价；
- CasFormer和Autoencoder真实checkpoint已在Windows单GPU环境加载并完成推理；
- 可在真实CasCast输出上筛查VIL强回波区域、近似坐标和持续时间；
- 支持用户上传384×384的SEVIR VIL图片，并针对图片进行自然语言提问；
- 支持一次上传13张连续VIL图，分析过去60分钟演变、连通区域轨迹并运行Persistence基线；
- 上传的单张图片支持当前强回波统计和hazard区域分析，不冒充未来预测；
- 页面内置3张由官方SEVIR事件`S696651`导出的原始VIL灰度帧，可直接点击作为上传示例；
- CasCast预测页展示未来`+5`至`+60`分钟的12个时效，可点击切换空间预览；
- Hazard阈值提供CasCast论文所用的SEVIR阈值预设`74/133/160/181/219`；
- Data Indexer可检索20,393条SEVIR VIL catalog记录并报告本地HDF5可用性；
- Historical Statistics可统计年份、月份、storm type、data_max分位数和强度排序；
- Geolocator可返回事件覆盖框与近似中心；
- History Comparison可将事件catalog data_max与历史参考组进行经验分位比较；
- satellite-to-radar和模拟器仍为预留工具接口；
- 已实现pixel-wise标准差、超阈值概率、storm-object位置/面积差异和ensemble成员柱状图；
- 已完成10事件聚合评价：6个标准VIL阈值、12个预测时效、CasCast与Persistence对比；
- 已用ground truth验证ensemble：spread-error相关、80%区间覆盖率、spread-skill ratio、可靠性曲线和rank histogram；
- 已实现逐帧storm-object匹配及跨时效连通区域轨迹分析；
- 评价结果可导出为CSV和Markdown，位于`data/evaluation_exports`；
- Agent可在复杂问题中连续调用最多3个工具，并在技术记录中展示调用链；
- 网页显示Agent回答、预测摘要、VIL空间预览、工具参数和完整返回。

## 重要限制

- 当前CasCast结果来自已保存并验收的真实推理数据包，网页不会在每次提问时重新启动模型；
- 每个事件有10个ensemble成员，可分析成员分歧，但ensemble不确定性不等于真实误差；
- 10个已接入事件已有ground truth，可计算CSI、POD、FAR、MSE、RMSE和MAE；但不能冒充完整SEVIR测试集评价；
- 当前只支持未来60分钟；2至3小时滚动预测尚未执行；
- VIL阈值仅作为强回波和潜在降雨危险代理，不能直接等同于地面降雨量或官方灾害预警；
- 当前接入10个CasCast测试事件，仍不能作为完整测试集实验结论；
- 历史统计来自CATALOG metadata；没有本地HDF5的事件不能进行逐帧强度或持续时间分析；
- 可上传图片应来自SEVIR VIL数组的单个384×384、uint8时间帧；普通雷达截图或伪彩色图不保留原始VIL数值；
- 上传单张图片只能分析当前状态；上传13帧可分析历史演变并运行Persistence，但任意序列现场CasCast仍缺EarthFormer确定性预测和latent压缩链路；
- satellite-to-radar仍需课题组模型、checkpoint及配对卫星通道数据。
- 当前CasCast样本对应的公开HDF5对象约16GB；不建议整文件下载，优先按`data/cascast_sample/requested_export.json`导出单样本输入、GT和ensemble。

## 运行

```powershell
cd "D:\Profile\专业实习\sevir-agent-demo"
python server.py
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## Demo任务

1. 未来危险区域与持续时间；
2. 未来VIL临近预报；
3. 历史强回波过程分析；
4. SEVIR事件查询；
5. satellite-to-radar synthesis；
6. 气象模拟器接口状态。

## 后端工具

- `search_sevir_catalog(query, year, month, event_type, limit)`
- `historical_sevir_statistics(year, event_type)`
- `geolocate_sevir_event(event_id)`
- `compare_event_with_history(event_id)`

- `get_event_summary(event_id)`
- `analyze_threshold(event_id, threshold)`
- `analyze_hazard(event_id, threshold, min_area_ratio)`
- `analyze_ensemble_uncertainty(event_id, threshold, minimum_region_pixels)`
- `evaluate_cascast_dataset(primary_threshold)`
- `analyze_uploaded_vil_sequence(upload_id, threshold, minimum_region_pixels)`
- `run_nowcast(event_id, model_name, input_frames, forecast_frames)`
- `forecast_hazards(event_id, model_name, lead_minutes, threshold)`
- `run_satellite_to_radar(event_id)`
- `run_simulator(event_id, simulator_name)`

## CasCast接入契约

任意上传序列的现场CasCast最终应封装成统一函数：

```python
predict(input_frames) -> forecast_frames
```

输入为13帧VIL，EarthFormer先生成确定性预测，再经Autoencoder压缩为latent，CasCast生成多路概率预测，最后解码为未来12帧VIL。当前网页已能接收这种输出格式并复用hazard、不确定性和评价逻辑，但上述任意事件现场推理链尚未接通。

## 当前10事件结果（VIL≥133）

- CasCast ensemble mean：CSI `0.498555`，POD `0.621622`，FAR `0.284233`，RMSE `16.697`；
- Persistence：CSI `0.333487`，POD `0.501043`，FAR `0.500694`，RMSE `26.5521`；
- pixel std与绝对误差Pearson相关：`0.585496`；
- 80%中心预测区间实际覆盖率：`0.491213`；
- spread-skill ratio：`0.276141`，说明ensemble离散度偏小；
- storm-object CSI：`0.26285`，匹配对象中心平均误差`8.171`像素。

## 安全

API Key只存放在`.env`中。不要提交`.env`，也不要把Key写入前端代码。
