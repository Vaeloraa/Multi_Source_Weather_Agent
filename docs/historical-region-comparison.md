# 地区历史同期对比

## 功能定位

系统现在可以将一个 CasCast 事件的未来预测，与同一 SEVIR 地理范围内、相近月份的历史 VIL 事件进行同口径比较。比较对象包括：

1. CasCast ensemble mean 的未来 12 帧；
2. 对齐的 Ground Truth 未来 12 帧；
3. CATALOG 中空间包围盒相交、月份相近的历史事件。

历史候选先从 CATALOG 元数据筛选，随后才按需读取对应 HDF5；不会因为查询一个地区而下载完整数据集。

## Agent 调用方式

如果用户输入的是地点名称，Agent 可以先解析为代表点，再检索候选事件：

```python
place = geolocator.resolve_place("Los Angeles")
if place["status"] == "ok":
    region_events = data.search_region(
        latitude=place["latitude"],
        longitude=place["longitude"],
        radius_degrees=1.0,
        month=6,
        limit=20,
    )
```

地点解析依赖可配置的地理编码服务（`SEVIR_GEOCODER_URL`），也支持直接传入
`"latitude,longitude"` 而不访问网络。`ambiguous`、`not_found` 或
`geocoder_unavailable` 都必须交给 Agent 解释，不能静默选择一个地点。

```python
region_events = data.search_region(
    latitude=40.0,
    longitude=-90.0,
    radius_degrees=1.0,
    month=6,
    limit=20,
)

comparison = statistics.compare_cascast_history(
    event_id="R19100105107994",
    month_window=1,
    threshold=40,
    candidate_limit=10,
    # 可选：像素区域 [row0, col0, row1, col1]
    region=[80, 80, 300, 300],
)
```

比较结果包含 `forecast.metrics`、`ground_truth.metrics`、`historical` 和 `historical_events`。指标采用相同定义：最大 VIL、均值、P95、阈值覆盖率。`historical.*.current_percentile` 是有限历史样本中的经验百分位，不是校准概率。

如果参考事件和候选事件都具有完整投影元数据，参考区域会先通过
`geolocator.pixel_box_to_latlon_bounds()` 转换为 WGS84，再通过
`geolocator.latlon_box_to_pixel_box()` 映射到每个候选事件自己的网格；因此候选事件不会直接复用参考事件的像素坐标。返回的 `provenance.historical_region_alignment` 会说明使用了逐事件地理对齐还是缺少投影时的像素框回退。

Unified Agent 只接收已注入工具的名称，并在需要时用 `dir()` 和 `inspect.signature()` 发现方法；关键词不再决定它能否使用历史、地理或 Ensemble 能力。Agent 在持久 Python 环境中按问题选择实际调用的工具和指标。

## 数据边界

- CATALOG 地理筛选使用事件包围盒相交，不能保证天气核心正好位于用户地点。
- 历史样本可能很少；`sample_count=0` 时不能给出历史百分位。
- VIL 是编码值，阈值是分析阈值，不等价于官方降雨量、预警等级或灾害概率。
- 普通上传图片如果没有事件匹配或地理信息，系统不能仅凭图像内容推断地点。

## Demo 案例

`statistics.demo_cases()` 会根据当前缓存中的 Ensemble 和 Ground Truth 自动标注弱天气、强天气、集合差异大、真实结果超出集合范围和时间变化明显等案例。案例标签用于演示和回归测试，不是写死的预测结论。

推荐演示流程：

```text
输入地区或上传 13 帧观测
→ 确认地理范围
→ CasCast 预测未来 12 帧
→ 计算地区指标
→ 检索相近月份的历史事件
→ 比较历史分布与当前预测
→ 展示 Ensemble 差异
→ 用户继续追问阈值、成员或重点地点
```
