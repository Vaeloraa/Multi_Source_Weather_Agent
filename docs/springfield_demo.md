# Springfield, Missouri 端到端案例定义

这个案例把老师提出的“区域发现—局地追踪”流程固定到一个真实地点和一个可核验的历史事件，但**案例定义不等于数据和推理结果已经准备完成**。唯一可信的现场状态来自每次演示前运行的只读预检。

案例配置位于 `data/demo_cases/springfield_mo_20180811.json`，事件为 `S781667`，参考时刻为 `2018-08-11T00:05:00Z`，地点写作 `Springfield, Missouri`。不要只写 `Springfield`，因为美国有多个同名城市。

## 已核验的事件元数据

本地 `data/sevir_complete/CATALOG.csv` 包含同一事件的五条对齐记录：

| 通道 | 文件 | 索引 | 当前集成状态 |
| --- | --- | ---: | --- |
| VIL | `vil/2018/SEVIR_VIL_STORMEVENTS_2018_0701_1231.h5` | 105 | 当前 VIL 加载器支持；本案例数据是否已在本地仍由预检判断 |
| VIS | `vis/2018/SEVIR_VIS_STORMEVENTS_2018_0801_0831.h5` | 63 | 仅确认目录元数据，尚未接入案例加载器 |
| IR107 | `ir107/2018/SEVIR_IR107_STORMEVENTS_2018_0701_1231.h5` | 108 | 仅确认目录元数据，尚未接入案例加载器 |
| IR069 | `ir069/2018/SEVIR_IR069_STORMEVENTS_2018_0701_1231.h5` | 108 | 仅确认目录元数据，尚未接入案例加载器 |
| LGHT | `lght/2018/SEVIR_LGHT_ALLEVENTS_2018_0801_0901.h5` | 0 | 仅确认目录元数据，尚未接入案例加载器 |

五条记录的事件类型都是 `Thunderstorm Wind`，边界框为纬度 `35.5452–38.8176°N`、经度 `95.5243–90.9729°W`。Springfield, Missouri 的案例坐标位于该边界框内，距离边界框中心约 4.9 km。

源目录中的 IR069 行以重复的 `+115` 分钟结束，而不是 `+120` 分钟。配置与预检会保留这个源数据异常，不会静默修正。

## CasCast 时间窗为什么必须是 segment 1

SEVIR 事件共有 49 个、间隔 5 分钟的 VIL 时次。当前 CasCast 远程事件路径对 segment 1 使用零基切片 `[12:37]`：

- 13 帧历史：`2018-08-10T23:05:00Z` 至 `2018-08-11T00:05:00Z`，即相对时刻 `-60` 至 `0` 分钟；
- 12 帧验证：`2018-08-11T00:10:00Z` 至 `2018-08-11T01:05:00Z`，即相对时刻 `+5` 至 `+60` 分钟。

因此这个案例必须传 `segment=1`。默认 `segment=0` 会使用更早的一小时，无法表达“事件参考时刻此前一小时 → 此后一小时验证”的目标。

## 当前能力边界

| 部分 | 可以准确陈述 | 不能陈述 |
| --- | --- | --- |
| SEVIR | 本地目录已确认五种对齐通道；现有代码可按事件加载 VIL | 五个 HDF5 已下载，或五通道已经全部接入 |
| CasCast | 现有推理链可在 VIL 与模型资产可用时运行 segment 1 | `S781667` 的预测已经运行或已有缓存，除非预检找到对应文件 |
| ERA5/Aurora | 仓库已有通用 ERA5/Aurora 能力，案例目标是 6、12、18、24 小时区域发现 | 本案例已有对齐的 ERA5、Aurora 输出或验证结果，除非预检找到案例清单 |
| NEXRAD/GOES | SEVIR VIL 与 VIS/IR 可作为本次历史案例的雷达/卫星观测 | 已实现任意时间地点的 NEXRAD/GOES 下载与到 CasCast 的通用转换 |

尤其要区分两件事：SEVIR 中包含历史对齐的 VIL、VIS 和 IR 记录，不代表“原始 NEXRAD/GOES 在线下载适配器”已经实现。后者仍需要独立的数据检索、投影、时序对齐、质量控制和 CasCast 输入标定。

## 演示前预检

在仓库根目录执行：

```powershell
python .\scripts\preflight_springfield_demo.py
```

该命令只读，不下载、不推理、不生成缓存。重点查看：

- `metadata_verified`：五条目录记录是否仍与案例定义一致；
- `cascast_input_present_locally`：VIL HDF5 或事件级缓存是否存在；
- `cascast_segment_1_cache_present`：是否存在 `S781667_s1_ens*.npz`；
- `aurora_case_manifest_present`：是否存在本案例专用 Aurora 清单；
- `cached_end_to_end_demo_ready`：两段模型缓存是否都满足现场快速展示条件。

默认情况下，只要元数据正确，脚本即以 0 退出；这并不代表 Demo 已准备好。CI 或演示前门禁应使用：

```powershell
python .\scripts\preflight_springfield_demo.py --require-ready
```

在缓存不完整时该命令返回 2。不要用截图、其他事件的缓存或通用模型资产替代本案例状态。

## 建议的现场叙事

准备完整后，现场输入可固定为：

```text
地点：Springfield, Missouri
UTC 时间：2018-08-11 00:05
任务：先展示 Aurora 对目标时段的大尺度区域发现，再用此前一小时 VIL/卫星观测启动 CasCast segment 1，最后用此后一小时观测验证局地演变。
```

页面进度应分开报告“目录定位”“观测准备”“CasCast 推理/缓存命中”“ERA5 准备”“Aurora 推理/缓存命中”“后验验证”，不能在一个阶段成功后把整个链路显示为完成。

如果预检仍显示 `metadata_only`，本案例只适合演示地点解析、事件定位、通道目录和时间窗规划；不得现场宣称 CasCast 或 Aurora 已产出结果。如果只准备了其中一段，应明确说“部分准备完成”，并展示缺失阶段和现场运行入口。
