# ERA5–Aurora 专项 Agent 重测（Round 2）

事件：`R19080121067677`

说明：本轮测试读取已经生成的本地 Aurora rollout 和 ERA5 验证结果，没有重新运行 Aurora 模型。

## 结果摘要

| 测试 | 结果 | 主要表现 |
|---|---|---|
| local_observation | 部分通过 | 能识别 SEVIR，但偶尔声称原始数据不可访问 |
| local_nowcast | fallback | 回退到 CasCast 本地结果，未完成 foundation 相关判断 |
| environment_analysis | 未通过 | 一次回答 ERA5 不可用 |
| environment_forecast | 未通过 | 未稳定引用 Aurora +6/+12/+18/+24h rollout |
| full_case | 部分通过 | 能列出四源，但出现不可信的 0.0 数值 |
| id_conflict | 超时 | 并行压力下未完成冲突处理 |
| unavailable_model | 部分通过 | 能说明 Aurora 不提供 VIL，但事件识别不稳定 |
| evidence_conflict | 通过边界解释 | 能降低结论强度，但缺少当前 case 的具体数值 |
| era5_profile | 部分通过 | 正确引用 850/500/250 hPa 位势高度和850 hPa水汽 |
| era5_timeseries | 通过 | 正确引用事件前后 ERA5 温度、气压和风速变化 |
| aurora_value | 未通过 | 曾报告 `foundation.case` 缺少 `evidence` 字段 |
| aurora_cost | 部分通过 | 基本判断 +30 分钟局地 VIL 不需要 Aurora，但时效表述不稳定 |

## 本轮发现的问题

1. `foundation.case()` 原先只返回 `case_evidence`，而 Agent 某些路径读取 `evidence`。
2. 部分回答没有读取当前事件证据，而是生成了通用天气解释。
3. 完整四模型回答出现 `0.0`，属于证据读取失败后的不可信填充。
4. 并行调用造成部分请求超时并进入 fallback。

## 已完成修复

`sevir_world/tools/foundation.py` 现在同时返回：

```text
evidence
case_evidence
artifacts
manifest
```

下一轮复测必须确认：ERA5/Aurora 数值来自当前 case，不能出现“数据不可用”或 `0.0` 占位值。
