# ERA5–Aurora 接入收尾报告

## 已完成

- `R19080121067677` 已完成 Aurora `+6/+12/+18/+24 h` rollout，并与后续 ERA5 做全域多时效验证。
- 验证产物已写入该 case 的 `artifacts`：`aurora_vs_era5_verification.json`、`aurora_vs_era5_timeseries.png`、`aurora_error_maps.png`。
- 已登记四个模型的 capability card，并定义统一 evidence bundle schema。
- 事件冲突、Aurora 缺失、证据冲突和“是否值得追加 Aurora”已进入确定性路径；local/reflective 两种模式均使用该路径。
- 模型选择评估集包含 12 道题，覆盖单模型、组合、冲突、缺失、证据不一致和成本决策。
- 最近一次 Reflective 回归：12/12 请求均在确定性/缓存路径完成；10 个 `ok`、1 个预期 `conflict`、1 个预期 `partial`，没有 `failure`；模型选择路由准确率为 1.0。

## 当前验证指标

| 时效 | 2 m 温度区域 MAE (°C) | 海平面气压区域 MAE (hPa) | 10 m 风速区域 MAE (m/s) |
|---|---:|---:|---:|
| +6 h | 1.444 | 0.940 | 0.800 |
| +12 h | 1.335 | 0.967 | 1.018 |
| +18 h | 1.273 | 1.129 | 1.382 |
| +24 h | 4.425 | 1.048 | 1.529 |

验证状态：`full_domain_multi_lead`。Aurora 结果只支持大尺度环境趋势，不能替代局地 VIL/CasCast。

组合题、环境题、局地观测/短临题现在分别使用 `deterministic_combined_foundation_summary`、`deterministic_foundation_summary` 和 `deterministic_local_case_summary`；成本题使用 `deterministic_model_addition_decision`。trace 会记录这些 route、termination reason 和 evidence bundle。

## 回归边界

- 上传事件与问题事件不一致时，系统停止复用旧 case，并返回冲突诊断。
- `R19050812577422` 等没有对应 ERA5/Aurora evidence 的事件只返回缺失诊断，不会自动套用 `R19080121067677`。
- 当前只有一个完整 ERA5–Aurora case，因此多事件泛化尚未完成；需要为 3–5 个事件准备对应 ERA5 输入、Aurora rollout 和后续 ERA5 targets。
- Aurora 预训练数据的严格时间重叠审计仍依赖官方训练数据时间范围；在获得元数据前，泛化结论必须标记为 `training_overlap_unknown`。

## 下一批可执行工作

1. 为 3–5 个 SEVIR 事件按同一 manifest 按需请求 ERA5，并生成独立 case artifacts。
2. 对每个事件重复 `+6/+12/+18/+24 h` 验证，汇总跨事件 MAE/RMSE/bias。
3. 在前端 trace 中展示确定性路由、工具调用、失败阶段、缺失证据和追加模型决策。
