# Agent 工具能力迁移

本次重构只收缩 Agent 顶层入口，不删除旧算法、缓存或产品功能。旧类继续作为内部实现和兼容 API；生成代码只看到 `ds` 与四个窄能力对象。

| 旧 Agent 入口 | 新位置 | 迁移原则 |
|---|---|---|
| `data` | `sevir` | 事件元数据、目录检索、局地观测和上传匹配迁入 `sevir`。上传像素本身继续由 `ds` 提供。 |
| `history` | `sevir` | 历史候选和原始历史观测迁入 `sevir`；比较标准由 Agent 根据问题决定。 |
| `geolocator` | `geolocator` | 保留空间转换职责，去掉与天气判断无关的业务含义。 |
| `forecaster` | `cascast` | 只保留事件/上传序列的原始集合预测。阈值、追踪、分歧和结论由 Agent 计算。 |
| `foundation` | `aurora` | 只保留缓存案例、Aurora预测和区域环境场。它不负责匹配SEVIR或串联Study Case。 |
| `statistics` | 普通 Python / 内部库 | NumPy负责问题相关统计；旧 `SevirStatistics` 与统计后端保留供内部和兼容测试使用。 |
| `uncertainty` | 普通 Python / 内部库 | Agent从CasCast成员原始场计算问题需要的均值、分位数、范围和分歧；旧实现保留。 |
| `metrics` | 普通 Python / 内部库 | CSI、POD、FAR等验证算法保留，但不再作为一个独立顶层工具影响工具选择。 |
| `visualizer` | matplotlib + Worker制品注册 | 旧可视化类保留；新Agent直接保存PNG到 `SEVIR_ARTIFACT_DIR`，运行时注册稳定URL。 |

## 保留保证

- `sevir_world/tools/data.py`、`history.py`、`forecast.py`、`foundation.py`、`statistics.py`、`uncertainty.py`、`metrics.py`、`visualization.py` 不删除。
- 单张上传和13帧上传仍由 `ds` 加载真实像素。
- CasCast、ERA5/Aurora、本地缓存、历史检索、指标算法和旧图表实现继续存在。
- 旧类仍可从 `sevir_world.tools` 导入，便于前端兼容代码和回归测试逐步迁移。
- 只有生成代码的顶层变量发生变化；旧变量不会继续注入Agent命名空间。
- 旧客户端提交的 `selected_tools` 会在唯一 API 入口归并为四个新能力名；`visualizer` 和 `arrays` 归入 Worker 的普通 Python 能力，不再成为顶层工具。
