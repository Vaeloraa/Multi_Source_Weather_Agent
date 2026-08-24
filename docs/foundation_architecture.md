# 当前统一 Agent 与 Foundation 数据架构

系统只有一条 Unified Agent 主链。它既可以直接回答能力边界问题，也可以在同一循环中多次执行 Python、观察真实结果并继续分析；项目不再保留 Local、Direct、Reflective 三套路由，也没有 TaskSpec 或问题级 ExecutionPlan 编译器。

## 请求与执行流程

```text
问题 + 可选事件/上传/时间地点
  → bind_request_assets：只绑定明确身份和可用工具名
  → Unified Agent：选择 <execute> 或 <solution>
  → 持久 Worker：同一请求共享完整 Python 命名空间
  → 按需发现数据结构、工具方法和签名
  → 工具 Observation / artifact / error
  → Agent 根据未完成目标继续执行或结束
```

HTTP 入口位于 `sevir_world/api/server.py`，编排位于 `sevir_world/api/unified_service.py`，唯一 Agent 循环位于 `sevir_world/agent/unified.py`。入口不根据问题关键词预写气象分析步骤；`selected_tools` 只限定可注入的能力对象，具体方法、指标和绘图由 Agent 在运行时按需发现。

## 执行环境

`ExecutionClient` 为一次 Agent 请求维持一个隔离 Worker。多次 `execute` 共享数组、导入、函数、变量和工具对象；取消、30 秒总截止时间、内存超限或请求绑定改变都会关闭会话。Worker 只导入被选中的工具实现，简单问题不会加载 Matplotlib、Torch 或未使用的预报工具。

## 数据与证据

`data_descs` 只描述已绑定的数据身份。Worker 内的 `ds` 在需要执行时加载真实记录；模型可以用 `len(ds)`、`ds[i].keys()`、`dir(tool)` 和 `inspect.signature()` 逐步查看所需内容，而不是在首轮提示中接收全部数据和方法说明。

科学缓存使用 `layer + source_role + request_key` 标识：

- `raw / era5_environment_input`：ERA5 原始输入；
- `standardized / era5_environment_analysis`：标准化环境分析；
- `standardized / aurora_environment_forecast`：Aurora 区域环境预测；
- `verification / model_forecast_verification`：模型与后续观测的验证；
- `presentation / presentation_artifact`：图表等展示资产。

同一层的 ERA5 与 Aurora 条目不会互相覆盖。自然语言回答只保存在会话历史中，用作对话上下文，不允许写入科学证据 manifest。

## 模型与失败边界

服务固定使用 `deepseek-v4-flash`。模型/API 失败立即返回真实错误，不切换其他模型。工具业务失败、Python 异常、取消和超时在 trace 中保持不同状态；已有有效 Observation 时，后续模型失败会返回明确的 partial 结果而不会抹掉证据。

## Study Case 边界

Aurora 用于 6—24 小时区域环境发现，CasCast 用于对应位置未来 1—2 小时局地演变；系统不比较两者数值是否“相等”。当前 R190 缓存案例正在收口为“区域发现→局地观测匹配→短临追踪→双轨验证”。NEXRAD、GOES 自动下载、投影对齐和 CasCast 输入转换按本轮要求暂缓。
