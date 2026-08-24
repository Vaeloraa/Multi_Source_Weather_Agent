# SEVIR Unified Agent

本项目使用一个统一的代码 Agent，围绕 SEVIR、CasCast、ERA5 与 Aurora 数据完成气象案例分析。当前主案例为 `R19062708588472`。

> 当前架构仍在重构中，没有任何重构阶段已经验收。持久状态、完成定义和证据要求以 `ARCHITECTURE_REFACTOR_PLAN.md` 与 `architecture_refactor_state.json` 为准。通用测试通过或 R190 缓存齐全均不能单独证明阶段完成。

## 当前架构

```text
用户自然语言问题
→ 只绑定事件或上传数据的轻量身份
→ Unified Agent 先获得四个能力对象的准确方法签名与说明
→ 生成普通 Python
→ exec(code, namespace) 隔离执行
→ 观察真实返回值或完整错误
→ 继续分析或直接回答
```

系统运行入口已经收敛为一条 Unified Agent 主链，不再运行 Local、Direct、Reflective 三套路由。Agent 在同一请求内复用持久 Worker 和完整 Python 命名空间，四个能力对象按首次访问惰性加载；问题级预写分析与旧 render-jobs 链已删除，整条请求共享 30 秒硬截止时间。Aurora 区域发现—CasCast 局地追踪是 Demo 的案例设计，不是写死在工具层里的固定工作流。

生成代码可直接使用注入变量并把结果赋给 `result`：

```python
event = sevir.event()  # 未显式传参时使用本请求绑定的事件
result = {"event_id": event.data["record"]["id"]}
```

兼容旧代码的 `run(...)` 仍可执行，但不是必需协议。

## R19062708588472 Study Case

案例按“区域发现—局地追踪”组织，而不是比较 Aurora 与 CasCast 数值是否相等：

```text
ERA5 当前环境
→ Aurora 判断未来 6—24 小时的大尺度区域与时段
→ 定位 R19062708588472 对应区域
→ 读取此前一小时连续 SEVIR VIL 观测
→ CasCast 分析未来 1—2 小时局地演变
→ 用后续 SEVIR 观测验证局地结果
→ 分别验证 Aurora 环境信号和 CasCast 局地演变
→ Agent 综合解释两类证据如何衔接
```

两类模型的证据角色保持独立：

- Aurora：较长时效、较大区域的环境风险发现；
- CasCast：目标区域未来 1—2 小时的局地短临演变；
- 后续观测：只在用户明确要求验证时读取，不会泄漏给普通预测问题。

当前已经准备的是模型输入、模型预测数组及必要的环境数据缓存，不是缓存好的自然语言答案。Agent 每次仍需按问题读取所需数据、现场计算并生成回答。

NEXRAD、GOES 自动下载、投影对齐和 CasCast 输入转换按当前约定暂缓。

## Agent 的四个能力对象

- `sevir`：事件、目录、连续局地观测、历史候选和上传匹配；
- `geolocator`：事件像素、经纬度、地名和行政区之间的空间转换；
- `cascast`：返回事件或上传序列的原始局地集合预测；
- `aurora`：返回 ERA5/Aurora 案例资料或指定时间区域的环境预测场。

`ds` 是请求绑定的数据句柄，不计入工具。统计、集合分歧、验证指标和绘图由 Agent 使用 NumPy、matplotlib 与普通 Python 按问题完成，不再由多个边界重叠的业务工具替 Agent 决策。

代码生成前，系统从这四个真实类的源码提取完整公共方法签名、参数默认值、方法说明、绑定事件规则和稳定返回约定。Agent 不必先花一轮调用 `dir()` 或 `inspect.signature()` 猜接口；能力对象本身仍在首次被代码访问时才加载。旧工具类、算法和缓存没有删除，迁移关系见 `docs/tool-capability-migration.md`。

## 稳定性设计

- 同一 Agent 请求复用一个 `ExecutionClient`；
- 工具调用成功但后处理失败时，下一轮可以直接复用成功的工具结果；
- 成功执行的 JSON-safe `result` 会作为下一轮命名空间中的 `result`；
- CasCast、Aurora 等大数组继续使用各自的模型/NPZ 缓存，不复制进 LLM 上下文；
- PNG 文件由运行时自动登记为 `/api/artifacts/<id>.png`；
- Agent 固定使用 `deepseek-v4-flash`；一次调用失败后立即返回错误，不探测或切换其他模型；
- Agent 代码错误不会错误地触发模型供应商切换，而是把完整执行错误反馈给同一个 Agent。

## 日志

网页只显示人可读步骤，例如：

```text
已绑定案例数据
模型生成分析代码
调用 cascast.forecast（命中模型数据缓存）
代码执行成功并生成 2 张图
Agent 完成回答
```

原始 JSON、schema 和 traceback 保留在技术记录中，不直接展示给普通用户。

## 启动

在项目根目录运行：

```powershell
python server.py
```

默认地址为 `http://127.0.0.1:8000`。模型 API 配置从项目 `.env` 和环境变量加载。

## 测试

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

测试覆盖统一 Agent 主链、普通 Python 执行、跨轮工具缓存、固定单模型失败退出、R190 数据绑定、科学数据契约、上传可选、日志显示和图像协议。测试超过 30 秒时应立即停止并定位底层原因，不能用增加重试或提示词掩盖问题。

## 重构治理与阶段验收

查看当前真实状态和自动反证：

```powershell
python scripts/refactor_gate.py
```

要求某一阶段已经完成时，必须使用严格验收门：

```powershell
python scripts/refactor_gate.py --require-phase phase_1
```

只要状态不是 `accepted`、必需证据不完整或自动反证仍失败，该命令就会返回失败。验收脚本只读，不会自动把任何项目改成完成状态。非阻塞问题记录在 `ARCHITECTURE_BACKLOG.md`。

`scripts/project_health_check.py` 只是语法、资产和已知残留的冒烟检查，不再作为“架构完成”的证明。
