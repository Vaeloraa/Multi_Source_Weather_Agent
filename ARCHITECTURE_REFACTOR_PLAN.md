# SEVIR Agent 架构重构计划契约

本文件是重构工作的唯一人类可读计划源；机器状态位于 `architecture_refactor_state.json`。聊天中的“完成”“通过”或口头总结不能改变这里的状态。

## 当前结论

- 当前保留并验收的范围只有 `phase_1` 和 `phase_2`。
- `phase_3` 与 `phase_4` 已按用户要求全部撤出，不再属于当前代码计划或验收状态。
- R19062708588472、Aurora 与 CasCast 仍可作为未来 Demo 构思和 Agent 评测材料，但不得再次实现为案例专用编排代码。
- 原始观测、模型输出与科学缓存继续保留；任何后续阶段必须重新提出并经用户确认。
## 状态定义

验收项只能按以下含义使用状态：

1. `not_started`：尚未实施。
2. `implemented`：已有代码修改，但尚未完成全部验证。
3. `verified`：专项验证通过，但尚未完成代表场景和 diff 审查。
4. `accepted`：全部必需证据存在、自动反证检查通过、代表场景通过。
5. `blocked`：存在明确阻塞条件，并保留失败证据。

阶段只能使用 `not_started`、`in_progress`、`verified`、`accepted` 或 `blocked`。只有阶段内每个验收项均为 `accepted`，阶段才能标记为 `accepted`。

## 不可绕过的规则

1. 不得用语法测试、通用单元测试、健康检查或文件存在性代替阶段验收。
2. 不得因为某一项已经实现，就宣布整个阶段完成。
3. 状态修改必须通过可见的文件 diff；验收脚本是只读的，不会自动提升状态。
4. 每个 `accepted` 项必须包含机器状态文件中声明的全部证据类型。
5. 每个阶段结束必须运行 `python scripts/refactor_gate.py --require-phase <phase_id>`。
6. 自动验收门失败时，只能报告“未完成”或“阻塞”，不能报告“基本完成”。
7. 发现的新问题如果不阻塞当前验收，写入 `ARCHITECTURE_BACKLOG.md`，不得借此改变本阶段范围。
8. 如果新问题阻塞当前验收，必须在证据中说明它阻塞哪个验收项。
9. 未运行真实 API 时，必须明确写“未联调”；Agent 控制层代表场景允许使用真实缓存、真实 Worker 和确定性模型动作验收，但不得冒充外部模型联调。
10. 删除代码前必须先确认正式入口、测试、脚本和文档引用；当前脏工作区中的 `??` 文件不能仅因未跟踪而删除。

## 当前冻结基线

截至 2026-08-24，下列事实均未解决：

| 编号 | 事实 | 状态 |
|---|---|---|
| B-01 | 多次 execute 仍使用新 Worker，只传递前一结果和小型缓存 | 已解决并验收 |
| B-02 | `uploaded_ensemble_divergence` 仍存在 | 已解决并验收 |
| B-03 | 旧 API、render-jobs、失效脚本和过期文档仍存在 | 已解决并验收 |
| B-04 | 业务错误可能被进程执行成功状态掩盖 | 已解决并验收 |
| B-05 | 30 秒尚未成为整条请求链共享的硬预算 | 已解决并验收 |

这五项是反证基线。只要相应问题仍存在，就不得把对应阶段标记为 `accepted`。

## Phase 1：真实、稳定且有硬预算的 Agent 循环

本阶段只处理执行状态、截止时间、取消、失败保真和 trace。为让浏览器超时能够传播取消，允许修改 `web/app.js` 的请求控制代码；不得顺带重写气象算法、前端样式或 Study Case。

验收项：

- `P1-01`：业务错误进入失败计数，不会清零错误历史。
- `P1-02`：共享 30 秒截止时间覆盖地点解析、LLM、Worker 和整理阶段。
- `P1-03`：客户端取消或断开能终止剩余工作。
- `P1-04`：最终模型失败不会抹掉已经获得的有效 Observation。
- `P1-05`：R190 代表问题留下与真实状态一致的可读 trace。

阶段目标不是“让所有回答都专业完美”，而是保证 Agent 不盲目重试、不伪造成功、不超出硬预算。

### Phase 1 本轮重新执行计划（phase_1_iteration_1）

重新规划原因：2026-08-24 运行严格验收门失败，Phase 1 为 `in_progress`，`accepted 0/5`。P1-01 至 P1-04 自动反证未通过；P1-05 只满足“R190 资产和问题集存在”的前提，不等于代表场景已经验收。

本轮不得修改 Phase 2 的持久 Worker、问题级预写流程、render-jobs 清理、缓存重构，也不得修改 Phase 3 的 Aurora—CasCast Study Case 编排。发现相关问题时只写入 Backlog。

#### 执行顺序和依赖

```text
P1-01 统一业务失败语义
    ↓
P1-02 建立共享 30 秒硬截止时间
    ↓
P1-03 将取消信号贯穿 HTTP → Agent → Worker
    ↓
P1-04 保留最后有效 Observation 并诚实返回 partial/error
    ↓
P1-05 使用 R190 代表问题审查完整 trace
    ↓
严格阶段门：python scripts/refactor_gate.py --require-phase phase_1
```

P1-01 必须先完成，因为 P1-04 需要依赖正确的“有效 Observation”定义。P1-02 必须先于 P1-03，因为取消和超时应共用一种终止信号。P1-05 最后执行，不能用模拟结果代替真实代表场景。

#### P1-01：统一业务失败语义

允许修改：

- `sevir_world/agent/schema.py`
- `sevir_world/agent/unified.py`
- 必要时修改 `code_execution/client.py` 或 `worker.py` 的结果封装
- 对应专项测试

实施目标：

1. 明确区分 Python 进程执行成功和业务分析成功。
2. 接入现有 `domain_result_error()`，或建立等价的唯一规范化入口。
3. `result.status=error/failed`、工具调用 `result_status=error` 和 Worker 异常形成一致的 Agent Observation 状态。
4. 业务失败必须增加错误计数，不能重置 `consecutive_errors`。
5. trace 必须显示具体失败工具和业务错误，不显示“代码执行成功，已获得真实观测”。

专项测试必须包含精确测试名：

```text
test_domain_result_error_counts_as_execution_failure
```

停止条件：专项测试、相关既有统一 Agent 测试和 schema 测试通过后，记录 `source_diff`、`targeted_test`、`execution_trace` 三类证据；否则 P1-01 保持 `implemented` 或 `blocked`。

#### P1-02：共享 30 秒硬截止时间

允许修改：

- `sevir_world/api/server.py`
- `sevir_world/api/unified_service.py`
- `sevir_world/api/model_client.py`
- `sevir_world/code_execution/client.py`
- 对应专项测试

实施目标：

1. 截止时间从 `/api/run` 接收请求时开始，而不是进入 Agent 后才开始。
2. 地点解析、每次模型调用、每次 Worker 执行和最终整理读取同一个 deadline。
3. 模型调用超时使用 `min(配置超时, 剩余总预算)`。
4. 删除或停用默认 300 秒总预算；默认硬预算为 30 秒。
5. 达到 deadline 后不再开始新的模型或 Worker 调用。
6. 测试使用假时钟或受控阻塞，不调用真实 API。

专项测试必须包含精确测试名：

```text
test_shared_deadline_bounds_the_complete_request
```

停止条件：受控测试证明整条请求不超过共享预算，并记录 `source_diff`、`targeted_test`、`timing_result`。

#### P1-03：完整取消链

允许修改：

- `sevir_world/api/server.py`
- `sevir_world/api/unified_service.py`
- `sevir_world/agent/unified.py`
- `sevir_world/code_execution/client.py`
- 对应专项测试

实施目标：

1. 建立请求级 cancellation token，不用新增另一套路由。
2. HTTP 客户端断开、共享 deadline 到达和服务器关闭都能设置同一取消信号。
3. Agent 在每次 LLM/Worker 调用前检查取消信号。
4. ExecutionClient 收到取消后终止 Worker 进程树。
5. 取消不能被 `BaseException` 捕获后继续重试。

专项测试必须包含精确测试名：

```text
test_client_cancellation_stops_agent_and_worker
```

停止条件：测试证明取消后不再增加模型调用数、Worker 已退出，并记录 `source_diff`、`targeted_test`、`cancellation_trace`。

#### P1-04：最后有效 Observation 保真

允许修改：

- `sevir_world/agent/unified.py`
- `sevir_world/api/response_pipeline.py`
- 对应专项测试

实施目标：

1. 明确定义“有效 Observation”：执行与业务状态均成功，并保留必要来源。
2. 最终模型调用失败、超时或返回非法协议时，不删除此前 Observation、tool calls 和 artifacts。
3. 如果无法生成完整回答，返回明确 `partial` 或 `error`，不能伪装成 `ok`。
4. 返回内容只总结已经获得的事实，不生成新的气象结论。

专项测试必须包含精确测试名：

```text
test_last_valid_observation_survives_final_model_failure
```

停止条件：故障注入测试能看到最后有效 Observation、真实终止原因和保留 artifact，并记录 `source_diff`、`targeted_test`、`execution_trace`。

#### P1-05：R190 代表问题验收

固定代表问题：

```text
请问未来天气多路预测结果分歧大不大，请用bar-chart图和热力图可视化显示
```

执行要求：

1. 使用 `R19062708588472` 已准备缓存。
2. 固定 `deepseek-v4-flash`，失败时不得切换模型。
3. 记录请求开始、每次 LLM、每次 Worker、每次工具、缓存命中、artifact 和终止原因。
4. 总耗时超过 30 秒立即停止，并回到 P1-02/P1-03 定位，不通过增加重试解决。
5. 本阶段只审查执行真实性、是否回答用户目标和是否生成两个请求图表；不因气象结论不够专业要求 LLM 重写。
6. 如果真实 API 不可用，记录失败证据并将 P1-05 标为 `blocked`，不得使用 mock 宣称代表场景通过。

停止条件：记录 `representative_scenario`、`trace_review`、`git_diff_review`；五个验收项全部 accepted 后再运行严格阶段门。

#### 本轮统一回归顺序

每完成一个验收项，只运行与它有关的专项测试和受影响模块测试。P1-05 前再运行一次全量离线回归：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/project_health_check.py
python scripts/refactor_gate.py --require-phase phase_1
```

其中冒烟检查允许报告 Phase 2 的已知失败，但 Phase 1 严格验收门必须返回成功，才允许进入 Phase 2。

## Phase 2：Zephyrus 式执行环境和架构收口

必须在 Phase 1 已验收后开始。

本阶段已于 2026-08-24 通过验收：117/117 全量单元测试通过，健康检查 10/10，通过持久命名空间、通用分歧分析、按需导入、接口一致性和缓存协议五项自动反证。外部模型网关联调仍按 BL-011 保留为 Demo 前置，不混入本阶段结论。

验收项：

- `P2-01`：同一请求复用持久执行会话和完整 Python 命名空间。
- `P2-02`：删除问题级预写分析，代表问题由原始数据和通用绘图完成。
- `P2-03`：工具真正按需加载。
- `P2-04`：服务器、前端、测试、脚本和文档描述同一套真实接口。
- `P2-05`：缓存协议统一，最终文字答案不作为科学证据缓存。

## 每轮开始和结束流程

开始前：

```powershell
python scripts/refactor_gate.py
git status --short
```

执行中：

1. 只修改活动阶段允许范围内的文件。
2. 非阻塞问题写入 Backlog。
3. 每个验收项单独添加专项测试。
4. 不提前修改 `accepted` 状态。

结束前：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/refactor_gate.py --require-phase phase_1
git diff --stat
git status --short
```

只有最后一个命令对应的阶段门通过，才能将阶段称为“已验收”。

## 证据记录格式

机器状态中的每条证据必须至少包含：

```json
{
  "type": "targeted_test",
  "reference": "tests.test_x.TestX.test_behavior",
  "result": "pass",
  "recorded_at": "YYYY-MM-DDTHH:MM:SSZ",
  "notes": "验证了什么，而不是只写测试通过"
}
```

失败证据也应保留，`result` 使用 `fail`，验收项不能提升为 `accepted`。

## 阶段证据表模板

| 验收项 | 修改位置 | 专项验证 | 代表场景 | Git diff 审查 | 状态 |
|---|---|---|---|---|---|
| P1-01 | schema/client/unified | 专项测试通过 | 错误trace通过 | 通过 | accepted |
| P1-02 | server/unified_service/response/client | 专项测试通过 | 人工短截止时间通过 | 通过 | accepted |
| P1-03 | web/server/unified/client | 专项测试通过 | socket断开与Worker取消通过 | 通过 | accepted |
| P1-04 | unified | 专项测试通过 | Observation保真通过 | 通过 | accepted |
| P1-05 | R190真实缓存与真实Worker trace通过 | 控制链代表场景通过 | 外部LLM保留为Demo前联调 | 通过 | accepted |

Phase 1 已通过严格门禁。其边界是“执行状态、截止时间、取消、失败保真和 trace”；外部 `qwen` 网关上的 `deepseek-v4-flash` 尚未联调，失败记录保留在机器状态和 Backlog 中，不被写成已完成。

