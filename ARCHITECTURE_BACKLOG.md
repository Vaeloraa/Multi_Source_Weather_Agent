# 架构重构 Backlog

本文件只记录不阻塞当前阶段验收的问题。将问题写入这里不代表已经处理，也不能用 Backlog 数量变化证明阶段完成。

## 使用规则

- 新问题必须写明发现日期、影响、建议阶段和是否阻塞当前阶段。
- 如果问题阻塞当前阶段，应移入 `architecture_refactor_state.json` 对应验收项的失败证据，而不是继续留在这里。
- 活动阶段之外的问题只记录，不实现。
- 每个阶段通过验收门后，再审查哪些 Backlog 项进入下一阶段。

## 当前记录

| ID | 问题 | 建议阶段 | 当前状态/是否阻塞当前基线 |
|---|---|---|---|
| BL-001 | `backends/forecast.py` 和 `backends/statistics.py` 体积过大、职责重复 | Phase 2 | 否 |
| BL-002 | `cache_layers.py` 的统一读取器未接入实际 Foundation 读取流程 | Phase 2 | 已解决并验收 |
| BL-003 | ERA5 与 Aurora 可能覆盖同一个 standardized manifest | Phase 2 | 已解决并验收 |
| BL-004 | 前端 `app.js` 超过 2600 行并混合聊天、旧卡片和渲染任务 | Phase 2 | 否 |
| BL-005 | `model_router.py` 保留大量固定单模型策略不再使用的逻辑 | Phase 2 | 否 |
| BL-006 | `ques.md` 存在多份重复和过期问题集 | Phase 2 | 否 |
| BL-007 | `examples_backup_20260812`、旧 session、execution 和 artifact 需要保留策略 | Phase 2 | 否 |
| BL-008 | `fetch_era5_live` 跨月窗口可能形成不正确的日期笛卡尔积 | 后续数据审查 | 否 |
| BL-009 | 普通外部未来两天天气查询尚无正式数据工具 | 后续能力 | 否 |
| BL-010 | Worker 对外部不可信用户不构成严格安全沙箱 | 部署安全阶段 | 否 |
