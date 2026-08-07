## 二、如何深度理解现有工作

建议按以下顺序阅读，不要一上来逐行看全部代码。

### 第一层：理解系统解决什么问题

先用一句话概括：

> 用户通过自然语言选择 SEVIR 分析任务，Agent 调用真实 CasCast 预测或 Python 统计工具，最后同时返回气象图像、定量指标和风险解释。

然后明确系统中四个角色：

- **Qwen**：理解问题、选择工具、解释结果；
- **CasCast**：生成多个可能的未来 VIL 序列；
- **Python 工具**：进行确定性的统计和 hazard 计算；
- **网页**：展示输入、预测、概率和文字解释。

关键原则是：**Qwen 不负责计算数值，它只能解释 Python 工具计算出的结果。**

### 第二层：理解 CasCast 数据

重点掌握这些维度：

```
history:          13 × 384 × 384
ensemble_members: 10 × 12 × 384 × 384
ensemble_mean:    12 × 384 × 384
```

含义分别是：

- 13 帧过去一小时观测；
- 10 种可能的未来；
- 每种未来包含未来一小时的 12 个五分钟时刻；
- ensemble mean 是中心预测；
- ensemble 差异用于描述不确定性。

你应该能够不看材料回答：

- 为什么需要多个 ensemble？
- 为什么不能只返回均值？
- ensemble 标准差代表什么？
- 为什么 ensemble 概率不一定校准？
- 为什么 +60 分钟通常比 +5 分钟更不确定？

### 第三层：理解后端工具

按这个顺序阅读 [sevir_tools.py](/D:/Profile/专业实习/sevir-agent-demo/sevir_tools.py)：

1. 数据路径、manifest 和事件映射；
2. `_load_cascast_ensemble`；
3. `run_nowcast`；
4. `forecast_hazards`；
5. `analyze_ensemble_uncertainty`；
6. `analyze_uploaded_vil`；
7. `historical_sevir_statistics`；
8. `geolocate_sevir_event`；
9. `compare_event_with_history`；
10. `execute_tool` 和工具定义。

对每个函数都回答四个问题：

- 输入是什么？
- 从哪里读取数据？
- 计算公式是什么？
- 输出字段最终在哪里显示？

### 第四层：理解 Agent 调用

阅读 [server.py](/D:/Profile/专业实习/sevir-agent-demo/server.py) 时重点看：

- 给 Qwen 注册了哪些工具；
- Qwen 如何返回 tool call；
- 参数如何被限制和补全；
- Qwen 调用失败后如何降级；
- 为什么最多连续调用 3 个工具；
- 最终回答如何结合工具结果生成；
- 如何防止 Qwen 调用未注册工具。

### 第五层：理解前端状态

阅读 [app.js](/D:/Profile/专业实习/sevir-agent-demo/web/app.js) 时跟踪一次完整请求：

```
用户点击提交
→ 读取事件/阈值/问题
→ POST /api/run
→ 获取tool_result
→ 更新摘要
→ 绘制空间预览
→ 显示Qwen回答和调用链
```

重点检查事件 ID、阈值、预测时刻和上传 ID 如何保存，避免不同请求之间串数据。

------

## 三、待完成工作如何推进

建议按以下优先级推进。

### 优先级 1：补齐自动测试和实验记录

这是成本最低、收益最高的一步。

需要增加：

- 数据形状测试；
- ensemble mean 复算测试；
- Persistence 测试；
- 阈值概率测试；
- hazard 持续时间测试；
- 错误事件和错误上传测试；
- API 返回结构测试；
- 10 个事件批量冒烟测试。

同时固定一份实验配置：

- checkpoint 版本；
- 随机种子；
- diffusion steps；
- guidance weight；
- 阈值；
- 软件和硬件环境；
- 事件编号；
- 输出文件校验值。

完成标准：换一台机器后，别人可以根据 README 重复得到相同结果。

### 优先级 2：获取 ground truth，建立定量评价

这是从“可展示 Demo”升级为“可信研究结果”的关键。

需要让 CasCast 合作者补充每个样本真实未来 12 帧，然后计算：

- MSE/MAE；
- CSI；
- POD；
- FAR；
- 不同 VIL 阈值下的表现；
- CasCast ensemble mean 与 Persistence 的对比；
- 不同 lead time 的误差曲线；
- ensemble spread 与实际误差的关系。

完成标准：能够回答“CasCast 相比 Persistence 是否更好、在哪些时效和阈值下更好”。

### 优先级 3：实现 2～3 小时滚动预测

先明确模型接口：

```
predict(history_13_frames, ensemble_size) -> future_12_frames
```

滚动过程可以是：

```
过去13帧 → 预测未来1小时
取预测结果更新输入 → 预测第2小时
继续更新输入 → 预测第3小时
```

必须同时处理：

- 每个 ensemble 成员独立滚动；
- 不要在第一小时结束后只用 ensemble mean；
- 记录每一轮输入来源；
- 明确误差会随滚动累积；
- 页面按 1、2、3 小时分组显示；
- 防止一次请求耗时过长或显存溢出。

建议先离线生成 2～3 个样本验证，不要直接对全部事件运行。

### 优先级 4：接入 satellite-to-radar

当前真正的阻塞项不是网页，而是模型材料。需要向立欣学姐取得：

- 模型代码和运行环境；
- checkpoint；
- 所需卫星通道，如 `IR069、IR107、VIS`；
- 输入形状、数值范围和预处理；
- 输出是 VIL、雷达反射率还是其他产品；
- 时间和空间配准规则；
- 一组可运行示例及对应 ground truth；
- 许可和引用信息。

拿到材料后分四步接入：

1. 独立脚本能够运行；
2. 包装成稳定的 `predict()` 接口；
3. 保存一组验证结果；
4. 再注册为 Agent 工具并加入网页。

不要在模型尚未独立跑通时先改 Agent。

### 优先级 5：扩展上传格式

按研究价值排序：

1. `.npy/.npz` 单帧 VIL；
2. 连续 13 帧 VIL 序列；
3. SEVIR HDF5 中的指定事件；
4. 普通 PNG/伪彩图。

连续 13 帧上传完成后，才有条件对用户数据调用 CasCast。普通 PNG 缺少原始数值映射，只适合视觉分析，不能可靠恢复 VIL。

### 优先级 6：改进 hazard 科学定义

需要与老师确认：

- hazard 是像素超过阈值，还是连通区域超过面积阈值；
- 持续多少分钟才算 hazard；
- 事件概率如何由 ensemble 汇总；
- 使用最大概率、区域平均概率还是成员事件比例；
- 是否需要引入地理行政区、地形或历史灾害资料。

建议把规则写成显式配置，而不是散落在代码中，例如：

```
VIL阈值
最小区域面积
最小持续时间
ensemble概率定义
风险等级映射
```

------

## 四、最推荐的推进顺序

```
自动测试
→ 补ground truth
→ CasCast与Persistence定量比较
→ 2～3小时滚动预测
→ 获取并独立跑通satellite-to-radar
→ 接入Agent
→ 扩展上传格式
→ 完善hazard定义和论文实验
```

现阶段不要急着继续增加很多 Agent 工具。当前最需要提升的不是功能数量，而是**可复现性、ground truth 评价、科学定义和长时效预测**。这些补齐后，现有工作才能从“功能较完整的 Demo”进一步变成“可以支撑论文或答辩结论的系统”。