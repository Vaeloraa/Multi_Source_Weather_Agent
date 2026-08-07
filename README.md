# SEVIR Agent Demo

面向 SEVIR 雷达回波临近预报的代码生成 Agent。系统参考 Zephyrus 的工具注入与
代码执行方式，同时保留更严格的隔离执行和完整调用记录。

## 核心流程

```text
自然语言问题
  -> data_descs + selected_tools
  -> 动态提取所选工具的签名和 docstring
  -> Qwen 生成唯一 run(ds, ...)
  -> 独立子进程安全执行
  -> 记录工具调用、结果与错误
  -> 根据真实 observation 回答
```

- **Direct**：生成完整 `run()`，执行失败时修复代码，成功后回答。
- **Reflective**：按 `execute -> observe -> refine -> finish` 循环逐步分析。

生成代码不能导入模块、访问文件、调用网络或系统命令。执行器限制 AST、运行时间、
内存、标准输出和结果大小。

## 启动

在项目根目录配置本机 `.env`，然后运行：

```powershell
python server.py
```

默认地址：<http://127.0.0.1:8000>

`.env` 包含本机 API 密钥，已被 Git 忽略，不应对外发送。

## 目录

```text
sevir-agent-demo/
|-- server.py                 # 启动入口
|-- sevir_world/
|   |-- agent/                # Direct、Reflective、请求契约、动态工具文档
|   |-- code_execution/       # AST 安全策略、子进程客户端与 Worker
|   |-- tools/                # 注入 run() 的公开气象工具
|   |-- data_access/          # CATALOG、HDF5、ensemble 与上传存储
|   |-- backends/             # 预测、评估、统计和地理算法
|   `-- api/                  # HTTP 服务和兼容工具网关
|-- data/                     # 现场演示所需真实数据
|-- web/                      # 浏览器界面
`-- vendor/                   # 本地 Python 依赖
```

## 核心数据

- `data/cascast_ensemble_cache`：10 个事件的 CasCast ensemble、历史输入与 ground truth。
- `data/sevir_complete/CATALOG.csv`：SEVIR VIL 目录，用于检索、统计与投影定位。
- `data/boundaries`：US Census 2025 州/县边界，用于离线行政区反查。
- `data/model_inputs`：EarthFormer/CasCast 输入。
- `data/cascast_evaluation_report.json`：10 事件聚合评估结果。

## 科学边界

- VIL 阈值是强回波代理量，不等于地面降雨量或官方灾害预警。
- VIL 值 `255` 表示缺测，评估时必须排除。
- ensemble 成员支持率不是校准后的概率。
- 历史统计来自 CATALOG metadata，记录数不等于独立灾害数。
- 当前 CasCast 页面读取已保存并验证的预测包，不会为任意上传现场运行完整模型。
