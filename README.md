# 多源气象分析 Agent

通过自然语言调用气象数据和预测模型，执行 Python 计算，生成图表与分析结论。

支持 SEVIR 观测、ERA5 再分析、Aurora 区域预测、CasCast 短临预测、地理定位和文献检索；支持连续追问及分析记录保存。

Agent 的代码执行与反馈循环参考 [Zephyrus](https://arxiv.org/abs/2510.04017)。

## 安装

```powershell
python -m pip install -r requirements.txt
Copy-Item -LiteralPath .env.example -Destination .env
```

服务端依赖统一在 `requirements.txt`。CasCast、Aurora 和 RAG 使用各自的模型解释器，环境配置见[准备文档](docs/DATA_AND_MODELS.md)。

## 数据与模型

| 组件 | 需要准备 | 路径或配置 |
| --- | --- | --- |
| [SEVIR](https://registry.opendata.aws/sevir/) | `CATALOG.csv`、目标范围的观测 HDF5；CasCast 输入为 13 帧 VIL | `data/sevir_complete/` |
| [ERA5](https://cds.climate.copernicus.eu/how-to-api) | CDS 账号与令牌；按任务获取地面、高空及静态场 | `~/.cdsapirc`；数据按请求下载 |
| [CasCast](https://github.com/OpenEarthLab/CasCast) | 官方工程、EarthFormer／CasFormer／自编码器三份权重 | `CASCAST_ROOT` |
| [Aurora](https://github.com/microsoft/aurora/blob/v2.0.1/docs/models.md) | `microsoft-aurora==2.0.1`、`AuroraPretrained` 权重 | Aurora 环境的 Hugging Face 缓存 |
| RAG | BGE-M3、bge-reranker-v2-m3、来源清单中的文献，随后建立索引 | `sevir_world/tools/knowledge/` |
| [行政边界](https://www2.census.gov/geo/tiger/GENZ2025/kml/) | 2025 年美国州、县全国边界，解压后的 `.kmz` | `data/boundaries/` |

下载链接、权重文件名、目录及准备命令见[数据与模型准备](docs/DATA_AND_MODELS.md)。

## 配置与运行

编辑 `.env`，填写模型服务的 API 地址、Key、模型名称，以及推理解释器路径：

- `MODEL_FIXED_PROVIDER`、`MODEL_FIXED_NAME`：LLM 服务与模型。
- `CASCAST_ROOT`、`CASCAST_PYTHON`：CasCast 工程与解释器。
- `AURORA_PYTHON`：Aurora 解释器。
- `SEVIR_RAG_PYTHON`：RAG 解释器；使用 WSL 时同时设置发行版及模型目录。

```powershell
python server.py
```

浏览器打开 `http://127.0.0.1:8000`。

## 项目结构

```text
sevir_world/       Agent、数据与模型工具、执行器、任务与存储
web/               网页界面
docs/              数据与模型准备说明
server.py          启动入口
requirements.txt   Python 依赖
.env.example       配置模板
```

知识来源清单位于 `sevir_world/tools/knowledge/sources.json`，文献、模型和索引在同一模块下准备。观测资料和运行缓存按需生成到 `data/`。

## 文档

- [数据与模型准备](docs/DATA_AND_MODELS.md)：来源、权重、边界、存放位置和准备命令。
- [项目说明](PROJECT_OVERVIEW.md)：系统架构、已完成内容和后续工作。

## 参考与引用

| 用途 | 研究来源 |
| --- | --- |
| Agent 执行思路 | [Zephyrus — ICLR 2026](https://arxiv.org/abs/2510.04017) |
| 观测数据 | [SEVIR — NeurIPS 2020](https://papers.nips.cc/paper_files/paper/2020/hash/fa78a16157fed00d7a80515818432169-Abstract.html) |
| 再分析数据 | [The ERA5 global reanalysis — QJRMS 2020](https://doi.org/10.1002/qj.3803) |
| 局地集合预测 | [CasCast — ICML 2024](https://proceedings.mlr.press/v235/gong24a.html) |
| CasCast 确定性预测子模型 | [Earthformer — NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a2affd71d15e8fedffe18d0219f4837a-Abstract-Conference.html) |
| 区域环境预测 | [Aurora — Nature 2025](https://doi.org/10.1038/s41586-025-09005-y) |
| 文献向量检索 | [M3-Embedding — Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.137/) |

重排模型来源：[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)。论文 BibTeX 见 [references.bib](docs/references.bib)。
