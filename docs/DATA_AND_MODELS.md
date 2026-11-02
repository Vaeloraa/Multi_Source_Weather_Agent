# 数据与模型准备


在 `.env` 填写 LLM 的 API 信息，以及 `CASCAST_ROOT`、`CASCAST_PYTHON`、`AURORA_PYTHON`、`SEVIR_RAG_PYTHON`。模型环境准备 CUDA 版 PyTorch；CasCast 使用 einops、timm，RAG 使用 Transformers 4.38.2、safetensors。

## 1. 观测与环境数据

| 数据 | 获取方式 | 保存位置 |
| --- | --- | --- |
| SEVIR 目录 | 下载 [CATALOG.csv](https://sevir.s3.us-west-2.amazonaws.com/CATALOG.csv) | `data/sevir_complete/CATALOG.csv` |
| SEVIR 观测 | 从 [官方开放数据](https://registry.opendata.aws/sevir/) 获取目录中 `file_name` 对应的 VIL、VIS、IR069、IR107、LGHT HDF5，按任务选择 | `data/sevir_complete/data/`，保留原相对路径 |
| ERA5 | 注册 [CDS](https://cds.climate.copernicus.eu/how-to-api)、接受数据集许可，配置令牌后由项目按需下载 | `data/foundation_live_cache/` |

CasCast 输入为 **13 帧、5 分钟间隔、384×384 的 VIL**；事后验证另取后续 12 帧观测。上传 PNG 时保留原始灰度编码并按时间排序。

CDS 令牌写入下载用户的 `.cdsapirc`（Windows：`%USERPROFILE%/.cdsapirc`；Linux：`~/.cdsapirc`）：

```yaml
url: https://cds.climate.copernicus.eu/api
key: 自己的个人访问令牌
```

## 2. CasCast

获取 [官方工程](https://github.com/OpenEarthLab/CasCast)，保留 `networks/`、`src/`，将工程路径填入 `CASCAST_ROOT`。以下路径均相对于该工程：

| 权重下载 | 目标文件 |
| --- | --- |
| [EarthFormer](https://drive.google.com/file/d/1ZrUgJMbx4hRE4uA0Y43m4Q1OPwA_R_Ow/view) | `experiments/EarthFormer/world_size1-ckpt/earthformer_ckpt.pth` |
| [CasFormer](https://drive.google.com/file/d/1CoUsHmBcTiv1vqj8I88c16RIlNOCSuZi/view) | `experiments/cascast_diffusion/world_size1-ckpt/checkpoint_best.pth` |
| [自编码器](https://drive.google.com/file/d/18HaCuLpiX15Os5XCOqoenj2r__2cRHm7/view) | `ckpts/autoencoder/ckpt.pth` |

## 3. Aurora

在 `AURORA_PYTHON` 对应环境执行：

```powershell
python -m pip install "microsoft-aurora==2.0.1"
python -c "from aurora import AuroraPretrained; m = AuroraPretrained(); m.load_checkpoint()"
```

使用 [官方 AuroraPretrained](https://github.com/microsoft/aurora/blob/v2.0.1/docs/models.md)，默认下载 `aurora-0.25-pretrained.ckpt` 到该用户的 Hugging Face 缓存。

项目从 ERA5 准备起报时刻及前 6 小时的输入：地面变量 `2t/10u/10v/msl`，高空变量 `t/u/v/q/z`，静态变量 `z/slt/lsm`，对应 `surface.nc`、`pressure.nc`、`static.nc`。气压层为 50、100、150、200、250、300、400、500、600、700、850、925、1000 hPa。

## 4. RAG

准备 [BGE-M3](https://huggingface.co/BAAI/bge-m3)、[bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) 和 `sevir_world/tools/knowledge/sources.json` 中的文献。下载程序已固定模型版本。

在项目根目录依次执行：

```powershell
python -m sevir_world.tools.knowledge.prepare_sources --download
python -m sevir_world.tools.knowledge.download_models
# Windows/WSL 推理方式执行此步；先配置 SEVIR_RAG_PYTHON 和 SEVIR_RAG_WSL_DISTRO
python -m sevir_world.tools.knowledge.prepare_runtime
python -m sevir_world.tools.knowledge.cli ingest
python -m sevir_world.tools.knowledge.cli describe
```

文献、模型和索引统一保存在 `sevir_world/tools/knowledge/` 下，分别为 `originals/`、`models/` 和 `index.sqlite3`。WSL 运行副本由准备命令生成，并写入 `.env` 的 `SEVIR_RAG_INFERENCE_MODEL_DIR`。

## 5. 行政区边界

从 [Census 2025 KML 目录](https://www2.census.gov/geo/tiger/GENZ2025/kml/) 下载以下两份全国数据，解压并将同名 `.kmz` 放入 `data/boundaries/`：

| 下载 | 目标文件 |
| --- | --- |
| [州级边界](https://www2.census.gov/geo/tiger/GENZ2025/kml/cb_2025_us_state_20m.zip) | `cb_2025_us_state_20m.kmz` |
| [县级边界](https://www2.census.gov/geo/tiger/GENZ2025/kml/cb_2025_us_county_20m.zip) | `cb_2025_us_county_20m.kmz` |

当前读取器支持美国州、县两级。其他国家和行政层级需另接数据与读取接口。
