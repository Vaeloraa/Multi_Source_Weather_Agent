# Data and Model Setup

Set the LLM API credentials and `CASCAST_ROOT`, `CASCAST_PYTHON`, `AURORA_PYTHON` and `SEVIR_RAG_PYTHON` in `.env`. Model environments require CUDA-enabled PyTorch. CasCast also uses einops and timm; RAG uses Transformers 4.38.2 and safetensors.

## 1. Observations and reanalysis

| Data | Source | Location |
| --- | --- | --- |
| SEVIR catalog | Download [CATALOG.csv](https://sevir.s3.us-west-2.amazonaws.com/CATALOG.csv) | `data/sevir_complete/CATALOG.csv` |
| SEVIR observations | Download the required VIL, VIS, IR069, IR107 and LGHT HDF5 files from [SEVIR Open Data](https://registry.opendata.aws/sevir/), using catalog `file_name` entries | `data/sevir_complete/data/`; preserve relative paths |
| ERA5 | Register with [CDS](https://cds.climate.copernicus.eu/how-to-api), accept the dataset terms and configure a token; the project downloads data on demand | `data/foundation_live_cache/` |

CasCast requires **13 VIL frames at 5-minute intervals, each 384 × 384 pixels**. Retrospective verification uses the following 12 observations. Upload PNGs in time order with their original grayscale encoding.

Save the CDS token in the downloading user's `.cdsapirc`: `%USERPROFILE%/.cdsapirc` on Windows or `~/.cdsapirc` on Linux.

```yaml
url: https://cds.climate.copernicus.eu/api
key: YOUR_PERSONAL_ACCESS_TOKEN
```

## 2. CasCast

Get the [official repository](https://github.com/OpenEarthLab/CasCast), retain `networks/` and `src/`, and set `CASCAST_ROOT` to its path. Place these checkpoints relative to that repository:

| Download | Target file |
| --- | --- |
| [EarthFormer](https://drive.google.com/file/d/1ZrUgJMbx4hRE4uA0Y43m4Q1OPwA_R_Ow/view) | `experiments/EarthFormer/world_size1-ckpt/earthformer_ckpt.pth` |
| [CasFormer](https://drive.google.com/file/d/1CoUsHmBcTiv1vqj8I88c16RIlNOCSuZi/view) | `experiments/cascast_diffusion/world_size1-ckpt/checkpoint_best.pth` |
| [Autoencoder](https://drive.google.com/file/d/18HaCuLpiX15Os5XCOqoenj2r__2cRHm7/view) | `ckpts/autoencoder/ckpt.pth` |

## 3. Aurora

Run these commands in the environment selected by `AURORA_PYTHON`:

```powershell
python -m pip install "microsoft-aurora==2.0.1"
python -c "from aurora import AuroraPretrained; m = AuroraPretrained(); m.load_checkpoint()"
```

The project uses [AuroraPretrained](https://github.com/microsoft/aurora/blob/v2.0.1/docs/models.md). Its default checkpoint, `aurora-0.25-pretrained.ckpt`, downloads to the current user's Hugging Face cache.

ERA5 inputs cover the initialization time and six hours earlier: surface variables `2t/10u/10v/msl`, atmospheric variables `t/u/v/q/z`, and static variables `z/slt/lsm`, stored as `surface.nc`, `pressure.nc` and `static.nc`. Pressure levels are 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925 and 1000 hPa.

## 4. RAG

Prepare [BGE-M3](https://huggingface.co/BAAI/bge-m3), [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) and the documents listed in `sevir_world/tools/knowledge/sources.json`. The downloader pins model revisions.

From the project root, run:

```powershell
python -m sevir_world.tools.knowledge.prepare_sources --download
python -m sevir_world.tools.knowledge.download_models
# Windows/WSL only: configure SEVIR_RAG_PYTHON and SEVIR_RAG_WSL_DISTRO first
python -m sevir_world.tools.knowledge.prepare_runtime
python -m sevir_world.tools.knowledge.cli ingest
python -m sevir_world.tools.knowledge.cli describe
```

Documents, models and the index are stored under `sevir_world/tools/knowledge/` as `originals/`, `models/` and `index.sqlite3`. The WSL preparation command creates an inference copy and writes its path to `SEVIR_RAG_INFERENCE_MODEL_DIR` in `.env`.

## 5. Administrative boundaries

Download the following national datasets from [Census 2025 KML](https://www2.census.gov/geo/tiger/GENZ2025/kml/). Extract the `.kmz` files into `data/boundaries/`:

| Download | Target file |
| --- | --- |
| [States](https://www2.census.gov/geo/tiger/GENZ2025/kml/cb_2025_us_state_20m.zip) | `cb_2025_us_state_20m.kmz` |
| [Counties](https://www2.census.gov/geo/tiger/GENZ2025/kml/cb_2025_us_county_20m.zip) | `cb_2025_us_county_20m.kmz` |

The current reader supports US states and counties. Additional countries or administrative levels require corresponding data and reader support.
