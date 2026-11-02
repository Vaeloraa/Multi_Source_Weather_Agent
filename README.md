# Multi-Source Weather Agent

Ask weather questions in natural language. The agent retrieves data, runs forecast models, executes Python analysis and returns charts with supporting sources.

Integrates SEVIR observations, ERA5 reanalysis, Aurora regional forecasts, CasCast nowcasts, geolocation and document retrieval. Conversations and analysis records support follow-up questions.

The code execution and feedback loop draws on [Zephyrus](https://arxiv.org/abs/2510.04017).

## Installation

```powershell
python -m pip install -r requirements.txt
Copy-Item -LiteralPath .env.example -Destination .env
```

Server dependencies are listed in `requirements.txt`. CasCast, Aurora and RAG use separate model environments; see [Data and models](docs/DATA_AND_MODELS.md).

## Data and models

| Component | Requirements | Location or setting |
| --- | --- | --- |
| [SEVIR](https://registry.opendata.aws/sevir/) | `CATALOG.csv` and observation HDF5 files; CasCast requires 13 VIL frames | `data/sevir_complete/` |
| [ERA5](https://cds.climate.copernicus.eu/how-to-api) | CDS account and token; surface, pressure-level and static fields | `~/.cdsapirc`; downloaded on demand |
| [CasCast](https://github.com/OpenEarthLab/CasCast) | Official repository and EarthFormer, CasFormer and autoencoder checkpoints | `CASCAST_ROOT` |
| [Aurora](https://github.com/microsoft/aurora/blob/v2.0.1/docs/models.md) | `microsoft-aurora==2.0.1` and `AuroraPretrained` weights | Model environment's Hugging Face cache |
| RAG | BGE-M3, bge-reranker-v2-m3, source documents and a local index | `sevir_world/tools/knowledge/` |
| [Administrative boundaries](https://www2.census.gov/geo/tiger/GENZ2025/kml/) | 2025 US state and county boundaries, extracted as `.kmz` | `data/boundaries/` |

Download links, checkpoint names, paths and setup commands are in [Data and models](docs/DATA_AND_MODELS.md).

## Configuration and startup

Edit `.env` with the model API endpoint, key and model name, then set the inference paths:

- `MODEL_FIXED_PROVIDER`, `MODEL_FIXED_NAME`: language model provider and model.
- `CASCAST_ROOT`, `CASCAST_PYTHON`: CasCast repository and Python executable.
- `AURORA_PYTHON`: Aurora Python executable.
- `SEVIR_RAG_PYTHON`: RAG Python executable. For WSL, also set the distribution and inference model directory.

```powershell
python server.py
```

Open `http://127.0.0.1:8000`.

## Project structure

```text
sevir_world/       Agent, tools, execution, tasks and persistence
web/               Web interface
docs/              Data and model setup, references and diagrams
server.py          Server entry point
requirements.txt   Python dependencies
.env.example       Configuration template
```

The source manifest is `sevir_world/tools/knowledge/sources.json`. Documents, retrieval models and the index live alongside it. Observations and runtime caches are created under `data/` as needed.

## Documentation

- [Data and models](docs/DATA_AND_MODELS.md): downloads, checkpoints, boundaries and setup commands.
- [Project overview](PROJECT_OVERVIEW.md): architecture, implemented features and next steps.

## References

| Use | Source |
| --- | --- |
| Agent execution approach | [Zephyrus - ICLR 2026](https://arxiv.org/abs/2510.04017) |
| Observation data | [SEVIR - NeurIPS 2020](https://papers.nips.cc/paper_files/paper/2020/hash/fa78a16157fed00d7a80515818432169-Abstract.html) |
| Reanalysis data | [The ERA5 global reanalysis - QJRMS 2020](https://doi.org/10.1002/qj.3803) |
| Local ensemble nowcasting | [CasCast - ICML 2024](https://proceedings.mlr.press/v235/gong24a.html) |
| CasCast deterministic component | [Earthformer - NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a2affd71d15e8fedffe18d0219f4837a-Abstract-Conference.html) |
| Regional forecasting | [Aurora - Nature 2025](https://doi.org/10.1038/s41586-025-09005-y) |
| Document embeddings | [M3-Embedding - Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.137/) |

Reranker: [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3). BibTeX entries are in [references.bib](docs/references.bib).
