# Project Architecture

The package follows the same responsibility boundaries as Zephyrus while keeping
the stricter SEVIR subprocess sandbox.

```text
sevir-agent-demo/
|-- server.py                  Compatibility launcher
|-- sevir_world/
|   |-- agent/                 Direct/Reflective orchestration and prompt contracts
|   |-- code_execution/        AST policy, subprocess client, isolated worker
|   |-- tools/                 Public run() arguments exposed to generated code
|   |-- data_access/           CATALOG, HDF5, ensemble and upload persistence
|   |-- backends/              Forecast, verification and geospatial algorithms
|   `-- api/                   HTTP server, status and legacy comparison gateway
|-- data/
|   |-- source_packages/       Inputs used to rebuild ensemble caches
|   |-- model_inputs/          EarthFormer/CasCast model inputs
|   |-- examples/uploads/      Manual upload test fixtures
|   |-- boundaries/            Offline Census administrative boundaries
|   `-- cascast_ensemble_cache Runtime forecast packages
|-- web/                       Browser application
|-- scripts/                   Reproducible data, evaluation and QA commands
|-- tests/                     Unit and subprocess integration tests
`-- docs/                      Architecture and project notes
```

## Zephyrus Mapping

| Zephyrus responsibility | This project |
| --- | --- |
| `src/models` | `sevir_world/agent` |
| `src/code_execution` | `sevir_world/code_execution` |
| `src/tools` | `sevir_world/tools` |
| `DataIndexer` and data utilities | `agent.runtime.DataIndexer` + `data_access` |
| `ToolDocs` and docstring extraction | `agent.tool_docs` |
| execution server | `api.server` + isolated `code_execution.worker` |

Generated code receives `ds` and selected tool objects as independent `run()`
arguments. It never receives repositories, backends, paths, or the HTTP server.
