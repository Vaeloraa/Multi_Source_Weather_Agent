from __future__ import annotations

"""Build the versioned local SEVIR signature index and register it."""

import argparse
import json
from pathlib import Path

from sevir_world.tools.retrieval.index import (
    DEFAULT_EVENT_CACHE,
    DEFAULT_INDEX_PATH,
    build_local_sevir_index,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-cache", type=Path, default=DEFAULT_EVENT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    index, status = build_local_sevir_index(
        event_cache=arguments.event_cache,
        output_path=arguments.output,
        force=arguments.force,
    )
    print(json.dumps({
        "status": status["status"],
        "index_id": index.index_id,
        "schema_version": index.schema_version,
        "signature_count": len(index.metadata),
        "feature_count": len(index.feature_names),
        "artifact": status["artifact"],
        "registry_entry_key": status["registry_entry_key"],
        "build_elapsed_ms": status.get("build_elapsed_ms"),
        "build_report": status["build_report"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
