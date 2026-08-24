from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sevir_world.data_access.uploads import build_hdf5_history_index  # noqa: E402


def main() -> None:
    result = build_hdf5_history_index(force="--force" in sys.argv)
    print(json.dumps({
        "status": "ok",
        "indexed_rows": result.get("indexed_rows", 0),
        "source_count": len(result.get("sources", [])),
        "method": result.get("method"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
