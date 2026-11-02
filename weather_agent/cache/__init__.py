"""Reusable artifact identities, registry and atomic metadata writes."""

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish complete metadata only after its contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
