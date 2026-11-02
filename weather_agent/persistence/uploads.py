"""Shared upload storage location, process-local caches and write lock."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ROOT


UPLOADED_VIL: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
UPLOADED_VIL_SEQUENCES: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
_DEFAULT_UPLOAD_STORAGE_DIR = ROOT / "data" / "uploads"
_FALLBACK_UPLOAD_STORAGE_DIR = Path(tempfile.gettempdir()) / "weather_agent_uploads"
try:
    _DEFAULT_UPLOAD_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    _upload_probe = _DEFAULT_UPLOAD_STORAGE_DIR / ".write_probe"
    _upload_probe.write_bytes(b"ok")
    _upload_probe.unlink(missing_ok=True)
    UPLOAD_STORAGE_DIR = _DEFAULT_UPLOAD_STORAGE_DIR
except (OSError, PermissionError):
    _FALLBACK_UPLOAD_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_STORAGE_DIR = _FALLBACK_UPLOAD_STORAGE_DIR
UPLOAD_STORAGE_LOCK = threading.Lock()
