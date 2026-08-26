from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor"
# The bundled vendor wheel is a portability fallback.  Prefer the active
# environment's compiled packages (notably h5py) unless explicitly forced;
# the vendored h5py binary can be incompatible with the current Python.
if VENDOR.exists() and os.getenv("SEVIR_USE_VENDOR", "0").lower() in {"1", "true", "yes"}:
    sys.path.insert(0, str(VENDOR))

import numpy as np  # noqa: E402

from sevir_world.domain.thresholds import CASCAST_VIL_THRESHOLDS  # noqa: E402


SAMPLE_DIR = ROOT / "data" / "sevir_sample"
SAMPLE_NPZ = SAMPLE_DIR / "sevir_vil_event.npz"
SAMPLE_METADATA = SAMPLE_DIR / "metadata.json"
CASCAST_ROOT = Path(os.getenv("CASCAST_ROOT", str(ROOT.parent / "CasCast")))
EARTHFORMER_LATENT_ROOT = Path(
    os.getenv(
        "EARTHFORMER_LATENT_ROOT",
        str(ROOT / "data" / "model_inputs" / "earthformer"),
    )
)
CASCAST_SAMPLE_NAME = "vil-2019-SEVIR_VIL_RANDOMEVENTS_2019_0501_0831.h5-0-0.npy"
CASCAST_OUTPUT_NPZ = (
    CASCAST_ROOT
    / "experiments"
    / "cascast_diffusion"
    / "world_size1-ckpt"
    / "local_ens1_outputs"
    / CASCAST_SAMPLE_NAME.replace(".npy", ".npz")
)
CASCAST_METADATA = ROOT / "data" / "cascast_sample" / "metadata.json"
CASCAST_ENSEMBLE_CACHE = ROOT / "data" / "cascast_ensemble_cache"
CASCAST_ENSEMBLE_MANIFEST = CASCAST_ENSEMBLE_CACHE / "manifest.json"
CASCAST_LIVE_CACHE = ROOT / "data" / "live_cascast_cache"


def _default_cascast_python() -> Path:
    """Prefer a project-local real interpreter over uv's venv trampoline."""
    runtime_root = CASCAST_ROOT / ".runtime" / "python"
    local_runtimes = sorted(
        runtime_root.glob("cpython-3.10*-windows-x86_64-none/python.exe"),
        reverse=True,
    )
    if local_runtimes:
        return local_runtimes[0]
    return CASCAST_ROOT / ".venv" / "Scripts" / "python.exe"


CASCAST_PYTHON = Path(
    os.getenv("CASCAST_PYTHON", str(_default_cascast_python()))
)
CASCAST_VENV_SITE_PACKAGES = CASCAST_ROOT / ".venv" / "Lib" / "site-packages"
CASCAST_LIVE_WORKER = ROOT / "scripts" / "run_local_cascast.py"
EARTHFORMER_CHECKPOINT = (
    CASCAST_ROOT / "experiments" / "EarthFormer" / "world_size1-ckpt" / "earthformer_ckpt.pth"
)
CASFORMER_CHECKPOINT = (
    CASCAST_ROOT / "experiments" / "cascast_diffusion" / "world_size1-ckpt" / "checkpoint_best.pth"
)
AUTOENCODER_CHECKPOINT = CASCAST_ROOT / "ckpts" / "autoencoder" / "ckpt.pth"
UPLOADED_VIL: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
UPLOADED_VIL_SEQUENCES: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
_DEFAULT_UPLOAD_STORAGE_DIR = ROOT / "data" / "uploads"
_FALLBACK_UPLOAD_STORAGE_DIR = Path(tempfile.gettempdir()) / "sevir_world_uploads"
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
CATALOG_CSV = ROOT / "data" / "sevir_complete" / "CATALOG.csv"
LOCAL_SEVIR_DATA_ROOT = ROOT / "data" / "sevir_complete" / "data"
SEVIR_EVENT_CACHE = ROOT / "data" / "sevir_event_cache"
SEVIR_REMOTE_BASE_URL = os.getenv(
    "SEVIR_REMOTE_BASE_URL",
    "https://sevir.s3.us-west-2.amazonaws.com/data",
).rstrip("/")
SEVIR_REMOTE_ENABLED = os.getenv("SEVIR_REMOTE_ENABLED", "1").lower() not in {
    "0", "false", "no", "off"
}
SEVIR_REMOTE_BLOCK_SIZE = max(
    1 << 20,
    int(os.getenv("SEVIR_REMOTE_BLOCK_SIZE", str(8 << 20))),
)
CASCAST_EVALUATION_REPORT = ROOT / "data" / "cascast_evaluation_report.json"
SEVIR_EVALUATION_THRESHOLDS = CASCAST_VIL_THRESHOLDS
