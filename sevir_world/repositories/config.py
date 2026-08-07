from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np  # noqa: E402


SAMPLE_DIR = ROOT / "data" / "sevir_sample"
SAMPLE_NPZ = SAMPLE_DIR / "sevir_vil_event.npz"
SAMPLE_METADATA = SAMPLE_DIR / "metadata.json"
CASCAST_ROOT = Path(os.getenv("CASCAST_ROOT", str(ROOT.parent / "CasCast")))
EARTHFORMER_LATENT_ROOT = Path(
    os.getenv("EARTHFORMER_LATENT_ROOT", str(ROOT / "earthformer_example"))
)
CASCAST_EVENT_ID = "R19100105107994"
CASCAST_LEGACY_EVENT_ID = "R19082500047459"
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
UPLOADED_VIL: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
UPLOADED_VIL_SEQUENCES: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
UPLOAD_STORAGE_DIR = ROOT / "data" / "uploads"
UPLOAD_STORAGE_LOCK = threading.Lock()
CATALOG_CSV = ROOT / "data" / "sevir_complete" / "CATALOG.csv"
LOCAL_SEVIR_DATA_ROOT = ROOT / "data" / "sevir_complete" / "data"
CASCAST_EVALUATION_REPORT = ROOT / "data" / "cascast_evaluation_report.json"
SEVIR_EVALUATION_THRESHOLDS = (16.0, 74.0, 133.0, 160.0, 181.0, 219.0)
