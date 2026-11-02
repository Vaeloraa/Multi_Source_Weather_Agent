"""Local model assets and execution paths owned by CasCast."""

from __future__ import annotations

import os
from pathlib import Path

from ...config import ROOT


CASCAST_ROOT = Path(os.getenv("CASCAST_ROOT", str(ROOT.parent / "CasCast")))
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


CASCAST_PYTHON = Path(os.getenv("CASCAST_PYTHON", str(_default_cascast_python())))
CASCAST_VENV_SITE_PACKAGES = CASCAST_ROOT / ".venv" / "Lib" / "site-packages"
CASCAST_LIVE_WORKER = Path(__file__).resolve().with_name("worker.py")
EARTHFORMER_CHECKPOINT = (
    CASCAST_ROOT / "experiments" / "EarthFormer" / "world_size1-ckpt" / "earthformer_ckpt.pth"
)
CASFORMER_CHECKPOINT = (
    CASCAST_ROOT / "experiments" / "cascast_diffusion" / "world_size1-ckpt" / "checkpoint_best.pth"
)
AUTOENCODER_CHECKPOINT = CASCAST_ROOT / "ckpts" / "autoencoder" / "ckpt.pth"
