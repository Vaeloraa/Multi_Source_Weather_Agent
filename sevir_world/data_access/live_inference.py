from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..tools.sevir.Catalog import (
    _catalog_subject_key,
    _catalog_vil_event,
    _load_vil_array,
    _local_hdf5_path,
)
from ..domain.cascast_alignment import build_validation_manifest
from .config import (
    AUTOENCODER_CHECKPOINT,
    CASCAST_LIVE_CACHE,
    CASCAST_LIVE_WORKER,
    CASCAST_PYTHON,
    CASCAST_ROOT,
    CASCAST_VENV_SITE_PACKAGES,
    CASFORMER_CHECKPOINT,
    EARTHFORMER_CHECKPOINT,
    UPLOAD_STORAGE_DIR,
)


MAX_LIVE_ENSEMBLE_MEMBERS = max(
    1, int(os.getenv("SEVIR_MAX_LIVE_ENSEMBLE_MEMBERS", "10"))
)


def _validate_ensemble_members(value: int) -> int:
    count = int(value)
    if not 1 <= count <= MAX_LIVE_ENSEMBLE_MEMBERS:
        raise ValueError(
            f"ensemble_members must be between 1 and {MAX_LIVE_ENSEMBLE_MEMBERS}"
        )
    return count


_INFERENCE_LOCK = threading.Lock()
try:
    _WORKER_TMP = Path(tempfile.mkdtemp(prefix="sevir_cascast_worker_"))
except OSError:
    _WORKER_TMP = Path(tempfile.gettempdir())


def _uploaded_input_path(upload_id: str) -> Path:
    """Resolve the raw uploaded sequence, including the durable demo store."""
    primary = UPLOAD_STORAGE_DIR / f"{upload_id}.npz"
    if primary.is_file():
        return primary
    durable = Path(__file__).resolve().parents[2] / "data" / "uploads" / f"{upload_id}.npz"
    return durable


def _worker_environment() -> dict[str, str]:
    """Expose CasCast's venv packages when using a project-local CPython."""
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    python_paths = [str(CASCAST_VENV_SITE_PACKAGES)]
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["VIRTUAL_ENV"] = str(CASCAST_ROOT / ".venv")
    # wandb and the CasCast worker create temporary media directories. The
    # managed Windows Temp can deny cleanup after a long GPU run, so keep the
    # worker's temporary files in the project-owned writable area.
    environment["TEMP"] = str(_WORKER_TMP)
    environment["TMP"] = str(_WORKER_TMP)
    environment["TMPDIR"] = str(_WORKER_TMP)
    for variable in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR"):
        environment[variable] = str(_WORKER_TMP)
    scripts = str(CASCAST_ROOT / ".venv" / "Scripts")
    environment["PATH"] = scripts + os.pathsep + environment.get("PATH", "")
    return environment


def live_cascast_readiness() -> dict[str, Any]:
    assets = {
        "python": CASCAST_PYTHON.is_file(),
        "worker": CASCAST_LIVE_WORKER.is_file(),
        "earthformer_checkpoint": EARTHFORMER_CHECKPOINT.is_file(),
        "casformer_checkpoint": CASFORMER_CHECKPOINT.is_file(),
        "autoencoder_checkpoint": AUTOENCODER_CHECKPOINT.is_file(),
    }
    return {
        "ready": all(assets.values()),
        "assets": assets,
        "python_path": str(CASCAST_PYTHON),
        "cache_dir": str(CASCAST_LIVE_CACHE),
    }


def _cache_path(subject_key: str, segment: int, ensemble_members: int) -> Path:
    """Return the canonical cache path for one exact SEVIR catalog row."""
    record = _catalog_vil_event(subject_key)
    token = _catalog_subject_key(record).split(":", 1)[1]
    return CASCAST_LIVE_CACHE / f"sevir_{token}_s{segment}_ens{ensemble_members}.npz"


def _matching_cache(
    subject_key: str,
    segment: int = 0,
    ensemble_members: int | None = None,
) -> Path | None:
    """Find a canonical or verified legacy cache for one exact catalog row."""
    record = _catalog_vil_event(subject_key)
    token = _catalog_subject_key(record).split(":", 1)[1]
    member_pattern = str(int(ensemble_members)) if ensemble_members is not None else "*"
    patterns = (
        f"sevir_{token}_s{segment}_ens{member_pattern}.npz",
        f"{record['id']}_s{segment}_ens{member_pattern}.npz",
    )
    candidates = sorted(
        (path for pattern in patterns for path in CASCAST_LIVE_CACHE.glob(pattern)),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ) if CASCAST_LIVE_CACHE.exists() else []
    for path in candidates:
        try:
            with np.load(path, allow_pickle=False) as payload:
                cached_row = int(payload["hdf5_row"].item())
                cached_source = str(payload["source_id"].item()).strip().lower()
                exact_sources = {
                    str(record["id"]).strip().lower(),
                    _catalog_subject_key(record).strip().lower(),
                }
                # Remote SEVIR inputs are materialized as NPZ files rather
                # than read from the original HDF5, so the worker records
                # hdf5_row=-1.  Their canonical filename plus source_id still
                # verifies the exact catalog subject and must be accepted.
                if cached_row == int(record["file_index"]) or (
                    cached_row == -1 and cached_source in exact_sources
                ):
                    return path
        except (OSError, ValueError, KeyError):
            continue
    return None


def _uploaded_cache_path(
    upload_id: str, ensemble_members: int, rolling_segments: int = 1
) -> Path:
    suffix = "" if int(rolling_segments) == 1 else f"_roll{int(rolling_segments)}"
    return CASCAST_LIVE_CACHE / f"upload_{upload_id}_ens{ensemble_members}{suffix}.npz"


def _remote_event_input_path(record: dict[str, Any], segment: int) -> Path:
    """Materialize one remote 25-frame segment for the existing CasCast worker."""
    frames, _source = _load_vil_array(record)
    start = (0, 12, 24)[segment]
    sequence = np.asarray(frames[start:start + 25], dtype=np.uint8)
    if sequence.shape != (25, 384, 384):
        raise ValueError(f"unexpected remote event segment shape: {sequence.shape}")
    CASCAST_LIVE_CACHE.mkdir(parents=True, exist_ok=True)
    path = CASCAST_LIVE_CACHE / f".remote_{record['id']}_s{segment}_{uuid.uuid4().hex}.npz"
    np.savez_compressed(
        path,
        source_id=np.asarray(record["id"]),
        pixels=sequence[:13],
        ground_truth=sequence[13:],
    )
    return path


def run_local_cascast_inference(
    subject_key: str,
    segment: int = 0,
    ensemble_members: int = 10,
    force: bool = False,
    allow_live: bool = False,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Run the official local EarthFormer -> CasCast -> VAE chain for one HDF5 event.

    The event may be present in a local HDF5 file or be loaded through the
    remote event cache. Results are cached by event, 25-frame segment and
    ensemble size. GPU inference is serialized so concurrent web requests
    cannot exhaust VRAM.
    """
    if segment not in {0, 1, 2}:
        raise ValueError("segment must be 0, 1 or 2")
    ensemble_members = _validate_ensemble_members(ensemble_members)
    readiness = live_cascast_readiness()
    if not readiness["ready"]:
        missing = [name for name, available in readiness["assets"].items() if not available]
        raise RuntimeError(f"Local CasCast assets are incomplete: {', '.join(missing)}")
    record = _catalog_vil_event(subject_key)
    exact_key = _catalog_subject_key(record)
    output = _cache_path(exact_key, segment, ensemble_members)
    existing = _matching_cache(exact_key, segment, ensemble_members)
    if existing is not None and not force:
        return {"status": "ok", "cache_hit": True, "output": str(existing)}
    if os.getenv("SEVIR_CACHE_ONLY", "").strip().lower() in {"1", "true", "yes", "on"} and not allow_live:
        raise RuntimeError(
            f"cache_only_miss: refusing to run CasCast for event {record['id']}, "
            f"segment {segment}, ensemble {ensemble_members}"
        )

    CASCAST_LIVE_CACHE.mkdir(parents=True, exist_ok=True)
    temporary = CASCAST_LIVE_CACHE / f".{output.stem}.{uuid.uuid4().hex}.npz"
    temporary_input: Path | None = None
    hdf5_path = _local_hdf5_path(record)
    if hdf5_path.is_file():
        command = [
            str(CASCAST_PYTHON), str(CASCAST_LIVE_WORKER),
            "--cascast-root", str(CASCAST_ROOT),
            "--hdf5", str(hdf5_path),
            "--event-id", record["id"],
            "--hdf5-row", str(record["file_index"]),
            "--source-id", exact_key,
            "--segment", str(segment),
            "--ensemble-members", str(ensemble_members),
            "--precision", "fp16",
            "--output", str(temporary),
        ]
    else:
        temporary_input = _remote_event_input_path(record, segment)
        command = [
            str(CASCAST_PYTHON), str(CASCAST_LIVE_WORKER),
            "--cascast-root", str(CASCAST_ROOT),
            "--input-npz", str(temporary_input),
            "--source-id", record["id"],
            "--segment", str(segment),
            "--ensemble-members", str(ensemble_members),
            "--precision", "fp16",
            "--output", str(temporary),
        ]
    with _INFERENCE_LOCK:
        existing = _matching_cache(exact_key, segment, ensemble_members)
        if existing is not None and not force:
            if temporary_input is not None:
                temporary_input.unlink(missing_ok=True)
            return {"status": "ok", "cache_hit": True, "output": str(existing)}
        try:
            completed = subprocess.run(
                command,
                cwd=CASCAST_ROOT,
                env=_worker_environment(),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Local CasCast inference exceeded {timeout_seconds} seconds") from error
        finally:
            if temporary_input is not None:
                temporary_input.unlink(missing_ok=True)
        if completed.returncode != 0 or not temporary.is_file():
            temporary.unlink(missing_ok=True)
            detail = (completed.stderr or completed.stdout or "unknown worker error")[-2000:]
            raise RuntimeError(f"Local CasCast inference failed: {detail}")
        temporary.replace(output)
        if not output.is_file():
            raise RuntimeError("CasCast inference completed but its cache artifact was not persisted")
        load_live_cascast_ensemble.cache_clear()
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return {**payload, "cache_hit": False, "output": str(output)}


def run_uploaded_cascast_inference(
    upload_id: str,
    ensemble_members: int = 10,
    rolling_segments: int = 1,
    force: bool = False,
    allow_live: bool = False,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Run local CasCast for a persisted 13-frame user upload."""
    if len(upload_id) != 32 or any(character not in "0123456789abcdef" for character in upload_id):
        raise ValueError("Invalid sequence upload_id")
    ensemble_members = _validate_ensemble_members(ensemble_members)
    rolling_segments = int(rolling_segments)
    if rolling_segments not in {1, 2, 3}:
        raise ValueError("rolling_segments must be 1, 2, or 3")
    readiness = live_cascast_readiness()
    if not readiness["ready"]:
        missing = [name for name, available in readiness["assets"].items() if not available]
        raise RuntimeError(f"Local CasCast assets are incomplete: {', '.join(missing)}")
    input_path = _uploaded_input_path(upload_id)
    if not input_path.is_file():
        raise ValueError("The uploaded 13-frame sequence is no longer available")
    output = _uploaded_cache_path(upload_id, ensemble_members, rolling_segments)
    if output.is_file() and not force:
        return {"status": "ok", "cache_hit": True, "output": str(output)}
    if os.getenv("SEVIR_CACHE_ONLY", "").strip().lower() in {"1", "true", "yes", "on"} and not allow_live:
        raise RuntimeError(
            f"cache_only_miss: refusing to run CasCast for upload {upload_id}, "
            f"ensemble {ensemble_members}, rolling_segments {rolling_segments}"
        )
    CASCAST_LIVE_CACHE.mkdir(parents=True, exist_ok=True)
    temporary = CASCAST_LIVE_CACHE / f".{output.stem}.{uuid.uuid4().hex}.npz"
    command = [
        str(CASCAST_PYTHON), str(CASCAST_LIVE_WORKER),
        "--cascast-root", str(CASCAST_ROOT),
        "--input-npz", str(input_path),
        "--source-id", upload_id,
        "--ensemble-members", str(int(ensemble_members)),
        "--rolling-segments", str(rolling_segments),
        "--precision", "fp16",
        "--output", str(temporary),
    ]
    with _INFERENCE_LOCK:
        if output.is_file() and not force:
            return {"status": "ok", "cache_hit": True, "output": str(output)}
        try:
            completed = subprocess.run(
                command, cwd=CASCAST_ROOT, env=_worker_environment(),
                capture_output=True, text=True,
                timeout=timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Uploaded CasCast inference exceeded {timeout_seconds} seconds") from error
        if completed.returncode != 0 or not temporary.is_file():
            temporary.unlink(missing_ok=True)
            detail = (completed.stderr or completed.stdout or "unknown worker error")[-2000:]
            raise RuntimeError(f"Uploaded CasCast inference failed: {detail}")
        temporary.replace(output)
        if not output.is_file():
            raise RuntimeError("Uploaded CasCast inference completed but its cache artifact was not persisted")
        load_uploaded_cascast_ensemble.cache_clear()
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return {**payload, "cache_hit": False, "output": str(output)}


@lru_cache(maxsize=16)
def load_live_cascast_ensemble(
    subject_key: str,
    segment: int = 0,
    ensemble_members: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = _matching_cache(
        subject_key,
        segment,
        _validate_ensemble_members(ensemble_members) if ensemble_members is not None else None,
    )
    if path is None:
        raise ValueError(
            f"No local CasCast cache exists for subject {subject_key}, segment {segment}"
        )
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key])
            for key in (
                "history", "ground_truth", "ensemble_members", "ensemble_mean",
                "pixel_std", "q10", "q90", "member_seed",
            )
        }
        hdf5_row = int(payload["hdf5_row"])
    record = _catalog_vil_event(subject_key)
    metadata = {
        "subject_key": _catalog_subject_key(record),
        "event_id": record["id"],
        "segment": segment,
        "h5_file": record["file_name"],
        "file_index": record["file_index"],
        "hdf5_row": hdf5_row,
        "ensemble_count": int(arrays["ensemble_members"].shape[0]),
        "ground_truth_available": True,
        "cache_file": str(path),
        "catalog_row": record,
    }
    metadata["validation_manifest"] = build_validation_manifest(
        metadata,
        forecast_mode="standard_single_window",
        requested_lead_minutes=list(range(5, 65, 5)),
    )
    return arrays, metadata


@lru_cache(maxsize=16)
def load_uploaded_cascast_ensemble(
    upload_id: str,
    ensemble_members: int | None = None,
    rolling_segments: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rolling_segments = int(rolling_segments)
    if rolling_segments not in {1, 2, 3}:
        raise ValueError("rolling_segments must be 1, 2, or 3")
    if ensemble_members is not None:
        requested = _uploaded_cache_path(
            upload_id, _validate_ensemble_members(ensemble_members), rolling_segments
        )
        files = [requested] if requested.is_file() else []
    else:
        files = sorted(
            [
                path for path in CASCAST_LIVE_CACHE.glob(
                    f"upload_{upload_id}_ens*{'_roll' + str(rolling_segments) if rolling_segments > 1 else ''}.npz"
                )
                if rolling_segments > 1 or "_roll" not in path.stem
            ],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ) if CASCAST_LIVE_CACHE.exists() else []
    if not files:
        raise ValueError(f"No local CasCast result exists for upload {upload_id}")
    path = files[0]
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key])
            for key in (
                "history", "ensemble_members", "ensemble_mean", "pixel_std",
                "q10", "q90", "member_seed",
            )
        }
    metadata = {
        "upload_id": upload_id,
        "ensemble_count": int(arrays["ensemble_members"].shape[0]),
        "ground_truth_available": False,
        "cache_file": str(path),
        "rolling_segments": rolling_segments,
    }
    return arrays, metadata


def live_cascast_source(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "cascast_local_live_inference",
        "event_id": metadata["event_id"],
        "segment": metadata["segment"],
        "h5_file": metadata.get("h5_file"),
        "file_index": metadata.get("file_index"),
        "hdf5_row": metadata["hdf5_row"],
        "prediction_file": metadata["cache_file"],
        "validation_manifest": metadata.get("validation_manifest"),
        "note": "本机从官方SEVIR HDF5读取13帧历史，现场运行EarthFormer、CasCast和VAE解码。",
    }
