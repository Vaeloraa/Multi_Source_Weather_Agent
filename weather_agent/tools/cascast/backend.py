from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..sevir.repository import (
    catalog_subject_key,
    catalog_vil_event,
    _load_vil_array,
    _local_hdf5_path,
)
from .alignment import build_validation_manifest
from ..gpu import gpu_slot
from ...cache.registry import (
    CacheRegistry,
    file_stat_fingerprint,
    registry_entry_key,
    versioned_artifact_identity,
)
from ...persistence.uploads import UPLOAD_STORAGE_DIR
from .config import (
    AUTOENCODER_CHECKPOINT,
    CASCAST_LIVE_CACHE,
    CASCAST_LIVE_WORKER,
    CASCAST_PYTHON,
    CASCAST_ROOT,
    CASCAST_VENV_SITE_PACKAGES,
    CASFORMER_CHECKPOINT,
    EARTHFORMER_CHECKPOINT,
)


MAX_LIVE_ENSEMBLE_MEMBERS = max(
    1, int(os.getenv("WEATHER_AGENT_MAX_LIVE_ENSEMBLE_MEMBERS", "10"))
)
MAX_ARRAY_MEMORY_CACHE_ITEMS = max(
    0, int(os.getenv("WEATHER_AGENT_ARRAY_MEMORY_CACHE_ITEMS", "2"))
)
CASCAST_CACHE_NAMESPACE = "cascast.live.v2"
CASCAST_PIPELINE_VERSION = "earthformer-cascast-vae-v2"
CASCAST_PRECISION = "fp16"
CASCAST_GUIDANCE = 2.0
CASCAST_SEED = 0


def _validate_ensemble_members(value: int) -> int:
    count = int(value)
    if not 1 <= count <= MAX_LIVE_ENSEMBLE_MEMBERS:
        raise ValueError(
            f"ensemble_members must be between 1 and {MAX_LIVE_ENSEMBLE_MEMBERS}"
        )
    return count


_INFERENCE_LOCK = threading.Lock()
try:
    _WORKER_TMP = Path(tempfile.mkdtemp(prefix="weather_agent_cascast_worker_"))
except OSError:
    _WORKER_TMP = Path(tempfile.gettempdir())


def _uploaded_input_path(upload_id: str) -> Path:
    """Resolve the raw uploaded sequence, including the durable upload store."""
    primary = UPLOAD_STORAGE_DIR / f"{upload_id}.npz"
    if primary.is_file():
        return primary
    durable = Path(__file__).resolve().parents[2] / "data" / "uploads" / f"{upload_id}.npz"
    return durable


def _cascast_model_fingerprint() -> dict[str, Any]:
    return {
        "earthformer_checkpoint": file_stat_fingerprint(EARTHFORMER_CHECKPOINT),
        "casformer_checkpoint": file_stat_fingerprint(CASFORMER_CHECKPOINT),
        "autoencoder_checkpoint": file_stat_fingerprint(AUTOENCODER_CHECKPOINT),
        "worker": file_stat_fingerprint(CASCAST_LIVE_WORKER),
    }


def _catalog_source_fingerprint(record: dict[str, Any]) -> dict[str, Any]:
    local = _local_hdf5_path(record)
    return {
        "event_id": str(record.get("id") or ""),
        "catalog_subject": catalog_subject_key(record),
        "h5_file": str(record.get("file_name") or ""),
        "file_index": int(record.get("file_index", -1)),
        "local_hdf5": file_stat_fingerprint(local),
    }


def _catalog_cache_identity(
    record: dict[str, Any], segment: int, ensemble_members: int
) -> dict[str, Any]:
    return versioned_artifact_identity(
        CASCAST_CACHE_NAMESPACE,
        source=_catalog_source_fingerprint(record),
        parameters={
            "segment": int(segment),
            "ensemble_members": int(ensemble_members),
            "rolling_segments": 1,
            "guidance": CASCAST_GUIDANCE,
            "seed": CASCAST_SEED,
            "precision": CASCAST_PRECISION,
        },
        pipeline_version=CASCAST_PIPELINE_VERSION,
        model="EarthFormer+CasCast+VAE",
        model_fingerprint=_cascast_model_fingerprint(),
    )


def _uploaded_cache_identity(
    upload_id: str,
    input_path: Path,
    ensemble_members: int,
    rolling_segments: int,
) -> dict[str, Any]:
    return versioned_artifact_identity(
        CASCAST_CACHE_NAMESPACE,
        source={
            "upload_id": str(upload_id),
            "input": file_stat_fingerprint(input_path),
        },
        parameters={
            "segment": 0,
            "ensemble_members": int(ensemble_members),
            "rolling_segments": int(rolling_segments),
            "guidance": CASCAST_GUIDANCE,
            "seed": CASCAST_SEED,
            "precision": CASCAST_PRECISION,
        },
        pipeline_version=CASCAST_PIPELINE_VERSION,
        model="EarthFormer+CasCast+VAE",
        model_fingerprint=_cascast_model_fingerprint(),
    )


def _register_cascast_artifact(
    path: Path,
    identity: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return CacheRegistry().register(
        CASCAST_CACHE_NAMESPACE,
        identity,
        path,
        metadata={
            **metadata,
            "scientific_validity": "versioned_identity",
        },
    )


@contextmanager
def _cache_build_lease(
    artifact: Path, *, timeout_seconds: int
):
    """Prevent duplicate workers from publishing to one cache path."""

    lease_identity = versioned_artifact_identity(
        "cache.build.lease",
        source={"namespace": CASCAST_CACHE_NAMESPACE, "artifact": str(artifact.resolve())},
    )
    lease_key = registry_entry_key("cache.build.lease", lease_identity)
    owner = uuid.uuid4().hex
    registry = CacheRegistry()
    acquired = registry.acquire_lease(
        lease_key,
        owner,
        lease_seconds=max(3600, int(timeout_seconds) * 4 + 600),
    )
    if not acquired:
        raise RuntimeError(
            f"cache_build_in_progress: another process is building {artifact.name}"
        )
    try:
        yield
    finally:
        registry.release_lease(lease_key, owner)


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
    record = catalog_vil_event(subject_key)
    token = catalog_subject_key(record).split(":", 1)[1]
    return CASCAST_LIVE_CACHE / f"sevir_{token}_s{segment}_ens{ensemble_members}.npz"


def _matching_cache(
    subject_key: str,
    segment: int = 0,
    ensemble_members: int | None = None,
) -> Path | None:
    """Find an artifact whose stored request key matches its current identity."""
    record = catalog_vil_event(subject_key)
    token = catalog_subject_key(record).split(":", 1)[1]
    member_pattern = str(int(ensemble_members)) if ensemble_members is not None else "*"
    expected_identity = (
        _catalog_cache_identity(record, segment, int(ensemble_members))
        if ensemble_members is not None else None
    )
    registered_path: Path | None = None
    if expected_identity is not None:
        registered = CacheRegistry().resolve(CASCAST_CACHE_NAMESPACE, expected_identity)
        if registered is not None:
            registered_path = Path(str(registered["artifact_uri"]))
    patterns = (
        f"sevir_{token}_s{segment}_ens{member_pattern}.npz",
        f"{record['id']}_s{segment}_ens{member_pattern}.npz",
    )
    discovered = sorted(
        (path for pattern in patterns for path in CASCAST_LIVE_CACHE.glob(pattern)),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ) if CASCAST_LIVE_CACHE.exists() else []
    candidates: list[Path] = []
    for path in ([registered_path] if registered_path is not None else []) + discovered:
        if path not in candidates:
            candidates.append(path)
    for path in candidates:
        try:
            with np.load(path, allow_pickle=False) as payload:
                cached_row = int(payload["hdf5_row"].item())
                cached_source = str(payload["source_id"].item()).strip().lower()
                actual_members = int(payload["ensemble_members"].shape[0])
                cached_segment = int(payload["segment"].item())
                cached_rolling = int(payload["rolling_segments"].item())
                actual_cache_key = (
                    str(payload["cache_request_key"].item()).strip()
                    if "cache_request_key" in payload.files else ""
                )
                exact_sources = {
                    str(record["id"]).strip().lower(),
                    catalog_subject_key(record).strip().lower(),
                }
                # Remote SEVIR inputs are materialized as NPZ files rather
                # than read from the original HDF5, so the worker records
                # hdf5_row=-1.  Their canonical filename plus source_id still
                # verifies the exact catalog subject and must be accepted.
                source_matches = (
                    cached_source in exact_sources
                    and cached_row in {int(record["file_index"]), -1}
                )
                if (
                    not source_matches
                    or cached_segment != int(segment)
                    or cached_rolling != 1
                    or (
                        ensemble_members is not None
                        and actual_members != int(ensemble_members)
                    )
                ):
                    continue
                candidate_identity = expected_identity or _catalog_cache_identity(
                    record, segment, actual_members
                )
                if not actual_cache_key or actual_cache_key != candidate_identity["request_key"]:
                    continue
                _register_cascast_artifact(
                    path,
                    candidate_identity,
                    metadata={
                        "event_id": record["id"],
                        "subject_key": catalog_subject_key(record),
                        "segment": int(segment),
                        "ensemble_members": actual_members,
                    },
                )
                return path
        except (OSError, ValueError, KeyError):
            continue
    return None


def _uploaded_cache_path(
    upload_id: str, ensemble_members: int, rolling_segments: int = 1
) -> Path:
    suffix = "" if int(rolling_segments) == 1 else f"_roll{int(rolling_segments)}"
    return CASCAST_LIVE_CACHE / f"upload_{upload_id}_ens{ensemble_members}{suffix}.npz"


def _matching_uploaded_cache(
    upload_id: str, ensemble_members: int, rolling_segments: int
) -> Path | None:
    input_path = _uploaded_input_path(upload_id)
    if not input_path.is_file():
        return None
    identity = _uploaded_cache_identity(
        upload_id, input_path, ensemble_members, rolling_segments
    )
    registered = CacheRegistry().resolve(CASCAST_CACHE_NAMESPACE, identity)
    requested = _uploaded_cache_path(upload_id, ensemble_members, rolling_segments)
    candidates = []
    if registered is not None:
        candidates.append(Path(str(registered["artifact_uri"])))
    if requested not in candidates:
        candidates.append(requested)
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as payload:
                count = int(payload["ensemble_members"].shape[0])
                rolling = int(payload["rolling_segments"].item())
                source = str(payload["source_id"].item()).strip()
                actual_cache_key = (
                    str(payload["cache_request_key"].item()).strip()
                    if "cache_request_key" in payload.files else ""
                )
            if count != int(ensemble_members) or rolling != int(rolling_segments) or source != upload_id:
                continue
            if not actual_cache_key or actual_cache_key != identity["request_key"]:
                continue
            _register_cascast_artifact(
                path,
                identity,
                metadata={
                    "upload_id": upload_id,
                    "ensemble_members": count,
                    "rolling_segments": rolling,
                },
            )
            return path
        except (OSError, ValueError, KeyError):
            continue
    return None


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
    timeout_seconds: int = 86_400,
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
    record = catalog_vil_event(subject_key)
    exact_key = catalog_subject_key(record)
    output = _cache_path(exact_key, segment, ensemble_members)
    existing = None if force else _matching_cache(
        exact_key, segment, ensemble_members
    )
    if existing is not None and not force:
        return {"status": "ok", "cache_hit": True, "output": str(existing)}
    if not allow_live:
        raise RuntimeError(
            f"cache_only_miss: refusing to run CasCast for event {record['id']}, "
            f"segment {segment}, ensemble {ensemble_members}"
        )
    readiness = live_cascast_readiness()
    if not readiness["ready"]:
        missing = [name for name, available in readiness["assets"].items() if not available]
        raise RuntimeError(f"Local CasCast assets are incomplete: {', '.join(missing)}")

    CASCAST_LIVE_CACHE.mkdir(parents=True, exist_ok=True)
    identity = _catalog_cache_identity(record, segment, ensemble_members)
    temporary = CASCAST_LIVE_CACHE / f".{output.stem}.{uuid.uuid4().hex}.npz"
    temporary_input: Path | None = None
    with _cache_build_lease(output, timeout_seconds=timeout_seconds):
        existing = None if force else _matching_cache(
            exact_key, segment, ensemble_members
        )
        if existing is not None and not force:
            return {"status": "ok", "cache_hit": True, "output": str(existing)}
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
                "--precision", CASCAST_PRECISION,
                "--cache-request-key", identity["request_key"],
                "--pipeline-version", CASCAST_PIPELINE_VERSION,
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
                "--precision", CASCAST_PRECISION,
                "--cache-request-key", identity["request_key"],
                "--pipeline-version", CASCAST_PIPELINE_VERSION,
                "--output", str(temporary),
            ]
        with _INFERENCE_LOCK:
            existing = None if force else _matching_cache(
                exact_key, segment, ensemble_members
            )
            if existing is not None and not force:
                if temporary_input is not None:
                    temporary_input.unlink(missing_ok=True)
                return {"status": "ok", "cache_hit": True, "output": str(existing)}
            try:
                with gpu_slot(timeout=timeout_seconds):
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
            _register_cascast_artifact(
                output,
                identity,
                metadata={
                    "event_id": record["id"],
                    "subject_key": exact_key,
                    "segment": int(segment),
                    "ensemble_members": int(ensemble_members),
                },
            )
            _load_npz_payload.cache_clear()
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return {**payload, "cache_hit": False, "output": str(output)}


def run_uploaded_cascast_inference(
    upload_id: str,
    ensemble_members: int = 10,
    rolling_segments: int = 1,
    force: bool = False,
    allow_live: bool = False,
    timeout_seconds: int = 86_400,
) -> dict[str, Any]:
    """Run local CasCast for a persisted 13-frame user upload."""
    if len(upload_id) != 32 or any(character not in "0123456789abcdef" for character in upload_id):
        raise ValueError("Invalid sequence upload_id")
    ensemble_members = _validate_ensemble_members(ensemble_members)
    rolling_segments = int(rolling_segments)
    if rolling_segments not in {1, 2, 3}:
        raise ValueError("rolling_segments must be 1, 2, or 3")
    input_path = _uploaded_input_path(upload_id)
    if not input_path.is_file():
        raise ValueError("The uploaded 13-frame sequence is no longer available")
    output = _uploaded_cache_path(upload_id, ensemble_members, rolling_segments)
    existing = None if force else _matching_uploaded_cache(
        upload_id, ensemble_members, rolling_segments
    )
    if existing is not None and not force:
        return {"status": "ok", "cache_hit": True, "output": str(existing)}
    if not allow_live:
        raise RuntimeError(
            f"cache_only_miss: refusing to run CasCast for upload {upload_id}, "
            f"ensemble {ensemble_members}, rolling_segments {rolling_segments}"
        )
    readiness = live_cascast_readiness()
    if not readiness["ready"]:
        missing = [name for name, available in readiness["assets"].items() if not available]
        raise RuntimeError(f"Local CasCast assets are incomplete: {', '.join(missing)}")
    CASCAST_LIVE_CACHE.mkdir(parents=True, exist_ok=True)
    identity = _uploaded_cache_identity(
        upload_id, input_path, ensemble_members, rolling_segments
    )
    temporary = CASCAST_LIVE_CACHE / f".{output.stem}.{uuid.uuid4().hex}.npz"
    command = [
        str(CASCAST_PYTHON), str(CASCAST_LIVE_WORKER),
        "--cascast-root", str(CASCAST_ROOT),
        "--input-npz", str(input_path),
        "--source-id", upload_id,
        "--ensemble-members", str(int(ensemble_members)),
        "--rolling-segments", str(rolling_segments),
        "--precision", CASCAST_PRECISION,
        "--cache-request-key", identity["request_key"],
        "--pipeline-version", CASCAST_PIPELINE_VERSION,
        "--output", str(temporary),
    ]
    with _cache_build_lease(output, timeout_seconds=timeout_seconds):
        existing = None if force else _matching_uploaded_cache(
            upload_id, ensemble_members, rolling_segments
        )
        if existing is not None and not force:
            return {"status": "ok", "cache_hit": True, "output": str(existing)}
        with _INFERENCE_LOCK:
            try:
                with gpu_slot(timeout=timeout_seconds):
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
            _register_cascast_artifact(
                output,
                identity,
                metadata={
                    "upload_id": upload_id,
                    "ensemble_members": int(ensemble_members),
                    "rolling_segments": int(rolling_segments),
                },
            )
            _load_npz_payload.cache_clear()
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return {**payload, "cache_hit": False, "output": str(output)}


@lru_cache(maxsize=MAX_ARRAY_MEMORY_CACHE_ITEMS)
def _load_npz_payload(
    path_value: str,
    mtime_ns: int,
    size_bytes: int,
    array_keys: tuple[str, ...],
    scalar_keys: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load an immutable in-memory view keyed by the physical file version."""

    del mtime_ns, size_bytes  # They intentionally participate in the LRU key.
    with np.load(path_value, allow_pickle=False) as payload:
        arrays: dict[str, np.ndarray] = {}
        for key in array_keys:
            value = np.array(payload[key], copy=True)
            value.setflags(write=False)
            arrays[key] = value
        scalars = {
            key: payload[key].item()
            for key in scalar_keys
            if key in payload.files
        }
    return arrays, scalars


def _load_versioned_npz(
    path: Path,
    *,
    array_keys: tuple[str, ...],
    scalar_keys: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stat = path.stat()
    arrays, scalars = _load_npz_payload(
        str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size),
        array_keys, scalar_keys,
    )
    # Callers may add/remove dictionary entries, but cannot mutate shared
    # ndarray contents. A replaced file receives a new mtime/size cache key.
    return dict(arrays), dict(scalars)


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
    arrays, scalars = _load_versioned_npz(
        path,
        array_keys=(
            "history", "ground_truth", "ensemble_members", "ensemble_mean",
            "pixel_std", "q10", "q90", "member_seed",
        ),
        scalar_keys=("hdf5_row", "cache_request_key", "pipeline_version"),
    )
    hdf5_row = int(scalars["hdf5_row"])
    cache_request_key = str(scalars.get("cache_request_key") or "").strip()
    pipeline_version = str(scalars.get("pipeline_version") or "").strip()
    if not cache_request_key or not pipeline_version:
        raise ValueError("CasCast cache is missing versioned identity metadata")
    record = catalog_vil_event(subject_key)
    metadata = {
        "subject_key": catalog_subject_key(record),
        "event_id": record["id"],
        "segment": segment,
        "h5_file": record["file_name"],
        "file_index": record["file_index"],
        "hdf5_row": hdf5_row,
        "ensemble_count": int(arrays["ensemble_members"].shape[0]),
        "ground_truth_available": True,
        "cache_file": str(path),
        "cache_request_key": cache_request_key,
        "cache_identity_status": "versioned",
        "pipeline_version": pipeline_version,
        "catalog_row": record,
    }
    metadata["validation_manifest"] = build_validation_manifest(
        metadata,
        forecast_mode="standard_single_window",
        requested_lead_minutes=list(range(5, 65, 5)),
    )
    return arrays, metadata


def load_uploaded_cascast_ensemble(
    upload_id: str,
    ensemble_members: int | None = None,
    rolling_segments: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rolling_segments = int(rolling_segments)
    if rolling_segments not in {1, 2, 3}:
        raise ValueError("rolling_segments must be 1, 2, or 3")
    if ensemble_members is not None:
        matched = _matching_uploaded_cache(
            upload_id, _validate_ensemble_members(ensemble_members), rolling_segments
        )
        files = [matched] if matched is not None else []
    else:
        discovered = sorted(
            [
                path for path in CASCAST_LIVE_CACHE.glob(
                    f"upload_{upload_id}_ens*{'_roll' + str(rolling_segments) if rolling_segments > 1 else ''}.npz"
                )
                if rolling_segments > 1 or "_roll" not in path.stem
            ],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ) if CASCAST_LIVE_CACHE.exists() else []
        files = []
        for candidate in discovered:
            try:
                with np.load(candidate, allow_pickle=False) as payload:
                    candidate_members = int(payload["ensemble_members"].shape[0])
            except (OSError, ValueError, KeyError):
                continue
            matched = _matching_uploaded_cache(
                upload_id, candidate_members, rolling_segments
            )
            if matched is not None:
                files = [matched]
                break
    if not files:
        raise ValueError(f"No local CasCast result exists for upload {upload_id}")
    path = files[0]
    arrays, scalars = _load_versioned_npz(
        path,
        array_keys=(
            "history", "ensemble_members", "ensemble_mean", "pixel_std",
            "q10", "q90", "member_seed",
        ),
        scalar_keys=("cache_request_key", "pipeline_version"),
    )
    cache_request_key = str(scalars.get("cache_request_key") or "").strip()
    pipeline_version = str(scalars.get("pipeline_version") or "").strip()
    if not cache_request_key or not pipeline_version:
        raise ValueError("CasCast upload cache is missing versioned identity metadata")
    metadata = {
        "upload_id": upload_id,
        "ensemble_count": int(arrays["ensemble_members"].shape[0]),
        "ground_truth_available": False,
        "cache_file": str(path),
        "cache_request_key": cache_request_key,
        "cache_identity_status": "versioned",
        "pipeline_version": pipeline_version,
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
        "note": "Read 13 historical frames from official SEVIR HDF5 files and ran local EarthFormer, CasCast and VAE decoding.",
    }
