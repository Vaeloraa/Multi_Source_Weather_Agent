from __future__ import annotations

"""Exact NumPy index and offline builder for SEVIR analogue signatures."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping
import uuid

import numpy as np

from ...cache.registry import (
    CacheRegistry,
    file_stat_fingerprint,
    registry_entry_key,
    versioned_artifact_identity,
)
from ...config import ROOT
from .contracts import (
    ANALOGUE_INDEX_SCHEMA_VERSION,
    AnalogueContractError,
    EventSignature,
)
from .signature import (
    DEFAULT_THRESHOLD,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURE_UNITS,
    SIGNATURE_PIPELINE_VERSION,
    extract_event_signature,
)


INDEX_PIPELINE_VERSION = "sevir-analogue-exact-index-1.0"
INDEX_NAMESPACE = "sevir.analogue.index.v1"
DEFAULT_INDEX_PATH = ROOT / "data" / "indices" / "sevir_analogue_signature_v1.npz"
DEFAULT_EVENT_CACHE = ROOT / "data" / "sevir_event_cache"


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _metadata_without_vector(signature: EventSignature) -> dict[str, Any]:
    value = signature.to_dict()
    for key in ("feature_names", "features", "feature_units", "feature_groups"):
        value.pop(key, None)
    return value


@dataclass(frozen=True, slots=True)
class AnalogueIndex:
    """In-memory exact index with robust per-feature normalization."""

    index_id: str
    feature_names: tuple[str, ...]
    feature_groups: tuple[str, ...]
    feature_units: tuple[str, ...]
    matrix: np.ndarray
    median: np.ndarray
    scale: np.ndarray
    metadata: tuple[Mapping[str, Any], ...]
    source_manifest_sha256: str
    build_report: Mapping[str, Any]
    schema_version: str = ANALOGUE_INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ANALOGUE_INDEX_SCHEMA_VERSION:
            raise AnalogueContractError(
                f"unsupported index schema {self.schema_version!r}"
            )
        matrix = np.asarray(self.matrix, dtype=np.float64)
        median = np.asarray(self.median, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        count = len(self.feature_names)
        if matrix.ndim != 2 or matrix.shape[1] != count:
            raise AnalogueContractError("index matrix shape does not match feature schema")
        if matrix.shape[0] != len(self.metadata):
            raise AnalogueContractError("index metadata row count does not match matrix")
        if median.shape != (count,) or scale.shape != (count,):
            raise AnalogueContractError("index normalization vectors have invalid shape")
        if not np.all(np.isfinite(median)) or not np.all(np.isfinite(scale)):
            raise AnalogueContractError("index normalization vectors must be finite")
        if np.any(scale <= 0):
            raise AnalogueContractError("index scales must be positive")
        matrix.setflags(write=False)
        median.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "median", median)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "feature_groups", tuple(self.feature_groups))
        object.__setattr__(self, "feature_units", tuple(self.feature_units))
        if len({str(item.get("subject_key") or "") for item in self.metadata}) != len(self.metadata):
            raise AnalogueContractError("index subject keys must be unique")

    @classmethod
    def from_signatures(
        cls,
        signatures: Iterable[EventSignature],
        *,
        source_manifest_sha256: str,
        build_report: Mapping[str, Any],
    ) -> "AnalogueIndex":
        rows = tuple(signatures)
        if not rows:
            raise AnalogueContractError("cannot build an empty analogue index")
        expected = rows[0].feature_names
        if any(item.feature_names != expected for item in rows):
            raise AnalogueContractError("index signatures use different feature schemas")
        matrix = np.asarray([
            [np.nan if item is None else float(item) for item in signature.features]
            for signature in rows
        ], dtype=np.float64)
        median = np.zeros(matrix.shape[1], dtype=np.float64)
        scale = np.ones(matrix.shape[1], dtype=np.float64)
        for column in range(matrix.shape[1]):
            finite = matrix[np.isfinite(matrix[:, column]), column]
            if not finite.size:
                continue
            median[column] = float(np.median(finite))
            q25, q75 = np.quantile(finite, (0.25, 0.75))
            candidate = float(q75 - q25)
            if not np.isfinite(candidate) or candidate <= 1e-12:
                candidate = float(np.std(finite))
            if not np.isfinite(candidate) or candidate <= 1e-12:
                candidate = 1.0
            scale[column] = candidate
        identity = {
            "schema_version": ANALOGUE_INDEX_SCHEMA_VERSION,
            "pipeline_version": INDEX_PIPELINE_VERSION,
            "signature_pipeline_version": SIGNATURE_PIPELINE_VERSION,
            "source_manifest_sha256": source_manifest_sha256,
            "feature_names": list(expected),
            "signature_ids": [item.signature_id for item in rows],
        }
        return cls(
            index_id=_sha(identity),
            feature_names=expected,
            feature_groups=rows[0].feature_groups,
            feature_units=rows[0].feature_units,
            matrix=matrix,
            median=median,
            scale=scale,
            metadata=tuple(_metadata_without_vector(item) for item in rows),
            source_manifest_sha256=source_manifest_sha256,
            build_report=dict(build_report),
        )

    def signature_at(self, index: int) -> EventSignature:
        metadata = dict(self.metadata[int(index)])
        metadata.update({
            "feature_names": self.feature_names,
            "features": tuple(
                None if not np.isfinite(item) else float(item)
                for item in self.matrix[int(index)]
            ),
            "feature_units": self.feature_units,
            "feature_groups": self.feature_groups,
        })
        return EventSignature.from_mapping(metadata)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.stem}.{uuid.uuid4().hex}.tmp.npz"
        )
        payload = {
            "schema_version": np.asarray(self.schema_version),
            "index_id": np.asarray(self.index_id),
            "feature_names": np.asarray(self.feature_names, dtype=str),
            "feature_groups": np.asarray(self.feature_groups, dtype=str),
            "feature_units": np.asarray(self.feature_units, dtype=str),
            "matrix": self.matrix.astype(np.float32),
            "median": self.median.astype(np.float64),
            "scale": self.scale.astype(np.float64),
            "metadata_json": np.asarray(json.dumps(
                [_canonical(item) for item in self.metadata],
                ensure_ascii=False, sort_keys=True,
            )),
            "source_manifest_sha256": np.asarray(self.source_manifest_sha256),
            "build_report_json": np.asarray(json.dumps(
                _canonical(self.build_report), ensure_ascii=False, sort_keys=True,
            )),
        }
        try:
            np.savez_compressed(temporary, **payload)
            with temporary.open("rb+") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "AnalogueIndex":
        with np.load(Path(path), allow_pickle=False) as raw:
            metadata = json.loads(str(raw["metadata_json"]))
            build_report = json.loads(str(raw["build_report_json"]))
            return cls(
                schema_version=str(raw["schema_version"]),
                index_id=str(raw["index_id"]),
                feature_names=tuple(str(item) for item in raw["feature_names"].tolist()),
                feature_groups=tuple(str(item) for item in raw["feature_groups"].tolist()),
                feature_units=tuple(str(item) for item in raw["feature_units"].tolist()),
                matrix=np.asarray(raw["matrix"], dtype=np.float64),
                median=np.asarray(raw["median"], dtype=np.float64),
                scale=np.asarray(raw["scale"], dtype=np.float64),
                metadata=tuple(metadata),
                source_manifest_sha256=str(raw["source_manifest_sha256"]),
                build_report=build_report,
            )


def _catalog_maps() -> tuple[dict[str, Mapping[str, Any]], dict[tuple[str, int], Mapping[str, Any]]]:
    from ..sevir.repository import (
        _catalog_subject_index,
        _catalog_subject_key,
        _catalog_vil_records,
    )

    by_subject = dict(_catalog_subject_index())
    by_pair: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in _catalog_vil_records():
        if record.get("file_index") is None:
            continue
        enriched = {**record, "subject_key": _catalog_subject_key(record)}
        by_pair[(str(record.get("id") or "").upper(), int(record["file_index"]))] = enriched
    return by_subject, by_pair


def _bounds(record: Mapping[str, Any]) -> dict[str, float] | None:
    values = (
        record.get("llcrnrlat"), record.get("llcrnrlon"),
        record.get("urcrnrlat"), record.get("urcrnrlon"),
    )
    if any(item is None for item in values):
        return None
    return dict(zip(
        ("south", "west", "north", "east"),
        (float(item) for item in values),
    ))


def _source_manifest(paths: list[Path], cache_root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for path in paths:
        stat = path.stat()
        rows.append({
            "relative_path": str(path.relative_to(cache_root)),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    return _sha(rows), rows


def _build_signatures(paths: list[Path]) -> tuple[list[EventSignature], dict[str, Any]]:
    by_subject, by_pair = _catalog_maps()
    signatures: list[EventSignature] = []
    exclusions: dict[str, int] = {}

    def exclude(reason: str) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as raw:
                files = set(raw.files)
                explicit_type = str(raw["img_type"]).lower() if "img_type" in files else ""
                if explicit_type and explicit_type != "vil":
                    exclude("non_vil_channel")
                    continue
                frames = np.asarray(raw["frames"])
                if frames.ndim != 3 or frames.shape[1:] != (384, 384):
                    exclude("noncanonical_grid_shape")
                    continue
                event_id = str(raw["event_id"]).strip().upper()
                file_index = int(raw["file_index"]) if "file_index" in files else -1
                embedded_subject = (
                    str(raw["subject_key"]).strip().lower()
                    if "subject_key" in files else ""
                )
            record = by_subject.get(embedded_subject) if embedded_subject else None
            if record is None:
                record = by_pair.get((event_id, file_index))
            if record is None:
                exclude("catalog_identity_unresolved")
                continue
            subject_key = str(record.get("subject_key") or embedded_subject).lower()
            if not subject_key:
                exclude("subject_key_unresolved")
                continue
            bounds = _bounds(record)
            context: dict[str, Any] = {
                "bounding_box": bounds,
                "month": record.get("month"),
                "catalog_file_name": record.get("file_name"),
                "catalog_file_index": record.get("file_index"),
            }
            signatures.append(extract_event_signature(
                frames,
                subject_key=subject_key,
                event_id=event_id,
                source="SEVIR event cache",
                source_fingerprint=file_stat_fingerprint(path),
                minute_offsets=record.get("minute_offsets"),
                event_anchor_time_utc=str(record.get("time_utc") or ""),
                event_type=str(record.get("event_type") or "Unlabeled"),
                context=context,
                threshold=DEFAULT_THRESHOLD,
            ))
        except Exception as error:
            exclude(f"read_or_signature_error:{type(error).__name__}")
    signatures.sort(key=lambda item: item.subject_key)
    duplicate_count = len(signatures) - len({item.subject_key for item in signatures})
    if duplicate_count:
        unique: dict[str, EventSignature] = {}
        for item in signatures:
            unique.setdefault(item.subject_key, item)
        signatures = sorted(unique.values(), key=lambda item: item.subject_key)
        exclusions["duplicate_subject_key"] = duplicate_count
    return signatures, {
        "source_file_count": len(paths),
        "included_signature_count": len(signatures),
        "excluded_file_count": len(paths) - len(signatures),
        "exclusions": exclusions,
        "corpus_claim": "only QC-passing locally readable event-cache arrays are indexed",
        "catalog_wide_coverage_claimed": False,
        "future_frames_used": False,
    }


def build_local_sevir_index(
    *,
    event_cache: str | Path = DEFAULT_EVENT_CACHE,
    output_path: str | Path = DEFAULT_INDEX_PATH,
    registry: CacheRegistry | None = None,
    force: bool = False,
) -> tuple[AnalogueIndex, dict[str, Any]]:
    """Build or resolve the exact local index through the unified registry."""

    cache_root = Path(event_cache)
    output = Path(output_path)
    paths = sorted(cache_root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no SEVIR event caches found below {cache_root}")
    manifest_sha, manifest_rows = _source_manifest(paths, cache_root)
    identity = versioned_artifact_identity(
        INDEX_NAMESPACE,
        source={
            "corpus_manifest_sha256": manifest_sha,
            "source_file_count": len(paths),
        },
        parameters={
            "signature_schema": "sevir-event-signature-v1",
            "feature_names": list(FEATURE_NAMES),
            "feature_groups": list(FEATURE_GROUPS),
            "feature_units": list(FEATURE_UNITS),
            "window_minutes": [-60, 0],
            "threshold": DEFAULT_THRESHOLD,
        },
        pipeline_version=INDEX_PIPELINE_VERSION,
    )
    cache_registry = registry or CacheRegistry()
    if not force:
        cached = cache_registry.resolve(INDEX_NAMESPACE, identity)
        if cached is not None:
            index = AnalogueIndex.load(cached["artifact_uri"])
            return index, {
                "status": "cache_hit",
                "cache_identity": identity,
                "registry_entry_key": cached["entry_key"],
                "artifact": cached["artifact_uri"],
                "build_report": index.build_report,
            }
    entry_key = registry_entry_key(INDEX_NAMESPACE, identity)
    owner = f"index-builder:{os.getpid()}:{uuid.uuid4().hex}"
    acquired = cache_registry.acquire_lease(
        entry_key, owner, lease_seconds=300.0
    )
    if not acquired:
        for _ in range(40):
            time.sleep(0.25)
            cached = cache_registry.resolve(INDEX_NAMESPACE, identity)
            if cached is not None:
                index = AnalogueIndex.load(cached["artifact_uri"])
                return index, {
                    "status": "cache_hit_after_wait",
                    "cache_identity": identity,
                    "registry_entry_key": cached["entry_key"],
                    "artifact": cached["artifact_uri"],
                    "build_report": index.build_report,
                }
        raise TimeoutError("another process holds the analogue-index build lease")
    started = time.perf_counter()
    try:
        signatures, report = _build_signatures(paths)
        report = {
            **report,
            "source_manifest_sha256": manifest_sha,
            "manifest_entry_count": len(manifest_rows),
            "signature_pipeline_version": SIGNATURE_PIPELINE_VERSION,
            "index_pipeline_version": INDEX_PIPELINE_VERSION,
            "feature_count": len(FEATURE_NAMES),
        }
        index = AnalogueIndex.from_signatures(
            signatures,
            source_manifest_sha256=manifest_sha,
            build_report=report,
        )
        index.save(output)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        registered = cache_registry.register(
            INDEX_NAMESPACE,
            identity,
            output,
            metadata={
                "index_id": index.index_id,
                "schema_version": index.schema_version,
                "signature_count": len(index.metadata),
                "feature_count": len(index.feature_names),
                "source_manifest_sha256": manifest_sha,
                "build_elapsed_ms": round(elapsed_ms, 3),
                "scientific_artifact": True,
            },
        )
        return index, {
            "status": "built",
            "cache_identity": identity,
            "registry_entry_key": registered["entry_key"],
            "artifact": registered["artifact_uri"],
            "build_elapsed_ms": round(elapsed_ms, 3),
            "build_report": report,
        }
    finally:
        cache_registry.release_lease(entry_key, owner)


__all__ = [
    "DEFAULT_EVENT_CACHE",
    "DEFAULT_INDEX_PATH",
    "INDEX_NAMESPACE",
    "INDEX_PIPELINE_VERSION",
    "AnalogueIndex",
    "build_local_sevir_index",
]
