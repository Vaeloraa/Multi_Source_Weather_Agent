from __future__ import annotations

"""Request-bound analogue service with exact query-result caching."""

import json
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from ...cache import atomic_json
from ...cache.registry import (
    CacheRegistry,
    registry_entry_key,
    versioned_artifact_identity,
)
from ...config import ROOT
from .contracts import (
    ANALOGUE_RESULT_SCHEMA_VERSION,
    AnalogueQuery,
    AnalogueResult,
    EventSignature,
)
from .index import AnalogueIndex
from .retriever import RETRIEVER_VERSION, retrieve_analogues
from .signature import DEFAULT_THRESHOLD, extract_event_signature


QUERY_CACHE_NAMESPACE = "sevir.analogue.query.v1"
QUERY_PIPELINE_VERSION = "sevir-analogue-query-cache-2.0-catalog-corpus"
DEFAULT_QUERY_CACHE = ROOT / "data" / "analogue_query_cache"


def _event_reference(
    subject_key: str, *, allow_remote: bool | None = None
) -> EventSignature:
    from ..sevir.repository import get_sevir_event_metadata, load_sevir_vil_event

    metadata = get_sevir_event_metadata(subject_key)
    record = dict(metadata["record"])
    loaded = load_sevir_vil_event(
        subject_key, minutes_before=60, minutes_after=0,
        allow_remote=allow_remote,
    )
    if str(loaded.get("status")) != "ok":
        raise RuntimeError(str(loaded.get("reason") or "reference observations unavailable"))
    return extract_event_signature(
        loaded["frames"],
        subject_key=str(record["subject_key"]),
        event_id=str(record["id"]),
        source=str(loaded.get("source_hdf5") or metadata.get("data_source") or "SEVIR"),
        source_fingerprint={
            "subject_key": str(record["subject_key"]),
            "catalog_file_name": record.get("file_name"),
            "file_index": record.get("file_index"),
        },
        minute_offsets=loaded.get("relative_minutes"),
        event_anchor_time_utc=str(record.get("time_utc") or ""),
        event_type=str(record.get("event_type") or "Unlabeled"),
        context={
            "month": int(str(record.get("time_utc") or "0000-00")[5:7]),
            "bounding_box": record.get("bounding_box"),
            "spatial_metadata": loaded.get("spatial_metadata"),
            "catalog_file_name": record.get("file_name"),
            "catalog_file_index": record.get("file_index"),
        },
        threshold=DEFAULT_THRESHOLD,
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reference_bounds(reference: EventSignature) -> dict[str, float] | None:
    context = reference.context if isinstance(reference.context, Mapping) else {}
    metadata = context.get("spatial_metadata")
    value = metadata.get("bounds") if isinstance(metadata, Mapping) else context.get("bounding_box")
    if not isinstance(value, Mapping):
        return None
    try:
        return {
            name: float(value[name]) for name in ("south", "west", "north", "east")
        }
    except (KeyError, TypeError, ValueError):
        return None


def _catalog_candidate_index(
    reference: EventSignature,
    *,
    candidate_subject_keys: Sequence[str] | None,
    same_event_type: bool,
    max_month_distance: int | None,
    max_geographic_km: float | None,
    region: Mapping[str, float] | None,
    latitude: float | None,
    longitude: float | None,
    query_time_utc: str,
    allow_remote: bool,
) -> tuple[AnalogueIndex | None, dict[str, Any]]:
    """Discover candidates in Catalog, then load and sign each exact row."""

    from ..sevir.repository import (
        discover_sevir_catalog_candidates,
        load_sevir_vil_event,
    )
    from ..geospatial.projection import bounding_box_center

    spatial_region = dict(region) if region is not None else None
    reference_region = _reference_bounds(reference)
    if spatial_region is None and latitude is None:
        spatial_region = reference_region
    if latitude is None and max_geographic_km is not None and reference_region is not None:
        centre = bounding_box_center(reference_region)
        latitude, longitude = centre.latitude, centre.longitude
    cutoff = str(query_time_utc or reference.event_anchor_time_utc or "")
    discovery = discover_sevir_catalog_candidates(
        region=spatial_region,
        latitude=latitude,
        longitude=longitude,
        max_distance_km=max_geographic_km,
        before_time_utc=cutoff or None,
        exclude_subject_key=reference.subject_key,
        event_type=reference.event_type if same_event_type else "",
        max_month_distance=max_month_distance,
        require_prior_minutes=60,
    )
    frozen = {
        str(value).strip().lower() for value in (candidate_subject_keys or ())
        if str(value).strip()
    }
    records = list(discovery["records"])
    frozen_excluded = 0
    if frozen:
        selected = []
        for record in records:
            if str(record["subject_key"]).lower() in frozen:
                selected.append(record)
            else:
                frozen_excluded += 1
        records = selected

    signatures: list[EventSignature] = []
    ledger: list[dict[str, Any]] = []
    loaded_count = 0
    failure_reasons: dict[str, int] = {}
    for record in records:
        item = {
            "subject_key": str(record["subject_key"]),
            "event_id": str(record["id"]),
            "time": str(record.get("time_utc") or ""),
            "catalog_candidate": True,
            "loaded": False,
            "signature_success": False,
            "failed": False,
            "excluded": False,
            "status": "catalog_candidate",
            "reason": "",
        }
        try:
            loaded = load_sevir_vil_event(
                str(record["subject_key"]),
                minutes_before=60,
                minutes_after=0,
                allow_remote=allow_remote,
            )
            if str(loaded.get("status") or "") != "ok":
                raise RuntimeError(str(loaded.get("reason") or "data unavailable"))
            item["loaded"] = True
            item["status"] = "loaded"
            loaded_count += 1
            signatures.append(extract_event_signature(
                loaded["frames"],
                subject_key=str(record["subject_key"]),
                event_id=str(record["id"]),
                source=str(loaded.get("source_hdf5") or "SEVIR repository"),
                source_fingerprint={
                    "subject_key": str(record["subject_key"]),
                    "catalog_file_name": record.get("file_name"),
                    "file_index": record.get("file_index"),
                },
                minute_offsets=loaded.get("relative_minutes"),
                event_anchor_time_utc=str(record.get("time_utc") or ""),
                event_type=str(record.get("event_type") or "Unlabeled"),
                context={
                    "month": record.get("month"),
                    "bounding_box": record.get("bounding_box"),
                    "spatial_metadata": loaded.get("spatial_metadata"),
                    "catalog_file_name": record.get("file_name"),
                    "catalog_file_index": record.get("file_index"),
                },
                threshold=DEFAULT_THRESHOLD,
            ))
            item["signature_success"] = True
            item["status"] = "signature_success"
        except Exception as error:
            item["failed"] = True
            item["status"] = "failed"
            item["reason"] = f"{type(error).__name__}: {error}"
            failure_reasons[item["reason"]] = failure_reasons.get(item["reason"], 0) + 1
        ledger.append(item)

    catalog_excluded = int(discovery.get("excluded_count") or 0) + frozen_excluded
    report = {
        "corpus_source": "SEVIR Catalog spatial candidate discovery",
        "catalog_wide_coverage_claimed": False,
        "catalog_record_count": int(discovery["catalog_record_count"]),
        "catalog_candidate_count": len(records),
        "loaded_count": loaded_count,
        "signature_success_count": len(signatures),
        "ranking_candidate_count": len(signatures),
        "failed_count": len(records) - len(signatures),
        "excluded_count": catalog_excluded,
        "lifecycle_counts": {
            "catalog_candidate": len(records),
            "loaded": loaded_count,
            "signature_success": len(signatures),
            "failed": len(records) - len(signatures),
            "excluded": catalog_excluded,
        },
        "discovery_filters": discovery.get("filters", {}),
        "discovery_exclusions": {
            **dict(discovery.get("exclusions") or {}),
            **({"outside_frozen_candidate_set": frozen_excluded} if frozen_excluded else {}),
        },
        "failure_reasons": failure_reasons,
        "candidate_records": ledger,
        "future_frames_used": False,
        "strict_historical_cutoff_applied": bool(cutoff),
        "remote_access_allowed": bool(allow_remote),
    }
    if not signatures:
        return None, report
    manifest = _digest({
        "catalog_candidates": [str(record["subject_key"]) for record in records],
        "signature_ids": [signature.signature_id for signature in signatures],
        "discovery_filters": report["discovery_filters"],
    })
    index = AnalogueIndex.from_signatures(
        signatures,
        source_manifest_sha256=manifest,
        build_report=report,
    )
    return index, report


def _upload_reference(upload_id: str) -> EventSignature:
    from ...request.uploads import _load_uploaded_vil_sequence, uploaded_event_match

    frames, metadata = _load_uploaded_vil_sequence(upload_id)
    match = uploaded_event_match(upload_id)
    event = match.get("event_match") if isinstance(match.get("event_match"), Mapping) else {}
    event_id = str(event.get("event_id") or metadata.get("matched_event_id") or f"UPLOAD-{upload_id[:8]}")
    return extract_event_signature(
        frames,
        subject_key=f"upload:{upload_id}",
        event_id=event_id,
        source=str(metadata.get("source") or "user uploaded VIL sequence"),
        source_fingerprint={
            "upload_id": upload_id,
            "sha256": metadata.get("sha256") or metadata.get("sequence_sha256"),
        },
        minute_offsets=metadata.get("relative_minutes") or tuple(range(-60, 5, 5)),
        event_anchor_time_utc=str(metadata.get("event_anchor_time_utc") or ""),
        event_type=str(metadata.get("event_type") or "Unlabeled"),
        context={
            "month": metadata.get("month"),
            "bounding_box": metadata.get("bounding_box"),
        },
        threshold=DEFAULT_THRESHOLD,
    )


def _query_identity(index_id: str, query: AnalogueQuery) -> dict[str, Any]:
    value = query.to_dict()
    value.pop("reference", None)
    return versioned_artifact_identity(
        QUERY_CACHE_NAMESPACE,
        source={
            "query_signature_id": query.reference.signature_id,
            "index_id": index_id,
        },
        parameters=value,
        pipeline_version=QUERY_PIPELINE_VERSION,
        model=RETRIEVER_VERSION,
    )


def find_bound_analogues(
    *,
    subject_key: str = "",
    upload_id: str = "",
    top_k: int = 5,
    recall_count: int = 40,
    candidate_subject_keys: Sequence[str] | None = None,
    same_event_type: bool = False,
    max_month_distance: int | None = None,
    max_geographic_km: float | None = None,
    feature_weights: Mapping[str, float] | None = None,
    region: Mapping[str, float] | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    query_time_utc: str = "",
    allow_remote: bool = False,
    registry: CacheRegistry | None = None,
) -> dict[str, Any]:
    """Run Catalog discovery, repository loading, signatures and ranking."""

    cache_registry = registry or CacheRegistry()
    reference = (
        _upload_reference(str(upload_id).strip())
        if str(upload_id).strip()
        else _event_reference(subject_key, allow_remote=allow_remote)
    )
    index, build_report = _catalog_candidate_index(
        reference,
        candidate_subject_keys=candidate_subject_keys,
        same_event_type=bool(same_event_type),
        max_month_distance=max_month_distance,
        max_geographic_km=max_geographic_km,
        region=region,
        latitude=latitude,
        longitude=longitude,
        query_time_utc=query_time_utc,
        allow_remote=bool(allow_remote),
    )
    index_id = index.index_id if index is not None else "catalog-empty:" + _digest(build_report)
    query = AnalogueQuery(
        reference=reference,
        top_k=int(top_k),
        recall_count=int(recall_count),
        # Catalog discovery above owns corpus and hard constraints.  The
        # retriever receives only successfully signed candidates and ranks them.
        candidate_subject_keys=(),
        same_event_type=False,
        max_month_distance=None,
        max_geographic_km=None,
        feature_weights=dict(feature_weights or {}),
    )
    identity = _query_identity(index_id, query)
    cached = cache_registry.resolve(QUERY_CACHE_NAMESPACE, identity)
    if cached is not None:
        value = json.loads(Path(cached["artifact_uri"]).read_text(encoding="utf-8"))
        value["cache_hit"] = True
        value["cache_identity"] = {
            "namespace": QUERY_CACHE_NAMESPACE,
            "request_key": identity["request_key"],
            "index_id": index_id,
        }
        value["index_cache_status"] = "catalog_candidate_index"
        return value
    entry_key = registry_entry_key(QUERY_CACHE_NAMESPACE, identity)
    owner = f"analogue-query:{os.getpid()}:{uuid.uuid4().hex}"
    acquired = cache_registry.acquire_lease(entry_key, owner, lease_seconds=60.0)
    if not acquired:
        for _ in range(40):
            time.sleep(0.1)
            cached = cache_registry.resolve(QUERY_CACHE_NAMESPACE, identity)
            if cached is not None:
                value = json.loads(Path(cached["artifact_uri"]).read_text(encoding="utf-8"))
                value["cache_hit"] = True
                value["cache_identity"] = {
                    "namespace": QUERY_CACHE_NAMESPACE,
                    "request_key": identity["request_key"],
                    "index_id": index_id,
                }
                value["index_cache_status"] = "catalog_candidate_index"
                return value
        raise TimeoutError("another process holds the analogue-query lease")
    try:
        if index is None:
            value = {
                "schema_version": ANALOGUE_RESULT_SCHEMA_VERSION,
                "query_signature_id": reference.signature_id,
                "index_id": index_id,
                "records": [],
                "candidate_scope": {
                    "index_entry_count": 0,
                    "eligible_count": 0,
                    "stage1_recalled_count": 0,
                    "returned_count": 0,
                    "corpus": build_report,
                },
                "distance_definition": {
                    "retriever_version": RETRIEVER_VERSION,
                    "ranking_performed": False,
                },
                "provenance": [{
                    "role": "catalog_candidate_discovery",
                    "source": build_report["corpus_source"],
                }],
                "elapsed_ms": 0.0,
                "warnings": ["Catalog candidates were found, but none produced a usable signature."],
                "limitations": [
                    "Catalog discovery does not imply that every referenced HDF5 object is readable."
                ],
            }
            returned_count = 0
        else:
            result = retrieve_analogues(index, query)
            value = result.to_dict()
            returned_count = len(result.records)
        artifact = DEFAULT_QUERY_CACHE / f"{identity['request_key']}.json"
        atomic_json(artifact, value)
        cache_registry.register(
            QUERY_CACHE_NAMESPACE,
            identity,
            artifact,
            metadata={
                "query_signature_id": reference.signature_id,
                "index_id": index_id,
                "returned_count": returned_count,
                "scientific_artifact": True,
            },
        )
        value["cache_hit"] = False
        value["cache_identity"] = {
            "namespace": QUERY_CACHE_NAMESPACE,
            "request_key": identity["request_key"],
            "index_id": index_id,
        }
        value["index_cache_status"] = "catalog_candidate_index"
        return value
    finally:
        cache_registry.release_lease(entry_key, owner)


__all__ = [
    "DEFAULT_QUERY_CACHE",
    "QUERY_CACHE_NAMESPACE",
    "QUERY_PIPELINE_VERSION",
    "find_bound_analogues",
]
