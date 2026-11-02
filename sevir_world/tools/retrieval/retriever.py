from __future__ import annotations

"""Two-stage exact recall and meteorological-constraint reranking."""

import math
import time
from typing import Any, Mapping

import numpy as np

from .contracts import (
    AnalogueContractError,
    AnalogueQuery,
    AnalogueResult,
    EventSignature,
)
from .index import AnalogueIndex


RETRIEVER_VERSION = "sevir-analogue-retriever-1.0"

DEFAULT_GROUP_WEIGHTS: Mapping[str, float] = {
    "intensity": 1.0,
    "area": 1.2,
    "temporal": 1.3,
    "temporal_curve": 1.0,
    "motion": 0.8,
    "objects": 0.7,
    "season": 0.15,
    "geography": 0.10,
    "quality": 0.25,
}
RECALL_GROUPS = frozenset({
    "intensity", "area", "temporal", "temporal_curve", "motion",
})


def _month(signature: EventSignature | Mapping[str, Any]) -> int | None:
    context = signature.context if isinstance(signature, EventSignature) else signature.get("context", {})
    if not isinstance(context, Mapping) or context.get("month") is None:
        return None
    try:
        value = int(context["month"])
        return value if 1 <= value <= 12 else None
    except (TypeError, ValueError):
        return None


def _center(signature: EventSignature | Mapping[str, Any]) -> tuple[float, float] | None:
    context = signature.context if isinstance(signature, EventSignature) else signature.get("context", {})
    if not isinstance(context, Mapping):
        return None
    metadata = context.get("spatial_metadata")
    bounds = metadata.get("bounds") if isinstance(metadata, Mapping) else context.get("bounding_box")
    if not isinstance(bounds, Mapping):
        return None
    try:
        return (
            (float(bounds["south"]) + float(bounds["north"])) / 2.0,
            (float(bounds["west"]) + float(bounds["east"])) / 2.0,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lon1 = (math.radians(item) for item in first)
    lat2, lon2 = (math.radians(item) for item in second)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return float(6371.0088 * 2 * math.asin(min(1.0, math.sqrt(value))))


def _month_distance(first: int, second: int) -> int:
    difference = abs(int(first) - int(second))
    return min(difference, 12 - difference)


def _weights(query: AnalogueQuery, index: AnalogueIndex) -> np.ndarray:
    result = np.asarray([
        float(DEFAULT_GROUP_WEIGHTS.get(group, 1.0))
        for group in index.feature_groups
    ], dtype=np.float64)
    for position, (name, group) in enumerate(zip(index.feature_names, index.feature_groups)):
        if group in query.feature_weights:
            result[position] = float(query.feature_weights[group])
        if name in query.feature_weights:
            result[position] = float(query.feature_weights[name])
    return result


def _distances(
    query_vector: np.ndarray,
    matrix: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
    selected_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(matrix)
        & np.isfinite(query_vector)[None, :]
        & (weights > 0)[None, :]
        & selected_features[None, :]
    )
    total_weight = float(np.sum(weights[selected_features & np.isfinite(query_vector)]))
    valid_weight = np.sum(valid * weights[None, :], axis=1)
    differences = (matrix - query_vector[None, :]) / scale[None, :]
    squared = np.where(valid, differences ** 2 * weights[None, :], 0.0)
    factor = np.divide(
        total_weight,
        valid_weight,
        out=np.full(valid_weight.shape, np.nan, dtype=np.float64),
        where=valid_weight > 0,
    )
    distance_squared = np.sum(squared, axis=1) * factor
    matched = np.sum(valid, axis=1)
    minimum = max(4, int(math.ceil(np.count_nonzero(selected_features) * 0.35)))
    distance_squared[matched < minimum] = np.nan
    return np.sqrt(distance_squared), valid


def _eligible(query: AnalogueQuery, index: AnalogueIndex) -> tuple[np.ndarray, dict[str, int], dict[int, float]]:
    allowed = set(query.candidate_subject_keys)
    mask = np.ones(len(index.metadata), dtype=bool)
    exclusions: dict[str, int] = {}
    geography: dict[int, float] = {}
    reference_month = _month(query.reference)
    reference_center = _center(query.reference)

    def exclude(position: int, reason: str) -> None:
        if mask[position]:
            mask[position] = False
            exclusions[reason] = exclusions.get(reason, 0) + 1

    for position, metadata in enumerate(index.metadata):
        subject_key = str(metadata.get("subject_key") or "").lower()
        if allowed and subject_key not in allowed:
            exclude(position, "outside_frozen_candidate_set")
            continue
        if query.exclude_same_event and (
            subject_key == query.reference.subject_key.lower()
            or str(metadata.get("event_id") or "").upper() == query.reference.event_id.upper()
        ):
            exclude(position, "reference_event")
            continue
        if query.same_event_type:
            reference_type = query.reference.event_type.strip().lower()
            candidate_type = str(metadata.get("event_type") or "").strip().lower()
            if reference_type and reference_type != "unlabeled" and candidate_type != reference_type:
                exclude(position, "different_event_type")
                continue
        if query.max_month_distance is not None:
            candidate_month = _month(metadata)
            if reference_month is None or candidate_month is None:
                exclude(position, "month_unavailable")
                continue
            if _month_distance(reference_month, candidate_month) > int(query.max_month_distance):
                exclude(position, "outside_month_window")
                continue
        if query.max_geographic_km is not None:
            candidate_center = _center(metadata)
            if reference_center is None or candidate_center is None:
                exclude(position, "geography_unavailable")
                continue
            distance = _haversine_km(reference_center, candidate_center)
            geography[position] = distance
            if distance > float(query.max_geographic_km):
                exclude(position, "outside_geographic_radius")
    return mask, exclusions, geography


def _rank_positions(distances: np.ndarray, metadata: tuple[Mapping[str, Any], ...]) -> list[int]:
    return sorted(
        (index for index, item in enumerate(distances) if np.isfinite(item)),
        key=lambda index: (
            float(distances[index]),
            str(metadata[index].get("subject_key") or ""),
        ),
    )


def retrieve_analogues(
    index: AnalogueIndex,
    query: AnalogueQuery | EventSignature,
    **query_options: Any,
) -> AnalogueResult:
    """Retrieve explainable analogues without reading post-anchor observations."""

    started = time.perf_counter()
    compiled = query if isinstance(query, AnalogueQuery) else AnalogueQuery(
        reference=query, **query_options
    )
    reference = compiled.reference
    if reference.feature_names != index.feature_names:
        raise AnalogueContractError("query and index feature schemas differ")
    query_vector = np.asarray([
        np.nan if item is None else float(item) for item in reference.features
    ], dtype=np.float64)
    weights = _weights(compiled, index)
    eligible, exclusions, geography = _eligible(compiled, index)
    recall_features = np.asarray([
        group in RECALL_GROUPS for group in index.feature_groups
    ], dtype=bool)
    stage1, _ = _distances(
        query_vector, index.matrix, index.scale, weights, recall_features
    )
    stage1[~eligible] = np.nan
    recalled = _rank_positions(stage1, index.metadata)[: min(
        int(compiled.recall_count), int(np.count_nonzero(eligible))
    )]
    rerank_mask = np.zeros(len(index.metadata), dtype=bool)
    rerank_mask[recalled] = True
    all_features = np.ones(len(index.feature_names), dtype=bool)
    final_distance, final_valid = _distances(
        query_vector, index.matrix, index.scale, weights, all_features
    )
    final_distance[~rerank_mask] = np.nan
    ranked = _rank_positions(final_distance, index.metadata)[: int(compiled.top_k)]
    records: list[dict[str, Any]] = []
    total_weight = float(np.sum(weights[np.isfinite(query_vector) & (weights > 0)]))
    for rank, position in enumerate(ranked, start=1):
        candidate = index.matrix[position]
        valid = final_valid[position]
        valid_weight = float(np.sum(weights[valid]))
        factor = total_weight / valid_weight if valid_weight else 0.0
        raw_contributions = np.where(
            valid,
            weights * ((candidate - query_vector) / index.scale) ** 2 * factor,
            0.0,
        )
        contributions = {
            name: float(raw_contributions[item])
            for item, name in enumerate(index.feature_names)
            if valid[item] and raw_contributions[item] > 0
        }
        group_contributions: dict[str, float] = {}
        for feature_position, group in enumerate(index.feature_groups):
            if valid[feature_position]:
                group_contributions[group] = group_contributions.get(group, 0.0) + float(
                    raw_contributions[feature_position]
                )
        important = sorted(
            (item for item in range(len(index.feature_names)) if valid[item]),
            key=lambda item: (-float(raw_contributions[item]), index.feature_names[item]),
        )[:8]
        differences = [{
            "feature": index.feature_names[item],
            "group": index.feature_groups[item],
            "unit": index.feature_units[item],
            "query_value": float(query_vector[item]),
            "candidate_value": float(candidate[item]),
            "signed_difference": float(candidate[item] - query_vector[item]),
            "distance_squared_contribution": float(raw_contributions[item]),
        } for item in important]
        metadata = dict(index.metadata[position])
        records.append({
            "rank": rank,
            "subject_key": str(metadata.get("subject_key") or ""),
            "event_id": str(metadata.get("event_id") or ""),
            "event_anchor_time_utc": str(metadata.get("event_anchor_time_utc") or ""),
            "event_type": str(metadata.get("event_type") or "Unlabeled"),
            "distance": float(final_distance[position]),
            "distance_squared": float(final_distance[position] ** 2),
            "stage1_recall_distance": float(stage1[position]),
            "matched_feature_count": int(np.count_nonzero(valid)),
            "feature_contributions": contributions,
            "group_contributions": group_contributions,
            "largest_differences": differences,
            "geographic_distance_km": geography.get(position),
            "context": metadata.get("context", {}),
            "quality": metadata.get("quality", {}),
            "signature_id": str(metadata.get("signature_id") or ""),
        })
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    candidate_scope = {
        "index_entry_count": len(index.metadata),
        "eligible_count": int(np.count_nonzero(eligible)),
        "stage1_recalled_count": len(recalled),
        "returned_count": len(records),
        "exclusions": exclusions,
        "constraints": {
            "candidate_subject_key_count": len(compiled.candidate_subject_keys),
            "exclude_same_event": compiled.exclude_same_event,
            "same_event_type": compiled.same_event_type,
            "max_month_distance": compiled.max_month_distance,
            "max_geographic_km": compiled.max_geographic_km,
        },
        "corpus": index.build_report,
    }
    return AnalogueResult(
        query_signature_id=reference.signature_id,
        index_id=index.index_id,
        records=tuple(records),
        candidate_scope=candidate_scope,
        distance_definition={
            "retriever_version": RETRIEVER_VERSION,
            "stage1": "robust-scaled weighted L2 on intensity/area/temporal/motion features",
            "stage2": "hard meteorological constraints then robust-scaled missing-aware weighted L2 on all features",
            "normalization": "index median and IQR; standard deviation then 1.0 fallback",
            "missing_policy": "pairwise finite features with total-weight renormalization and minimum overlap",
            "group_weights": dict(DEFAULT_GROUP_WEIGHTS),
            "query_weight_overrides": dict(compiled.feature_weights),
            "feature_scales": {
                name: float(index.scale[position])
                for position, name in enumerate(index.feature_names)
            },
            "future_observations_allowed": False,
        },
        provenance=(
            {
                "role": "reference_signature",
                "signature_id": reference.signature_id,
                "subject_key": reference.subject_key,
                "window_minutes": [reference.window_start_minute, reference.window_end_minute],
                "source": reference.source,
            },
            {
                "role": "candidate_index",
                "index_id": index.index_id,
                "schema_version": index.schema_version,
                "source_manifest_sha256": index.source_manifest_sha256,
            },
        ),
        elapsed_ms=elapsed_ms,
        warnings=(() if records else ("no candidate satisfied the declared scope and overlap rules",)),
        limitations=(
            "This is an interpretable project baseline, not expert proof of meteorological equivalence.",
            (
                "The ranked corpus comes from Catalog candidates that passed independent load and signature QC; "
                "see candidate_scope.corpus for failures and exclusions."
                if index.build_report.get("corpus_source") == "SEVIR Catalog spatial candidate discovery"
                else "The offline index covers only locally readable QC-passing event-cache arrays, not the full SEVIR catalog."
            ),
            "Encoded VIL similarity does not establish identical hazards, impacts or future evolution.",
        ),
    )


__all__ = [
    "DEFAULT_GROUP_WEIGHTS",
    "RECALL_GROUPS",
    "RETRIEVER_VERSION",
    "retrieve_analogues",
]
