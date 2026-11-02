from __future__ import annotations

"""Small deterministic executor for the first contract batch.

High-frequency, independently testable primitives are implemented here.  The
analogue operators call the same versioned signature/index implementation as
the public SEVIR capability; they do not maintain a second demo algorithm.
"""

from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from ..retrieval import AnalogueIndex, AnalogueQuery, extract_event_signature, retrieve_analogues

from .contracts import (
    AnalysisContractError,
    AnalysisRequest,
    AnalysisResult,
    DataRef,
    OperatorSpec,
)
from .schema import DEFAULT_PROFILE_REGISTRY, ProfileRegistry


IMPLEMENTED_OPERATORS = {
    "quality.inspect",
    "statistics.describe",
    "verification.continuous",
    "verification.threshold",
    "temporal.trend",
    "spatial.threshold",
    "objects.describe",
    "analogue.signature",
    "retrieval.analogue",
}


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    result = np.asarray(value)
    if result.size == 0:
        raise AnalysisContractError("analytics input is empty")
    if result.dtype.kind not in "biufc":
        raise AnalysisContractError(f"analytics input dtype {result.dtype} is not numeric")
    return result.astype(np.float64, copy=False)


def _input_handle(operator: OperatorSpec, request: AnalysisRequest) -> str:
    return str(operator.parameters.get("input_handle") or request.inputs[0].data_handle)


def _data_ref(request: AnalysisRequest, handle: str) -> DataRef:
    for item in request.inputs:
        if item.data_handle == handle:
            return item
    raise AnalysisContractError(f"operator references unknown input handle {handle!r}")


def _finite(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(value)
    return value[mask], mask


def _quality(value: np.ndarray, reference: DataRef) -> dict[str, Any]:
    finite, _ = _finite(value)
    result: dict[str, Any] = {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "input_count": int(value.size),
        "valid_count": int(finite.size),
        "excluded_count": int(value.size - finite.size),
        "missing_fraction": float(1.0 - finite.size / value.size),
        "units": reference.units,
    }
    if finite.size:
        result.update({"minimum": float(finite.min()), "maximum": float(finite.max())})
    return result


def _describe(value: np.ndarray, parameters: Mapping[str, Any]) -> dict[str, Any]:
    finite, _ = _finite(value)
    if not finite.size:
        raise AnalysisContractError("statistics.describe has no finite sample")
    quantiles = parameters.get("quantiles", (0.05, 0.25, 0.5, 0.75, 0.95))
    levels = tuple(float(item) for item in quantiles)
    if any(item < 0 or item > 1 for item in levels):
        raise AnalysisContractError("quantiles must be in [0, 1]")
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "standard_deviation": float(np.std(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "quantiles": {
            f"q{int(round(level * 100)):02d}": float(np.quantile(finite, level))
            for level in levels
        },
    }


def _paired(
    forecast: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    if forecast.shape != truth.shape:
        raise AnalysisContractError(
            f"forecast shape {forecast.shape} does not match reference {truth.shape}"
        )
    mask = np.isfinite(forecast) & np.isfinite(truth)
    if not np.any(mask):
        raise AnalysisContractError("verification has no aligned finite values")
    return forecast[mask], truth[mask], int(mask.size - np.count_nonzero(mask))


def _continuous(forecast: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    predicted, observed, _ = _paired(forecast, truth)
    error = predicted - observed
    mse = float(np.mean(error ** 2))
    return {
        "sample_count": int(error.size),
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.mean(error)),
    }


def _threshold(
    forecast: np.ndarray, truth: np.ndarray, threshold: float
) -> dict[str, Any]:
    predicted, observed, _ = _paired(forecast, truth)
    predicted_event = predicted >= threshold
    observed_event = observed >= threshold
    hits = int(np.count_nonzero(predicted_event & observed_event))
    misses = int(np.count_nonzero(~predicted_event & observed_event))
    false_alarms = int(np.count_nonzero(predicted_event & ~observed_event))

    def ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    return {
        "threshold": float(threshold),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "csi": ratio(hits, hits + misses + false_alarms),
        "pod": ratio(hits, hits + misses),
        "far": ratio(false_alarms, hits + false_alarms),
    }


def _time_cube(value: np.ndarray, operator: str) -> np.ndarray:
    if value.ndim < 2 or value.shape[0] < 2:
        raise AnalysisContractError(
            f"{operator} requires at least two frames with time as the first axis"
        )
    return value


def _temporal(value: np.ndarray) -> dict[str, Any]:
    cube = _time_cube(value, "temporal.trend")
    axes = tuple(range(1, cube.ndim))
    with np.errstate(invalid="ignore"):
        series = np.nanmean(cube, axis=axes)
    valid = np.isfinite(series)
    if np.count_nonzero(valid) < 2:
        raise AnalysisContractError("temporal.trend has fewer than two valid frames")
    positions = np.arange(series.size, dtype=np.float64)[valid]
    slope = float(np.polyfit(positions, series[valid], 1)[0])
    first, last = float(series[valid][0]), float(series[valid][-1])
    return {
        "frame_count": int(series.size),
        "valid_frame_count": int(np.count_nonzero(valid)),
        "start_mean": first,
        "end_mean": last,
        "growth": last - first,
        "slope_per_frame": slope,
        "series": [None if not np.isfinite(item) else float(item) for item in series],
    }


def _spatial_threshold(value: np.ndarray, threshold: float) -> dict[str, Any]:
    cube = _time_cube(value, "spatial.threshold")
    axes = tuple(range(1, cube.ndim))
    finite = np.isfinite(cube)
    valid_count = np.sum(finite, axis=axes)
    active_count = np.sum(finite & (cube >= threshold), axis=axes)
    coverage = np.divide(
        active_count, valid_count,
        out=np.full(active_count.shape, np.nan, dtype=np.float64),
        where=valid_count > 0,
    )
    valid = np.isfinite(coverage)
    if not np.any(valid):
        raise AnalysisContractError("spatial.threshold has no valid frame")
    return {
        "threshold": float(threshold),
        "coverage_fraction_series": [
            None if not np.isfinite(item) else float(item) for item in coverage
        ],
        "target_coverage_fraction": float(coverage[valid][-1]),
        "maximum_coverage_fraction": float(np.max(coverage[valid])),
        "coverage_growth": float(coverage[valid][-1] - coverage[valid][0]),
    }


def _objects(value: np.ndarray, threshold: float) -> dict[str, Any]:
    frame = value[-1] if value.ndim >= 3 else value
    if frame.ndim != 2:
        raise AnalysisContractError("objects.describe requires a 2-D grid or time-first grid cube")
    mask = np.isfinite(frame) & (frame >= threshold)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(np.max(sizes)) if sizes.size else 0
    return {
        "threshold": float(threshold),
        "object_count": int(count),
        "largest_object_pixel_count": largest,
        "largest_object_grid_fraction": float(largest / frame.size),
        "connectivity": 8,
    }


def _signature(
    value: np.ndarray,
    reference: DataRef,
    parameters: Mapping[str, Any],
):
    if "threshold" not in parameters:
        raise AnalysisContractError("analogue.signature requires an explicit threshold")
    window_raw = parameters.get("window_minutes", (-60, 0))
    window = tuple(int(item) for item in window_raw)
    if len(window) != 2:
        raise AnalysisContractError("analogue signature window_minutes requires two values")
    minutes = parameters.get("minute_offsets")
    if minutes is None:
        minutes = reference.coordinates.get("minute_offset")
    event_id = str(reference.provenance.get("event_id") or reference.subject_key)
    context = {
        "bounding_box": dict(reference.spatial_metadata.get("bounds", {})),
        "month": reference.provenance.get("month"),
    }
    return extract_event_signature(
        value,
        subject_key=reference.subject_key,
        event_id=event_id,
        source=reference.source,
        source_fingerprint=dict(reference.provenance),
        minute_offsets=minutes,
        window=(window[0], window[1]),
        threshold=float(parameters["threshold"]),
        event_anchor_time_utc=str(reference.provenance.get("event_anchor_time_utc") or reference.time_end_utc),
        event_type=str(reference.provenance.get("event_type") or "Unlabeled"),
        context=context,
    )


def execute_analysis(
    request: AnalysisRequest | Mapping[str, Any],
    datasets: Mapping[str, Any],
    *,
    profiles: ProfileRegistry = DEFAULT_PROFILE_REGISTRY,
) -> AnalysisResult:
    """Validate and execute deterministic operators on worker-local datasets."""

    compiled = (
        request
        if isinstance(request, AnalysisRequest)
        else AnalysisRequest.from_mapping(request)
    )
    values: dict[str, Any] = {}
    parameters: dict[str, Any] = {}
    first_sample: dict[str, int] | None = None
    provenance: list[dict[str, Any]] = []
    warnings: list[str] = []
    limitations: list[str] = []
    refs_by_handle = {item.data_handle: item for item in compiled.inputs}
    for reference in compiled.inputs:
        if reference.data_handle not in datasets:
            raise AnalysisContractError(
                f"dataset for handle {reference.data_handle!r} is unavailable"
            )
        profile = profiles.get(reference.profile_id)
        if reference.representation != profile.canonical_representation:
            raise AnalysisContractError(
                f"{reference.profile_id} requires {profile.canonical_representation}, "
                f"not {reference.representation}"
            )
        provenance.append({
            "subject_key": reference.subject_key,
            "source": reference.source,
            "profile_id": reference.profile_id,
            "profile_version": profile.version,
            **dict(reference.provenance),
        })
        limitations.extend(profile.semantic_limits)

    for operator in compiled.operators:
        if operator.op not in IMPLEMENTED_OPERATORS:
            raise AnalysisContractError(
                f"operator {operator.op!r} is not implemented in this release"
            )
        handle = _input_handle(operator, compiled)
        reference = _data_ref(compiled, handle)
        if not profiles.operation_allowed(reference.profile_id, operator.op):
            raise AnalysisContractError(
                f"{operator.op} is not allowed for {reference.profile_id}"
            )
        value = _array(datasets[handle])
        quality = _quality(value, reference)
        if first_sample is None:
            first_sample = {
                key: int(quality[key])
                for key in ("input_count", "valid_count", "excluded_count")
            }
        parameters[operator.output_name] = dict(operator.parameters)
        if operator.op == "quality.inspect":
            output = quality
        elif operator.op == "statistics.describe":
            output = _describe(value, operator.parameters)
        elif operator.op == "temporal.trend":
            output = _temporal(value)
        elif operator.op == "spatial.threshold":
            if "threshold" not in operator.parameters:
                raise AnalysisContractError("spatial.threshold requires an explicit threshold")
            output = _spatial_threshold(value, float(operator.parameters["threshold"]))
        elif operator.op == "objects.describe":
            if "threshold" not in operator.parameters:
                raise AnalysisContractError("objects.describe requires an explicit threshold")
            output = _objects(value, float(operator.parameters["threshold"]))
        elif operator.op in {"analogue.signature", "retrieval.analogue"}:
            signature = _signature(value, reference, operator.parameters)
            if operator.op == "analogue.signature":
                output = signature.to_dict()
            else:
                index_handle = str(operator.parameters.get("index_handle") or "")
                if not index_handle or index_handle not in datasets:
                    raise AnalysisContractError(
                        "retrieval.analogue requires a declared index_handle dataset"
                    )
                candidate_index = datasets[index_handle]
                if not isinstance(candidate_index, AnalogueIndex):
                    raise AnalysisContractError(
                        "retrieval.analogue index_handle must contain AnalogueIndex"
                    )
                output = retrieve_analogues(candidate_index, AnalogueQuery(
                    reference=signature,
                    top_k=int(operator.parameters.get("top_k", 5)),
                    recall_count=int(operator.parameters.get("recall_count", 40)),
                    candidate_subject_keys=tuple(operator.parameters.get("candidate_subject_keys", ())),
                    same_event_type=bool(operator.parameters.get("same_event_type", False)),
                    max_month_distance=operator.parameters.get("max_month_distance"),
                    max_geographic_km=operator.parameters.get("max_geographic_km"),
                    feature_weights=dict(operator.parameters.get("feature_weights", {})),
                )).to_dict()
        else:
            reference_handle = str(operator.parameters.get("reference_handle") or "")
            if not reference_handle:
                raise AnalysisContractError(
                    f"{operator.op} requires reference_handle (Ground Truth)"
                )
            if reference_handle not in refs_by_handle or reference_handle not in datasets:
                raise AnalysisContractError(
                    f"Ground Truth handle {reference_handle!r} is not a declared input"
                )
            truth = _array(datasets[reference_handle])
            if operator.op == "verification.continuous":
                output = _continuous(value, truth)
            else:
                if "threshold" not in operator.parameters:
                    raise AnalysisContractError(
                        "verification.threshold requires an explicit threshold"
                    )
                output = _threshold(value, truth, float(operator.parameters["threshold"]))
        values[operator.output_name] = output
    return AnalysisResult(
        status="ok",
        method="analytics.executor.v1",
        request_id=compiled.request_id,
        parameters=parameters,
        sample=first_sample or {"input_count": 0, "valid_count": 0, "excluded_count": 0},
        values=values,
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=tuple(provenance),
        limitations=tuple(dict.fromkeys(limitations)),
    )
