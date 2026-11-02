"""Minimal CasCast forecast transport used by the Agent-facing capability."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from ...request.uploads import _load_uploaded_vil_sequence
from .backend import (
    load_live_cascast_ensemble,
    load_uploaded_cascast_ensemble,
    run_local_cascast_inference,
    run_uploaded_cascast_inference,
)
from .alignment import load_uploaded_sequence_hdf5_ground_truth


def _evidence_id(subject: str, count: int, horizon: float, source: str) -> str:
    payload = {"subject": subject, "count": count, "horizon": horizon, "source": source}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return f"forecast-{digest}"


def _readonly(values: Any, dtype: Any = np.float32) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _transport(values: np.ndarray, arrays: bool) -> Any:
    return values if arrays else values.tolist()


def _members(
    arrays: dict[str, np.ndarray], leads: list[int], array_transport: bool
) -> list[dict[str, Any]]:
    values = np.clip(np.asarray(arrays["ensemble_members"], dtype=np.float32), 0, 254)
    seeds = np.asarray(arrays.get("member_seed", []), dtype=np.int64).tolist()
    return [{
        "member": index + 1,
        "seed": int(seeds[index]) if index < len(seeds) else None,
        "lead_minutes": leads,
        "frames": _transport(_readonly(member), array_transport),
    } for index, member in enumerate(values)]


def get_uploaded_vil_sequence_forecast(
    upload_id: str,
    model_name: str = "cascast_live",
    ensemble_members: int = 10,
    horizon_hours: float = 1.0,
    include_ground_truth: bool = False,
    array_transport: bool = False,
    force_live: bool = False,
) -> dict[str, Any]:
    """Run CasCast for an uploaded 13-frame history and return raw grids only."""
    if str(model_name).lower() not in {"cascast", "cascast_live"}:
        raise ValueError("CasCast does not provide baseline forecast models")
    horizon = float(horizon_hours)
    if horizon not in {1.0, 2.0, 3.0}:
        raise ValueError("horizon_hours must be 1.0, 2.0, or 3.0")
    sequence, upload_metadata = _load_uploaded_vil_sequence(upload_id)
    inference = run_uploaded_cascast_inference(
        upload_id, ensemble_members=int(ensemble_members),
        rolling_segments=int(horizon), allow_live=bool(force_live),
    )
    arrays, metadata = load_uploaded_cascast_ensemble(
        upload_id, ensemble_members=int(ensemble_members), rolling_segments=int(horizon)
    )
    history = _readonly(sequence)
    forecast = _readonly(np.clip(arrays["ensemble_mean"], 0, 254))
    leads = list(range(5, forecast.shape[0] * 5 + 1, 5))
    members = _members(arrays, leads, array_transport)
    truth = (
        load_uploaded_sequence_hdf5_ground_truth(upload_id, leads)
        if include_ground_truth else
        {"available": False, "frames": [], "lead_minutes": [], "reason": "ground_truth_not_requested"}
    )
    source = str(metadata["cache_file"])
    return {
        "status": "ok", "task": "cascast_forecast",
        "forecast_evidence_id": _evidence_id(upload_id, len(members), horizon, source),
        "upload_id": upload_id, "model_name": "cascast_live",
        "initialization_time_utc": upload_metadata.get("initialization_time_utc"),
        "history_relative_minutes": [int(v) for v in upload_metadata["history_minutes"]],
        "forecast_relative_minutes": leads, "lead_minutes": leads,
        "history_frames": _transport(history, array_transport),
        "forecast_frames": _transport(forecast, array_transport),
        "ensemble": {"member_count": len(members), "members": members},
        "ensemble_count": len(members), "cache_hit": bool(inference.get("cache_hit")),
        "prediction_source": source, "horizon_hours": horizon,
        "rolling_segments": int(horizon),
        "forecast_mode": "free_running" if horizon > 1 else "standard_single_window",
        "uses_future_observations": False,
        "ground_truth_requested": bool(include_ground_truth),
        "ground_truth_available": bool(truth.get("available")), "ground_truth": truth,
        "raw_data": True, "statistics_computed": [], "threshold_applied": None,
    }


def get_rolling_forecast_raw_values(
    event_id: str,
    horizon_hours: float = 1.0,
    model_name: str = "cascast_live",
    ensemble_members: int = 10,
    include_ground_truth: bool = False,
    array_transport: bool = False,
    force_live: bool = False,
) -> dict[str, Any]:
    """Run request-bound CasCast windows and return unanalysed forecast grids."""
    if str(model_name).lower() not in {"cascast", "cascast_live"}:
        raise ValueError("CasCast does not provide baseline forecast models")
    horizon = float(horizon_hours)
    if horizon not in {1.0, 2.0, 3.0}:
        raise ValueError("horizon_hours must be 1.0, 2.0, or 3.0")
    loaded = []
    inference_records = []
    for segment in range(int(horizon)):
        inference_records.append(run_local_cascast_inference(
            event_id, segment=segment, ensemble_members=int(ensemble_members),
            allow_live=bool(force_live),
        ))
        loaded.append(load_live_cascast_ensemble(
            event_id, segment=segment, ensemble_members=int(ensemble_members)
        ))
    first, metadata = loaded[0]
    arrays = {
        "history": first["history"],
        "ground_truth": np.concatenate([item[0]["ground_truth"] for item in loaded]),
        "ensemble_members": np.concatenate([item[0]["ensemble_members"] for item in loaded], axis=1),
        "ensemble_mean": np.concatenate([item[0]["ensemble_mean"] for item in loaded]),
        "member_seed": first["member_seed"],
    }
    history = _readonly(arrays["history"])
    forecast = _readonly(np.clip(arrays["ensemble_mean"], 0, 254))
    leads = list(range(5, forecast.shape[0] * 5 + 1, 5))
    members = _members(arrays, leads, array_transport)
    truth = _readonly(arrays["ground_truth"]) if include_ground_truth else None
    source = ",".join(str(item[1].get("cache_file") or "") for item in loaded)
    return {
        "status": "ok", "task": "cascast_forecast",
        "forecast_evidence_id": _evidence_id(str(metadata.get("event_id") or event_id), len(members), horizon, source),
        "event_id": metadata.get("event_id", event_id), "model_name": "cascast_live",
        "history_relative_minutes": list(range(-5 * (history.shape[0] - 1), 5, 5)),
        "forecast_relative_minutes": leads, "lead_minutes": leads,
        "history_frames": _transport(history, array_transport),
        "forecast_frames": _transport(forecast, array_transport),
        "ensemble": {"member_count": len(members), "members": members},
        "ensemble_count": len(members),
        "cache_hit": all(bool(item.get("cache_hit")) for item in inference_records),
        "prediction_source": source, "horizon_hours": horizon,
        "rolling_segments": int(horizon),
        "forecast_mode": "observation_reset" if horizon > 1 else "standard_single_window",
        "uses_future_observations": horizon > 1,
        "ground_truth_requested": bool(include_ground_truth),
        "ground_truth_available": truth is not None,
        "ground_truth": {
            "available": truth is not None,
            "frames": _transport(truth, array_transport) if truth is not None else [],
            "lead_minutes": leads if truth is not None else [], "missing_value": 255,
        },
        "raw_data": True, "statistics_computed": [], "threshold_applied": None,
    }
