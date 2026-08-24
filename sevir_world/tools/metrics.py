from __future__ import annotations

from typing import Any

import numpy as np

from ..backends.forecast import _forecast_evidence_id
from ..data_access import ensemble
from ..data_access.live_inference import (
    load_uploaded_cascast_ensemble,
    run_uploaded_cascast_inference,
)
from ..data_access.uploads import load_uploaded_sequence_hdf5_ground_truth

from .base import WorldComponent
from ..domain.thresholds import CASCAST_VIL_THRESHOLDS


DEFAULT_THRESHOLDS = CASCAST_VIL_THRESHOLDS
DEFAULT_SCALES = (1, 4, 8, 16, 32)


def _selected_lead_indices(
    lead_minutes: list[int] | None,
    frame_count: int,
    *,
    maximum_selected: int = 6,
) -> tuple[list[int], list[int]]:
    available = list(range(5, int(frame_count) * 5 + 5, 5))
    selected = list(dict.fromkeys(int(value) for value in (lead_minutes or available)))
    if not selected:
        raise ValueError("lead_minutes must contain at least one lead")
    invalid = [value for value in selected if value not in available]
    if invalid:
        raise ValueError(
            f"lead_minutes must be available 5-minute leads through {available[-1] if available else 0}"
        )
    if len(selected) > maximum_selected:
        raise ValueError(
            f"at most {maximum_selected} raw prediction/truth leads may be loaded per call"
        )
    return selected, [value // 5 - 1 for value in selected]


def _prediction_from_arrays(
    arrays: dict[str, np.ndarray], prediction_source: str, member: int
) -> tuple[np.ndarray, dict[str, Any]]:
    source = str(prediction_source or "ensemble_mean").strip().lower()
    if source == "ensemble_mean":
        return np.asarray(arrays["ensemble_mean"], dtype=np.float32), {
            "prediction_source": "cascast_ensemble_mean",
            "member": None,
        }
    if source == "member":
        members = np.asarray(arrays["ensemble_members"], dtype=np.float32)
        selected_member = int(member)
        if not 1 <= selected_member <= int(members.shape[0]):
            raise ValueError(f"member must be between 1 and {members.shape[0]}")
        return members[selected_member - 1], {
            "prediction_source": "cascast_stochastic_member",
            "member": selected_member,
        }
    if source == "persistence":
        truth_frame_count = int(np.asarray(arrays["ensemble_mean"]).shape[0])
        return np.repeat(
            np.asarray(arrays["history"][-1], dtype=np.float32)[None],
            truth_frame_count,
            axis=0,
        ), {"prediction_source": "persistence", "member": None}
    raise ValueError("prediction_source must be ensemble_mean, member, or persistence")


def _raw_pair_result(
    *,
    identifier: dict[str, Any],
    prediction: np.ndarray,
    truth: np.ndarray,
    selected_leads: list[int],
    indices: list[int],
    source_metadata: dict[str, Any],
    prediction_provenance: dict[str, Any],
) -> dict[str, Any]:
    selected_prediction = np.asarray(prediction[indices], dtype=np.float32)
    selected_truth = np.asarray(truth, dtype=np.uint8)
    if selected_prediction.shape != selected_truth.shape:
        raise ValueError(
            f"prediction/truth shapes are not aligned: {selected_prediction.shape} vs {selected_truth.shape}"
        )
    valid = selected_truth != 255
    selected_prediction = np.ascontiguousarray(selected_prediction, dtype=np.float32)
    selected_truth = np.ascontiguousarray(selected_truth, dtype=np.uint8)
    valid = np.ascontiguousarray(valid, dtype=bool)
    selected_prediction.setflags(write=False)
    selected_truth.setflags(write=False)
    valid.setflags(write=False)
    return {
        "status": "ok",
        "task": "raw_prediction_truth_pair",
        "capabilities": [
            "raw_prediction",
            "raw_ground_truth",
            "valid_mask",
            "agent_defined_verification_metric",
        ],
        **identifier,
        **prediction_provenance,
        "ground_truth_requested": True,
        "ground_truth_available": True,
        "raw_data": True,
        "lead_minutes": selected_leads,
        "prediction_frames": selected_prediction,
        "truth_frames": selected_truth,
        "valid_mask": valid,
        "array_transport": "read_only_ndarray_inside_isolated_worker",
        "shape": [int(value) for value in selected_prediction.shape],
        "prediction_dtype": "float32",
        "truth_dtype": "uint8",
        "prediction_value_space": "CasCast decoded encoded-VIL scale",
        "truth_value_space": "SEVIR uint8 encoded VIL",
        "missing_truth_value": 255,
        "valid_pixel_count_by_lead": [
            int(item.sum()) for item in valid
        ],
        "statistics_computed": [],
        "threshold_applied": None,
        "provenance": {
            "prediction": source_metadata.get("prediction_source")
            or source_metadata.get("cache_file"),
            "truth": source_metadata.get("truth_source")
            or "SEVIR held-out observations",
            "lead_alignment": "exact equality of lead_minutes positions",
            "truth_was_loaded_explicitly_for_verification": True,
        },
        "verification_rule": (
            "Compare only positionally aligned leads and exclude valid_mask=false pixels. "
            "The tool chooses no RMSE, MAE, bias, threshold, or other score."
        ),
    }


def _contingency(prediction: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, int]:
    valid = target != 255
    predicted = (prediction >= threshold) & valid
    observed = (target >= threshold) & valid
    return {
        "hits": int((predicted & observed).sum()),
        "misses": int((~predicted & observed).sum()),
        "false_alarms": int((predicted & ~observed & valid).sum()),
        "correct_negatives": int((~predicted & ~observed & valid).sum()),
        "valid": int(valid.sum()),
    }


def _scores(counts: dict[str, int]) -> dict[str, float]:
    hits = counts["hits"]
    misses = counts["misses"]
    false_alarms = counts["false_alarms"]
    correct_negatives = counts["correct_negatives"]
    csi_denominator = hits + misses + false_alarms
    hss_numerator = 2 * (hits * correct_negatives - misses * false_alarms)
    hss_denominator = (
        (hits + misses) * (misses + correct_negatives)
        + (hits + false_alarms) * (false_alarms + correct_negatives)
    )
    return {
        "csi": round(hits / csi_denominator, 6) if csi_denominator else 0.0,
        "pod": round(hits / (hits + misses), 6) if hits + misses else 0.0,
        "far": round(false_alarms / (hits + false_alarms), 6)
        if hits + false_alarms
        else 0.0,
        "hss": round(hss_numerator / hss_denominator, 6) if hss_denominator else 0.0,
    }


def _fraction_field(binary: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return binary.astype(np.float32)
    before = (window - 1) // 2
    after = window // 2
    padded = np.pad(binary.astype(np.float32), ((0, 0), (before, after), (before, after)))
    integral = np.pad(padded, ((0, 0), (1, 0), (1, 0))).cumsum(1).cumsum(2)
    sums = (
        integral[:, window:, window:]
        - integral[:, :-window, window:]
        - integral[:, window:, :-window]
        + integral[:, :-window, :-window]
    )
    return sums / float(window * window)


def _fss_terms(
    prediction: np.ndarray, target: np.ndarray, threshold: float, scale: int
) -> tuple[float, float]:
    valid = target != 255
    predicted_fraction = _fraction_field((prediction >= threshold) & valid, scale)
    observed_fraction = _fraction_field((target >= threshold) & valid, scale)
    difference = predicted_fraction[valid] - observed_fraction[valid]
    numerator = float(np.square(difference).sum(dtype=np.float64))
    denominator = float(
        (np.square(predicted_fraction[valid]) + np.square(observed_fraction[valid])).sum(
            dtype=np.float64
        )
    )
    return numerator, denominator


class VerificationMetrics(WorldComponent):
    """Compute multi-threshold categorical and neighborhood verification.

    CSI/POD/FAR/HSS use all valid pixels. FSS uses square neighborhoods in
    pixels. Ground-truth value 255 is excluded. Scores describe the local
    cached events, not full SEVIR. Pixel scales are not converted to kilometres.
    """

    def raw_pair(
        self,
        event_id: str,
        lead_minutes: list[int],
        prediction_source: str = "ensemble_mean",
        member: int = 1,
    ) -> dict[str, Any]:
        """Return aligned raw cached-event prediction/truth arrays.

        This is the open-ended verification primitive. Generated code chooses
        RMSE, MAE, bias, correlation, threshold scores, spatial summaries, or
        any other statistic. At most six leads are returned per call to bound
        worker memory. Calling this method is an explicit request to load held-
        out truth; forecasting and uncertainty methods do not need to do so.
        """
        arrays, metadata = ensemble._load_cascast_ensemble(str(event_id))
        prediction, prediction_provenance = _prediction_from_arrays(
            arrays, prediction_source, member
        )
        target = np.asarray(arrays["ground_truth"], dtype=np.uint8)
        selected, indices = _selected_lead_indices(
            lead_minutes, min(int(prediction.shape[0]), int(target.shape[0]))
        )
        result = _raw_pair_result(
            identifier={"event_id": str(event_id)},
            prediction=prediction,
            truth=target[indices],
            selected_leads=selected,
            indices=indices,
            source_metadata={
                "cache_file": metadata.get("cache_file"),
                "truth_source": metadata.get("cache_file"),
            },
            prediction_provenance=prediction_provenance,
        )
        ensemble_count = int(np.asarray(arrays["ensemble_members"]).shape[0])
        member_seeds = [
            int(value)
            for value in np.asarray(arrays.get("member_seed", []), dtype=np.int64).tolist()
        ]
        model_name = (
            "persistence_baseline"
            if prediction_provenance["prediction_source"] == "persistence"
            else "cascast"
        )
        evidence_ensemble_count = 1 if model_name == "persistence_baseline" else ensemble_count
        horizon_hours = float(prediction.shape[0] * 5 / 60.0)
        evidence_id = _forecast_evidence_id(
            subject_id=str(event_id),
            model_name=model_name,
            ensemble_members=evidence_ensemble_count,
            horizon_hours=horizon_hours,
            prediction_source=metadata.get("cache_file"),
        )
        result.update({
            "forecast_evidence_id": evidence_id,
            "forecast_evidence": {
                "id": evidence_id,
                "subject_kind": "catalog_event",
                "subject_id": str(event_id),
                "model_name": model_name,
                "ensemble_count": evidence_ensemble_count,
                "member_seeds": member_seeds if model_name == "cascast" else [],
                "horizon_hours": horizon_hours,
                "lead_minutes": list(range(5, int(prediction.shape[0]) * 5 + 5, 5)),
                "selected_lead_minutes": selected,
                "prediction_source": metadata.get("cache_file"),
                "ground_truth_attached": True,
            },
        })
        self._recorder(
            "metrics.raw_pair",
            {
                "event_id": event_id,
                "lead_minutes": lead_minutes,
                "prediction_source": prediction_source,
                "member": member,
            },
            result,
        )
        return result

    def uploaded_raw_pair(
        self,
        upload_id: str,
        lead_minutes: list[int],
        ensemble_members: int = 10,
        horizon_hours: float = 1.0,
        prediction_source: str = "ensemble_mean",
        member: int = 1,
    ) -> dict[str, Any]:
        """Return aligned upload forecast and exact-match held-out truth grids.

        Truth is loaded only because this verification method was explicitly
        called. The upload history must exactly match one local SEVIR HDF5 row.
        The exact Agent-facing array keys are ``prediction_frames``,
        ``truth_frames`` and ``valid_mask``; the aligned selected times are the
        top-level ``lead_minutes`` list. There are no ``prediction``,
        ``predictions``, ``truth`` or ``ground_truth`` array aliases. At most
        six requested leads are returned. Leads beyond the matched segment's
        event boundary are excluded from the arrays and listed as unverified.
        The t=0 history frame is never substituted for missing future truth.
        """
        horizon = float(horizon_hours)
        if horizon not in {1.0, 2.0, 3.0}:
            raise ValueError("horizon_hours must be 1.0, 2.0, or 3.0")
        rolling_segments = int(horizon)
        run = run_uploaded_cascast_inference(
            upload_id,
            ensemble_members=int(ensemble_members),
            rolling_segments=rolling_segments,
        )
        arrays, metadata = load_uploaded_cascast_ensemble(
            upload_id,
            ensemble_members=int(ensemble_members),
            rolling_segments=rolling_segments,
        )
        prediction, prediction_provenance = _prediction_from_arrays(
            arrays, prediction_source, member
        )
        selected, _ = _selected_lead_indices(lead_minutes, int(prediction.shape[0]))
        truth_payload = load_uploaded_sequence_hdf5_ground_truth(upload_id, selected)
        if not truth_payload.get("available"):
            raise ValueError(
                str(truth_payload.get("reason") or "held-out ground truth is unavailable")
            )
        verified_leads = [
            int(value) for value in truth_payload.get("lead_minutes", selected)
        ]
        verified_indices = [value // 5 - 1 for value in verified_leads]
        truth = np.asarray(truth_payload["frames"], dtype=np.uint8)
        result = _raw_pair_result(
            identifier={
                "upload_id": upload_id,
                "matched_event_id": truth_payload.get("event_id"),
                "horizon_hours": horizon,
                "rolling_segments": rolling_segments,
                "cache_hit": bool(run.get("cache_hit", False)),
            },
            prediction=prediction,
            truth=truth,
            selected_leads=verified_leads,
            indices=verified_indices,
            source_metadata={
                "cache_file": metadata.get("cache_file"),
                "truth_source": truth_payload.get("source"),
            },
            prediction_provenance=prediction_provenance,
        )
        ensemble_count = int(np.asarray(arrays["ensemble_members"]).shape[0])
        member_seeds = [
            int(value)
            for value in np.asarray(arrays.get("member_seed", []), dtype=np.int64).tolist()
        ]
        model_name = (
            "persistence_baseline"
            if prediction_provenance["prediction_source"] == "persistence"
            else "cascast_live"
        )
        evidence_ensemble_count = 1 if model_name == "persistence_baseline" else ensemble_count
        evidence_id = _forecast_evidence_id(
            subject_id=upload_id,
            model_name=model_name,
            ensemble_members=evidence_ensemble_count,
            horizon_hours=horizon,
            prediction_source=metadata.get("cache_file"),
        )
        result.update({
            "sample_identity": {
                key: truth_payload.get(key)
                for key in ("event_id", "h5_file", "file_index", "segment")
            },
            "validation_manifest": truth_payload.get("validation_manifest"),
            "requested_lead_minutes": selected,
            "verified_lead_minutes": verified_leads,
            "unverified_lead_minutes": truth_payload.get("unverified_lead_minutes", []),
            "forecast_evidence_id": evidence_id,
            "forecast_evidence": {
                "id": evidence_id,
                "subject_kind": "uploaded_sequence",
                "subject_id": upload_id,
                "model_name": model_name,
                "ensemble_count": evidence_ensemble_count,
                "member_seeds": member_seeds if model_name == "cascast_live" else [],
                "horizon_hours": horizon,
                "rolling_segments": rolling_segments,
                "lead_minutes": list(range(5, int(prediction.shape[0]) * 5 + 5, 5)),
                "selected_lead_minutes": verified_leads,
                "unverified_lead_minutes": truth_payload.get("unverified_lead_minutes", []),
                "prediction_source": metadata.get("cache_file"),
                "cache_hit": bool(run.get("cache_hit", False)),
                "ground_truth_attached": True,
                "ground_truth_event_id": truth_payload.get("event_id"),
            },
        })
        self._recorder(
            "metrics.uploaded_raw_pair",
            {
                "upload_id": upload_id,
                "lead_minutes": lead_minutes,
                "ensemble_members": ensemble_members,
                "horizon_hours": horizon_hours,
                "prediction_source": prediction_source,
                "member": member,
            },
            result,
        )
        return result

    def evaluate_uploaded(
        self,
        upload_id: str,
        threshold: float = 133.0,
        ensemble_members: int = 10,
        horizon_hours: float = 3.0,
        prediction_source: str = "ensemble_mean",
        member: int = 1,
    ) -> dict[str, Any]:
        """Evaluate every available lead and label event-boundary overflow.

        Multi-hour CasCast upload forecasts use recursive free-running output.
        The matched event and segment determine which requested leads have
        observed future frames; metrics are never emitted for the remainder.
        """
        horizon = float(horizon_hours)
        if horizon not in {1.0, 2.0, 3.0}:
            raise ValueError("horizon_hours must be 1.0, 2.0, or 3.0")
        rolling_segments = int(horizon)
        run = run_uploaded_cascast_inference(
            upload_id,
            ensemble_members=int(ensemble_members),
            rolling_segments=rolling_segments,
        )
        arrays, metadata = load_uploaded_cascast_ensemble(
            upload_id,
            ensemble_members=int(ensemble_members),
            rolling_segments=rolling_segments,
        )
        prediction, prediction_provenance = _prediction_from_arrays(
            arrays, prediction_source, member
        )
        requested_leads = list(range(5, int(prediction.shape[0]) * 5 + 5, 5))
        truth_payload = load_uploaded_sequence_hdf5_ground_truth(
            upload_id, requested_leads
        )
        if not truth_payload.get("available"):
            raise ValueError(
                str(truth_payload.get("reason") or "held-out ground truth is unavailable")
            )
        verified_leads = [int(value) for value in truth_payload["lead_minutes"]]
        truth = np.asarray(truth_payload["frames"], dtype=np.uint8)
        verified_prediction = np.asarray(
            prediction[[lead // 5 - 1 for lead in verified_leads]], dtype=np.float32
        )
        threshold_value = float(threshold)
        per_lead: list[dict[str, Any]] = []
        aggregate_counts = {
            key: 0
            for key in ("hits", "misses", "false_alarms", "correct_negatives", "valid")
        }
        absolute_error_sum = 0.0
        signed_error_sum = 0.0
        valid_total = 0
        for lead, forecast_frame, truth_frame in zip(
            verified_leads, verified_prediction, truth
        ):
            valid = truth_frame != 255
            error = forecast_frame[valid] - truth_frame[valid].astype(np.float32)
            counts = _contingency(forecast_frame, truth_frame, threshold_value)
            for key, value in counts.items():
                aggregate_counts[key] += value
            absolute_error_sum += float(np.abs(error).sum(dtype=np.float64))
            signed_error_sum += float(error.sum(dtype=np.float64))
            valid_total += int(error.size)
            per_lead.append({
                "lead_minutes": lead,
                "absolute_frame_index": truth_payload["hdf5_frame_indices"][len(per_lead)],
                "mae": round(float(np.abs(error).mean()), 6) if error.size else None,
                "bias": round(float(error.mean()), 6) if error.size else None,
                **_scores(counts),
                "valid_pixel_count": int(error.size),
            })
        result = {
            "status": "ok",
            "task": "uploaded_cascast_accuracy_evaluation",
            "upload_id": upload_id,
            "sample_identity": {
                key: truth_payload.get(key)
                for key in ("event_id", "h5_file", "file_index", "segment")
            },
            "forecast_mode": "free_running" if rolling_segments > 1 else "standard_single_window",
            "horizon_hours": horizon,
            "rolling_segments": rolling_segments,
            "threshold": threshold_value,
            "requested_lead_minutes": requested_leads,
            "verified_lead_minutes": verified_leads,
            "unverified_lead_minutes": truth_payload.get("unverified_lead_minutes", []),
            "max_valid_lead_minutes": truth_payload.get("max_valid_lead_minutes"),
            "ground_truth_complete_for_horizon": not truth_payload.get(
                "unverified_lead_minutes", []
            ),
            "validation_manifest": truth_payload.get("validation_manifest"),
            "aggregate": {
                "mae": round(absolute_error_sum / valid_total, 6) if valid_total else None,
                "bias": round(signed_error_sum / valid_total, 6) if valid_total else None,
                **_scores(aggregate_counts),
                "valid_pixel_count": valid_total,
            },
            "per_lead": per_lead,
            **prediction_provenance,
            "prediction_source_file": metadata.get("cache_file"),
            "cache_hit": bool(run.get("cache_hit", False)),
            "missing_truth_value": 255,
            "interpretation_limit": (
                "Metrics apply only to exact event/segment-aligned observed frames. "
                "Leads listed as unverified have no Ground Truth and are forecast display only."
            ),
        }
        self._recorder(
            "metrics.evaluate_uploaded",
            {
                "upload_id": upload_id,
                "threshold": threshold,
                "ensemble_members": ensemble_members,
                "horizon_hours": horizon_hours,
                "prediction_source": prediction_source,
                "member": member,
            },
            result,
        )
        return result

    def evaluate_event(
        self,
        event_id: str,
        thresholds: list[float] | None = None,
        fss_scales: list[int] | None = None,
    ) -> dict[str, Any]:
        """Evaluate CasCast ensemble mean and Persistence for one cached event."""
        arrays, _ = ensemble._load_cascast_ensemble(event_id)
        target = arrays["ground_truth"].astype(np.float32)
        predictions = {
            "cascast_ensemble_mean": arrays["ensemble_mean"].astype(np.float32),
            "persistence": np.repeat(arrays["history"][-1][None], target.shape[0], axis=0).astype(
                np.float32
            ),
        }
        threshold_values = [float(value) for value in (thresholds or DEFAULT_THRESHOLDS)]
        scale_values = [int(value) for value in (fss_scales or DEFAULT_SCALES)]
        models: dict[str, Any] = {}
        for model_name, prediction in predictions.items():
            by_threshold: dict[str, Any] = {}
            for threshold in threshold_values:
                counts = _contingency(prediction, target, threshold)
                fss = {}
                for scale in scale_values:
                    numerator, denominator = _fss_terms(prediction, target, threshold, scale)
                    fss[str(scale)] = round(1.0 - numerator / denominator, 6) if denominator else 1.0
                per_lead = []
                for index in range(target.shape[0]):
                    lead_counts = _contingency(prediction[index], target[index], threshold)
                    per_lead.append(
                        {
                            "lead_minutes": (index + 1) * 5,
                            **_scores(lead_counts),
                        }
                    )
                by_threshold[str(int(threshold) if threshold.is_integer() else threshold)] = {
                    **counts,
                    **_scores(counts),
                    "fss": fss,
                    "per_lead": per_lead,
                }
            models[model_name] = by_threshold
        result = {
            "status": "ok",
            "task": "multi_threshold_verification",
            "capabilities": ["standard_categorical_scores", "pixel_neighborhood_fss"],
            "provenance": {
                "prediction": "saved CasCast ensemble cache",
                "truth": "saved aligned SEVIR held-out observations",
                "truth_was_loaded_explicitly_for_verification": True,
                "spatial_units": "pixels",
            },
            "scope": "single_cached_event",
            "event_id": event_id,
            "thresholds": threshold_values,
            "fss_scales_pixels": scale_values,
            "models": models,
            "missing_value_code": 255,
        }
        self._recorder(
            "metrics.evaluate_event",
            {"event_id": event_id, "thresholds": thresholds, "fss_scales": fss_scales},
            result,
        )
        return result

    def evaluate_dataset(
        self,
        thresholds: list[float] | None = None,
        fss_scales: list[int] | None = None,
    ) -> dict[str, Any]:
        """Aggregate CSI/HSS/FSS across all ten cached events."""
        threshold_values = [float(value) for value in (thresholds or DEFAULT_THRESHOLDS)]
        scale_values = [int(value) for value in (fss_scales or DEFAULT_SCALES)]
        model_names = ("cascast_ensemble_mean", "persistence")
        aggregate = {
            model: {
                threshold: {
                    "counts": {key: 0 for key in ("hits", "misses", "false_alarms", "correct_negatives", "valid")},
                    "fss": {scale: [0.0, 0.0] for scale in scale_values},
                }
                for threshold in threshold_values
            }
            for model in model_names
        }
        event_ids = [
            item["event_id"] for item in ensemble._ensemble_manifest()["events"]
        ]
        for event_id in event_ids:
            arrays, _ = ensemble._load_cascast_ensemble(event_id)
            target = arrays["ground_truth"].astype(np.float32)
            predictions = {
                "cascast_ensemble_mean": arrays["ensemble_mean"].astype(np.float32),
                "persistence": np.repeat(arrays["history"][-1][None], target.shape[0], axis=0).astype(
                    np.float32
                ),
            }
            for model, prediction in predictions.items():
                for threshold in threshold_values:
                    counts = _contingency(prediction, target, threshold)
                    for key, value in counts.items():
                        aggregate[model][threshold]["counts"][key] += value
                    for scale in scale_values:
                        numerator, denominator = _fss_terms(prediction, target, threshold, scale)
                        aggregate[model][threshold]["fss"][scale][0] += numerator
                        aggregate[model][threshold]["fss"][scale][1] += denominator
        models: dict[str, Any] = {}
        for model in model_names:
            models[model] = {}
            for threshold in threshold_values:
                item = aggregate[model][threshold]
                fss = {
                    str(scale): round(1.0 - terms[0] / terms[1], 6) if terms[1] else 1.0
                    for scale, terms in item["fss"].items()
                }
                key = str(int(threshold) if threshold.is_integer() else threshold)
                models[model][key] = {**item["counts"], **_scores(item["counts"]), "fss": fss}
        result = {
            "status": "ok",
            "task": "multi_threshold_verification",
            "capabilities": ["standard_categorical_scores", "pixel_neighborhood_fss"],
            "provenance": {
                "prediction": "saved CasCast ensemble cache",
                "truth": "saved aligned SEVIR held-out observations",
                "truth_was_loaded_explicitly_for_verification": True,
                "spatial_units": "pixels",
            },
            "scope": "ten_cached_events",
            "event_count": len(event_ids),
            "thresholds": threshold_values,
            "fss_scales_pixels": scale_values,
            "models": models,
            "missing_value_code": 255,
        }
        self._recorder(
            "metrics.evaluate_dataset",
            {"thresholds": thresholds, "fss_scales": fss_scales},
            result,
        )
        return result
