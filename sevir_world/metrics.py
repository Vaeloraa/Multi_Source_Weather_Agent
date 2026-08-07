from __future__ import annotations

from typing import Any

import numpy as np

import sevir_tools

from .base import WorldComponent


DEFAULT_THRESHOLDS = (16.0, 74.0, 133.0, 160.0, 181.0, 219.0)
DEFAULT_SCALES = (1, 4, 8, 16, 32)


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
    pixels; SEVIR VIL is approximately 1 km per pixel. Ground-truth value 255
    is excluded. Scores describe the local ten-event cache, not full SEVIR.
    """

    def evaluate_event(
        self,
        event_id: str,
        thresholds: list[float] | None = None,
        fss_scales: list[int] | None = None,
    ) -> dict[str, Any]:
        """Evaluate CasCast ensemble mean and Persistence for one cached event."""
        arrays, _ = sevir_tools._load_cascast_ensemble(event_id)
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
            item["event_id"] for item in sevir_tools._ensemble_manifest()["events"]
        ]
        for event_id in event_ids:
            arrays, _ = sevir_tools._load_cascast_ensemble(event_id)
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
