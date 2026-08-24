from __future__ import annotations

import base64
import binascii
import csv
import json
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..data_access.config import *
from ..domain.thresholds import DEFAULT_CASCAST_VIL_THRESHOLD
from ..data_access.ensemble import _ensemble_manifest, _load_cascast_ensemble
from .forecast import (
    _deterministic_metrics,
    _storm_object_verification,
    _storm_tracks,
)

import h5py
import numpy as np
from pyproj import CRS, Transformer


def _empty_metric_accumulator() -> dict[str, float]:
    return {
        "valid": 0.0,
        "hits": 0.0,
        "misses": 0.0,
        "false_alarms": 0.0,
        "squared_error": 0.0,
        "absolute_error": 0.0,
    }


def _accumulate_metrics(
    accumulator: dict[str, float],
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> None:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    valid = target != 255
    difference = prediction[valid] - target[valid]
    predicted = (prediction >= threshold) & valid
    observed = (target >= threshold) & valid
    accumulator["valid"] += float(valid.sum())
    accumulator["hits"] += float(np.logical_and(predicted, observed).sum())
    accumulator["misses"] += float(np.logical_and(~predicted, observed).sum())
    accumulator["false_alarms"] += float(
        np.logical_and(predicted, ~observed & valid).sum()
    )
    accumulator["squared_error"] += float(np.square(difference).sum(dtype=np.float64))
    accumulator["absolute_error"] += float(np.abs(difference).sum(dtype=np.float64))


def _finalize_metrics(accumulator: dict[str, float], threshold: float) -> dict[str, Any]:
    valid = int(accumulator["valid"])
    hits = int(accumulator["hits"])
    misses = int(accumulator["misses"])
    false_alarms = int(accumulator["false_alarms"])
    mse = accumulator["squared_error"] / valid if valid else 0.0
    mae = accumulator["absolute_error"] / valid if valid else 0.0
    return {
        "metric_threshold": float(threshold),
        "valid_pixel_count": valid,
        "csi": round(hits / (hits + misses + false_alarms), 6)
        if hits + misses + false_alarms
        else 0.0,
        "pod": round(hits / (hits + misses), 6) if hits + misses else 0.0,
        "far": round(false_alarms / (hits + false_alarms), 6)
        if hits + false_alarms
        else 0.0,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "mse_raw_vil": round(mse, 4),
        "rmse_raw_vil": round(float(np.sqrt(mse)), 4),
        "mae_raw_vil": round(mae, 4),
        "mse_normalized_0_1": round(mse / (255.0**2), 8),
    }


def build_cascast_evaluation_report(primary_threshold: float = DEFAULT_CASCAST_VIL_THRESHOLD) -> dict[str, Any]:
    """Evaluate all locally available CasCast events against their aligned ground truth."""
    manifest = _ensemble_manifest()
    thresholds = list(SEVIR_EVALUATION_THRESHOLDS)
    if float(primary_threshold) not in thresholds:
        thresholds.append(float(primary_threshold))
        thresholds.sort()
    model_accumulators = {
        model: {threshold: _empty_metric_accumulator() for threshold in thresholds}
        for model in ("cascast_ensemble_mean", "persistence")
    }
    lead_accumulators = {
        model: {
            threshold: [_empty_metric_accumulator() for _ in range(12)]
            for threshold in thresholds
        }
        for model in ("cascast_ensemble_mean", "persistence")
    }
    member_accumulators = [_empty_metric_accumulator() for _ in range(int(manifest["members_per_sample"]))]
    reliability = {
        threshold: [{"count": 0, "observed": 0} for _ in range(11)]
        for threshold in thresholds
    }
    rank_histogram = np.zeros(int(manifest["members_per_sample"]) + 1, dtype=np.int64)
    coverage_count = interval_valid = 0
    interval_width_sum = 0.0
    spread_sum = squared_error_sum = 0.0
    correlation = {key: 0.0 for key in ("n", "sum_x", "sum_y", "sum_xx", "sum_yy", "sum_xy")}
    event_results: list[dict[str, Any]] = []
    aggregate_object = {"hits": 0, "misses": 0, "false_alarms": 0, "centroid_error_sum": 0.0, "centroid_error_count": 0}

    for event in manifest["events"]:
        arrays, metadata = _load_cascast_ensemble(event["event_id"])
        members = arrays["ensemble_members"].astype(np.float32)
        mean_prediction = arrays["ensemble_mean"].astype(np.float32)
        target = arrays["ground_truth"].astype(np.float32)
        persistence = np.repeat(arrays["history"][-1][None, :, :], target.shape[0], axis=0).astype(np.float32)
        valid = target != 255

        for threshold in thresholds:
            for model, prediction in (
                ("cascast_ensemble_mean", mean_prediction),
                ("persistence", persistence),
            ):
                _accumulate_metrics(model_accumulators[model][threshold], prediction, target, threshold)
                for frame_index in range(target.shape[0]):
                    _accumulate_metrics(
                        lead_accumulators[model][threshold][frame_index],
                        prediction[frame_index],
                        target[frame_index],
                        threshold,
                    )
            probability = (members >= threshold).mean(axis=0)
            observed = target >= threshold
            probability_index = np.clip(np.rint(probability * 10).astype(np.int8), 0, 10)
            for bin_index in range(11):
                selected = (probability_index == bin_index) & valid
                reliability[threshold][bin_index]["count"] += int(selected.sum())
                reliability[threshold][bin_index]["observed"] += int(observed[selected].sum())

        for member_index, member in enumerate(members):
            _accumulate_metrics(member_accumulators[member_index], member, target, primary_threshold)

        q10 = arrays["q10"].astype(np.float32)
        q90 = arrays["q90"].astype(np.float32)
        interval_valid += int(valid.sum())
        coverage_count += int(((target >= q10) & (target <= q90) & valid).sum())
        interval_width_sum += float((q90 - q10)[valid].sum(dtype=np.float64))
        std = arrays["pixel_std"].astype(np.float32)[valid].astype(np.float64)
        absolute_error = np.abs(mean_prediction[valid] - target[valid]).astype(np.float64)
        spread_sum += float(std.sum())
        squared_error_sum += float(np.square(mean_prediction[valid] - target[valid]).sum(dtype=np.float64))
        correlation["n"] += float(std.size)
        correlation["sum_x"] += float(std.sum())
        correlation["sum_y"] += float(absolute_error.sum())
        correlation["sum_xx"] += float(np.square(std).sum())
        correlation["sum_yy"] += float(np.square(absolute_error).sum())
        correlation["sum_xy"] += float((std * absolute_error).sum())
        ranks = np.sum(members < target[None, :, :, :], axis=0)
        rank_histogram += np.bincount(ranks[valid].ravel(), minlength=len(rank_histogram))

        predicted_tracks = _storm_tracks(mean_prediction, metadata, primary_threshold)
        observed_tracks = _storm_tracks(target, metadata, primary_threshold)
        object_verification = _storm_object_verification(
            mean_prediction, target, metadata, primary_threshold
        )
        for key in ("hits", "misses", "false_alarms"):
            aggregate_object[key] += int(object_verification[key])
        if object_verification["mean_matched_centroid_error_pixels"] is not None:
            matched = int(object_verification["hits"])
            aggregate_object["centroid_error_sum"] += float(
                object_verification["mean_matched_centroid_error_pixels"] * matched
            )
            aggregate_object["centroid_error_count"] += matched
        event_results.append(
            {
                "event_id": metadata["event_id"],
                "sample_index": metadata["sample_index"],
                "event_type": metadata.get("event_type") or "Unlabeled",
                "cascast": _deterministic_metrics(
                    mean_prediction.transpose(1, 2, 0),
                    target.transpose(1, 2, 0),
                    primary_threshold,
                ),
                "persistence": _deterministic_metrics(
                    persistence.transpose(1, 2, 0),
                    target.transpose(1, 2, 0),
                    primary_threshold,
                ),
                "predicted_storm_tracks": predicted_tracks,
                "observed_storm_tracks": observed_tracks,
                "storm_object_verification": object_verification,
            }
        )

    n = correlation["n"]
    numerator = n * correlation["sum_xy"] - correlation["sum_x"] * correlation["sum_y"]
    denominator = np.sqrt(
        max(n * correlation["sum_xx"] - correlation["sum_x"] ** 2, 0.0)
        * max(n * correlation["sum_yy"] - correlation["sum_y"] ** 2, 0.0)
    )
    member_metrics = [
        _finalize_metrics(accumulator, primary_threshold) for accumulator in member_accumulators
    ]
    member_metric_summary = {
        key: {
            "min": round(min(float(item[key]) for item in member_metrics), 6),
            "mean": round(float(np.mean([float(item[key]) for item in member_metrics])), 6),
            "max": round(max(float(item[key]) for item in member_metrics), 6),
        }
        for key in ("csi", "pod", "far", "mse_raw_vil", "mae_raw_vil")
    }
    object_denominator = aggregate_object["hits"] + aggregate_object["misses"] + aggregate_object["false_alarms"]
    object_pod_denominator = aggregate_object["hits"] + aggregate_object["misses"]
    object_far_denominator = aggregate_object["hits"] + aggregate_object["false_alarms"]
    report = {
        "status": "ok",
        "task": "cascast_dataset_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "10 aligned SEVIR VIL test events with CasCast ensembles and ground truth",
        "event_count": len(event_results),
        "ensemble_count": int(manifest["members_per_sample"]),
        "forecast_frames": 12,
        "lead_minutes": 60,
        "primary_threshold": float(primary_threshold),
        "thresholds": thresholds,
        "aggregate_metrics": {
            model: {
                str(int(threshold) if threshold.is_integer() else threshold): _finalize_metrics(accumulator, threshold)
                for threshold, accumulator in by_threshold.items()
            }
            for model, by_threshold in model_accumulators.items()
        },
        "lead_time_metrics": {
            model: {
                str(int(threshold) if threshold.is_integer() else threshold): [
                    {"lead_minutes": (index + 1) * 5, **_finalize_metrics(accumulator, threshold)}
                    for index, accumulator in enumerate(accumulators)
                ]
                for threshold, accumulators in by_threshold.items()
            }
            for model, by_threshold in lead_accumulators.items()
        },
        "ensemble_member_performance": {
            "threshold": float(primary_threshold),
            "members": [
                {"member": index + 1, **metrics} for index, metrics in enumerate(member_metrics)
            ],
            "summary": member_metric_summary,
        },
        "ensemble_validation": {
            "mean_pixel_std": round(spread_sum / interval_valid, 4) if interval_valid else 0.0,
            "ensemble_mean_rmse": round(np.sqrt(squared_error_sum / interval_valid), 4) if interval_valid else 0.0,
            "spread_skill_ratio": round(
                (spread_sum / interval_valid) / np.sqrt(squared_error_sum / interval_valid), 6
            ) if interval_valid and squared_error_sum else 0.0,
            "pixel_std_absolute_error_pearson": round(float(numerator / denominator), 6)
            if denominator else 0.0,
            "central_80_interval_coverage": round(coverage_count / interval_valid, 6)
            if interval_valid else 0.0,
            "mean_central_80_interval_width": round(interval_width_sum / interval_valid, 4)
            if interval_valid else 0.0,
            "rank_histogram": rank_histogram.astype(int).tolist(),
            "rank_bins": list(range(len(rank_histogram))),
            "reliability": {
                str(int(threshold) if threshold.is_integer() else threshold): [
                    {
                        "forecast_probability": bin_index / 10,
                        "pixel_count": item["count"],
                        "observed_frequency": round(item["observed"] / item["count"], 6)
                        if item["count"] else None,
                    }
                    for bin_index, item in enumerate(bins)
                ]
                for threshold, bins in reliability.items()
            },
            "interpretation": (
                "spread-error相关、区间覆盖率、spread-skill ratio、可靠性曲线和rank histogram"
                "均由10路ensemble与真实未来帧共同计算；它们比只展示标准差更能检验不确定性是否可信。"
            ),
        },
        "storm_object_evaluation": {
            "threshold": float(primary_threshold),
            "minimum_region_pixels": 1,
            "hits": aggregate_object["hits"],
            "misses": aggregate_object["misses"],
            "false_alarms": aggregate_object["false_alarms"],
            "object_csi": round(aggregate_object["hits"] / object_denominator, 6)
            if object_denominator else 0.0,
            "object_pod": round(aggregate_object["hits"] / object_pod_denominator, 6)
            if object_pod_denominator else 0.0,
            "object_far": round(aggregate_object["false_alarms"] / object_far_denominator, 6)
            if object_far_denominator else 0.0,
            "mean_matched_centroid_error_pixels": round(
                aggregate_object["centroid_error_sum"] / aggregate_object["centroid_error_count"], 3
            ) if aggregate_object["centroid_error_count"] else None,
            "scope": "connected-region tracks and framewise bbox-IoU object verification",
        },
        "events": event_results,
        "interpretation_limit": (
            "这是当前本地10个测试事件的聚合评价，不是SEVIR完整测试集论文复现结果；"
            "storm-object匹配使用连通域外接框IoU和中心距离，尚未处理复杂分裂与合并。"
        ),
    }
    return report


def _evaluation_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    threshold_key = str(int(float(report["primary_threshold"])))
    validation = dict(report["ensemble_validation"])
    reliability = validation.pop("reliability", {})
    validation["reliability_at_primary_threshold"] = reliability.get(threshold_key, [])
    events = []
    for event in report["events"]:
        verification = dict(event["storm_object_verification"])
        verification.pop("per_frame", None)
        events.append(
            {
                "event_id": event["event_id"],
                "sample_index": event["sample_index"],
                "event_type": event["event_type"],
                "cascast": event["cascast"],
                "persistence": event["persistence"],
                "predicted_track_count": event["predicted_storm_tracks"]["track_count"],
                "observed_track_count": event["observed_storm_tracks"]["track_count"],
                "longest_predicted_track_minutes": event["predicted_storm_tracks"]["longest_track_duration_minutes"],
                "longest_observed_track_minutes": event["observed_storm_tracks"]["longest_track_duration_minutes"],
                "storm_object_verification": verification,
            }
        )
    return {
        key: report[key]
        for key in (
            "status", "task", "generated_at_utc", "data_source", "event_count",
            "ensemble_count", "forecast_frames", "lead_minutes", "primary_threshold",
            "thresholds", "aggregate_metrics", "ensemble_member_performance",
            "storm_object_evaluation", "interpretation_limit",
        )
    } | {
        "lead_time_metrics": {
            model: values[threshold_key] for model, values in report["lead_time_metrics"].items()
        },
        "ensemble_validation": validation,
        "events": events,
        "full_report_file": str(CASCAST_EVALUATION_REPORT.relative_to(ROOT)),
    }


def evaluate_cascast_dataset(primary_threshold: float = DEFAULT_CASCAST_VIL_THRESHOLD) -> dict[str, Any]:
    """Load a reproducible cached report and return an agent-friendly summary."""
    if CASCAST_EVALUATION_REPORT.exists():
        report = json.loads(CASCAST_EVALUATION_REPORT.read_text(encoding="utf-8"))
        if float(report.get("primary_threshold", -1)) == float(primary_threshold):
            return _evaluation_report_summary(report)
    report = build_cascast_evaluation_report(primary_threshold)
    CASCAST_EVALUATION_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return _evaluation_report_summary(report)
