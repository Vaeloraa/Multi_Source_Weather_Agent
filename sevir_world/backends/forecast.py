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

from ..repositories.config import *
from ..repositories.ensemble import (
    _cascast_source,
    _ensemble_manifest,
    _event,
    _load_cascast_ensemble,
    _source,
    _vil_display_values,
)
from ..repositories.uploads import _load_uploaded_vil, _load_uploaded_vil_sequence

import h5py
import numpy as np
from pyproj import CRS, Transformer


def analyze_uploaded_vil(
    upload_id: str,
    threshold: float,
    minimum_region_pixels: int = 16,
) -> dict[str, Any]:
    frame, metadata = _load_uploaded_vil(upload_id)
    mask = frame > threshold
    regions = [
        region
        for region in _hazard_regions(mask, metadata)
        if region["pixel_count"] >= minimum_region_pixels
    ]
    return {
        "status": "ok",
        "task": "uploaded_sevir_vil_analysis",
        "analysis_scope": "single_uploaded_frame",
        "upload_id": upload_id,
        "filename": metadata["filename"],
        "data_source": metadata["source"],
        "model_name": "uploaded_vil_analysis",
        "spatial_shape": [384, 384],
        "threshold": threshold,
        "min_vil": int(frame.min()),
        "max_vil": int(frame.max()),
        "mean_vil": round(float(frame.mean()), 4),
        "exceeding_pixel_count": int(mask.sum()),
        "exceeding_area_ratio": round(float(mask.mean()), 6),
        "hazard_detected": bool(regions),
        "hazard_region_count": len(regions),
        "hazard_regions": regions,
        "full_resolution_viewer": True,
        "grayscale_source": metadata["grayscale_source"],
        "interpretation_limit": (
            "这是用户上传的单张SEVIR VIL图像，只能分析当前强回波与空间分布，"
            "不能据此声称完成未来预测或持续时间分析。阈值是0到255缩放像素值，"
            "不是dBZ、降雨量或官方灾害预警。"
        ),
    }


def analyze_uploaded_vil_sequence(
    upload_id: str,
    threshold: float,
    minimum_region_pixels: int = 16,
) -> dict[str, Any]:
    sequence, metadata = _load_uploaded_vil_sequence(upload_id)
    history_summaries = [
        {
            "frame_index": index + 1,
            "relative_minutes": int(metadata["history_minutes"][index]),
            "min_vil": int(frame.min()),
            "max_vil": int(frame.max()),
            "mean_vil": round(float(frame.mean()), 3),
            "exceeding_area_ratio": round(float((frame >= threshold).mean()), 6),
        }
        for index, frame in enumerate(sequence)
    ]
    persistence = np.repeat(sequence[-1][None, :, :], 12, axis=0).astype(np.float32)
    current_regions = [
        item for item in _hazard_regions(sequence[-1] >= threshold, metadata)
        if item["pixel_count"] >= minimum_region_pixels
    ]
    tracks = _storm_tracks(sequence.astype(np.float32), metadata, threshold, minimum_region_pixels)
    return {
        "status": "ok",
        "task": "uploaded_sevir_vil_sequence_analysis",
        "analysis_scope": "13_uploaded_history_frames_plus_persistence_baseline",
        "upload_id": upload_id,
        "data_source": metadata["source"],
        "model_name": "persistence_baseline",
        "input_frames": 13,
        "forecast_frames": 12,
        "lead_minutes": 60,
        "threshold": threshold,
        "history_available": True,
        "history_frames": 13,
        "history_frame_summaries": history_summaries,
        "frame_summaries": _prediction_frame_summaries(persistence.transpose(1, 2, 0), threshold),
        "ground_truth_available": False,
        "ground_truth_frame_summaries": [],
        "hazard_detected": bool(current_regions),
        "hazard_region_count": len(current_regions),
        "hazard_regions": current_regions,
        "historical_storm_tracks": tracks,
        "full_resolution_viewer": True,
        "spatial_shape": [384, 384],
        "interpretation_limit": (
            "13张图仅用于历史演变分析；网页给出的未来12帧是最后观测帧持续不变的Persistence基线，"
            "不是CasCast。要对任意上传序列现场运行CasCast，还需EarthFormer生成确定性未来、"
            "Autoencoder压缩latent后再送入CasCast概率模型。"
        ),
    }


def get_event_summary(event_id: str) -> dict[str, Any]:
    vil, metadata = _event(event_id)
    return {
        "status": "ok",
        "data_source": _cascast_source() if model_name == "cascast" else _source(),
        "requested_event_id": event_id,
        "event_id": metadata["event_id"],
        "event_type": metadata.get("catalog_row", {}).get("event_type"),
        "channel": "VIL",
        "frame_count": int(vil.shape[-1]),
        "spatial_shape": [int(vil.shape[0]), int(vil.shape[1])],
        "min_vil": int(vil.min()),
        "max_vil": int(vil.max()),
        "mean_vil": round(float(vil.mean()), 4),
        "preview_grid": _preview(vil.max(axis=2)),
    }


def analyze_threshold(event_id: str, threshold: float) -> dict[str, Any]:
    vil, metadata = _event(event_id)
    mask = vil > threshold
    per_frame_ratio = mask.mean(axis=(0, 1))
    peak_index = int(per_frame_ratio.argmax())
    return {
        "status": "ok",
        "data_source": _source(),
        "event_id": metadata["event_id"],
        "threshold": threshold,
        "pixel_frame_count": int(mask.size),
        "exceeding_pixel_frame_count": int(mask.sum()),
        "exceeding_ratio": round(float(mask.mean()), 6),
        "peak_frame_index": peak_index,
        "peak_frame_exceeding_ratio": round(float(per_frame_ratio[peak_index]), 6),
        "preview_grid": _preview(vil[:, :, peak_index]),
        "interpretation_limit": (
            "这里使用SEVIR文件内0到255的缩放VIL像素值。阈值40是数据级阈值，"
            "不能直接写成40 dBZ，也不能直接等同于地面降雨量。"
        ),
    }


def analyze_hazard(
    event_id: str,
    threshold: float,
    min_area_ratio: float = 0.001,
) -> dict[str, Any]:
    vil, metadata = _event(event_id)
    per_frame_ratio = (vil > threshold).mean(axis=(0, 1))
    active = per_frame_ratio >= min_area_ratio
    longest = 0
    current = 0
    end_index = -1
    for index, is_active in enumerate(active.tolist()):
        current = current + 1 if is_active else 0
        if current > longest:
            longest = current
            end_index = index
    start_index = end_index - longest + 1 if longest else -1
    minutes_per_frame = 5
    return {
        "status": "ok",
        "data_source": _source(),
        "event_id": metadata["event_id"],
        "analysis_scope": "observed_vil_frames",
        "threshold": threshold,
        "minimum_affected_area_ratio": min_area_ratio,
        "active_frame_count": int(active.sum()),
        "longest_active_start_frame": start_index,
        "longest_active_end_frame": end_index,
        "longest_active_duration_minutes": int(longest * minutes_per_frame),
        "peak_affected_area_ratio": round(float(per_frame_ratio.max()), 6),
        "preview_grid": _preview(vil[:, :, int(per_frame_ratio.argmax())]),
        "interpretation_limit": (
            "这是基于观测VIL阈值的强回波筛查，不是未来预报，也不是官方灾害定级。"
        ),
    }


def _binary_csi(prediction: np.ndarray, target: np.ndarray, threshold: float) -> float:
    valid = target != 255
    predicted = (prediction >= threshold) & valid
    observed = (target >= threshold) & valid
    true_positive = np.logical_and(predicted, observed).sum()
    union = np.logical_or(predicted, observed).sum()
    return float(true_positive / union) if union else 0.0


def _deterministic_metrics(
    prediction: np.ndarray, target: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Evaluate raw CasCast ensemble mean using the official encoded VIL scale."""
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError(
            f"预测和ground truth形状不一致：{prediction.shape} vs {target.shape}"
        )
    valid = target != 255
    if not valid.any():
        raise ValueError("Ground truth没有可用于评价的有效像素。")
    differences = prediction[valid] - target[valid]
    mse = float(np.mean(differences**2))
    mae = float(np.mean(np.abs(differences)))
    predicted = (prediction >= threshold) & valid
    observed = (target >= threshold) & valid
    hits = int(np.logical_and(predicted, observed).sum())
    misses = int(np.logical_and(~predicted, observed).sum())
    false_alarms = int(np.logical_and(predicted, ~observed & valid).sum())
    csi_denominator = hits + misses + false_alarms
    pod_denominator = hits + misses
    far_denominator = hits + false_alarms
    return {
        "ground_truth_available": True,
        "prediction_value_space": "raw CasCast VAE-decoded float on encoded VIL scale",
        "ground_truth_value_space": "SEVIR uint8 encoded VIL",
        "missing_value_code": 255,
        "valid_pixel_count": int(valid.sum()),
        "metric_threshold": threshold,
        "csi_at_threshold": round(hits / csi_denominator, 6) if csi_denominator else 0.0,
        "pod_at_threshold": round(hits / pod_denominator, 6) if pod_denominator else 0.0,
        "far_at_threshold": round(false_alarms / far_denominator, 6) if far_denominator else 0.0,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "mse_raw_vil": round(mse, 4),
        "rmse_raw_vil": round(float(np.sqrt(mse)), 4),
        "mae_raw_vil": round(mae, 4),
        "mse_normalized_0_1": round(mse / (255.0**2), 8),
        "postprocessing": "raw prediction; no clipping before metrics",
    }


def _longest_true_run(active: np.ndarray) -> tuple[int, int, int]:
    """Return start index, end index and length of the longest consecutive True run."""
    longest = 0
    current = 0
    end_index = -1
    for index, value in enumerate(np.asarray(active, dtype=bool).tolist()):
        current = current + 1 if value else 0
        if current > longest:
            longest = current
            end_index = index
    start_index = end_index - longest + 1 if longest else -1
    return start_index, end_index, longest


def _persistence_prediction(
    vil: np.ndarray, input_frames: int, forecast_frames: int
) -> np.ndarray:
    last_observation = vil[:, :, input_frames - 1].astype(np.float32)
    return np.repeat(last_observation[:, :, None], forecast_frames, axis=2)


def _hazard_regions(
    mask: np.ndarray, metadata: dict[str, Any], max_regions: int = 3
) -> list[dict[str, Any]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for row, column in np.argwhere(mask):
        row_i, column_i = int(row), int(column)
        if visited[row_i, column_i]:
            continue
        queue = deque([(row_i, column_i)])
        visited[row_i, column_i] = True
        component: list[tuple[int, int]] = []
        while queue:
            current_row, current_column = queue.popleft()
            component.append((current_row, current_column))
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and mask[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    queue.append((next_row, next_column))
        components.append(component)

    catalog = metadata.get("catalog_row", {})
    try:
        min_lat = float(catalog["llcrnrlat"])
        min_lon = float(catalog["llcrnrlon"])
        max_lat = float(catalog["urcrnrlat"])
        max_lon = float(catalog["urcrnrlon"])
    except (KeyError, TypeError, ValueError):
        min_lat = min_lon = max_lat = max_lon = None

    regions = []
    for rank, component in enumerate(
        sorted(components, key=len, reverse=True)[:max_regions], start=1
    ):
        rows = [point[0] for point in component]
        columns = [point[1] for point in component]
        center_row = sum(rows) / len(rows)
        center_column = sum(columns) / len(columns)
        region: dict[str, Any] = {
            "region_id": f"H{rank}",
            "pixel_count": len(component),
            "area_ratio": round(len(component) / mask.size, 6),
            "pixel_bbox": [min(columns), min(rows), max(columns), max(rows)],
            "pixel_centroid": [round(center_column, 1), round(center_row, 1)],
        }
        if min_lat is not None:
            longitude = min_lon + center_column / max(width - 1, 1) * (max_lon - min_lon)
            latitude = max_lat - center_row / max(height - 1, 1) * (max_lat - min_lat)
            region["approx_centroid_latlon"] = [round(latitude, 4), round(longitude, 4)]
            region["location_note"] = "由SEVIR事件边界线性映射得到的近似坐标"
        regions.append(region)
    return regions


def _prepare_prediction_data(
    event_id: str,
    model_name: str,
    input_frames: int,
    forecast_frames: int,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None, dict[str, Any], np.ndarray | None]:
    """Load the shared prediction material without formatting a public tool result."""
    if input_frames < 1 or forecast_frames < 1:
        raise ValueError("输入帧数和预测帧数必须大于0。")
    if model_name == "cascast":
        if forecast_frames != 12:
            raise ValueError("当前CasCast输出仅支持未来60分钟（12帧）；2/3小时滚动推理尚未执行。")
        arrays, metadata = _load_cascast_ensemble(event_id)
        prediction = arrays["ensemble_mean"].astype(np.float32).transpose(1, 2, 0)
        target = arrays["ground_truth"].astype(np.float32).transpose(1, 2, 0)
        return prediction, arrays, metadata, target
    if model_name != "persistence_baseline":
        raise ValueError(f"不支持的预测模型：{model_name}")
    try:
        arrays, metadata = _load_cascast_ensemble(event_id)
    except ValueError:
        arrays = None
    if arrays is not None:
        history = arrays["history"].astype(np.float32)
        if input_frames > history.shape[0]:
            raise ValueError("当前CasCast事件没有足够的历史输入帧。")
        prediction = np.repeat(history[input_frames - 1][:, :, None], forecast_frames, axis=2)
        target = arrays["ground_truth"].astype(np.float32).transpose(1, 2, 0)
        return prediction, arrays, metadata, target
    vil, metadata = _event(event_id)
    if input_frames + forecast_frames > vil.shape[-1]:
        raise ValueError("当前事件没有足够帧数用于指定的输入与验证区间。")
    prediction = _persistence_prediction(vil, input_frames, forecast_frames)
    target = vil[:, :, input_frames : input_frames + forecast_frames].astype(np.float32)
    return prediction, None, metadata, target


def _prediction_frame_summaries(prediction: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": index + 1,
            "lead_minutes": (index + 1) * 5,
            "min_vil": round(float(frame.min()), 3),
            "max_vil": round(float(frame.max()), 3),
            "mean_vil": round(float(frame.mean()), 3),
            "exceeding_area_ratio": round(float((frame >= threshold).mean()), 6),
            "area_threshold": threshold,
        }
        for index, frame in enumerate(prediction.transpose(2, 0, 1))
    ]


def _history_frame_summaries(arrays: dict[str, np.ndarray] | None) -> list[dict[str, Any]]:
    if arrays is None:
        return []
    history_minutes = _ensemble_manifest().get("history_minutes", list(range(-60, 5, 5)))
    return [
        {
            "frame_index": index + 1,
            "relative_minutes": int(history_minutes[index]),
            "min_vil": int(frame.min()),
            "max_vil": int(frame.max()),
            "mean_vil": round(float(frame.mean()), 3),
        }
        for index, frame in enumerate(arrays["history"])
    ]


def run_nowcast(
    event_id: str,
    model_name: str,
    threshold: float,
    input_frames: int = 13,
    forecast_frames: int = 12,
) -> dict[str, Any]:
    prediction, arrays, metadata, target = _prepare_prediction_data(
        event_id, model_name, input_frames, forecast_frames
    )
    analysis_prediction = _vil_display_values(prediction)
    frame_summaries = _prediction_frame_summaries(analysis_prediction, threshold)
    if arrays is not None:
        result = {
            "status": "ok",
            "task": "nowcast",
            "data_source": _cascast_source(str(metadata["event_id"])),
            "event_id": metadata["event_id"],
            "model_name": model_name,
            "input_frames": input_frames,
            "forecast_frames": forecast_frames,
            "lead_minutes": forecast_frames * 5,
            "inference_mode": (
                "precomputed_multi_event_ensemble"
                if model_name == "cascast"
                else "persistence_from_matching_cascast_history"
            ),
            "ensemble_count": metadata.get("ensemble_count", 10) if model_name == "cascast" else None,
            "ground_truth_available": target is not None,
            "history_available": True,
            "history_frames": int(arrays["history"].shape[0]),
            "history_frame_summaries": _history_frame_summaries(arrays),
            "full_resolution_viewer": True,
            "spatial_shape": [384, 384],
            "min_vil": round(float(analysis_prediction.min()), 4),
            "max_vil": round(float(analysis_prediction.max()), 4),
            "mean_vil": round(float(analysis_prediction.mean()), 4),
            "frame_summaries": frame_summaries,
            "ground_truth_frame_summaries": _prediction_frame_summaries(target, threshold) if target is not None else [],
            "model_metrics": _deterministic_metrics(prediction, target, threshold) if target is not None else {"ground_truth_available": False},
            "interpretation_limit": "指标仅评价当前所选SEVIR测试样本；不能代表完整测试集总体性能。",
        }
        if model_name == "cascast":
            result.update(
                {
                    "available_event_count": int(_ensemble_manifest()["samples"]),
                    "code_available": (CASCAST_ROOT / "networks" / "casformer.py").exists(),
                    "checkpoint_available": (CASCAST_ROOT / "experiments" / "cascast_diffusion" / "world_size1-ckpt" / "checkpoint_best.pth").exists(),
                    "autoencoder_checkpoint_available": (CASCAST_ROOT / "ckpts" / "autoencoder" / "ckpt.pth").exists(),
                }
            )
        return result

    assert target is not None
    mse_raw = float(np.mean((prediction - target) ** 2))
    return {
        "status": "ok",
        "task": "nowcast",
        "data_source": _source(),
        "event_id": metadata["event_id"],
        "model_name": model_name,
        "input_frames": input_frames,
        "forecast_frames": forecast_frames,
        "ground_truth_available": True,
        "evaluation_target": "同一事件中紧随输入序列的真实VIL帧",
        "metric_threshold": threshold,
        "csi_at_threshold": round(_binary_csi(prediction, target, threshold), 6),
        "mse_raw_vil": round(mse_raw, 4),
        "mse_normalized_0_1": round(mse_raw / (255.0**2), 8),
        "frame_summaries": frame_summaries,
        "interpretation_limit": "这是可复现的persistence基线，不是CasCast结果。",
    }


def forecast_hazards(
    event_id: str,
    model_name: str,
    threshold: float,
    lead_minutes: int = 60,
    minimum_region_pixels: int = 16,
) -> dict[str, Any]:
    if lead_minutes not in {60, 120, 180}:
        raise ValueError("当前Demo支持未来1、2或3小时预测时效。")
    forecast_frames = lead_minutes // 5
    prediction, arrays, metadata, target = _prepare_prediction_data(
        event_id, model_name, 13, forecast_frames
    )
    analysis_prediction = _vil_display_values(prediction)
    frame_summaries = _prediction_frame_summaries(analysis_prediction, threshold)
    hazard_mask = analysis_prediction.max(axis=2) >= threshold
    regions = [
        region
        for region in _hazard_regions(hazard_mask, metadata)
        if region["pixel_count"] >= minimum_region_pixels
    ]
    for region in regions:
        min_x, min_y, max_x, max_y = region["pixel_bbox"]
        region_active = (
            analysis_prediction[min_y : max_y + 1, min_x : max_x + 1, :] >= threshold
        ).any(axis=(0, 1))
        start_index, end_index, longest = _longest_true_run(region_active)
        if longest:
            region["start_minute"] = int((start_index + 1) * 5)
            region["end_minute"] = int((end_index + 1) * 5)
            region["duration_minutes"] = int(longest * 5)
            region["total_active_minutes"] = int(region_active.sum() * 5)
            region["duration_is_lower_bound"] = bool(region_active.all())

    exceeding_pixel_counts = (analysis_prediction >= threshold).sum(axis=(0, 1))
    any_pixel_active = exceeding_pixel_counts > 0
    hazard_area_active = exceeding_pixel_counts >= minimum_region_pixels
    _, _, any_pixel_longest = _longest_true_run(any_pixel_active)
    _, _, hazard_longest = _longest_true_run(hazard_area_active)
    duration = int(hazard_longest * 5)
    return {
        "status": "ok",
        "data_source": _cascast_source(str(metadata["event_id"])) if arrays is not None else _source(),
        "event_id": metadata["event_id"],
        "task": "future_hazard_analysis",
        "model_name": model_name,
        "lead_minutes": lead_minutes,
        "threshold": threshold,
        "hazard_detected": bool(regions),
        "hazard_region_count": len(regions),
        "hazard_regions": regions,
        "threshold_exceeded": bool(any_pixel_active.any()),
        "maximum_exceeding_pixel_count": int(exceeding_pixel_counts.max()),
        "any_pixel_exceedance_total_minutes": int(any_pixel_active.sum() * 5),
        "any_pixel_exceedance_longest_consecutive_minutes": int(any_pixel_longest * 5),
        "hazard_area_minimum_pixels": minimum_region_pixels,
        "hazard_area_total_minutes": int(hazard_area_active.sum() * 5),
        "strong_echo_duration_minutes_within_horizon": duration,
        "duration_is_lower_bound": bool(hazard_area_active.all()),
        "frame_summaries": frame_summaries,
        "ground_truth_frame_summaries": _prediction_frame_summaries(target, threshold) if target is not None else [],
        "history_available": arrays is not None,
        "history_frames": int(arrays["history"].shape[0]) if arrays is not None else 0,
        "history_frame_summaries": _history_frame_summaries(arrays),
        "full_resolution_viewer": arrays is not None,
        "spatial_shape": [384, 384],
        "ensemble_count": metadata.get("ensemble_count") if model_name == "cascast" else None,
        "model_metrics": (
            {
                **_deterministic_metrics(prediction, target, threshold),
            }
            if target is not None
            else {"ground_truth_available": False}
        ),
        "interpretation_limit": (
            f"危险区域来自{model_name}预测的VIL强回波代理。"
            f"阈值{threshold:g}是0至255范围的缩放像素值，不是dBZ或降雨量；"
            "这不是官方灾害预警。"
        ),
    }


def analyze_ensemble_uncertainty(
    event_id: str,
    threshold: float,
    minimum_region_pixels: int = 16,
) -> dict[str, Any]:
    arrays, metadata = _load_cascast_ensemble(event_id)
    members = arrays["ensemble_members"].astype(np.float32)
    mean_prediction = arrays["ensemble_mean"]
    ground_truth = arrays["ground_truth"].astype(np.float32)
    pixel_std = arrays["pixel_std"]
    exceedance_probability = (members >= threshold).mean(axis=0)

    member_summaries: list[dict[str, Any]] = []
    object_centroids: list[list[float]] = []
    largest_object_area_ratios: list[float] = []
    for member_index, member in enumerate(members):
        per_frame_area = (member >= threshold).mean(axis=(1, 2))
        active_frames = (member >= threshold).sum(axis=(1, 2)) >= minimum_region_pixels
        regions = [
            region
            for region in _hazard_regions((member >= threshold).any(axis=0), metadata, max_regions=1)
            if region["pixel_count"] >= minimum_region_pixels
        ]
        largest = regions[0] if regions else None
        if largest:
            object_centroids.append(largest["pixel_centroid"])
            largest_object_area_ratios.append(float(largest["area_ratio"]))
        member_summaries.append(
            {
                "member": member_index + 1,
                "seed": int(arrays["member_seed"][member_index]),
                "max_vil": round(float(member.max()), 3),
                "mean_vil": round(float(member.mean()), 3),
                "peak_exceeding_area_ratio": round(float(per_frame_area.max()), 6),
                "active_duration_minutes": int(active_frames.sum() * 5),
                "largest_object_area_ratio": round(float(largest["area_ratio"]), 6) if largest else 0.0,
                "largest_object_centroid": largest["pixel_centroid"] if largest else None,
            }
        )

    if object_centroids:
        centroid_array = np.asarray(object_centroids, dtype=np.float32)
        centroid_spread = float(np.sqrt(np.var(centroid_array, axis=0).sum()))
    else:
        centroid_spread = 0.0

    uncertainty_frames = []
    for frame_index in range(12):
        frame_mean = mean_prediction[frame_index]
        frame_std = pixel_std[frame_index]
        frame_probability = exceedance_probability[frame_index]
        uncertainty_frames.append(
            {
                "frame_index": frame_index + 1,
                "lead_minutes": (frame_index + 1) * 5,
                "mean_vil": round(float(frame_mean.mean()), 3),
                "max_mean_vil": round(float(frame_mean.max()), 3),
                "max_vil": round(float(frame_mean.max()), 3),
                "mean_pixel_std": round(float(frame_std.mean()), 3),
                "p95_pixel_std": round(float(np.quantile(frame_std, 0.95)), 3),
                "expected_exceeding_area_ratio": round(float(frame_probability.mean()), 6),
                "exceeding_area_ratio": round(float(frame_probability.mean()), 6),
                "area_threshold": threshold,
                "max_exceedance_probability": round(float(frame_probability.max()), 3),
            }
        )

    member_areas = np.asarray(
        [item["peak_exceeding_area_ratio"] for item in member_summaries], dtype=np.float32
    )
    member_durations = np.asarray(
        [item["active_duration_minutes"] for item in member_summaries], dtype=np.float32
    )
    mean_field = mean_prediction.transpose(1, 2, 0)
    mean_regions = [
        region
        for region in _hazard_regions((mean_field >= threshold).any(axis=2), metadata)
        if region["pixel_count"] >= minimum_region_pixels
    ]
    for region in mean_regions:
        min_x, min_y, max_x, max_y = region["pixel_bbox"]
        active = (mean_prediction[:, min_y : max_y + 1, min_x : max_x + 1] >= threshold).any(axis=(1, 2))
        start_index, end_index, longest = _longest_true_run(active)
        if longest:
            region["start_minute"] = int((start_index + 1) * 5)
            region["end_minute"] = int((end_index + 1) * 5)
            region["duration_minutes"] = int(longest * 5)
            region["total_active_minutes"] = int(active.sum() * 5)
            region["duration_is_lower_bound"] = bool(active.all())
    history_minutes = _ensemble_manifest().get("history_minutes", list(range(-60, 5, 5)))
    history_summaries = [
        {
            "frame_index": index + 1,
            "relative_minutes": int(history_minutes[index]),
            "min_vil": int(frame.min()),
            "max_vil": int(frame.max()),
            "mean_vil": round(float(frame.mean()), 3),
        }
        for index, frame in enumerate(arrays["history"])
    ]
    return {
        "status": "ok",
        "task": "ensemble_uncertainty_analysis",
        "data_source": _cascast_source(str(metadata["event_id"])),
        "event_id": metadata["event_id"],
        "model_name": "cascast",
        "lead_minutes": 60,
        "forecast_frames": 12,
        "ensemble_count": int(members.shape[0]),
        "threshold": threshold,
        "ground_truth_available": True,
        "history_available": True,
        "history_frames": int(arrays["history"].shape[0]),
        "history_frame_summaries": history_summaries,
        "full_resolution_viewer": True,
        "spatial_shape": [384, 384],
        "hazard_regions": mean_regions,
        "ground_truth_frame_summaries": _prediction_frame_summaries(
            ground_truth.transpose(1, 2, 0), threshold
        ),
        "model_metrics": _deterministic_metrics(
            mean_prediction.transpose(1, 2, 0),
            ground_truth.transpose(1, 2, 0),
            threshold,
        ),
        "pixel_uncertainty": {
            "mean_std": round(float(pixel_std.mean()), 3),
            "p95_std": round(float(np.quantile(pixel_std, 0.95)), 3),
            "max_std": round(float(pixel_std.max()), 3),
            "mean_80_percent_interval_width": round(float((arrays["q90"] - arrays["q10"]).mean()), 3),
        },
        "storm_object_uncertainty": {
            "members_with_object": len(object_centroids),
            "centroid_spread_pixels": round(centroid_spread, 3),
            "largest_object_area_ratio_min": round(min(largest_object_area_ratios), 6) if largest_object_area_ratios else 0.0,
            "largest_object_area_ratio_mean": round(float(np.mean(largest_object_area_ratios)), 6) if largest_object_area_ratios else 0.0,
            "largest_object_area_ratio_max": round(max(largest_object_area_ratios), 6) if largest_object_area_ratios else 0.0,
        },
        "member_peak_area_ratio": {
            "min": round(float(member_areas.min()), 6),
            "mean": round(float(member_areas.mean()), 6),
            "max": round(float(member_areas.max()), 6),
        },
        "member_active_duration_minutes": {
            "min": int(member_durations.min()),
            "mean": round(float(member_durations.mean()), 1),
            "max": int(member_durations.max()),
        },
        "member_summaries": member_summaries,
        "frame_summaries": uncertainty_frames,
        "uncertainty_frame_summaries": uncertainty_frames,
        "interpretation_limit": (
            "不确定性来自10个CasCast随机ensemble成员之间的离散程度；"
            "它描述模型预测分歧，不等于真实误差或灾害概率。"
            "准确率指标只对应当前所选测试样本。"
        ),
    }


def _bbox_iou(first: list[int], second: list[int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right < left or bottom < top:
        return 0.0
    intersection = (right - left + 1) * (bottom - top + 1)
    first_area = (first[2] - first[0] + 1) * (first[3] - first[1] + 1)
    second_area = (second[2] - second[0] + 1) * (second[3] - second[1] + 1)
    return float(intersection / (first_area + second_area - intersection))


def _storm_tracks(
    fields: np.ndarray,
    metadata: dict[str, Any],
    threshold: float,
    minimum_region_pixels: int = 16,
) -> dict[str, Any]:
    """Track thresholded connected regions across consecutive five-minute frames."""
    fields = np.asarray(fields, dtype=np.float32)
    tracks: list[dict[str, Any]] = []
    active_track_ids: list[int] = []
    frame_object_counts: list[int] = []
    for frame_index, frame in enumerate(fields):
        objects = [
            region
            for region in _hazard_regions(frame >= threshold, metadata, max_regions=20)
            if region["pixel_count"] >= minimum_region_pixels
        ]
        frame_object_counts.append(len(objects))
        candidates: list[tuple[float, float, int, int]] = []
        for track_id in active_track_ids:
            previous = tracks[track_id]["objects"][-1]
            for object_index, region in enumerate(objects):
                iou = _bbox_iou(previous["pixel_bbox"], region["pixel_bbox"])
                distance = float(
                    np.linalg.norm(
                        np.asarray(previous["pixel_centroid"], dtype=np.float32)
                        - np.asarray(region["pixel_centroid"], dtype=np.float32)
                    )
                )
                if iou > 0 or distance <= 32.0:
                    candidates.append((iou, -distance, track_id, object_index))
        matched_tracks: set[int] = set()
        matched_objects: set[int] = set()
        for _, _, track_id, object_index in sorted(candidates, reverse=True):
            if track_id in matched_tracks or object_index in matched_objects:
                continue
            tracks[track_id]["objects"].append(objects[object_index])
            tracks[track_id]["frame_indices"].append(frame_index)
            matched_tracks.add(track_id)
            matched_objects.add(object_index)
        new_active = list(matched_tracks)
        for object_index, region in enumerate(objects):
            if object_index in matched_objects:
                continue
            tracks.append({"frame_indices": [frame_index], "objects": [region]})
            new_active.append(len(tracks) - 1)
        active_track_ids = new_active

    public_tracks: list[dict[str, Any]] = []
    for track_index, track in enumerate(tracks, start=1):
        frames = track["frame_indices"]
        objects = track["objects"]
        centroids = np.asarray([item["pixel_centroid"] for item in objects], dtype=np.float32)
        path_length = float(
            np.linalg.norm(np.diff(centroids, axis=0), axis=1).sum()
        ) if len(centroids) > 1 else 0.0
        public_tracks.append(
            {
                "track_id": f"T{track_index}",
                "start_minute": int((frames[0] + 1) * 5),
                "end_minute": int((frames[-1] + 1) * 5),
                "duration_minutes": int(len(frames) * 5),
                "frame_count": len(frames),
                "maximum_pixel_count": max(item["pixel_count"] for item in objects),
                "maximum_area_ratio": round(max(item["area_ratio"] for item in objects), 6),
                "path_length_pixels": round(path_length, 3),
                "start_centroid": objects[0]["pixel_centroid"],
                "end_centroid": objects[-1]["pixel_centroid"],
            }
        )
    public_tracks.sort(key=lambda item: (item["duration_minutes"], item["maximum_pixel_count"]), reverse=True)
    return {
        "track_count": len(public_tracks),
        "frame_object_counts": frame_object_counts,
        "longest_track_duration_minutes": max(
            (item["duration_minutes"] for item in public_tracks), default=0
        ),
        "tracks": public_tracks[:10],
        "association_method": "consecutive-frame connected components matched by bbox overlap or centroid distance <=32 pixels",
    }


def _storm_object_verification(
    prediction: np.ndarray,
    target: np.ndarray,
    metadata: dict[str, Any],
    threshold: float,
    minimum_region_pixels: int = 16,
) -> dict[str, Any]:
    hits = misses = false_alarms = 0
    centroid_errors: list[float] = []
    per_frame: list[dict[str, Any]] = []
    for frame_index, (predicted_frame, target_frame) in enumerate(zip(prediction, target)):
        predicted_objects = [
            item for item in _hazard_regions(predicted_frame >= threshold, metadata, max_regions=20)
            if item["pixel_count"] >= minimum_region_pixels
        ]
        target_objects = [
            item for item in _hazard_regions(target_frame >= threshold, metadata, max_regions=20)
            if item["pixel_count"] >= minimum_region_pixels
        ]
        candidates: list[tuple[float, int, int]] = []
        for pred_index, predicted_object in enumerate(predicted_objects):
            for target_index, target_object in enumerate(target_objects):
                iou = _bbox_iou(predicted_object["pixel_bbox"], target_object["pixel_bbox"])
                if iou >= 0.1:
                    candidates.append((iou, pred_index, target_index))
        matched_predicted: set[int] = set()
        matched_target: set[int] = set()
        for _, pred_index, target_index in sorted(candidates, reverse=True):
            if pred_index in matched_predicted or target_index in matched_target:
                continue
            matched_predicted.add(pred_index)
            matched_target.add(target_index)
            centroid_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(predicted_objects[pred_index]["pixel_centroid"], dtype=np.float32)
                        - np.asarray(target_objects[target_index]["pixel_centroid"], dtype=np.float32)
                    )
                )
            )
        frame_hits = len(matched_predicted)
        frame_misses = len(target_objects) - frame_hits
        frame_false_alarms = len(predicted_objects) - frame_hits
        hits += frame_hits
        misses += frame_misses
        false_alarms += frame_false_alarms
        per_frame.append(
            {
                "lead_minutes": (frame_index + 1) * 5,
                "predicted_objects": len(predicted_objects),
                "observed_objects": len(target_objects),
                "hits": frame_hits,
                "misses": frame_misses,
                "false_alarms": frame_false_alarms,
            }
        )
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "object_csi": round(hits / (hits + misses + false_alarms), 6)
        if hits + misses + false_alarms else 0.0,
        "object_pod": round(hits / (hits + misses), 6) if hits + misses else 0.0,
        "object_far": round(false_alarms / (hits + false_alarms), 6)
        if hits + false_alarms else 0.0,
        "mean_matched_centroid_error_pixels": round(float(np.mean(centroid_errors)), 3)
        if centroid_errors else None,
        "per_frame": per_frame,
        "matching_method": "connected regions matched one-to-one using bounding-box IoU >= 0.1",
    }
