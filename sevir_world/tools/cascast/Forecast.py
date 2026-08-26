from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import json
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from ...data_access.config import *
from ...data_access.ensemble import (
    _cascast_source,
    _ensemble_manifest,
    _event,
    _load_cascast_ensemble,
    _load_cascast_rolling_ensemble,
    _source,
    _vil_display_values,
)
from ...data_access.uploads import (
    _load_uploaded_vil,
    _load_uploaded_vil_sequence,
    load_uploaded_sequence_hdf5_ground_truth,
)
from ...data_access.live_inference import (
    live_cascast_source,
    load_live_cascast_ensemble,
    load_uploaded_cascast_ensemble,
    run_local_cascast_inference,
    run_uploaded_cascast_inference,
)
from ..geolocator.Geospatial import geolocate_sevir_event, reverse_geocode_us

import h5py
import numpy as np
from pyproj import CRS, Geod, Transformer


def _scientific_contract(
    *,
    inference_type: str,
    ground_truth_available: bool,
    input_frames: int,
    forecast_frames: int,
    lead_minutes: int,
    location_precision: str = "sevir_grid_approximate",
) -> dict[str, Any]:
    """Return machine-checkable provenance for a forecast result.

    Parameters
    ----------
    inference_type : str
        Model or baseline inference category.
    ground_truth_available : bool
        Whether aligned future observations are included.
    input_frames, forecast_frames : int
        History and prediction frame counts.
    lead_minutes : int
        Maximum forecast lead.
    location_precision : str, default "sevir_grid_approximate"
        Spatial precision label.

    Returns
    -------
    dict
        Scientific scope, frame counts and interpretation constraints.
    """
    return {
        "inference_type": inference_type,
        "ground_truth_available": bool(ground_truth_available),
        "metrics_available": bool(ground_truth_available),
        "input_frame_count": int(input_frames),
        "forecast_frame_count": int(forecast_frames),
        "lead_minutes": int(lead_minutes),
        "location_precision": location_precision,
    }


def _forecast_evidence_id(
    *,
    subject_id: str,
    model_name: str,
    ensemble_members: int,
    horizon_hours: float,
    prediction_source: Any,
) -> str:
    """Return a stable, opaque identity for one reusable forecast cube.

    The identifier contains no pixel data and is safe to retain in a chat
    session.  It lets follow-up turns prove that ensemble, visualization and
    verification calculations refer to the same model run instead of silently
    switching from ten members to one member or recomputing another horizon.

    Parameters
    ----------
    subject_id : str
        Event or upload identity.
    model_name : str
        Forecast implementation.
    ensemble_members : int
        Requested ensemble size.
    horizon_hours : float
        Forecast horizon.
    prediction_source : Any
        Stable source identity used in the digest.

    Returns
    -------
    str
        Opaque deterministic evidence identifier.
    """
    payload = {
        "subject_id": str(subject_id),
        "model_name": str(model_name),
        "ensemble_members": int(ensemble_members),
        "horizon_hours": float(horizon_hours),
        "prediction_source": str(prediction_source or ""),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return f"forecast-{digest}"


def analyze_uploaded_vil(
    upload_id: str,
    threshold: float,
    minimum_region_pixels: int = 1,
) -> dict[str, Any]:
    """Describe thresholded structure in one uploaded VIL image.

    Parameters
    ----------
    upload_id : str
        Identifier assigned to the uploaded image.
    threshold : float
        Encoded VIL threshold defining active pixels.
    minimum_region_pixels : int, default 1
        Smallest connected region retained in the result.

    Returns
    -------
    dict
        Image statistics and connected-region geometry. No future forecast is
        inferred from a single frame.
    """
    frame, metadata = _load_uploaded_vil(upload_id)
    valid = frame != 255
    mask = valid & (frame >= threshold)
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
        **_scientific_contract(
            inference_type="observation_only",
            ground_truth_available=False,
            input_frames=1,
            forecast_frames=0,
            lead_minutes=0,
            location_precision="uploaded_grid_only",
        ),
        "spatial_shape": [384, 384],
        "physical_area_available": False,
        "pixel_area_km2": None,
        "area_calibration": "unavailable_for_standalone_upload",
        "threshold": threshold,
        "threshold_source": "caller_selected_compatibility_only",
        "threshold_space": "encoded_vil_0_254",
        "missing_value": 255,
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
        "analysis_provenance": {
            "threshold_source": "caller_selected_compatibility_only",
            "threshold_space": "encoded_vil_0_254",
            "missing_value": 255,
            "region_method": "connected_components_8_neighbour",
            "minimum_region_pixels": int(minimum_region_pixels),
        },
        "interpretation_limit": (
            "这是用户上传的单张SEVIR VIL图像，只能分析当前强回波与空间分布，"
            "不能据此声称完成未来预测或持续时间分析。阈值是0到255缩放像素值，"
            "不是dBZ、降雨量或官方灾害预警。上传文件没有地理投影、实际覆盖范围"
            "或像元尺寸标定，因此只能报告超阈值像素数和网格面积比例，不能换算平方公里。"
        ),
    }


# Deprecated compatibility implementation.  The active Agent path does not
# expose this registry: it receives raw arrays through ``ds[0]["raw_data"]``
# and generates question-specific fields in its own ``run()`` code.  Keeping
# this private wrapper avoids breaking older benchmark fixtures only.
UPLOADED_HISTORY_FRAME_FIELDS = {
    "min_vil": "该观测帧整幅网格的最小VIL值（0至254缩放值）",
    "max_vil": "该观测帧整幅网格的最大VIL值（0至254缩放值）",
    "mean_vil": "该观测帧整幅网格的VIL空间均值，不代表强回波核心强度",
    "exceeding_pixel_count": "该观测帧达到所选阈值的像素数",
    "exceeding_area_ratio": "该观测帧达到所选阈值的网格面积比例",
    "threshold_centroid_x": "全部阈值超标像素的横向网格质心；无超标像素时为null",
    "threshold_centroid_y": "全部阈值超标像素的纵向网格质心；无超标像素时为null",
    "region_count": "达到最小像素数的阈值连通区数量",
    "largest_region_pixel_count": "最大阈值连通区的像素数",
    "largest_region_area_ratio": "最大阈值连通区的网格面积比例",
    "largest_region_centroid_x": "最大阈值连通区横向网格质心；无候选区域时为null",
    "largest_region_centroid_y": "最大阈值连通区纵向网格质心；无候选区域时为null",
}


def get_uploaded_vil_sequence_values(upload_id: str) -> dict[str, Any]:
    """Return the raw pixels for every frame in an uploaded history.

    This is intentionally a data primitive, not an analysis endpoint.  It
    does not accept a field list, threshold, region size, or tracking option
    and it computes no summary statistics.  The generated ``run()`` program
    must decide which quantities are relevant and calculate them itself.

    Parameters
    ----------
    upload_id : str
        Uploaded sequence identifier.

    Returns
    -------
    dict
        Raw history frames, relative minutes and upload provenance.
    """
    sequence, metadata = _load_uploaded_vil_sequence(upload_id)
    frames = [
        frame.reshape(-1).astype(np.uint8).tolist()
        for frame in np.asarray(sequence, dtype=np.uint8)
    ]
    available_minutes = [int(value) for value in metadata["history_minutes"]]
    return {
        "status": "ok",
        "task": "uploaded_sevir_vil_raw_sequence",
        "upload_id": upload_id,
        "filenames": list(metadata.get("filenames", [])),
        "data_source": metadata["source"],
        "shape": [int(sequence.shape[1]), int(sequence.shape[2])],
        "frame_count": int(sequence.shape[0]),
        "dtype": "uint8",
        "relative_minutes": available_minutes,
        "frames": frames,
        "statistics_computed": [],
        "fields_computed": [],
        "threshold_applied": None,
        "region_parameters_applied": None,
        "raw_data": True,
        "interpretation_limit": (
            "The tool returns only the uploaded 0-255 VIL pixel grids. "
            "It does not choose fields, thresholds, statistics, regions, "
            "tracks, or trends; generated Agent code must do that."
        ),
        "analysis_provenance": {
            "threshold_source": "not_applied",
            "threshold_space": "encoded_vil_0_254",
            "missing_value": 255,
            "region_method": "agent_selected",
        },
    }


def _legacy_query_uploaded_history_values(
    upload_id: str,
    fields: list[str],
    relative_minutes: list[int],
    threshold: float = 133.0,
    minimum_region_pixels: int = 1,
    include_tracks: bool = False,
    track_limit: int = 5,
) -> dict[str, Any]:
    """Return selected primitives from an uploaded history.

    Parameters
    ----------
    upload_id : str
        Uploaded sequence identifier.
    fields : list of str
        Primitive fields to calculate.
    relative_minutes : list of int
        History times to select.
    threshold : float, default 133.0
        Encoded VIL threshold.
    minimum_region_pixels : int, default 1
        Smallest connected region retained.
    include_tracks : bool, default False
        Whether to include object tracks.
    track_limit : int, default 5
        Maximum tracks returned.

    Returns
    -------
    dict
        Selected observation primitives and provenance.
    """
    selected_fields = list(dict.fromkeys(str(field) for field in fields))
    if not selected_fields:
        raise ValueError("fields must contain at least one uploaded history field")
    unknown_fields = sorted(set(selected_fields) - set(UPLOADED_HISTORY_FRAME_FIELDS))
    if unknown_fields:
        raise ValueError(
            "unsupported uploaded history field(s): "
            + ", ".join(unknown_fields)
            + "; available fields: "
            + ", ".join(sorted(UPLOADED_HISTORY_FRAME_FIELDS))
        )
    sequence, metadata = _load_uploaded_vil_sequence(upload_id)
    available_minutes = [int(value) for value in metadata["history_minutes"]]
    selected_minutes = list(dict.fromkeys(int(value) for value in relative_minutes))
    if not selected_minutes:
        raise ValueError("relative_minutes must contain at least one observation time")
    invalid_minutes = [value for value in selected_minutes if value not in available_minutes]
    if invalid_minutes:
        raise ValueError(
            "relative_minutes values must be selected from: "
            + ", ".join(str(value) for value in available_minutes)
        )
    if not 1 <= int(track_limit) <= 30:
        raise ValueError("track_limit must be between 1 and 30")

    needs_mask = any(
        field
        in {
            "exceeding_pixel_count",
            "exceeding_area_ratio",
            "threshold_centroid_x",
            "threshold_centroid_y",
            "region_count",
            "largest_region_pixel_count",
            "largest_region_area_ratio",
            "largest_region_centroid_x",
            "largest_region_centroid_y",
        }
        for field in selected_fields
    )
    needs_regions = any(
        field == "region_count" or field.startswith("largest_region_")
        for field in selected_fields
    )
    records: list[dict[str, Any]] = []
    for relative_minute in selected_minutes:
        frame_index = available_minutes.index(relative_minute)
        frame = _vil_display_values(sequence[frame_index])
        record: dict[str, Any] = {
            "frame_index": frame_index + 1,
            "relative_minutes": relative_minute,
        }
        mask = frame >= threshold if needs_mask else None
        mask_rows: np.ndarray | None = None
        mask_columns: np.ndarray | None = None
        regions: list[dict[str, Any]] = []
        if mask is not None and (
            "threshold_centroid_x" in selected_fields
            or "threshold_centroid_y" in selected_fields
        ):
            mask_rows, mask_columns = np.nonzero(mask)
        if mask is not None and needs_regions:
            regions = [
                region
                for region in _hazard_regions(mask, metadata, max_regions=30)
                if region["pixel_count"] >= minimum_region_pixels
            ]
        largest_region = regions[0] if regions else None
        for field in selected_fields:
            if field == "min_vil":
                value: Any = round(float(frame.min()), 3)
            elif field == "max_vil":
                value = round(float(frame.max()), 3)
            elif field == "mean_vil":
                value = round(float(frame.mean()), 3)
            elif field == "exceeding_pixel_count":
                value = int(mask.sum()) if mask is not None else 0
            elif field == "exceeding_area_ratio":
                value = round(float(mask.mean()), 6) if mask is not None else 0.0
            elif field == "threshold_centroid_x":
                value = (
                    round(float(mask_columns.mean()), 3)
                    if mask_columns is not None and mask_columns.size
                    else None
                )
            elif field == "threshold_centroid_y":
                value = (
                    round(float(mask_rows.mean()), 3)
                    if mask_rows is not None and mask_rows.size
                    else None
                )
            elif field == "region_count":
                value = len(regions)
            elif field == "largest_region_pixel_count":
                value = int(largest_region["pixel_count"]) if largest_region else 0
            elif field == "largest_region_area_ratio":
                value = float(largest_region["area_ratio"]) if largest_region else 0.0
            elif field == "largest_region_centroid_x":
                value = float(largest_region["pixel_centroid"][0]) if largest_region else None
            else:
                value = float(largest_region["pixel_centroid"][1]) if largest_region else None
            record[field] = value
        records.append(record)

    tracking: dict[str, Any] | None = None
    if include_tracks:
        raw_tracking = _storm_tracks(
            sequence.astype(np.float32),
            metadata,
            threshold,
            minimum_region_pixels,
            track_limit=int(track_limit),
        )
        tracks = []
        for item in raw_tracking["tracks"]:
            first_index = int(item["start_frame_index"])
            last_index = int(item["end_frame_index"])
            tracks.append(
                {
                    "track_id": item["track_id"],
                    "first_observed_relative_minute": available_minutes[first_index],
                    "last_observed_relative_minute": available_minutes[last_index],
                    "observed_span_minutes": (
                        available_minutes[last_index] - available_minutes[first_index]
                    ),
                    "detection_frame_count": item["frame_count"],
                    "maximum_pixel_count": item["maximum_pixel_count"],
                    "maximum_area_ratio": item["maximum_area_ratio"],
                    "start_centroid": item["start_centroid"],
                    "end_centroid": item["end_centroid"],
                }
            )
        tracking = {
            "candidate_track_count": raw_tracking["track_count"],
            "returned_track_count": len(tracks),
            "frame_candidate_region_counts": raw_tracking["frame_object_counts"],
            "tracks": tracks,
            "association_method": raw_tracking["association_method"],
            "tracking_limit": (
                "这些是阈值连通区的近似匹配轨迹，不等同于经人工确认的风暴单体。"
                "首次或末次匹配只表示当前上传窗口和算法内的首次或末次检测，"
                "不能直接解释为真实生成或消散。"
            ),
        }

    return {
        "status": "ok",
        "task": "uploaded_history_frame_values",
        "analysis_scope": "agent_selected_uploaded_history_observations",
        "upload_id": upload_id,
        "data_source": metadata["source"],
        "model_name": None,
        **_scientific_contract(
            inference_type="uploaded_sequence_observation_only",
            ground_truth_available=False,
            input_frames=int(sequence.shape[0]),
            forecast_frames=0,
            lead_minutes=0,
            location_precision="uploaded_grid_only",
        ),
        "history_window_minutes": [available_minutes[0], available_minutes[-1]],
        "history_elapsed_minutes": available_minutes[-1] - available_minutes[0],
        "available_relative_minutes": available_minutes,
        "selected_fields": selected_fields,
        "selected_relative_minutes": selected_minutes,
        "threshold": threshold,
        "minimum_region_pixels": minimum_region_pixels,
        "record_count": len(records),
        "records": records,
        # Backward-compatible semantic alias. Some model-generated programs
        # naturally call the selected historical rows "observations"; keeping
        # both names prevents a harmless naming mismatch from consuming all
        # Direct-mode repair attempts.
        "observations": records,
        "field_definitions": {
            field: UPLOADED_HISTORY_FRAME_FIELDS[field] for field in selected_fields
        },
        "tracks_included": bool(include_tracks),
        "tracking": tracking,
        "calculated_change_statistics": [],
        "interpretation_limit": (
            "工具只返回Agent选择的历史观测字段和可选的候选连通区轨迹，不运行CasCast，"
            "也不预先计算趋势、增减幅度或故事结论；这些必须由Agent生成的代码完成。"
            "上传图像没有地理投影元数据，x/y仅是图像网格坐标，不能写成东、西、南、北，"
            "也不能把像素距离直接写成公里。全图mean_vil的变化不等于核心强度变化。"
            "阈值面积下降只表示超阈值像素减少，不自动证明外围收缩或核心集中；"
            "除非Agent选择并计算了明确的形态或生命周期判据，否则不能称为成熟、组织化、"
            "生成、发展或消散。"
        ),
    }


# Compatibility name for pre-refactor tests/clients.  New Agent prompts never
# select this endpoint; they operate on the raw injected dataset instead.
query_uploaded_history_values = _legacy_query_uploaded_history_values


# Kept only for old callers and regression fixtures.  Agent-facing
# ``CasCastForecaster.uploaded_history`` uses the raw primitive above and never
# reaches this field/threshold-based compatibility implementation.
query_uploaded_history_values = _legacy_query_uploaded_history_values


def get_uploaded_vil_values(upload_id: str) -> dict[str, Any]:
    """Return raw uploaded VIL pixels without computing a fixed statistic.

    This primitive deliberately does not apply a threshold, identify regions,
    or calculate mean/max/area. Generated ``run()`` code must choose and
    calculate any requested metrics.

    Parameters
    ----------
    upload_id : str
        Uploaded single-frame identifier.

    Returns
    -------
    dict
        Raw encoded VIL pixels and upload provenance.
    """
    frame, metadata = _load_uploaded_vil(upload_id)
    return {
        "status": "ok",
        "task": "uploaded_sevir_vil_raw_values",
        "upload_id": upload_id,
        "filename": metadata["filename"],
        "data_source": metadata["source"],
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "dtype": "uint8",
        "values": frame.reshape(-1).astype(np.uint8).tolist(),
        "statistics_computed": [],
        "threshold_applied": None,
        "analysis_provenance": {
            "threshold_source": "agent_selected",
            "threshold_space": "encoded_vil_0_254",
            "missing_value": 255,
            "region_method": "agent_selected",
        },
        "interpretation_limit": (
            "工具只返回上传图像的原始0到255灰度像素，不代表dBZ、降雨量或官方预警；"
            "统计量和阈值应由Agent根据问题在run()中选择并计算。"
        ),
    }


def get_uploaded_vil_sequence_forecast(
    upload_id: str,
    model_name: str = "persistence_baseline",
    ensemble_members: int = 10,
    horizon_hours: float = 1.0,
    include_ground_truth: bool = False,
    array_transport: bool = False,
    force_live: bool = False,
) -> dict[str, Any]:
    """Run the selected forecast model and return raw history/forecast grids.

    No threshold, hazard region, trend, or summary statistic is calculated
    here.  The Agent-generated program must choose those from the returned
    pixels.  This keeps a future question on the same code-generation path as
    an observed-only question while retaining the model as an execution tool.

    Parameters
    ----------
    upload_id : str
        Uploaded history identifier.
    model_name : str, default "persistence_baseline"
        Forecast implementation.
    ensemble_members : int, default 10
        Requested ensemble size.
    horizon_hours : float, default 1.0
        Forecast horizon.
    include_ground_truth : bool, default False
        Whether to return aligned future observations when available.
    array_transport : bool, default False
        Whether to use the array transport representation.
    force_live : bool, default False
        Whether to bypass reusable inference output.

    Returns
    -------
    dict
        Raw history/forecast grids, leads and scientific provenance.
    """
    if str(model_name).strip().lower() in {"cascast", "cascast_live"}:
        model_name = "cascast_live"
    sequence, metadata = _load_uploaded_vil_sequence(upload_id)
    horizon = float(horizon_hours)
    if horizon not in {1.0, 2.0, 3.0}:
        raise ValueError("horizon_hours must be 1.0, 2.0, or 3.0")
    rolling_segments = int(horizon)
    live_metadata: dict[str, Any] | None = None
    if model_name == "cascast_live":
        inference_kwargs = {
            "ensemble_members": ensemble_members,
            "rolling_segments": rolling_segments,
        }
        # Keep the compatibility call shape stable for cached inference;
        # only send `force` when the caller explicitly requests a rerun.
        if force_live:
            inference_kwargs["force"] = True
        inference = run_uploaded_cascast_inference(upload_id, **inference_kwargs)
        arrays, live_metadata = load_uploaded_cascast_ensemble(
            upload_id,
            ensemble_members=ensemble_members,
            rolling_segments=rolling_segments,
        )
        prediction = arrays["ensemble_mean"].astype(np.float32)
        data_source: Any = {
            "kind": "user_uploaded_sequence_local_cascast",
            "upload_id": upload_id,
            "prediction_file": live_metadata["cache_file"],
            "cache_hit": inference["cache_hit"],
        }
    elif model_name == "persistence_baseline":
        prediction = np.repeat(
            sequence[-1][None, :, :], 12 * rolling_segments, axis=0
        ).astype(np.float32)
        data_source = metadata["source"]
    else:
        raise ValueError(f"Unsupported uploaded-sequence model: {model_name}")
    history_values = np.clip(np.rint(sequence), 0, 255).astype(np.uint8)
    forecast_values = _vil_display_values(prediction).astype(np.float32)
    truth_leads = list(range(5, 60 * rolling_segments + 5, 5))
    if include_ground_truth:
        ground_truth = load_uploaded_sequence_hdf5_ground_truth(upload_id, truth_leads)
    else:
        # Truth is intentionally isolated from ordinary forecasting.  Loading it
        # only for an explicit verification question prevents hindsight from
        # leaking into a forecast explanation.
        ground_truth = {
            "available": False,
            "requested": False,
            "lead_minutes": [],
            "frames": [],
            "reason": "ground_truth_not_requested",
        }
    stochastic = model_name == "cascast_live"
    ensemble_count = int(live_metadata.get("ensemble_count", 1)) if live_metadata else 1
    member_seeds = (
        [int(value) for value in np.asarray(arrays.get("member_seed", [])).tolist()]
        if live_metadata is not None
        else []
    )
    prediction_source = (
        live_metadata.get("cache_file") if live_metadata else metadata.get("source")
    )
    source_metadata = data_source if isinstance(data_source, dict) else {
        "source": data_source,
        "cache_hit": False,
    }
    forecast_evidence_id = _forecast_evidence_id(
        subject_id=upload_id,
        model_name=model_name,
        ensemble_members=ensemble_count,
        horizon_hours=horizon,
        prediction_source=prediction_source,
    )
    history_payload: Any
    forecast_payload: Any
    ensemble_payload: Any = []
    if live_metadata is not None:
        member_values = _vil_display_values(
            np.asarray(arrays["ensemble_members"], dtype=np.float32)
        )
        member_records = []
        for member_index, member in enumerate(member_values):
            member_frames = np.ascontiguousarray(member, dtype=np.float32)
            if array_transport:
                member_frames.setflags(write=False)
                frames_value: Any = member_frames
            else:
                frames_value = [frame.tolist() for frame in member_frames]
            member_records.append(
                {
                    "member": member_index + 1,
                    "seed": int(member_seeds[member_index]),
                    "output_type": "stochastic_ensemble_member",
                    "lead_minutes": list(range(5, 60 * rolling_segments + 5, 5)),
                    "frames": frames_value,
                }
            )
        ensemble_payload = {"member_count": len(member_records), "members": member_records}
    if array_transport:
        history_payload = np.ascontiguousarray(history_values)
        forecast_payload = np.ascontiguousarray(forecast_values)
        history_payload.setflags(write=False)
        forecast_payload.setflags(write=False)
    else:
        history_payload = [frame.tolist() for frame in history_values]
        forecast_payload = [frame.tolist() for frame in forecast_values]
    return {
        "status": "ok",
        "task": "uploaded_sevir_vil_raw_sequence_forecast",
        "capabilities": ["current_forecast", "future_forecast", "forecast_grid", "forecast_evidence"]
        + (["ground_truth"] if ground_truth["available"] else []),
        "forecast_evidence_id": forecast_evidence_id,
        "forecast_evidence": {
            "id": forecast_evidence_id,
            "subject_kind": "uploaded_sequence",
            "subject_id": upload_id,
            "model_name": model_name,
            "ensemble_count": ensemble_count,
            "member_seeds": member_seeds,
            "horizon_hours": horizon,
            "rolling_segments": rolling_segments,
            "lead_minutes": list(range(5, 60 * rolling_segments + 5, 5)),
            "selected_lead_minutes": list(range(5, 60 * rolling_segments + 5, 5)),
            "forecast_frame_count": int(forecast_values.shape[0]),
            "cache_hit": bool(source_metadata.get("cache_hit", False)),
            "inference_mode": "cache" if bool(source_metadata.get("cache_hit", False)) else "live",
            "prediction_source": prediction_source,
            "ground_truth_attached": bool(ground_truth["available"]),
        },
        "upload_id": upload_id,
        "initialization_time_utc": metadata.get("initialization_time_utc"),
        "data_source": data_source,
        "model_name": model_name,
        "inference_mode": (
            "cache" if model_name == "cascast_live" and bool(source_metadata.get("cache_hit", False))
            else "live" if model_name == "cascast_live" else "deterministic_baseline"
        ),
        "cache_hit": bool(source_metadata.get("cache_hit", False)),
        "prediction_source": source_metadata.get("prediction_file") or data_source,
        "ensemble_count": ensemble_count,
        "ensemble_members_available": bool(live_metadata is not None),
        "ensemble": ensemble_payload,
        "member_seeds": member_seeds,
        "output_type": (
            "ensemble_mean" if stochastic and ensemble_count > 1
            else "single_stochastic_member" if stochastic
            else "deterministic_baseline"
        ),
        "is_deterministic": not stochastic,
        "history_shape": [int(sequence.shape[1]), int(sequence.shape[2])],
        "forecast_shape": [int(forecast_values.shape[1]), int(forecast_values.shape[2])],
        "forecast_dtype": "float32",
        "analysis_value_space": "decoded encoded-VIL scale before display quantization",
        "history_relative_minutes": [int(v) for v in metadata["history_minutes"]],
        "forecast_relative_minutes": list(range(5, 60 * rolling_segments + 5, 5)),
        # Keep each frame as a normal HxW JSON matrix.  This mirrors the raw
        # uploaded arrays injected into ds and prevents generated code from
        # accidentally iterating an integer pixel as if it were a row.
        "history_frames": history_payload,
        "forecast_frames": forecast_payload,
        "history_frame_count": int(history_values.shape[0]),
        "forecast_frame_count": int(forecast_values.shape[0]),
        "lead_minutes": int(forecast_values.shape[0] * 5),
        "forecast_mode": (
            "free_running" if model_name == "cascast_live" and rolling_segments > 1
            else "standard_single_window"
        ),
        "statistics_computed": [],
        "fields_computed": [],
        "threshold_applied": None,
        "region_parameters_applied": None,
        "raw_data": True,
        "ground_truth_requested": bool(include_ground_truth),
        "ground_truth_available": bool(ground_truth["available"]),
        "ground_truth": ground_truth,
        "sample_identity": (
            {
                key: ground_truth.get(key)
                for key in ("event_id", "h5_file", "file_index", "segment")
            }
            if ground_truth.get("available") else None
        ),
        "validation_manifest": ground_truth.get("validation_manifest"),
        "verified_lead_minutes": ground_truth.get("lead_minutes", []),
        "unverified_lead_minutes": ground_truth.get("unverified_lead_minutes", []),
        "ground_truth_complete_for_horizon": bool(
            ground_truth["available"]
            and not ground_truth.get("unverified_lead_minutes", [])
        ),
        "provenance": {
            "prediction_source": prediction_source,
            "rolling_conditioning": bool(rolling_segments > 1),
            "ground_truth_source": ground_truth.get("source") if ground_truth["available"] else None,
            "ground_truth_event_id": ground_truth.get("event_id") if ground_truth["available"] else None,
        },
        "horizon_hours": horizon,
        "rolling_segments": rolling_segments,
        "interpretation_limit": (
            "The model produced future pixel grids, but the tool did not decide "
            "what counts as severe weather or whether the user will be affected. "
            "Generated Agent code must calculate question-relevant metrics and "
            "state that the upload has no geospatial calibration."
        ),
    }


def analyze_uploaded_vil_sequence(
    upload_id: str,
    threshold: float,
    minimum_region_pixels: int = 1,
    model_name: str = "persistence_baseline",
    ensemble_members: int = 10,
    horizon_hours: float = 1.0,
    force_live: bool = False,
) -> dict[str, Any]:
    """Run and summarize a forecast from an uploaded VIL history sequence.

    Parameters
    ----------
    upload_id : str
        Identifier for an uploaded multi-frame sequence.
    threshold : float
        Encoded VIL threshold used in hazard summaries.
    minimum_region_pixels : int, default 1
        Smallest connected region retained.
    model_name : str, default "persistence_baseline"
        Forecast implementation selected by the caller.
    ensemble_members : int, default 10
        Requested ensemble size when supported.
    horizon_hours : float, default 1.0
        Forecast horizon in hours.
    force_live : bool, default False
        Whether to bypass reusable inference output when supported.

    Returns
    -------
    dict
        Input evidence, forecast arrays, summaries and provenance.
    """
    sequence, metadata = _load_uploaded_vil_sequence(upload_id)
    # This compatibility endpoint is an observation report, not the Agent's
    # open-ended analysis primitive.  Keep it from silently injecting future
    # data when a caller asks only about the uploaded past hour.
    horizon = float(horizon_hours)
    if horizon not in {1.0, 2.0, 3.0}:
        raise ValueError("horizon_hours must be 1.0, 2.0, or 3.0")
    rolling_segments = int(horizon)
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
    if model_name == "cascast_live":
        inference = run_uploaded_cascast_inference(
            upload_id,
            ensemble_members=ensemble_members,
            rolling_segments=rolling_segments,
            force=bool(force_live),
        )
        arrays, live_metadata = load_uploaded_cascast_ensemble(
            upload_id,
            ensemble_members=ensemble_members,
            rolling_segments=rolling_segments,
        )
        prediction = arrays["ensemble_mean"].astype(np.float32)
        data_source: Any = {
            "kind": "user_uploaded_sequence_local_cascast",
            "upload_id": upload_id,
            "prediction_file": live_metadata["cache_file"],
            "cache_hit": inference["cache_hit"],
            "inference_mode": "cache" if inference["cache_hit"] else "live",
            "prediction_source": live_metadata["cache_file"],
        }
        analysis_scope = "13_uploaded_history_frames_plus_local_cascast_forecast"
        limit = (
            "未来12帧来自本机EarthFormer、CasCast扩散和VAE解码。用户未上传未来12帧Ground Truth，"
            "因此不能计算CSI、HSS、FSS、MSE或准确率。"
        )
    elif model_name == "persistence_baseline":
        prediction = np.repeat(
            sequence[-1][None, :, :], 12 * rolling_segments, axis=0
        ).astype(np.float32)
        data_source = metadata["source"]
        analysis_scope = "13_uploaded_history_frames_plus_persistence_baseline"
        limit = "未来12帧是最后观测帧持续不变的Persistence基线，不是CasCast。"
    else:
        raise ValueError(f"Unsupported uploaded-sequence model: {model_name}")
    analysis_prediction = _vil_display_values(prediction)
    forecast_regions = [
        item for item in _hazard_regions((analysis_prediction >= threshold).any(axis=0), metadata)
        if item["pixel_count"] >= minimum_region_pixels
    ]
    tracks = _storm_tracks(sequence.astype(np.float32), metadata, threshold, minimum_region_pixels)
    return {
        "status": "ok",
        "task": "uploaded_sevir_vil_sequence_analysis",
        "analysis_scope": analysis_scope,
        "upload_id": upload_id,
        "data_source": data_source,
        "model_name": model_name,
        **_scientific_contract(
            inference_type=(
                "local_uploaded_sequence_cascast"
                if model_name == "cascast_live"
                else "uploaded_sequence_persistence"
            ),
            ground_truth_available=False,
            input_frames=int(sequence.shape[0]),
            forecast_frames=int(prediction.shape[0]),
            lead_minutes=int(prediction.shape[0] * 5),
            location_precision="uploaded_grid_only",
        ),
        "input_frames": 13,
        "forecast_frames": int(prediction.shape[0]),
        "lead_minutes": int(prediction.shape[0] * 5),
        "horizon_hours": horizon,
        "rolling_segments": rolling_segments,
        "threshold": threshold,
        "history_available": True,
        "history_frames": 13,
        "history_frame_summaries": history_summaries,
        "frame_summaries": _prediction_frame_summaries(analysis_prediction.transpose(1, 2, 0), threshold),
        "ground_truth_available": False,
        "ground_truth_frame_summaries": [],
        "hazard_detected": bool(forecast_regions),
        "hazard_region_count": len(forecast_regions),
        "hazard_regions": forecast_regions,
        "historical_storm_tracks": tracks,
        "full_resolution_viewer": True,
        "spatial_shape": [384, 384],
        "ensemble_count": live_metadata.get("ensemble_count") if model_name == "cascast_live" else None,
        "model_metrics": {"ground_truth_available": False},
        "interpretation_limit": limit,
    }


def get_event_summary(event_id: str) -> dict[str, Any]:
    """Return compact metadata and cached forecast availability for an event.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.

    Returns
    -------
    dict
        Event identity, timing, source data and forecast-cache status.
    """
    vil, metadata = _event(event_id)
    return {
        "status": "ok",
        "data_source": _cascast_source(str(metadata["event_id"])) if model_name == "cascast" else _source(),
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
    """Measure observed VIL exceedance at one explicit threshold.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    threshold : float
        Encoded VIL threshold.

    Returns
    -------
    dict
        Per-frame coverage and supporting provenance.
    """
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
    """Summarize whether observed coverage exceeds an explicit area rule.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    threshold : float
        Encoded VIL intensity threshold.
    min_area_ratio : float, default 0.001
        Minimum valid-grid fraction required for an active frame.

    Returns
    -------
    dict
        Active frames, coverage ratios and rule provenance.
    """
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


def analyze_area_ratio(
    event_id: str,
    model_name: str,
    threshold: float,
    area_ratio_rule: float = 0.001,
    lead_minutes: int = 60,
) -> dict[str, Any]:
    """Return the per-lead fraction of the valid event grid above threshold.

    This is deliberately separate from hazard detection: the rule is an explicit
    area fraction, never the legacy minimum-pixel shortcut.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    model_name : str
        Forecast implementation.
    threshold : float
        Encoded VIL threshold.
    area_ratio_rule : float, default 0.001
        Minimum active fraction used by the caller.
    lead_minutes : int, default 60
        Forecast horizon.

    Returns
    -------
    dict
        Per-lead valid-grid exceedance fractions and rule provenance.
    """
    if not 0.0 <= float(area_ratio_rule) <= 1.0:
        raise ValueError("area_ratio_rule must be between 0 and 1")
    if lead_minutes not in {60, 120, 180}:
        raise ValueError("lead_minutes must be 60, 120, or 180")
    forecast_frames = lead_minutes // 5
    prediction, arrays, metadata, target = _prepare_prediction_data(
        event_id, model_name, 13, forecast_frames
    )
    values = _vil_display_values(prediction)
    per_frame = (values >= threshold).mean(axis=(0, 1)).astype(float)
    active = per_frame >= float(area_ratio_rule)
    peak_index = int(per_frame.argmax())
    active_indices = [int(i) for i, flag in enumerate(active.tolist()) if flag]
    first = active_indices[0] if active_indices else None
    return {
        "status": "ok",
        "task": "area_ratio_analysis",
        "data_source": _cascast_source(str(metadata["event_id"])) if arrays is not None else _source(),
        "event_id": metadata["event_id"],
        "model_name": model_name,
        "lead_minutes": lead_minutes,
        "threshold": float(threshold),
        "region_scope": "full_valid_event_grid",
        "area_ratio_rule": float(area_ratio_rule),
        "per_frame_area_ratio": [round(float(item), 6) for item in per_frame.tolist()],
        "peak_area_ratio": round(float(per_frame[peak_index]), 6),
        "peak_frame_index": peak_index,
        "peak_minute": int((peak_index + 1) * 5),
        "rule_reached": bool(active.any()),
        "first_rule_reached_minute": int((first + 1) * 5) if first is not None else None,
        "active_frame_count": int(active.sum()),
        "full_resolution_viewer": True,
        "spatial_shape": [int(values.shape[0]), int(values.shape[1])],
        "frame_summaries": _prediction_frame_summaries(values, threshold),
        "interpretation_limit": "area is reported as fraction of the valid SEVIR event grid, not square kilometres; threshold exceedance is not an official hazard warning.",
    }


def analyze_duration(
    event_id: str,
    model_name: str,
    threshold: float,
    area_ratio_rule: float = 0.001,
    lead_minutes: int = 60,
) -> dict[str, Any]:
    """Measure cumulative and consecutive area-rule duration.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    model_name : str
        Forecast implementation.
    threshold : float
        Encoded VIL threshold.
    area_ratio_rule : float, default 0.001
        Minimum active fraction.
    lead_minutes : int, default 60
        Forecast horizon.

    Returns
    -------
    dict
        Total and longest consecutive active durations.
    """
    area = analyze_area_ratio(event_id, model_name, threshold, area_ratio_rule, lead_minutes)
    active = np.asarray(
        [ratio >= float(area_ratio_rule) for ratio in area["per_frame_area_ratio"]], dtype=bool
    )
    start, end, longest = _longest_true_run(active)
    return {
        "status": "ok",
        "task": "duration_analysis",
        "event_id": area["event_id"],
        "model_name": model_name,
        "threshold": float(threshold),
        "area_ratio_rule": float(area_ratio_rule),
        "lead_minutes": int(lead_minutes),
        "active_frames": int(active.sum()),
        "cumulative_duration_minutes": int(active.sum() * 5),
        "longest_consecutive_duration_minutes": int(longest * 5),
        "longest_consecutive_start_minute": int((start + 1) * 5) if longest else None,
        "longest_consecutive_end_minute": int((end + 1) * 5) if longest else None,
        "covers_horizon_end": bool(active[-1]) if active.size else False,
        "per_frame_area_ratio": area["per_frame_area_ratio"],
        "interpretation_limit": "duration is resolved at five-minute forecast intervals; it is not an official warning duration.",
    }


def _binary_csi(prediction: np.ndarray, target: np.ndarray, threshold: float) -> float:
    """Compute critical success index for aligned thresholded arrays.

    Parameters
    ----------
    prediction, target : np.ndarray
        Forecast and observed arrays in encoded VIL units.
    threshold : float
        Threshold defining predicted and observed positives.

    Returns
    -------
    float
        CSI in ``[0, 1]``; zero for an empty contingency denominator.
    """
    valid = target != 255
    predicted = (prediction >= threshold) & valid
    observed = (target >= threshold) & valid
    true_positive = np.logical_and(predicted, observed).sum()
    union = np.logical_or(predicted, observed).sum()
    return float(true_positive / union) if union else 0.0


def _deterministic_metrics(
    prediction: np.ndarray, target: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Evaluate prediction arrays on the encoded VIL scale.

    Parameters
    ----------
    prediction, target : np.ndarray
        Aligned time-last forecast and observed arrays.
    threshold : float
        Encoded VIL threshold for categorical scores.

    Returns
    -------
    dict
        Valid counts, contingency scores and continuous errors.
    """
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
    """Locate the longest consecutive active interval.

    Parameters
    ----------
    active : np.ndarray
        One-dimensional boolean sequence.

    Returns
    -------
    tuple of int
        Start index, end index and run length; ``(-1, -1, 0)`` if empty.
    """
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
    """Repeat the last input frame as a persistence forecast.

    Parameters
    ----------
    vil : np.ndarray
        Time-last VIL frame cube.
    input_frames : int
        Number of history frames used by the forecast.
    forecast_frames : int
        Number of future frames to create.

    Returns
    -------
    np.ndarray
        Persistence forecast with a time-last dimension.
    """
    last_observation = vil[:, :, input_frames - 1].astype(np.float32)
    return np.repeat(last_observation[:, :, None], forecast_frames, axis=2)


def _hazard_regions(
    mask: np.ndarray, metadata: dict[str, Any], max_regions: int = 3
) -> list[dict[str, Any]]:
    """Extract the largest connected regions from a binary hazard mask.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional boolean active-pixel mask.
    metadata : dict
        Event geometry used to attach spatial context.
    max_regions : int, default 3
        Maximum number of regions returned by descending area.

    Returns
    -------
    list of dict
        Region areas, centroids and pixel bounding boxes.
    """
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
            min_column, min_row, max_column, max_row = min(columns), min(rows), max(columns), max(rows)
            region["approx_latlon_bbox"] = {
                "south": round(max_lat - max_row / max(height - 1, 1) * (max_lat - min_lat), 4),
                "west": round(min_lon + min_column / max(width - 1, 1) * (max_lon - min_lon), 4),
                "north": round(max_lat - min_row / max(height - 1, 1) * (max_lat - min_lat), 4),
                "east": round(min_lon + max_column / max(width - 1, 1) * (max_lon - min_lon), 4),
            }
            longitude = min_lon + center_column / max(width - 1, 1) * (max_lon - min_lon)
            latitude = max_lat - center_row / max(height - 1, 1) * (max_lat - min_lat)
            region["approx_centroid_latlon"] = [round(latitude, 4), round(longitude, 4)]
            region["location_note"] = "由SEVIR事件边界线性映射得到的近似坐标"
            region["administrative_area"] = reverse_geocode_us(latitude, longitude)
        regions.append(region)
    return regions


def _prepare_prediction_data(
    event_id: str,
    model_name: str,
    input_frames: int,
    forecast_frames: int,
    ensemble_members: int = 10,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None, dict[str, Any], np.ndarray | None]:
    """Load prediction material without formatting a public result.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    model_name : str
        Forecast implementation.
    input_frames, forecast_frames : int
        Requested history and prediction lengths.
    ensemble_members : int, default 10
        Requested ensemble size.

    Returns
    -------
    tuple
        Prediction, optional cached arrays, metadata and optional ground truth.
    """
    if input_frames < 1 or forecast_frames < 1:
        raise ValueError("输入帧数和预测帧数必须大于0。")
    if model_name in {"cascast", "cascast_live"}:
        if forecast_frames not in {12, 24, 36}:
            raise ValueError("当前CasCast输出仅支持未来60分钟（12帧）；2/3小时滚动推理尚未执行。")
        if model_name == "cascast" and forecast_frames != 12:
            raise ValueError("precomputed CasCast cache only contains one 60-minute window; use cascast_live for same-event rolling prediction")
        if model_name == "cascast_live":
            segment_count = forecast_frames // 12
            loaded = []
            inference_records = []
            for segment in range(segment_count):
                inference_records.append(
                    run_local_cascast_inference(
                        event_id,
                        segment=segment,
                        ensemble_members=ensemble_members,
                    )
                )
                loaded.append(load_live_cascast_ensemble(event_id, segment=segment, ensemble_members=ensemble_members))
            first_arrays, first_metadata = loaded[0]
            arrays = {
                "history": first_arrays["history"],
                "ground_truth": np.concatenate([item[0]["ground_truth"] for item in loaded], axis=0),
                "ensemble_members": np.concatenate([item[0]["ensemble_members"] for item in loaded], axis=1),
                "ensemble_mean": np.concatenate([item[0]["ensemble_mean"] for item in loaded], axis=0),
                "pixel_std": np.concatenate([item[0]["pixel_std"] for item in loaded], axis=0),
                "q10": np.concatenate([item[0]["q10"] for item in loaded], axis=0),
                "q90": np.concatenate([item[0]["q90"] for item in loaded], axis=0),
                "member_seed": first_arrays["member_seed"],
            }
            metadata = {
                **first_metadata,
                "rolling": False,
                "forecast_mode": "observation_reset" if segment_count > 1 else "standard_single_window",
                "observation_reset": segment_count > 1,
                "rolling_segment_count": segment_count,
                "rolling_source_segments": [item[1].get("segment", index) for index, item in enumerate(loaded)],
                "ensemble_count": int(arrays["ensemble_members"].shape[0]),
                "cache_hit": bool(inference_records) and all(
                    bool(item.get("cache_hit", False)) for item in inference_records
                ),
                "inference_mode": (
                    "cache"
                    if inference_records and all(
                        bool(item.get("cache_hit", False)) for item in inference_records
                    )
                    else "live"
                ),
            }
        else:
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
    """Compute neutral scalar summaries for each prediction lead.

    Parameters
    ----------
    prediction : np.ndarray
        Time-last forecast cube.
    threshold : float
        Encoded VIL threshold used for coverage.

    Returns
    -------
    list of dict
        One summary record per forecast frame.
    """
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


def _ensemble_member_summaries(
    members: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    """Summarize ensemble members while leaving full grids in cache.

    Parameters
    ----------
    members : np.ndarray
        Member-first forecast array.
    threshold : float
        Encoded VIL threshold used for coverage.

    Returns
    -------
    list of dict
        Per-member intensity and coverage summaries.
    """
    values = np.asarray(members, dtype=np.float32)
    if values.ndim != 4:
        return []
    summaries: list[dict[str, Any]] = []
    for member_index, member in enumerate(values):
        active = member >= threshold
        per_frame_area = active.mean(axis=(1, 2))
        summaries.append(
            {
                "member": member_index + 1,
                "max_vil": round(float(member.max()), 3),
                "mean_vil": round(float(member.mean()), 3),
                "peak_exceeding_area_ratio": round(float(per_frame_area.max()), 6),
                "active_duration_minutes": int((active.sum(axis=(1, 2)) > 0).sum() * 5),
            }
        )
    return summaries


def _history_frame_summaries(arrays: dict[str, np.ndarray] | None) -> list[dict[str, Any]]:
    """Compute compact summaries for available input-history frames.

    Parameters
    ----------
    arrays : dict or None
        Loaded event arrays containing a ``history`` cube.

    Returns
    -------
    list of dict
        Per-frame timing and intensity summaries, or an empty list.
    """
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
    ensemble_members: int = 10,
) -> dict[str, Any]:
    """Return a 60-minute VIL forecast.

    ``model_name='cascast'`` reads the saved ten-member package,
    ``model_name='cascast_live'`` runs the local HDF5 event through
    EarthFormer, CasCast diffusion and VAE decoding, and
    ``model_name='persistence_baseline'`` repeats the final history frame.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    model_name : str
        ``cascast``, ``cascast_live`` or ``persistence_baseline``.
    threshold : float
        Encoded VIL threshold used for summaries.
    input_frames : int, default 13
        History-frame count.
    forecast_frames : int, default 12
        Forecast-frame count.
    ensemble_members : int, default 10
        Requested ensemble size.

    Returns
    -------
    dict
        Forecast arrays, lead summaries, optional truth and provenance.
    """
    prediction, arrays, metadata, target = _prepare_prediction_data(
        event_id, model_name, input_frames, forecast_frames, ensemble_members
    )
    analysis_prediction = _vil_display_values(prediction)
    frame_summaries = _prediction_frame_summaries(analysis_prediction, threshold)
    if arrays is not None:
        ground_truth_available = target is not None
        result = {
            "status": "ok",
            "task": "nowcast",
            "data_source": (
                live_cascast_source(metadata)
                if model_name == "cascast_live"
                else _cascast_source(str(metadata["event_id"]))
            ),
            "event_id": metadata["event_id"],
            "model_name": model_name,
            **_scientific_contract(
                inference_type=(
                    "local_hdf5_live_inference"
                    if model_name == "cascast_live"
                    else "precomputed_multi_event_ensemble"
                    if model_name == "cascast"
                    else "persistence_from_matching_cascast_history"
                ),
                ground_truth_available=ground_truth_available,
                input_frames=input_frames,
                forecast_frames=forecast_frames,
                lead_minutes=forecast_frames * 5,
            ),
            "input_frames": input_frames,
            "forecast_frames": forecast_frames,
            "lead_minutes": forecast_frames * 5,
            "rolling": bool(metadata.get("rolling", False)),
            "forecast_mode": metadata.get(
                "forecast_mode", "standard_single_window"
            ),
            "observation_reset": bool(metadata.get("observation_reset", False)),
            "rolling_segment_count": forecast_frames // 12,
            "inference_mode": (
                "local_hdf5_live_inference"
                if model_name == "cascast_live"
                else "precomputed_multi_event_ensemble"
                if model_name == "cascast"
                else "persistence_from_matching_cascast_history"
            ),
            "ensemble_count": metadata.get("ensemble_count") if model_name in {"cascast", "cascast_live"} else None,
            "output_type": (
                "ensemble_mean" if model_name in {"cascast", "cascast_live"} else "deterministic_baseline"
            ),
            "ensemble_members_available": bool(
                model_name in {"cascast", "cascast_live"}
                and arrays.get("ensemble_members") is not None
            ),
            "member_summaries": (
                _ensemble_member_summaries(arrays["ensemble_members"], threshold)
                if model_name in {"cascast", "cascast_live"}
                else []
            ),
            "ground_truth_available": ground_truth_available,
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
        if model_name in {"cascast", "cascast_live"}:
            result.update(
                {
                    "available_event_count": (
                        int(_ensemble_manifest()["samples"])
                        if model_name == "cascast"
                        else None
                    ),
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


def get_rolling_forecast_raw_values(
    event_id: str,
    horizon_hours: float = 2.0,
    model_name: str = "cascast_live",
    ensemble_members: int = 10,
    include_ground_truth: bool = False,
    array_transport: bool = False,
) -> dict[str, Any]:
    """Return raw 5-minute forecast grids for a saved 1-, 2-, or 3-hour forecast.

    This is a transport primitive only. It applies no threshold, region detector,
    trend label, or summary statistic; generated Agent code must choose and
    calculate those quantities from ``forecast_frames``.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    horizon_hours : float, default 2.0
        One-, two- or three-hour forecast horizon.
    model_name : str, default "cascast_live"
        Forecast implementation.
    ensemble_members : int, default 10
        Requested ensemble size.
    include_ground_truth : bool, default False
        Whether to return aligned observations when available.
    array_transport : bool, default False
        Whether to use the array transport representation.

    Returns
    -------
    dict
        Raw five-minute forecast grids, leads and provenance.
    """
    horizon = float(horizon_hours)
    if horizon not in {1.0, 2.0, 3.0}:
        raise ValueError("horizon_hours must be 1.0, 2.0 or 3.0")
    forecast_frames = int(horizon * 12)
    try:
        prediction, arrays, metadata, target = _prepare_prediction_data(
            event_id, model_name, 13, forecast_frames, ensemble_members
        )
    except RuntimeError as error:
        if "HDF5" in str(error) or "not downloaded" in str(error):
            return {
                "status": "awaiting_assets",
                "task": "rolling_forecast_raw",
                "event_id": event_id,
                "horizon_hours": horizon,
                "required_assets": ["same-event official SEVIR HDF5"],
                "error": str(error),
                "raw_data": False,
                "interpretation_limit": "Rolling raw forecasts require consecutive windows from the same event; unrelated cached events are never concatenated.",
            }
        raise
    values = _vil_display_values(prediction).astype(np.float32)
    frame_array = np.ascontiguousarray(values.transpose(2, 0, 1))
    frame_array.setflags(write=False)
    frames: Any = frame_array if array_transport else [frame.tolist() for frame in frame_array]
    history_array = np.ascontiguousarray(
        arrays["history"] if arrays is not None else np.empty((0, 0, 0)),
        dtype=np.float32,
    )
    history_array.setflags(write=False)
    history_frames: Any = (
        history_array
        if array_transport
        else [frame.tolist() for frame in history_array]
    )
    history_minutes = (
        [int(value) for value in _ensemble_manifest().get("history_minutes", [])]
        if arrays is not None
        else []
    )
    relative_minutes = list(range(5, forecast_frames * 5 + 1, 5))
    truth_array: np.ndarray | None = None
    if include_ground_truth and target is not None:
        truth_array = np.ascontiguousarray(target.transpose(2, 0, 1), dtype=np.float32)
        truth_array.setflags(write=False)
    member_records: list[dict[str, Any]] = []
    if arrays is not None and arrays.get("ensemble_members") is not None:
        member_values = _vil_display_values(
            np.asarray(arrays["ensemble_members"], dtype=np.float32)
        )
        seeds = np.asarray(arrays.get("member_seed", []), dtype=np.int64).tolist()
        for member_index, member in enumerate(member_values):
            member_frames = np.ascontiguousarray(member, dtype=np.float32)
            member_frames.setflags(write=False)
            member_records.append({
                "member": member_index + 1,
                "seed": int(seeds[member_index]) if member_index < len(seeds) else None,
                "lead_minutes": relative_minutes,
                "frames": member_frames if array_transport else member_frames.tolist(),
            })
    prediction_source = (
        metadata.get("cache_file")
        or metadata.get("prediction_file")
        or metadata.get("source")
    )
    actual_ensemble_count = int(metadata.get("ensemble_count", ensemble_members))
    forecast_evidence_id = _forecast_evidence_id(
        subject_id=str(metadata.get("event_id", event_id)),
        model_name=model_name,
        ensemble_members=actual_ensemble_count,
        horizon_hours=horizon,
        prediction_source=prediction_source,
    )
    return {
        "status": "ok",
        "task": "rolling_forecast_raw",
        "capabilities": [
            "current_forecast", "future_forecast", "forecast_grid", "forecast_evidence",
            *(["ensemble_agreement"] if member_records else []),
        ],
        "forecast_evidence_id": forecast_evidence_id,
        "forecast_evidence": {
            "id": forecast_evidence_id,
            "subject_kind": "sevir_event",
            "subject_id": str(metadata.get("event_id", event_id)),
            "model_name": model_name,
            "ensemble_count": actual_ensemble_count,
            "horizon_hours": horizon,
            "rolling_segments": int(horizon),
            "lead_minutes": relative_minutes,
            "forecast_frame_count": forecast_frames,
            "cache_hit": bool(metadata.get("cache_hit", False)),
            "ground_truth_attached": truth_array is not None,
        },
        "event_id": metadata.get("event_id", event_id),
        "model_name": model_name,
        "output_type": "ensemble_mean" if model_name == "cascast_live" and ensemble_members > 1 else "single_stochastic_member" if model_name == "cascast_live" else "deterministic_baseline",
        "is_deterministic": model_name != "cascast_live",
        "horizon_hours": horizon,
        # Keep time coordinates and forecast extent semantically distinct.
        # ``lead_minutes`` is always the coordinate aligned with the first
        # axis of every forecast grid; ``horizon_minutes`` is the scalar end
        # of the forecast.  The old scalar ``lead_minutes=60`` forced callers
        # to probe several aliases before they could safely index an array.
        "horizon_minutes": forecast_frames * 5,
        "lead_minutes": relative_minutes,
        "forecast_relative_minutes": relative_minutes,
        "forecast_frame_count": forecast_frames,
        "spatial_shape": [int(values.shape[0]), int(values.shape[1])],
        "dtype": "float32",
        "analysis_value_space": "decoded encoded-VIL scale before display quantization",
        "history_frames": history_frames,
        "history_relative_minutes": history_minutes,
        "forecast_frames": frames,
        "ensemble_members_available": bool(member_records),
        "ensemble": {
            "member_count": len(member_records),
            "members": member_records,
            "dimensions": ["member", "lead", "row", "column"],
            "semantics": "Configured CasCast members; member fractions are not calibrated probabilities.",
        },
        "cache_hit": bool(metadata.get("cache_hit", False)),
        "inference_mode": str(
            metadata.get("inference_mode")
            or ("cache" if metadata.get("cache_hit") else "live")
        ),
        "forecast_mode": str(metadata.get("forecast_mode") or "standard_single_window"),
        "observation_reset": bool(metadata.get("observation_reset", False)),
        "uses_future_observations": bool(metadata.get("observation_reset", False)),
        "rolling_segments": int(horizon),
        "raw_data": True,
        "ground_truth_requested": bool(include_ground_truth),
        "ground_truth_available": truth_array is not None,
        "ground_truth": {
            "available": truth_array is not None,
            "frames": truth_array if truth_array is not None else [],
            "lead_minutes": relative_minutes if truth_array is not None else [],
            "source": metadata.get("cache_file") if truth_array is not None else None,
            "event_id": metadata.get("event_id", event_id) if truth_array is not None else None,
            "missing_value": 255,
        },
        "interpretation_limit": "Raw encoded VIL grids only; no fixed fields, thresholds, regions, or statistics were computed. Ensemble-member fractions are model agreement, not calibrated probability.",
    }


def forecast_hazards(
    event_id: str,
    model_name: str,
    threshold: float,
    lead_minutes: int = 60,
    ensemble_members: int = 10,
) -> dict[str, Any]:
    """Run an event forecast and expose thresholded hazard evidence.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    model_name : str
        Forecast implementation selected by the caller.
    threshold : float
        Encoded VIL threshold for hazard evidence.
    lead_minutes : int, default 60
        Requested forecast horizon in minutes.
    ensemble_members : int, default 10
        Requested ensemble size.

    Returns
    -------
    dict
        Forecast provenance, lead summaries and connected hazard regions.
    """
    if lead_minutes not in {60, 120, 180}:
        raise ValueError("当前Demo支持未来1、2或3小时预测时效。")
    forecast_frames = lead_minutes // 5
    prediction, arrays, metadata, target = _prepare_prediction_data(
        event_id, model_name, 13, forecast_frames, ensemble_members
    )
    analysis_prediction = _vil_display_values(prediction)
    frame_summaries = _prediction_frame_summaries(analysis_prediction, threshold)
    hazard_mask = analysis_prediction.max(axis=2) >= threshold
    regions = _hazard_regions(hazard_mask, metadata)
    for region in regions:
        pixel_centroid = region.get("pixel_centroid")
        if pixel_centroid:
            pixel_location = geolocate_sevir_event(
                str(metadata["event_id"]),
                row=int(round(pixel_centroid[1])),
                column=int(round(pixel_centroid[0])),
            )["pixel_location"]
            centroid = [pixel_location["latitude"], pixel_location["longitude"]]
            region["approx_centroid_latlon"] = centroid
            region["location_note"] = "SEVIR投影坐标转换；州县为区域中心点的边界匹配结果"
            region["administrative_area"] = reverse_geocode_us(centroid[0], centroid[1])
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
    _, _, any_pixel_longest = _longest_true_run(any_pixel_active)
    duration = int(any_pixel_longest * 5)
    return {
        "status": "ok",
        "data_source": (
            live_cascast_source(metadata)
            if arrays is not None and model_name == "cascast_live"
            else _cascast_source(str(metadata["event_id"]))
            if arrays is not None
            else _source()
        ),
        "event_id": metadata["event_id"],
        "task": "future_hazard_analysis",
        "model_name": model_name,
        **_scientific_contract(
            inference_type=(
                "local_hdf5_live_inference"
                if model_name == "cascast_live"
                else "precomputed_multi_event_ensemble"
                if model_name == "cascast"
                else "persistence_baseline"
            ),
            ground_truth_available=target is not None,
            input_frames=int(arrays["history"].shape[0]) if arrays is not None else 13,
            forecast_frames=int(analysis_prediction.shape[2]),
            lead_minutes=lead_minutes,
        ),
        "lead_minutes": lead_minutes,
        "threshold": threshold,
        "hazard_detected": bool(any_pixel_active.any()),
        "hazard_region_count": len(regions),
        "hazard_regions": regions,
        "threshold_exceeded": bool(any_pixel_active.any()),
        "maximum_exceeding_pixel_count": int(exceeding_pixel_counts.max()),
        "any_pixel_exceedance_total_minutes": int(any_pixel_active.sum() * 5),
        "any_pixel_exceedance_longest_consecutive_minutes": int(any_pixel_longest * 5),
        "threshold_exceedance_rule": "any_valid_pixel_at_or_above_threshold",
        "strong_echo_duration_minutes_within_horizon": duration,
        "duration_is_lower_bound": bool(any_pixel_active.all()),
        "frame_summaries": frame_summaries,
        "ground_truth_frame_summaries": _prediction_frame_summaries(target, threshold) if target is not None else [],
        "history_available": arrays is not None,
        "history_frames": int(arrays["history"].shape[0]) if arrays is not None else 0,
        "history_frame_summaries": _history_frame_summaries(arrays),
        "full_resolution_viewer": arrays is not None,
        "spatial_shape": [384, 384],
        "ensemble_count": metadata.get("ensemble_count") if model_name in {"cascast", "cascast_live"} else None,
        "output_type": (
            "ensemble_mean" if model_name in {"cascast", "cascast_live"} else "deterministic_baseline"
        ),
        "ensemble_members_available": bool(
            model_name in {"cascast", "cascast_live"}
            and arrays is not None
            and arrays.get("ensemble_members") is not None
        ),
        "member_summaries": (
            _ensemble_member_summaries(arrays["ensemble_members"], threshold)
            if model_name in {"cascast", "cascast_live"} and arrays is not None
            else []
        ),
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
    minimum_region_pixels: int = 1,
) -> dict[str, Any]:
    """Describe disagreement among cached CasCast ensemble members.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    threshold : float
        Encoded VIL threshold for exceedance probabilities.
    minimum_region_pixels : int, default 1
        Smallest connected uncertainty region retained.

    Returns
    -------
    dict
        Spread fields, member summaries and spatial disagreement evidence.
    """
    arrays, metadata = _load_cascast_ensemble(event_id)
    members = arrays["ensemble_members"].astype(np.float32)
    mean_prediction = arrays["ensemble_mean"]
    ground_truth = arrays["ground_truth"].astype(np.float32)
    pixel_std = arrays["pixel_std"]
    member_hashes = [
        hashlib.sha1(np.ascontiguousarray(member).tobytes()).hexdigest()[:16]
        for member in members
    ]
    exceedance_probability = (members >= threshold).mean(axis=0)

    member_summaries: list[dict[str, Any]] = []
    object_centroids: list[list[float]] = []
    largest_object_area_ratios: list[float] = []
    for member_index, member in enumerate(members):
        valid = member != 255
        exceedance = valid & (member >= threshold)
        valid_counts = valid.sum(axis=(1, 2))
        per_frame_area = np.divide(
            exceedance.sum(axis=(1, 2)),
            valid_counts,
            out=np.zeros_like(valid_counts, dtype=np.float32),
            where=valid_counts > 0,
        )
        active_frames = exceedance.sum(axis=(1, 2)) >= minimum_region_pixels
        per_lead_metrics = []
        for lead_index, frame in enumerate(member):
            frame_valid = frame != 255
            valid_values = frame[frame_valid]
            per_lead_metrics.append(
                {
                    "lead_minutes": (lead_index + 1) * 5,
                    "max_vil": round(float(valid_values.max()), 3) if valid_values.size else None,
                    "mean_vil": round(float(valid_values.mean()), 3) if valid_values.size else None,
                    "p95_vil": round(float(np.quantile(valid_values, 0.95)), 3) if valid_values.size else None,
                    "threshold_coverage": round(float(per_frame_area[lead_index]), 6),
                    "exceeding_pixel_count": int(exceedance[lead_index].sum()),
                }
            )
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
                "per_lead_metrics": per_lead_metrics,
            }
        )

    if object_centroids:
        centroid_array = np.asarray(object_centroids, dtype=np.float32)
        centroid_spread = float(np.sqrt(np.var(centroid_array, axis=0).sum()))
        pairwise_distances = [
            float(np.linalg.norm(left - right))
            for index, left in enumerate(centroid_array)
            for right in centroid_array[index + 1 :]
        ]
    else:
        centroid_spread = 0.0
        pairwise_distances = []

    uncertainty_frames = []
    for frame_index in range(int(mean_prediction.shape[0])):
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
        **_scientific_contract(
            inference_type="precomputed_multi_event_ensemble",
            ground_truth_available=True,
            input_frames=int(arrays["history"].shape[0]),
            forecast_frames=int(mean_prediction.shape[0]),
            lead_minutes=int(mean_prediction.shape[0] * 5),
        ),
        "lead_minutes": int(mean_prediction.shape[0] * 5),
        "forecast_frames": int(mean_prediction.shape[0]),
        "ensemble_count": int(members.shape[0]),
        "member_validation": {
            "expected_member_count": 10,
            "loaded_member_count": int(members.shape[0]),
            "unique_member_count": len(set(member_hashes)),
            "member_arrays_independent": len(set(member_hashes)) > 1,
            "member_hashes": member_hashes,
            "metrics_source": "ensemble_members",
            "ensemble_mean_is_summary_only": True,
        },
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
            "pairwise_centroid_distance_mean_pixels": round(float(np.mean(pairwise_distances)), 3) if pairwise_distances else 0.0,
            "pairwise_centroid_distance_max_pixels": round(float(max(pairwise_distances)), 3) if pairwise_distances else 0.0,
            "consensus_rule": "centroid spread and pairwise pixel distances; not a physical error estimate",
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
            f"不确定性来自{members.shape[0]}个CasCast随机ensemble成员之间的离散程度；"
            "它描述模型预测分歧，不等于真实误差或灾害概率。"
            "准确率指标只对应当前所选测试样本。"
        ),
    }


UPLOADED_ENSEMBLE_MEMBER_FIELDS = {
    "mean_vil": "该成员该时效整幅网格的VIL均值（0至254缩放值）",
    "max_vil": "该成员该时效整幅网格的VIL最大值（0至254缩放值）",
    "exceeding_pixel_count": "该成员该时效达到所选阈值的像素数",
    "exceeding_area_ratio": "该成员该时效达到所选阈值的网格面积比例",
    "threshold_centroid_x": "全部阈值超标像素的横向质心；无超标像素时为null",
    "threshold_centroid_y": "全部阈值超标像素的纵向质心；无超标像素时为null",
    "largest_region_pixel_count": "最大阈值连通区的像素数",
    "largest_region_area_ratio": "最大阈值连通区的网格面积比例",
    "largest_region_centroid_x": "最大阈值连通区横向质心；无满足大小要求的区域时为null",
    "largest_region_centroid_y": "最大阈值连通区纵向质心；无满足大小要求的区域时为null",
}


def get_uploaded_ensemble_raw_values(
    upload_id: str,
    lead_minutes: list[int],
    ensemble_members: int = 10,
    horizon_hours: float = 1.0,
    include_ground_truth: bool = False,
    array_transport: bool = False,
    force_live: bool = False,
) -> dict[str, Any]:
    """Return raw CasCast member grids for selected leads.

    ``lead_minutes`` is a data slice, not a scientific field selection.  No
    thresholding, region detection, summary statistic, or comparison is done
    here.  The generated Agent program must decide what quantities to derive
    from these member pixels.  Leads are explicit to keep the JSON worker's
    memory bounded while preserving a Zephyrus-like raw-data contract.

    Parameters
    ----------
    upload_id : str
        Uploaded history identifier.
    lead_minutes : list of int
        Explicit forecast leads to return.
    ensemble_members : int, default 10
        Requested ensemble size.
    horizon_hours : float, default 1.0
        Forecast horizon.
    include_ground_truth : bool, default False
        Whether to include available future observations.
    array_transport : bool, default False
        Whether to use the array transport representation.
    force_live : bool, default False
        Whether to bypass reusable inference output.

    Returns
    -------
    dict
        Member grids at selected leads and scientific provenance.
    """
    horizon = float(horizon_hours)
    if horizon not in {1.0, 2.0, 3.0}:
        raise ValueError("horizon_hours must be 1.0, 2.0 or 3.0")
    maximum_lead = int(horizon * 60)
    selected_leads = list(dict.fromkeys(int(value) for value in lead_minutes))
    if not selected_leads:
        raise ValueError("lead_minutes must contain at least one forecast lead")
    invalid_leads = [
        value for value in selected_leads if value < 5 or value > maximum_lead or value % 5 != 0
    ]
    if invalid_leads:
        raise ValueError(
            f"lead_minutes values must be 5 through {maximum_lead} in 5-minute increments"
        )
    if len(selected_leads) > 4:
        raise ValueError("at most four forecast leads may be loaded as raw member grids")
    rolling_segments = int(horizon)
    inference = run_uploaded_cascast_inference(
        upload_id,
        ensemble_members=ensemble_members,
        rolling_segments=rolling_segments,
        force=bool(force_live),
    )
    arrays, metadata = load_uploaded_cascast_ensemble(
        upload_id, ensemble_members=ensemble_members, rolling_segments=rolling_segments
    )
    members = _vil_display_values(arrays["ensemble_members"])
    seeds = np.asarray(arrays["member_seed"]).astype(np.int64)
    records = []
    for member_index, member in enumerate(members):
        selected_frames = np.ascontiguousarray(
            [member[lead // 5 - 1] for lead in selected_leads], dtype=np.float32
        )
        selected_frames.setflags(write=False)
        records.append(
            {
                "member": member_index + 1,
                "seed": int(seeds[member_index]),
                "output_type": "stochastic_ensemble_member",
                "is_deterministic": False,
                "lead_minutes": selected_leads,
                "frames": selected_frames if array_transport else selected_frames.tolist(),
            }
        )
    if include_ground_truth:
        ground_truth = load_uploaded_sequence_hdf5_ground_truth(upload_id, selected_leads)
    else:
        ground_truth = {
            "available": False,
            "requested": False,
            "lead_minutes": [],
            "frames": [],
            "reason": "ground_truth_not_requested",
        }
    forecast_evidence_id = _forecast_evidence_id(
        subject_id=upload_id,
        model_name="cascast_live",
        ensemble_members=int(len(records)),
        horizon_hours=horizon,
        prediction_source=metadata.get("cache_file"),
    )
    return {
        "status": "ok",
        "task": "uploaded_cascast_raw_ensemble",
        "capabilities": [
            "current_forecast", "future_forecast", "forecast_grid",
            "ensemble_members", "forecast_evidence",
        ] + (["ground_truth"] if ground_truth["available"] else []),
        "forecast_evidence_id": forecast_evidence_id,
        "forecast_evidence": {
            "id": forecast_evidence_id,
            "subject_kind": "uploaded_sequence",
            "subject_id": upload_id,
            "model_name": "cascast_live",
            "ensemble_count": int(len(records)),
            "member_seeds": [int(value) for value in seeds.tolist()],
            "horizon_hours": horizon,
            "rolling_segments": rolling_segments,
            "lead_minutes": list(range(5, maximum_lead + 1, 5)),
            "selected_lead_minutes": selected_leads,
            "forecast_frame_count": int(members.shape[1]),
            "cache_hit": bool(inference.get("cache_hit", False)),
            "ground_truth_attached": bool(ground_truth["available"]),
        },
        "upload_id": upload_id,
        "model_name": "cascast_live",
        "output_type": "stochastic_ensemble",
        "is_deterministic": False,
        "ensemble_members": int(len(records)),
        "member_seeds": [int(value) for value in seeds.tolist()],
        "selected_lead_minutes": selected_leads,
        "horizon_hours": horizon,
        "rolling_segments": rolling_segments,
        "forecast_frame_count": int(members.shape[1]),
        "shape": [384, 384],
        "dtype": "float32",
        "analysis_value_space": "CasCast decoded encoded-VIL scale before display quantization",
        "frame_layout": "records[member]['frames'][index] aligns with records[member]['lead_minutes'][index]",
        "display_conversion": "clip to 0..255 and round only when rendering an image",
        "records": records,
        "statistics_computed": [],
        "fields_computed": [],
        "threshold_applied": None,
        "region_parameters_applied": None,
        "raw_data": True,
        "cache_hit": bool(inference.get("cache_hit", False)),
        "ground_truth_requested": bool(include_ground_truth),
        "ground_truth_available": bool(ground_truth["available"]),
        "ground_truth": ground_truth,
        "provenance": {
            "prediction_source": metadata.get("cache_file"),
            "rolling_segments": int(metadata.get("rolling_segments", 1)),
            "ground_truth_source": ground_truth.get("source") if ground_truth["available"] else None,
            "ground_truth_match_method": ground_truth.get("match_method") if ground_truth["available"] else None,
            "ground_truth_event_id": ground_truth.get("event_id") if ground_truth["available"] else None,
        },
        "verification_rule": (
            "Compare a forecast lead only with the ground_truth frame at the same "
            "lead_minutes position. Ignore pixels whose truth value is 255. Never "
            "use the final history frame (t=0) as future ground truth."
        ),
    }


def query_uploaded_ensemble_member_values(
    upload_id: str,
    fields: list[str],
    lead_minutes: list[int],
    threshold: float = 133.0,
    minimum_region_pixels: int = 1,
    ensemble_members: int = 10,
) -> dict[str, Any]:
    """Return selected member primitives for an uploaded forecast.

    Parameters
    ----------
    upload_id : str
        Uploaded history identifier.
    fields : list of str
        Primitive fields requested by the caller.
    lead_minutes : list of int
        Forecast leads to inspect.
    threshold : float, default 133.0
        Encoded VIL threshold.
    minimum_region_pixels : int, default 1
        Smallest connected region retained.
    ensemble_members : int, default 10
        Requested ensemble size.

    Returns
    -------
    dict
        Selected member/lead primitives and evidence identity.
    """
    selected_fields = list(dict.fromkeys(str(field) for field in fields))
    if not selected_fields:
        raise ValueError("fields must contain at least one ensemble member field")
    unknown_fields = sorted(set(selected_fields) - set(UPLOADED_ENSEMBLE_MEMBER_FIELDS))
    if unknown_fields:
        raise ValueError(
            "unsupported uploaded ensemble field(s): "
            + ", ".join(unknown_fields)
            + "; available fields: "
            + ", ".join(sorted(UPLOADED_ENSEMBLE_MEMBER_FIELDS))
        )
    selected_leads = list(dict.fromkeys(int(value) for value in lead_minutes))
    if not selected_leads:
        raise ValueError("lead_minutes must contain at least one forecast lead")
    invalid_leads = [
        value for value in selected_leads if value < 5 or value > 60 or value % 5 != 0
    ]
    if invalid_leads:
        raise ValueError("lead_minutes values must be 5 through 60 in 5-minute increments")

    inference = run_uploaded_cascast_inference(
        upload_id, ensemble_members=ensemble_members
    )
    arrays, metadata = load_uploaded_cascast_ensemble(
        upload_id, ensemble_members=ensemble_members
    )
    members = _vil_display_values(arrays["ensemble_members"])
    member_hashes = [
        hashlib.sha1(np.ascontiguousarray(member).tobytes()).hexdigest()[:16]
        for member in members
    ]
    member_seeds = np.asarray(arrays["member_seed"]).astype(np.int64)
    needs_mask = any(
        field
        in {
            "exceeding_pixel_count",
            "exceeding_area_ratio",
            "threshold_centroid_x",
            "threshold_centroid_y",
            "largest_region_pixel_count",
            "largest_region_area_ratio",
            "largest_region_centroid_x",
            "largest_region_centroid_y",
        }
        for field in selected_fields
    )
    needs_largest_region = any(
        field.startswith("largest_region_") for field in selected_fields
    )
    records: list[dict[str, Any]] = []
    for member_index, member in enumerate(members):
        for lead in selected_leads:
            frame = member[lead // 5 - 1]
            record: dict[str, Any] = {
                "member": member_index + 1,
                "seed": int(member_seeds[member_index]),
                "lead_minutes": lead,
            }
            mask = frame >= threshold if needs_mask else None
            mask_rows: np.ndarray | None = None
            mask_columns: np.ndarray | None = None
            largest_region: dict[str, Any] | None = None
            if mask is not None and (
                "threshold_centroid_x" in selected_fields
                or "threshold_centroid_y" in selected_fields
            ):
                mask_rows, mask_columns = np.nonzero(mask)
            if mask is not None and needs_largest_region:
                regions = [
                    region
                    for region in _hazard_regions(mask, metadata, max_regions=1)
                    if region["pixel_count"] >= minimum_region_pixels
                ]
                largest_region = regions[0] if regions else None
            for field in selected_fields:
                if field == "mean_vil":
                    value: Any = round(float(frame.mean()), 3)
                elif field == "max_vil":
                    value = round(float(frame.max()), 3)
                elif field == "exceeding_pixel_count":
                    value = int(mask.sum()) if mask is not None else 0
                elif field == "exceeding_area_ratio":
                    value = round(float(mask.mean()), 6) if mask is not None else 0.0
                elif field == "threshold_centroid_x":
                    value = (
                        round(float(mask_columns.mean()), 3)
                        if mask_columns is not None and mask_columns.size
                        else None
                    )
                elif field == "threshold_centroid_y":
                    value = (
                        round(float(mask_rows.mean()), 3)
                        if mask_rows is not None and mask_rows.size
                        else None
                    )
                elif field == "largest_region_pixel_count":
                    value = int(largest_region["pixel_count"]) if largest_region else 0
                elif field == "largest_region_area_ratio":
                    value = float(largest_region["area_ratio"]) if largest_region else 0.0
                elif field == "largest_region_centroid_x":
                    value = float(largest_region["pixel_centroid"][0]) if largest_region else None
                else:
                    value = float(largest_region["pixel_centroid"][1]) if largest_region else None
                record[field] = value
            records.append(record)

    return {
        "status": "ok",
        "task": "uploaded_ensemble_member_values",
        "analysis_scope": "agent_selected_cross_member_primitives",
        "upload_id": upload_id,
        "data_source": {
            "kind": "user_uploaded_sequence_local_cascast",
            "upload_id": upload_id,
            "prediction_file": metadata["cache_file"],
            "cache_hit": inference["cache_hit"],
        },
        "model_name": "cascast_live",
        **_scientific_contract(
            inference_type="local_uploaded_sequence_cascast_ensemble",
            ground_truth_available=False,
            input_frames=int(arrays["history"].shape[0]),
            forecast_frames=int(members.shape[1]),
            lead_minutes=int(members.shape[1] * 5),
            location_precision="uploaded_grid_only",
        ),
        "ensemble_count": int(members.shape[0]),
        "ensemble_integrity": {
            "member_count": int(members.shape[0]),
            "unique_member_count": len(set(member_hashes)),
            "unique_seed_count": len(set(int(value) for value in arrays["member_seed"])),
            "member_hashes": member_hashes,
            "metrics_source": "ensemble_members",
            "ensemble_mean_is_summary_only": True,
        },
        "selected_fields": selected_fields,
        "selected_lead_minutes": selected_leads,
        "threshold": threshold,
        "minimum_region_pixels": minimum_region_pixels,
        "record_count": len(records),
        "records": records,
        "field_definitions": {
            field: UPLOADED_ENSEMBLE_MEMBER_FIELDS[field] for field in selected_fields
        },
        "member_difference_cause": (
            "相同13帧输入与同一CasCast模型，使用不同随机种子采样得到不同集合成员。"
        ),
        "calculated_statistics": [],
        "interpretation_limit": (
            "工具只返回Agent所选成员字段的基础记录，不计算极差、标准差、分位数、"
            "成员间距离或差异等级；这些统计量及判断必须由Agent生成的代码完成。"
            "成员一致程度不是校准后的灾害概率。用户未上传未来Ground Truth，"
            "因此不能判断哪个成员正确，也不能把成员离散度直接当成真实预报误差。"
        ),
    }


def analyze_object_tracking(
    event_id: str,
    model_name: str,
    threshold: float,
    minimum_region_pixels: int = 1,
    lead_minutes: int = 60,
) -> dict[str, Any]:
    """Track thresholded connected objects across forecast frames.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    model_name : str
        Forecast implementation.
    threshold : float
        Encoded VIL object threshold.
    minimum_region_pixels : int, default 1
        Smallest object retained.
    lead_minutes : int, default 60
        Forecast horizon.

    Returns
    -------
    dict
        Object tracks, positions, lifetimes and provenance.
    """
    if lead_minutes not in {60, 120, 180}:
        raise ValueError("lead_minutes must be 60, 120, or 180")
    prediction, arrays, metadata, _ = _prepare_prediction_data(
        event_id, model_name, 13, lead_minutes // 5
    )
    fields = _vil_display_values(prediction).transpose(2, 0, 1)
    tracking = _storm_tracks(fields, metadata, threshold, minimum_region_pixels)
    return {
        "status": "ok",
        "task": "storm_object_tracking",
        "event_id": metadata["event_id"],
        "model_name": model_name,
        **_scientific_contract(
            inference_type="persistence_baseline",
            ground_truth_available=True,
            input_frames=input_frames,
            forecast_frames=forecast_frames,
            lead_minutes=forecast_frames * 5,
        ),
        "threshold": float(threshold),
        "minimum_region_pixels": int(minimum_region_pixels),
        "lead_minutes": int(lead_minutes),
        "tracking_rule": tracking["association_method"],
        **tracking,
        "data_source": _cascast_source(str(metadata["event_id"])) if arrays is not None else _source(),
        "interpretation_limit": "centroid displacement is in SEVIR pixels; it is not a calibrated physical storm track.",
    }


def analyze_historical_anomaly(
    event_id: str,
    threshold: float = 133.0,
    quantile: float = 0.9,
    minimum_region_pixels: int = 1,
) -> dict[str, Any]:
    """Compare observed history against the cached event-history baseline.

    The baseline is explicitly the available CasCast test-event cache, not a
    climatological claim about all SEVIR observations.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    threshold : float, default 133.0
        Encoded VIL threshold.
    quantile : float, default 0.9
        Baseline quantile used for anomaly comparison.
    minimum_region_pixels : int, default 1
        Smallest observed region retained.

    Returns
    -------
    dict
        Event-history placement against the explicitly scoped cache baseline.
    """
    if not 0.5 <= float(quantile) < 1.0:
        raise ValueError("quantile must be in [0.5, 1.0)")
    target_arrays, target_meta = _load_cascast_ensemble(event_id)
    manifest = _ensemble_manifest()
    baseline_max: list[float] = []
    baseline_area: list[float] = []
    for item in manifest.get("events", []):
        try:
            arrays, _ = _load_cascast_ensemble(str(item["event_id"]))
        except (RuntimeError, ValueError):
            continue
        history = _vil_display_values(arrays["history"])
        baseline_max.extend(history.max(axis=(1, 2)).astype(float).tolist())
        baseline_area.extend((history >= threshold).mean(axis=(1, 2)).astype(float).tolist())
    if not baseline_max:
        raise RuntimeError("historical event cache is unavailable")
    target_history = _vil_display_values(target_arrays["history"])
    target_max = target_history.max(axis=(1, 2)).astype(float)
    target_area = (target_history >= threshold).mean(axis=(1, 2)).astype(float)
    max_cutoff = float(np.quantile(np.asarray(baseline_max), quantile))
    area_cutoff = float(np.quantile(np.asarray(baseline_area), quantile))
    max_anomaly = target_max >= max_cutoff
    area_anomaly = target_area >= area_cutoff
    return {
        "status": "ok",
        "task": "historical_anomaly_analysis",
        "event_id": target_meta["event_id"],
        "threshold": float(threshold),
        "quantile": float(quantile),
        "baseline": {
            "source": "cached_cascast_event_history",
            "event_count": len(manifest.get("events", [])),
            "frame_count": len(baseline_max),
            "max_vil_cutoff": round(max_cutoff, 4),
            "area_ratio_cutoff": round(area_cutoff, 6),
            "minimum_region_pixels": int(minimum_region_pixels),
        },
        "target_history": {
            "max_vil": [round(float(value), 4) for value in target_max.tolist()],
            "area_ratio": [round(float(value), 6) for value in target_area.tolist()],
            "max_vil_anomaly": max_anomaly.tolist(),
            "area_ratio_anomaly": area_anomaly.tolist(),
            "anomalous_frame_count": int(np.logical_or(max_anomaly, area_anomaly).sum()),
        },
        "interpretation_limit": "This is an anomaly relative to the cached CasCast test-event histories, not a uniform climatology or official severe-weather probability.",
    }


def analyze_composite_hazard(
    event_id: str,
    threshold: float = 133.0,
    area_ratio_rule: float = 0.001,
    minimum_consecutive_minutes: int = 15,
    lead_minutes: int = 60,
) -> dict[str, Any]:
    """Combine explicit evidence without issuing an official warning.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    threshold : float, default 133.0
        Encoded VIL threshold.
    area_ratio_rule : float, default 0.001
        Minimum active fraction.
    minimum_consecutive_minutes : int, default 15
        Minimum persistence interval.
    lead_minutes : int, default 60
        Forecast horizon.

    Returns
    -------
    dict
        Independently calculated intensity, area and duration evidence.
    """
    area = analyze_area_ratio(event_id, "cascast", threshold, area_ratio_rule, lead_minutes)
    duration = analyze_duration(event_id, "cascast", threshold, area_ratio_rule, lead_minutes)
    ensemble = analyze_ensemble_uncertainty(event_id, threshold)
    anomaly = analyze_historical_anomaly(event_id, threshold, 0.9)
    evidence = {
        "area_rule_reached": bool(area["rule_reached"]),
        "peak_area_ratio": area["peak_area_ratio"],
        "longest_consecutive_duration_minutes": duration["longest_consecutive_duration_minutes"],
        "duration_rule_reached": duration["longest_consecutive_duration_minutes"] >= int(minimum_consecutive_minutes),
        "ensemble_count": ensemble["ensemble_count"],
        "centroid_spread_pixels": ensemble["storm_object_uncertainty"]["centroid_spread_pixels"],
        "historical_anomalous_frames": anomaly["target_history"]["anomalous_frame_count"],
    }
    candidate = bool(evidence["area_rule_reached"] and evidence["duration_rule_reached"])
    return {
        "status": "ok",
        "task": "composite_hazard_analysis",
        "event_id": event_id,
        "threshold": float(threshold),
        "lead_minutes": int(lead_minutes),
        "area_ratio_rule": float(area_ratio_rule),
        "minimum_consecutive_minutes": int(minimum_consecutive_minutes),
        "candidate_strong_echo": candidate,
        "evidence": evidence,
        "components": {"area_ratio": area, "duration": duration, "ensemble": ensemble, "historical_anomaly": anomaly},
        "interpretation_limit": "This is a transparent VIL evidence synthesis; it is not an official hazard, rainfall, flood, or warning classification.",
    }


def analyze_physical_area(
    event_id: str,
    model_name: str,
    threshold: float,
    lead_minutes: int = 60,
) -> dict[str, Any]:
    """Convert above-threshold pixels to approximate physical area.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    model_name : str
        Forecast implementation.
    threshold : float
        Encoded VIL threshold.
    lead_minutes : int, default 60
        Forecast horizon.

    Returns
    -------
    dict
        Per-lead active pixels and approximate square kilometres.
    """
    if lead_minutes not in {60, 120, 180}:
        raise ValueError("lead_minutes must be 60, 120, or 180")
    prediction, arrays, metadata, _ = _prepare_prediction_data(event_id, model_name, 13, lead_minutes // 5)
    bounds = metadata.get("catalog_row", {})
    required = [bounds.get(key) for key in ("llcrnrlat", "llcrnrlon", "urcrnrlat", "urcrnrlon")]
    if any(value is None for value in required):
        raise ValueError("event bounds are unavailable; physical area cannot be calibrated")
    ll_lat, ll_lon, ur_lat, ur_lon = map(float, required)
    geod = Geod(ellps="WGS84")
    _, _, width_m = geod.inv(ll_lon, ll_lat, ur_lon, ll_lat)
    _, _, height_m = geod.inv(ll_lon, ll_lat, ll_lon, ur_lat)
    rows, cols = prediction.shape[:2]
    pixel_area_km2 = abs(width_m * height_m) / (rows * cols) / 1_000_000.0
    values = _vil_display_values(prediction)
    counts = (values >= threshold).sum(axis=(0, 1)).astype(int)
    areas = counts.astype(float) * pixel_area_km2
    peak = int(areas.argmax())
    return {
        "status": "ok",
        "task": "physical_area_analysis",
        "event_id": metadata["event_id"],
        "model_name": model_name,
        "threshold": float(threshold),
        "lead_minutes": int(lead_minutes),
        "projection": "WGS84 event bounding-box approximation",
        "pixel_area_km2": round(float(pixel_area_km2), 6),
        "grid_shape": [int(rows), int(cols)],
        "per_frame_exceeding_pixel_count": counts.tolist(),
        "per_frame_area_km2": [round(float(value), 4) for value in areas.tolist()],
        "peak_area_km2": round(float(areas[peak]), 4),
        "peak_minute": int((peak + 1) * 5),
        "bounds": {"llcrnrlat": ll_lat, "llcrnrlon": ll_lon, "urcrnrlat": ur_lat, "urcrnrlon": ur_lon},
        "interpretation_limit": "Area uses a latitude/longitude bounding-box approximation; it is not a local projected-grid survey measurement.",
    }


def _bbox_iou(first: list[int], second: list[int]) -> float:
    """Compute intersection-over-union for two pixel bounding boxes.

    Parameters
    ----------
    first, second : list of int
        Boxes represented as ``[row_min, col_min, row_max, col_max]``.

    Returns
    -------
    float
        Intersection area divided by union area.
    """
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
    minimum_region_pixels: int = 1,
    track_limit: int = 10,
) -> dict[str, Any]:
    """Track connected regions across consecutive five-minute frames.

    Parameters
    ----------
    fields : np.ndarray
        Time-first VIL frames.
    metadata : dict
        Event geometry and timing metadata.
    threshold : float
        Encoded VIL object threshold.
    minimum_region_pixels : int, default 1
        Smallest connected region retained.
    track_limit : int, default 10
        Maximum tracks returned.

    Returns
    -------
    dict
        Linked object tracks, durations and displacement summaries.
    """
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
                "start_frame_index": int(frames[0]),
                "end_frame_index": int(frames[-1]),
                "start_minute": int((frames[0] + 1) * 5),
                "end_minute": int((frames[-1] + 1) * 5),
                "duration_minutes": int((frames[-1] - frames[0]) * 5),
                "duration_definition": "elapsed time between first and last matched frames",
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
        "tracks": public_tracks[: int(track_limit)],
        "association_method": "consecutive-frame connected components matched by bbox overlap or centroid distance <=32 pixels",
    }


def _storm_object_verification(
    prediction: np.ndarray,
    target: np.ndarray,
    metadata: dict[str, Any],
    threshold: float,
    minimum_region_pixels: int = 1,
) -> dict[str, Any]:
    """Verify thresholded forecast objects against aligned observations.

    Parameters
    ----------
    prediction, target : np.ndarray
        Time-aligned forecast and observed VIL frames.
    metadata : dict
        Event geometry and timing metadata.
    threshold : float
        Encoded VIL threshold defining objects.
    minimum_region_pixels : int, default 1
        Smallest connected object included in matching.

    Returns
    -------
    dict
        Framewise matches, misses, false alarms and centroid errors.
    """
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
