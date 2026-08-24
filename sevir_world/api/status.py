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
from ..data_access.catalog import _catalog_vil_records
from ..data_access.live_inference import live_cascast_readiness
from ..data_access.ensemble import (
    _earthformer_latent_status,
    _ensemble_manifest,
    _event,
    _load_cascast_prediction,
)

import h5py
import numpy as np
from pyproj import CRS, Transformer


def run_satellite_to_radar(event_id: str) -> dict[str, Any]:
    _, metadata = _event(event_id)
    return {
        "status": "awaiting_model_assets",
        "event_id": metadata["event_id"],
        "task": "satellite_to_radar_synthesis",
        "required_assets": [
            "卫星转雷达模型代码",
            "训练好的checkpoint",
            "所需SEVIR卫星通道及配对VIL数据",
            "预处理配置与示例推理命令",
        ],
        "integration_contract": "synthesize(satellite_frames) -> vil_or_radar_frames",
        "note": "当前本地样本只有VIL通道，因此不能执行卫星转雷达推理。",
    }


def run_simulator(event_id: str, simulator_name: str = "jcm") -> dict[str, Any]:
    _, metadata = _event(event_id)
    return {
        "status": "awaiting_simulator_assets",
        "event_id": metadata["event_id"],
        "simulator_name": simulator_name,
        "required_assets": ["模拟器代码", "运行环境", "初始场数据", "配置与调用示例"],
        "integration_contract": "simulate(initial_state, duration, parameters) -> fields",
        "note": "已预留Agent工具接口，但尚未接入真实模拟器。JCM不是CASCast的依赖项。",
    }


def runtime_status() -> dict[str, Any]:
    manifest: dict[str, Any] = {"samples": 0, "events": []}
    try:
        _, metadata = _load_cascast_prediction()
        manifest = _ensemble_manifest()
        sample = {
            "available": True,
            "event_id": metadata["event_id"],
            "kind": "official_cascast_test_sample",
            "available_event_count": int(manifest["samples"]),
            "event_ids": [event["event_id"] for event in manifest["events"]],
        }
    except RuntimeError as error:
        sample = {"available": False, "error": str(error)}
    cascast_code = (CASCAST_ROOT / "networks" / "casformer.py").exists()
    cascast_checkpoint = (
        CASCAST_ROOT
        / "experiments"
        / "cascast_diffusion"
        / "world_size1-ckpt"
        / "checkpoint_best.pth"
    ).exists()
    autoencoder_checkpoint = (CASCAST_ROOT / "ckpts" / "autoencoder" / "ckpt.pth").exists()
    earthformer_latent = _earthformer_latent_status()
    live_readiness = live_cascast_readiness()
    local_hdf5_records = [
        record
        for record in _catalog_vil_records()
        if (LOCAL_SEVIR_DATA_ROOT / record["file_name"]).is_file()
    ] if CATALOG_CSV.exists() else []
    local_hdf5_event_ids = list(dict.fromkeys(record["id"] for record in local_hdf5_records))
    remote_event_access = bool(SEVIR_REMOTE_ENABLED and CATALOG_CSV.exists())
    latent_data = earthformer_latent["available"]
    prediction_output = CASCAST_ENSEMBLE_MANIFEST.exists()
    cascast_assets_ready = bool(
        cascast_code and cascast_checkpoint and autoencoder_checkpoint and latent_data
    )
    event_latent_available = True
    missing_assets = []
    if not cascast_code:
        missing_assets.append("代码")
    if not cascast_checkpoint:
        missing_assets.append("CasFormer checkpoint")
    if not autoencoder_checkpoint:
        missing_assets.append("Autoencoder checkpoint")
    if not latent_data:
        missing_assets.append("latent data")
    if not prediction_output:
        missing_assets.append("CasCast多事件ensemble输出")
    return {
        "sevir_sample": sample,
        "catalog": {
            "available": CATALOG_CSV.exists(),
            "vil_record_count": len(_catalog_vil_records()) if CATALOG_CSV.exists() else 0,
            "tools": [
                "exact_event_metadata",
                "data_indexer",
                "local_hdf5_loader",
                "historical_statistics",
                "projected_pixel_geolocator",
                "noaa_storm_reference",
                "history_comparison",
            ],
        },
        "dataset_evaluation": {
            "available": CASCAST_EVALUATION_REPORT.exists(),
            "event_count": int(manifest["samples"]) if prediction_output else 0,
            "ground_truth_available": bool(manifest.get("ground_truth_available", False)) if prediction_output else False,
            "report_file": str(CASCAST_EVALUATION_REPORT.relative_to(ROOT)),
        },
        "persistence_baseline": {"available": True},
        "cascast": {
            "available": bool(prediction_output),
            "assets_ready": cascast_assets_ready,
            "inference_ready": bool(cascast_assets_ready),
            "code_available": cascast_code,
            "checkpoint_available": cascast_checkpoint,
            "autoencoder_checkpoint_available": autoencoder_checkpoint,
            "latent_data_available": latent_data,
            "event_latent_available": event_latent_available,
            "prediction_output_available": prediction_output,
            "display_mode": "saved_verified_ensemble_package",
            "live_inference_ready": bool(live_readiness["ready"] and (local_hdf5_event_ids or remote_event_access)),
            "live_inference_for_selected_event": bool(live_readiness["ready"]),
            "live_assets": live_readiness["assets"],
            "local_hdf5_event_count": len(local_hdf5_event_ids),
            "local_hdf5_event_ids": local_hdf5_event_ids,
            "remote_hdf5_enabled": remote_event_access,
            "remote_hdf5_base_url": SEVIR_REMOTE_BASE_URL if remote_event_access else None,
            "ensemble_count": int(manifest.get("members_per_sample", 0)) if prediction_output else 0,
            "available_event_count": int(manifest["samples"]) if prediction_output else 0,
            "event_ids": [event["event_id"] for event in manifest["events"]] if prediction_output else [],
            "earthformer_latent": earthformer_latent,
            "reason": (
                f"仍缺少{'、'.join(missing_assets)}"
                if missing_assets
                else (
                    f"{int(manifest.get('samples', 0))}个事件的"
                    f"{int(manifest.get('members_per_sample', 0))}路CasCast ensemble已接入Demo"
                )
            ),
        },
        "satellite_to_radar": {"available": False, "reason": "等待模型材料和卫星通道数据"},
        "simulator": {"available": False, "reason": "只预留接口，尚未指定具体模拟器"},
    }
