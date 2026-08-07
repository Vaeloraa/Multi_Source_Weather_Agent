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
from ..data_access.catalog import (
    _catalog_public_record,
    _catalog_vil_event,
    _catalog_vil_records,
)

import h5py
import numpy as np
from pyproj import CRS, Transformer


def historical_sevir_statistics(
    year: int | None = None,
    event_type: str = "",
) -> dict[str, Any]:
    event_type_normalized = str(event_type or "").strip().lower()
    records = [
        record
        for record in _catalog_vil_records()
        if (year is None or record["year"] == int(year))
        and (not event_type_normalized or event_type_normalized in record["event_type"].lower())
    ]
    if not records:
        raise ValueError("当前筛选条件下没有SEVIR VIL catalog记录。")
    counts_by_year: dict[str, int] = {}
    counts_by_month: dict[str, int] = {}
    counts_by_event_type: dict[str, int] = {}
    for record in records:
        year_key = str(record["year"] or "Unknown")
        month_key = f"{record['month']:02d}" if record["month"] else "Unknown"
        type_key = record["event_type"]
        counts_by_year[year_key] = counts_by_year.get(year_key, 0) + 1
        counts_by_month[month_key] = counts_by_month.get(month_key, 0) + 1
        counts_by_event_type[type_key] = counts_by_event_type.get(type_key, 0) + 1
    max_values = np.asarray(
        [record["data_max"] for record in records if record["data_max"] is not None],
        dtype=np.float64,
    )
    max_summary = None
    if max_values.size:
        max_summary = {
            "count": int(max_values.size),
            "mean": round(float(max_values.mean()), 4),
            "min": round(float(max_values.min()), 4),
            "p50": round(float(np.percentile(max_values, 50)), 4),
            "p90": round(float(np.percentile(max_values, 90)), 4),
            "p95": round(float(np.percentile(max_values, 95)), 4),
            "max": round(float(max_values.max()), 4),
        }
    ranked = sorted(
        (record for record in records if record["data_max"] is not None),
        key=lambda item: item["data_max"],
        reverse=True,
    )[:10]
    return {
        "status": "ok",
        "task": "historical_sevir_statistics",
        "data_source": str(CATALOG_CSV.relative_to(ROOT)),
        "channel": "vil",
        "filters": {"year": year, "event_type": event_type},
        "record_count": len(records),
        "counts_by_year": dict(sorted(counts_by_year.items())),
        "counts_by_month": dict(sorted(counts_by_month.items())),
        "counts_by_event_type": dict(
            sorted(counts_by_event_type.items(), key=lambda item: item[1], reverse=True)
        ),
        "data_max_summary": max_summary,
        "top_events_by_catalog_data_max": [_catalog_public_record(record) for record in ranked],
        "interpretation_limit": (
            "这是CATALOG记录级统计。data_max是每条VIL记录的catalog最大值；"
            "事件数量不等同于灾害发生次数，Unlabeled也不代表无灾害。"
        ),
    }


def get_noaa_storm_reference(event_id: str) -> dict[str, Any]:
    record = _catalog_vil_event(event_id)
    noaa_event_id = record["noaa_event_id"]
    noaa_episode_id = record["noaa_episode_id"]
    linked = noaa_event_id is not None
    return {
        "status": "ok",
        "task": "noaa_storm_reference",
        "sevir_id": record["id"],
        "event_type": record["event_type"],
        "noaa_linked": linked,
        "noaa_episode_id": noaa_episode_id,
        "noaa_event_id": noaa_event_id,
        "noaa_event_url": (
            f"https://www.ncei.noaa.gov/stormevents/eventdetails.jsp?id={noaa_event_id}"
            if linked
            else None
        ),
        "reason": (
            "CATALOG contains NOAA Storm Events identifiers for this storm event."
            if linked
            else "This record has no NOAA identifiers; random SEVIR events are commonly unlabeled."
        ),
        "interpretation_limit": (
            "This tool exposes the CATALOG linkage and official NOAA detail URL. It does not copy "
            "or independently verify the NOAA narrative, damage, or warning fields."
        ),
    }


def compare_event_with_history(event_id: str) -> dict[str, Any]:
    target = str(event_id).strip().lower()
    records = _catalog_vil_records()
    record = next((item for item in records if item["id"].lower() == target), None)
    if record is None:
        raise ValueError(f"SEVIR CATALOG中没有找到VIL事件{event_id}。")
    if record["data_max"] is None:
        raise ValueError(f"事件{event_id}没有可比较的catalog data_max。")
    reference = [
        item["data_max"]
        for item in records
        if item["data_max"] is not None
        and (record["event_type"] == "Unlabeled" or item["event_type"] == record["event_type"])
    ]
    values = np.asarray(reference, dtype=np.float64)
    percentile = float((values <= record["data_max"]).mean() * 100)
    return {
        "status": "ok",
        "task": "event_history_comparison",
        "event_id": record["id"],
        "event_type": record["event_type"],
        "time_utc": record["time_utc"],
        "catalog_data_max": record["data_max"],
        "reference_group": record["event_type"] if record["event_type"] != "Unlabeled" else "all_vil_records",
        "reference_count": int(values.size),
        "data_max_percentile": round(percentile, 2),
        "interpretation": f"该记录的catalog data_max高于或等于约{percentile:.2f}%的参考记录。",
        "interpretation_limit": "这是catalog data_max的经验分位比较，不是灾害概率或模型预测概率。",
    }
