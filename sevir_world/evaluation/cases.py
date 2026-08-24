"""Data-driven demo and regression case selection for the CasCast cache."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data_access import ensemble


def _case_tags(arrays: dict[str, np.ndarray]) -> list[str]:
    members = np.asarray(arrays["ensemble_members"], dtype=np.float32)
    truth = np.asarray(arrays["ground_truth"], dtype=np.float32)
    spread = np.nanmean(np.std(members, axis=0))
    member_min = np.nanmin(members, axis=0)
    member_max = np.nanmax(members, axis=0)
    valid_truth = truth != 255
    truth_covered = np.mean((truth >= member_min) & (truth <= member_max) & valid_truth)
    tags = ["cascast_demo"]
    if float(np.nanmax(members)) < 80:
        tags.append("weak_weather")
    if float(np.nanmax(members)) >= 180:
        tags.append("strong_weather")
    if float(spread) >= 8:
        tags.append("large_ensemble_spread")
    if float(truth_covered) < 0.5:
        tags.append("truth_outside_ensemble_range")
    if float(np.nanmax(np.abs(np.diff(np.nanmean(members, axis=(0, 2, 3)))))) >= 5:
        tags.append("strong_temporal_change")
    return tags


def list_demo_cases(limit: int = 50) -> dict[str, Any]:
    """Return reproducible case labels derived from current cached arrays.

    The labels are descriptive test fixtures, not hard-coded analysis results.
    """
    rows: list[dict[str, Any]] = []
    for event in ensemble._ensemble_manifest().get("events", []):
        event_id = str(event["event_id"])
        arrays, metadata = ensemble._load_cascast_ensemble(event_id)
        rows.append({
            "event_id": event_id,
            "sample_name": metadata.get("sample_name"),
            "tags": _case_tags(arrays),
            "ensemble_count": int(arrays["ensemble_members"].shape[0]),
            "forecast_frames": int(arrays["ensemble_members"].shape[1]),
        })
    rows.sort(key=lambda item: (len(item["tags"]), item["event_id"]), reverse=True)
    return {
        "status": "ok",
        "task": "cascast_demo_case_catalog",
        "case_count": len(rows),
        "cases": rows[: max(1, min(int(limit), len(rows) or 1))],
        "case_semantics": {
            "purpose": "demo selection and regression coverage",
            "selection": "derived from cached ensemble and ground-truth arrays",
            "not_a_forecast": True,
        },
    }
