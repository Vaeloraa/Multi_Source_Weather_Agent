"""Read-only preflight for the Springfield, Missouri S781667 demo case.

This script validates catalog identity and reports what is present locally.  It
does not download data, run a model, or create a cache, so a successful catalog
check must not be interpreted as an end-to-end-ready demo.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = ROOT / "data" / "demo_cases" / "springfield_mo_20180811.json"
CATALOG = ROOT / "data" / "sevir_complete" / "CATALOG.csv"
SEVIR_DATA = ROOT / "data" / "sevir_complete" / "data"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog_rows(event_id: str) -> dict[str, dict[str, str]]:
    if not CATALOG.is_file():
        return {}
    matches: dict[str, dict[str, str]] = {}
    with CATALOG.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("id") == event_id:
                matches[str(row.get("img_type", "")).lower()] = row
    return matches


def _float_equal(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    try:
        return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)
    except (TypeError, ValueError):
        return False


def _catalog_check(case: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    event = case["event"]
    rows = _catalog_rows(event["sevir_id"])
    bbox_columns = {
        "south": "llcrnrlat",
        "west": "llcrnrlon",
        "north": "urcrnrlat",
        "east": "urcrnrlon",
    }
    checks: dict[str, Any] = {}
    all_valid = CATALOG.is_file()
    for expected in case["sevir_catalog_records"]:
        channel = expected["img_type"]
        row = rows.get(channel)
        errors: list[str] = []
        if row is None:
            errors.append("missing catalog row")
        else:
            exact_fields = {
                "file_name": expected["file_name"],
                "file_index": str(expected["file_index"]),
                "time_utc": event["reference_time_utc"].replace("T", " ").removesuffix("Z"),
                "event_type": event["event_type"],
                "episode_id": str(event["noaa_episode_id"]) + ".0",
                "event_id": str(event["noaa_event_id"]) + ".0",
            }
            for field, value in exact_fields.items():
                if row.get(field) != value:
                    errors.append(f"{field}: expected {value!r}, got {row.get(field)!r}")
            for key, column in bbox_columns.items():
                if not _float_equal(row.get(column), event["bounding_box"][key]):
                    errors.append(
                        f"{column}: expected {event['bounding_box'][key]!r}, got {row.get(column)!r}"
                    )
            offsets = [item for item in str(row.get("minute_offsets") or "").split(":") if item]
            if expected["frame_count"] is not None and len(offsets) != expected["frame_count"]:
                errors.append(
                    f"frame_count: expected {expected['frame_count']}, got {len(offsets)}"
                )
        local_path = SEVIR_DATA / expected["file_name"]
        checks[channel] = {
            "catalog_match": not errors,
            "errors": errors,
            "local_hdf5_present": local_path.is_file(),
            "local_hdf5_path": str(local_path.relative_to(ROOT)),
            "integration_status": expected["integration_status"],
        }
        all_valid = all_valid and not errors
    return checks, all_valid


def _glob_relative(pattern: str) -> list[str]:
    pattern_path = Path(pattern)
    parent = ROOT / pattern_path.parent
    if not parent.is_dir():
        return []
    return sorted(
        str(item.relative_to(ROOT))
        for item in parent.iterdir()
        if item.is_file() and fnmatch.fnmatch(item.name, pattern_path.name)
    )


def _runtime_assets() -> dict[str, bool]:
    cascast_root = ROOT.parent / "CasCast"
    python_candidates = list((cascast_root / ".runtime" / "python").glob("*/python.exe"))
    if not python_candidates:
        python_candidates = [cascast_root / ".venv" / "Scripts" / "python.exe"]
    return {
        "python": any(path.is_file() for path in python_candidates),
        "worker": (ROOT / "scripts" / "run_local_cascast.py").is_file(),
        "earthformer_checkpoint": (
            cascast_root
            / "experiments"
            / "EarthFormer"
            / "world_size1-ckpt"
            / "earthformer_ckpt.pth"
        ).is_file(),
        "cascast_checkpoint": (
            cascast_root
            / "experiments"
            / "cascast_diffusion"
            / "world_size1-ckpt"
            / "checkpoint_best.pth"
        ).is_file(),
        "autoencoder_checkpoint": (
            cascast_root / "ckpts" / "autoencoder" / "ckpt.pth"
        ).is_file(),
    }


def build_report(case_path: Path) -> dict[str, Any]:
    case = _load_json(case_path)
    catalog_channels, metadata_verified = _catalog_check(case)
    expected = case["expected_case_artifacts"]
    vil_cache = ROOT / expected["vil_event_cache"]
    cascast_caches = _glob_relative(expected["cascast_cache_glob"])
    aurora_manifest = ROOT / expected["aurora_case_manifest"]
    runtime_assets = _runtime_assets()
    local_channels = {
        name: details["local_hdf5_present"]
        for name, details in catalog_channels.items()
    }
    cascast_input_ready = bool(local_channels.get("vil") or vil_cache.is_file())
    cascast_output_ready = bool(cascast_caches)
    multimodal_local_ready = bool(local_channels) and all(local_channels.values())
    aurora_output_ready = aurora_manifest.is_file()
    end_to_end_cached_ready = bool(
        metadata_verified and cascast_output_ready and aurora_output_ready
    )
    if not metadata_verified:
        status = "invalid_or_missing_metadata"
    elif end_to_end_cached_ready:
        status = "cached_end_to_end_ready"
    elif cascast_output_ready or aurora_output_ready:
        status = "partially_prepared"
    else:
        status = "metadata_only"
    return {
        "case_id": case["case_id"],
        "status": status,
        "read_only": True,
        "metadata_verified": metadata_verified,
        "catalog_path": str(CATALOG.relative_to(ROOT)),
        "catalog_channels": catalog_channels,
        "runtime_assets": runtime_assets,
        "readiness": {
            "cascast_input_present_locally": cascast_input_ready,
            "cascast_segment_1_cache_present": cascast_output_ready,
            "cascast_cache_files": cascast_caches,
            "all_listed_sevir_channels_present_locally": multimodal_local_ready,
            "aurora_case_manifest_present": aurora_output_ready,
            "cached_end_to_end_demo_ready": end_to_end_cached_ready,
        },
        "boundaries": {
            "nexrad_goes_generic_adapter_checked": False,
            "nexrad_goes_generic_adapter_ready": False,
            "note": (
                "This preflight checks the SEVIR historical case and named cache artifacts only; "
                "it does not treat SEVIR channels as proof of a generic NEXRAD/GOES adapter."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit with code 2 unless the named CasCast and Aurora caches are both present.",
    )
    args = parser.parse_args()
    report = build_report(args.case.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["metadata_verified"]:
        return 1
    if args.require_ready and not report["readiness"]["cached_end_to_end_demo_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
