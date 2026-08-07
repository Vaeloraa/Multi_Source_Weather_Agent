from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "data" / "source_packages" / "cascast_uncertainty"
SOURCE_H5 = PACKAGE_DIR / "cascast_ensemble_10samples_384.h5"
SOURCE_METADATA = PACKAGE_DIR / "metadata.json"
GROUND_TRUTH_H5 = (
    ROOT
    / "data"
    / "source_packages"
    / "cascast_ground_truth"
    / "cascast_ground_truth_10samples_384.h5"
)
TEST_LIST = ROOT.parent / "CasCast" / "datasets" / "sevir_list" / "test.txt"
CATALOG = ROOT / "data" / "sevir_complete" / "CATALOG.csv"
OUTPUT_DIR = ROOT / "data" / "cascast_ensemble_cache"
SAMPLE_PATTERN = re.compile(r"^vil-(.+\.h5)-(\d+)-([012])\.npy$")


def load_catalog() -> dict[tuple[str, int], dict[str, str]]:
    records: dict[tuple[str, int], dict[str, str]] = {}
    with CATALOG.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("img_type") != "vil":
                continue
            key = (Path(row["file_name"]).name, int(row["file_index"]))
            records[key] = row
    return records


def catalog_summary(sample_name: str, catalog: dict[tuple[str, int], dict[str, str]]) -> dict[str, object]:
    match = SAMPLE_PATTERN.match(sample_name)
    if not match:
        return {"sample_name": sample_name}
    encoded_file_name, file_index_text, segment_text = match.groups()
    # CasCast's preprocessed name encodes the original ``vil/<year>/`` path as
    # ``vil-<year>-`` before the real HDF5 basename.
    file_name = re.sub(r"^\d{4}-", "", encoded_file_name)
    row = catalog.get((file_name, int(file_index_text)), {})
    result: dict[str, object] = {
        "sample_name": sample_name,
        "source_h5_name": file_name,
        "file_index": int(file_index_text),
        "segment": int(segment_text),
    }
    for key in (
        "id",
        "time_utc",
        "event_type",
        "llcrnrlat",
        "llcrnrlon",
        "urcrnrlat",
        "urcrnrlon",
        "data_min",
        "data_max",
    ):
        value = row.get(key, "")
        if value == "":
            result[key] = None
        elif key in {"llcrnrlat", "llcrnrlon", "urcrnrlat", "urcrnrlon", "data_min", "data_max"}:
            result[key] = float(value)
        else:
            result[key] = value
    return result


def main() -> int:
    source_metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    test_names = TEST_LIST.read_text(encoding="utf-8").splitlines()
    catalog = load_catalog()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, object]] = []

    with h5py.File(SOURCE_H5, "r") as source, h5py.File(
        GROUND_TRUTH_H5, "r"
    ) as truth_source:
        sample_indices = source["sample_index"][:].astype(int)
        truth_sample_indices = truth_source["sample_index"][:].astype(int)
        if not np.array_equal(sample_indices, truth_sample_indices):
            raise RuntimeError("Ground truth sample_index与ensemble顺序不一致。")
        member_seeds = source["member_seed"][:].astype(np.int64)
        for position, sample_index in enumerate(sample_indices):
            sample_name = test_names[int(sample_index)]
            event = catalog_summary(sample_name, catalog)
            event_id = str(event.get("id") or f"SEVIR_TEST_{sample_index}")

            # Preserve the VAE decoder's raw float output.  Display clipping is a
            # derived view and must never replace the model output used for
            # ensemble statistics or paper-compatible evaluation.
            members_raw = source["ensemble_members"][position].astype(np.float32)
            source_mean = source["ensemble_mean"][position].astype(np.float32)
            pixel_std = members_raw.std(axis=0, ddof=1)
            q10 = np.quantile(members_raw, 0.1, axis=0)
            q90 = np.quantile(members_raw, 0.9, axis=0)
            history = source["history"][position].astype(np.uint8)
            ground_truth = truth_source["ground_truth"][position].astype(np.uint8)
            if ground_truth.shape != (12, 384, 384):
                raise RuntimeError(
                    f"样本{sample_index}的ground truth形状错误：{ground_truth.shape}"
                )

            output_name = f"sample_{int(sample_index):05d}.npz"
            output_path = OUTPUT_DIR / output_name
            np.savez_compressed(
                output_path,
                history=history,
                ground_truth=ground_truth,
                ensemble_members=members_raw,
                ensemble_mean=source_mean,
                pixel_std=pixel_std.astype(np.float16),
                q10=q10.astype(np.float16),
                q90=q90.astype(np.float16),
                member_seed=member_seeds,
                sample_index=np.asarray(sample_index, dtype=np.int64),
                sample_name=np.asarray(sample_name),
                event_id=np.asarray(event_id),
            )
            event.update(
                {
                    "event_id": event_id,
                    "sample_index": int(sample_index),
                    "sample_position": position,
                    "cache_file": output_name,
                    "ensemble_count": int(members_raw.shape[0]),
                    "history_frames": int(history.shape[0]),
                    "forecast_frames": int(members_raw.shape[1]),
                    "ground_truth_frames": int(ground_truth.shape[0]),
                }
            )
            events.append(event)
            print(f"[{position + 1}/{len(sample_indices)}] {event_id} -> {output_name}")

    manifest = {
        "format_version": 2,
        "source_h5": str(SOURCE_H5.relative_to(ROOT)),
        "source_metadata": str(SOURCE_METADATA.relative_to(ROOT)),
        "ground_truth_h5": str(GROUND_TRUTH_H5.relative_to(ROOT)),
        "source_model": source_metadata.get("model", "CasCast"),
        "source_dataset": source_metadata.get("source_dataset", "SEVIR VIL test split"),
        "samples": len(events),
        "members_per_sample": int(source_metadata["dimensions"]["members_per_sample"]),
        "history_minutes": source_metadata["time"]["history_minutes"],
        "forecast_minutes": source_metadata["time"]["forecast_minutes"],
        "member_seeds": source_metadata["ensemble"]["member_seeds"],
        "processing": {
            "raw_value_space": "VAE decoded float output on SEVIR encoded VIL scale",
            "input_normalization": "x / 255",
            "inverse_scaling": "decoded_output * 255",
            "decoder_output_bounded": False,
            "clipping_applied_to_cache": False,
            "display_clip_range": [0, 254],
            "missing_value_code": 255,
            "ensemble_members_cache_dtype": "float32",
            "ensemble_mean_cache_dtype": "float32",
            "pixel_std_cache_dtype": "float16",
            "quantiles": [0.1, 0.9],
            "note": "Raw float predictions are preserved for ensemble statistics. Clipping to [0, 254] is applied only when creating display or engineering-analysis views.",
        },
        "ground_truth_available": True,
        "events": events,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"manifest: {OUTPUT_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
