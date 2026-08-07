from __future__ import annotations

import csv
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
sys.path.insert(0, str(VENDOR))

import fsspec  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402


SOURCE_KEY = "data/vil/2017/SEVIR_VIL_STORMEVENTS_2017_0101_0630.h5"
SOURCE_URL = f"https://sevir.s3.us-west-2.amazonaws.com/{SOURCE_KEY}"
CATALOG_URL = "https://sevir.s3.us-west-2.amazonaws.com/CATALOG.csv"
OUTPUT_DIR = ROOT / "data" / "sevir_sample"
OUTPUT_NPZ = OUTPUT_DIR / "sevir_vil_event.npz"
OUTPUT_METADATA = OUTPUT_DIR / "metadata.json"


def decode_id(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def find_catalog_row(event_id: str) -> dict[str, str]:
    with requests.get(CATALOG_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        text_stream = io.TextIOWrapper(response.raw, encoding="utf-8", newline="")
        for row in csv.DictReader(text_stream):
            if row.get("id") == event_id and row.get("img_type") == "vil":
                return dict(row)
    return {}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Opening official SEVIR HDF5 with range requests: {SOURCE_URL}")
    with fsspec.open(
        SOURCE_URL,
        mode="rb",
        block_size=8 * 1024 * 1024,
        cache_type="readahead",
    ) as remote_file:
        with h5py.File(remote_file, "r") as h5_file:
            keys = list(h5_file.keys())
            if "vil" not in h5_file or "id" not in h5_file:
                raise RuntimeError(f"Unexpected SEVIR HDF5 keys: {keys}")
            event_id = decode_id(h5_file["id"][0])
            event = np.asarray(h5_file["vil"][0], dtype=np.uint8)

    if event.ndim != 3:
        raise RuntimeError(f"Expected a 3-D VIL event, got shape {event.shape}")

    catalog_row = find_catalog_row(event_id)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    np.savez_compressed(
        OUTPUT_NPZ,
        vil=event,
        event_id=np.asarray(event_id),
        source_url=np.asarray(SOURCE_URL),
        retrieved_at=np.asarray(retrieved_at),
    )

    metadata = {
        "event_id": event_id,
        "channel": "vil",
        "shape": list(event.shape),
        "dtype": str(event.dtype),
        "frame_count": int(event.shape[-1]),
        "min": int(event.min()),
        "max": int(event.max()),
        "mean": float(event.mean()),
        "source_bucket": "s3://sevir",
        "source_key": SOURCE_KEY,
        "source_url": SOURCE_URL,
        "catalog_url": CATALOG_URL,
        "retrieved_at": retrieved_at,
        "catalog_row": catalog_row,
    }
    OUTPUT_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
