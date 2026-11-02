"""Catalog, observation cache and remote-source settings owned by SEVIR."""

from __future__ import annotations

import os

from ...config import ROOT


CATALOG_CSV = ROOT / "data" / "sevir_complete" / "CATALOG.csv"
LOCAL_SEVIR_DATA_ROOT = ROOT / "data" / "sevir_complete" / "data"
SEVIR_EVENT_CACHE = ROOT / "data" / "sevir_event_cache"
SEVIR_REMOTE_BASE_URL = os.getenv(
    "SEVIR_REMOTE_BASE_URL",
    "https://sevir.s3.us-west-2.amazonaws.com/data",
).rstrip("/")
SEVIR_REMOTE_ENABLED = os.getenv("SEVIR_REMOTE_ENABLED", "1").lower() not in {
    "0", "false", "no", "off"
}
SEVIR_REMOTE_BLOCK_SIZE = max(
    1 << 20,
    int(os.getenv("SEVIR_REMOTE_BLOCK_SIZE", str(8 << 20))),
)
