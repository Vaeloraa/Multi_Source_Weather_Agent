"""Small runtime-readiness snapshot for the web configuration endpoint."""

from __future__ import annotations

from ..tools.sevir.config import CATALOG_CSV, LOCAL_SEVIR_DATA_ROOT, SEVIR_REMOTE_ENABLED
from ..capabilities.registry import capability_status
from ..tools.sevir.repository import catalog_vil_records


def runtime_status() -> dict[str, object]:
    """Report real source and model readiness with current runtime status."""
    records = list(catalog_vil_records()) if CATALOG_CSV.exists() else []
    local_records = [
        record for record in records
        if (LOCAL_SEVIR_DATA_ROOT / record["file_name"]).is_file()
    ]
    local_event_ids = list(dict.fromkeys(record["id"] for record in local_records))
    capabilities = capability_status()
    readiness = capabilities.get("cascast", {})  # Legacy configuration field only.
    return {
        "capabilities": capabilities,
        "catalog": {
            "available": CATALOG_CSV.exists(),
            "vil_record_count": len(records),
            "local_hdf5_event_count": len(local_event_ids),
            "remote_hdf5_enabled": bool(SEVIR_REMOTE_ENABLED),
        },
        "cascast": {
            "live_inference_ready": bool(
                readiness.get("ready") and (local_event_ids or SEVIR_REMOTE_ENABLED)
            ),
            "assets": readiness.get("assets", {}),
        },
    }
