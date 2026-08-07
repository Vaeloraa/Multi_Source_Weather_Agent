from __future__ import annotations

from typing import Any

from ..data_access import catalog, ensemble, uploads

from .base import WorldComponent


class SevirDataIndexer(WorldComponent):
    """Read SEVIR metadata, local HDF5 events, uploads, and CasCast cache metadata.

    Inputs use SEVIR event IDs or upload IDs. Raster dimensions are pixels and
    times are UTC/relative minutes. HDF5 pixel access only succeeds when the
    catalog-selected file exists below the configured local SEVIR data root.
    Returned dictionaries are JSON serializable and never expose open files.
    """

    def event(self, event_id: str) -> dict[str, Any]:
        """Return exact catalog metadata and the 49-frame UTC time axis."""
        return self._call(
            "data.event", {"event_id": event_id}, catalog.get_sevir_event_metadata
        )

    def search(
        self,
        query: str = "",
        year: int | None = None,
        month: int | None = None,
        event_type: str = "",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search VIL catalog rows; this returns metadata, not HDF5 pixels."""
        arguments = {
            "query": query,
            "year": year,
            "month": month,
            "event_type": event_type,
            "limit": limit,
        }
        return self._call("data.search", arguments, catalog.search_sevir_catalog)

    def load_hdf5(
        self, event_id: str, start_frame: int = 0, end_frame: int = 49
    ) -> dict[str, Any]:
        """Load real uint8 VIL frames from local HDF5 and return frame summaries.

        Frames are 384x384 for official VIL. Pixel value 255 is the SEVIR
        missing-value code and must be excluded from verification metrics.
        """
        arguments = {
            "event_id": event_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        return self._call("data.load_hdf5", arguments, catalog.load_sevir_vil_event)

    def cascast_cache(self, event_id: str) -> dict[str, Any]:
        """Describe one cached CasCast event without returning full arrays."""
        arrays, metadata = ensemble._load_cascast_ensemble(event_id)
        result = {
            "status": "ok",
            "event_id": metadata["event_id"],
            "history_shape": list(arrays["history"].shape),
            "member_shape": list(arrays["ensemble_members"].shape),
            "ensemble_mean_shape": list(arrays["ensemble_mean"].shape),
            "ground_truth_shape": list(arrays["ground_truth"].shape),
            "value_space": "SEVIR encoded VIL; 255 is missing in ground truth",
            "inference_mode": "saved_verified_ensemble_package",
        }
        self._recorder("data.cascast_cache", {"event_id": event_id}, result)
        return result

    def uploads(self) -> dict[str, Any]:
        """List persisted single-frame and 13-frame uploads."""
        return self._call("data.uploads", {}, uploads.list_uploaded_records)
