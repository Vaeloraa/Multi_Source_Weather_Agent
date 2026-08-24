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
    Upload contents are injected into ``ds`` by the runtime; generated analyses
    should use ``ds[0]`` rather than listing persisted uploads.
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

    def search_region(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 100.0,
        month: int | None = None,
        year: int | None = None,
        event_type: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Find metadata candidates within a physical WGS84 radius.

        ``limit`` controls only the records returned for display.  The catalog
        response still reports the full matched count and year distribution;
        pixel statistics must be computed separately from the selected HDF5
        events.
        """
        arguments = {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km,
            "month": month,
            "year": year,
            "event_type": event_type,
            "limit": limit,
        }
        return self._call("data.search_region", arguments, catalog.search_sevir_catalog_by_region)

    def load_hdf5(
        self, event_id: str, start_frame: int = 0, end_frame: int = 49
    ) -> dict[str, Any]:
        """Load VIL frames plus aligned relative_minutes and frame_statistics.

        ``frames``/``values`` is a read-only ``[time,row,column]`` array for
        generated calculations. Large pixels are omitted from the LLM/trace
        serialization. Pixel value 255 is missing and must be excluded.
        """
        arguments = {
            "event_id": event_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        return self._call("data.load_hdf5", arguments, catalog.load_sevir_vil_event)

    def hdf5_frame_histograms(
        self,
        event_ids: list[str],
        start_frame: int = 0,
        end_frame: int = 49,
    ) -> dict[str, Any]:
        """Return exact per-frame VIL histograms for up to 10 local HDF5 events.

        The Agent selects event IDs, encoded-value thresholds, active-pixel
        criteria, continuity rules, and duration statistics in generated Python.
        Each histogram has 256 counts aligned to encoded VIL values 0..255;
        bin 255 is missing. This primitive intentionally computes no duration.
        """
        arguments = {
            "event_ids": event_ids,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        return self._call(
            "data.hdf5_frame_histograms",
            arguments,
            catalog.load_sevir_vil_frame_histograms,
        )

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

    def match_uploaded_event(self, upload_id: str) -> dict[str, Any]:
        """Resolve an uploaded file to a SEVIR event without visual guessing.

        Explicit metadata/filename IDs are preferred. Otherwise, locally
        available HDF5 rows are looked up by SHA-256 and then verified by exact
        pixels: 13 history frames for a sequence, or canonical frame 24 for a
        single image. Read the stable nested ``event_match`` object: its
        ``status`` is matched, not_provided, not_found, or ambiguous; when
        matched, ``event_id``, ``method`` and ``confidence`` describe the match.
        Ambiguous and absent matches remain unmatched.
        """
        return self._call(
            "data.match_uploaded_event",
            {"upload_id": upload_id},
            uploads.match_uploaded_event,
        )
