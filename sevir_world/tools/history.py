from __future__ import annotations

from typing import Any

from ..backends import statistics
from .base import WorldComponent, resolve_event_id


class SevirHistory(WorldComponent):
    """History-focused facade over the existing statistics backend.

    The facade gives generated programs one stable history namespace without
    duplicating catalog or HDF5 logic. The legacy ``statistics`` tool remains
    available for compatibility.
    """

    def _call(self, name: str, arguments: dict[str, Any], function):
        def normalized(**kwargs: Any) -> Any:
            raw = function(**kwargs)
            if not isinstance(raw, dict):
                return raw
            result = dict(raw)
            result.setdefault("status", "ok")
            result.setdefault("task", name)
            result.setdefault("evidence", {})
            result.setdefault("provenance", {"tool": name, "backend": "statistics"})
            result.setdefault("limitations", [])
            result.setdefault("cache", {"hit": False, "key": name})
            return result

        return super()._call(name, arguments, normalized)

    def event_history_primitives(
        self, event_ids: list[str], start_frame: int = 0, end_frame: int = 49
    ) -> dict[str, Any]:
        """Return bounded per-frame history primitives for selected events."""
        return self._call(
            "history.event_history_primitives",
            {"event_ids": event_ids, "start_frame": start_frame, "end_frame": end_frame},
            statistics.load_event_history_frame_primitives,
        )

    # Stable public names from the product contract. The longer methods above
    # remain available for backwards compatibility with existing programs.
    def load_event_histories(self, event_ids: list[str], start_frame: int = 0, end_frame: int = 49, spatial_stride: int = 1, max_total_values: int = 8_000_000) -> dict[str, Any]:
        return self.event_histories_raw(event_ids, start_frame, end_frame, spatial_stride, max_total_values)

    def compare_region_history(self, event_id: str, month_window: int = 1, day_window: int | None = 15, threshold: float = 40.0, history_frame_count: int = 12, candidate_limit: int = 10, region: list[int] | None = None) -> dict[str, Any]:
        return self.compare_cascast_history(event_id, month_window, day_window, threshold, history_frame_count, candidate_limit, region)

    def catalog_distribution(self, event_id: str, field: str = "data_max", candidate_limit: int = 10, same_event_type: bool = True, local_hdf5_only: bool = True) -> dict[str, Any]:
        return self.similar_events(event_id, field, candidate_limit, same_event_type, local_hdf5_only)

    def event_histories_raw(
        self,
        event_ids: list[str],
        start_frame: int = 0,
        end_frame: int = 49,
        spatial_stride: int = 1,
        max_total_values: int = 8_000_000,
    ) -> dict[str, Any]:
        """Load bounded raw history arrays for agent-selected analysis."""
        return self._call(
            "history.event_histories_raw",
            {
                "event_ids": event_ids,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "spatial_stride": spatial_stride,
                "max_total_values": max_total_values,
            },
            statistics.load_event_history_raw,
        )

    def similar_events(
        self,
        reference_event_id: str,
        field: str = "data_max",
        candidate_limit: int = 10,
        same_event_type: bool = True,
        local_hdf5_only: bool = True,
    ) -> dict[str, Any]:
        """Rank history candidates before loading their raw frames."""
        return self._call(
            "history.similar_events",
            {
                "reference_event_id": reference_event_id,
                "field": field,
                "candidate_limit": candidate_limit,
                "same_event_type": same_event_type,
                "local_hdf5_only": local_hdf5_only,
            },
            statistics.find_similar_event_candidates,
        )

    def compare_event(self, event_id: Any, threshold: float = 133.0) -> dict[str, Any]:
        """Compare one event's observed history with catalog history."""
        normalized = resolve_event_id(event_id)
        return self._call(
            "history.compare_event",
            {"event_id": normalized, "threshold": threshold},
            statistics.compare_event_with_history,
        )

    def compare_observed_event_history(
        self,
        event_id: str,
        day_window: int = 15,
        threshold: float = 133.0,
        candidate_limit: int | None = None,
        display_limit: int = 20,
    ) -> dict[str, Any]:
        """Compare an observed event with same-region events from other years."""
        return self._call(
            "history.compare_observed_event_history",
            {
                "event_id": resolve_event_id(event_id),
                "day_window": day_window,
                "threshold": threshold,
                "candidate_limit": candidate_limit,
                "display_limit": display_limit,
            },
            statistics.compare_observed_event_with_historical_window,
        )

    def compare_cascast_history(
        self,
        event_id: str,
        month_window: int = 1,
        day_window: int | None = 15,
        threshold: float = 40.0,
        history_frame_count: int = 12,
        candidate_limit: int = 10,
        region: list[int] | None = None,
    ) -> dict[str, Any]:
        """Compare forecast/truth evidence with historical regional events."""
        return self._call(
            "history.compare_cascast_history",
            {
                "event_id": resolve_event_id(event_id),
                "month_window": month_window,
                "day_window": day_window,
                "threshold": threshold,
                "history_frame_count": history_frame_count,
                "candidate_limit": candidate_limit,
                "region": region,
            },
            statistics.compare_cascast_with_historical_region,
        )
