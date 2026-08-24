from __future__ import annotations

from typing import Any

import numpy as np

from ..backends import statistics
from ..data_access import catalog
from ..evaluation.cases import list_demo_cases

from .base import WorldComponent, resolve_event_id


class SevirStatistics(WorldComponent):
    """Catalog and optional HDF5 statistics with explicit metadata boundaries."""

    def query_records(
        self,
        year: int | None = None,
        event_type: str = "",
        fields: list[str] | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Return bounded catalog rows for agent-selected downstream analysis.

        Use this primitive when the question does not match a predefined report.
        The result is metadata/catalog data, not a per-pixel time series.
        """
        return self._call(
            "statistics.query_records",
            {"year": year, "event_type": event_type, "fields": fields, "limit": limit},
            statistics.query_sevir_records,
        )

    def schema(self) -> dict[str, Any]:
        """Discover all fields currently present in the SEVIR catalog records."""
        return self._call("statistics.schema", {}, statistics.catalog_schema)

    def aggregate(
        self,
        records: dict[str, Any] | list[dict[str, Any]],
        field: str,
        operations: list[str] | None = None,
        quantiles: list[float] | None = None,
    ) -> dict[str, Any]:
        """Compute only the aggregates selected by the agent over query rows."""
        return self._call(
            "statistics.aggregate",
            {
                "records": records,
                "field": field,
                "operations": operations,
                "quantiles": quantiles,
            },
            statistics.aggregate_sevir_records,
        )

    def values(
        self,
        field: str = "data_max",
        year: int | None = None,
        event_type: str = "",
        limit: int = 50000,
    ) -> dict[str, Any]:
        """Return filtered numeric values without selecting a statistic for the agent."""
        return self._call(
            "statistics.values",
            {
                "field": field,
                "year": year,
                "event_type": event_type,
                "limit": limit,
            },
            statistics.query_sevir_values,
        )

    def event_history_primitives(
        self,
        event_ids: list[str],
        start_frame: int = 0,
        end_frame: int = 49,
    ) -> dict[str, Any]:
        """Return per-frame source primitives for Agent-selected history analysis.

        The tool does not choose a threshold, duration rule, similarity rule,
        or summary statistic. Generated Agent code must calculate those from
        the returned frame maxima and timelines.
        """
        return self._call(
            "statistics.event_history_primitives",
            {
                "event_ids": event_ids,
                "start_frame": start_frame,
                "end_frame": end_frame,
            },
            statistics.load_event_history_frame_primitives,
        )

    def event_history_raw(
        self,
        event_ids: list[str],
        start_frame: int = 0,
        end_frame: int = 49,
        spatial_stride: int = 1,
        max_total_values: int = 8_000_000,
    ) -> dict[str, Any]:
        """Return bounded raw observed frame cubes for generated calculations.

        ``frame_values`` is shaped ``[time][row][column]`` and contains encoded
        uint8 VIL values; 255 is missing. ``relative_minutes`` is positionally
        aligned with time. No threshold, area, percentile, morphology,
        similarity rule, or duration statistic is selected by this tool.

        A full 49x384x384 event fits the default value budget, but several full
        events do not. Use separate calls, shorter windows, or
        ``spatial_stride`` 2/4/8 for a broader candidate screen. If a derived
        active run touches either selected time boundary, report the respective
        left/right censoring and treat its duration as a lower bound; this API
        never proves the complete storm lifecycle.
        """
        return self._call(
            "statistics.event_history_raw",
            {
                "event_ids": event_ids,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "spatial_stride": spatial_stride,
                "max_total_values": max_total_values,
            },
            statistics.load_event_history_raw,
        )

    def event_histories_raw(
        self,
        event_ids: list[str],
        start_frame: int = 0,
        end_frame: int = 49,
        spatial_stride: int = 1,
        max_total_values: int = 8_000_000,
    ) -> dict[str, Any]:
        """Load up to 20 event histories as one bounded raw frame cube.

        This is the preferred multi-event primitive for questions such as
        "similar events usually lasted how long?" First let generated code
        choose candidate event IDs and an explicit similarity definition, then
        call this method once with ``[reference_event_id, *candidate_ids]``.

        When the events share a shape/timeline, ``frame_cube`` is a read-only
        ndarray shaped ``[event,time,row,column]`` aligned with
        ``loaded_event_ids``; ``relative_minutes`` is aligned with time. Value
        255 is missing. Never test an ndarray with ``if frame_cube`` or
        ``if not frame_cube``; check ``frame_cube_available`` and use array
        operations. A robust per-frame valid mean can be composed as:
        ``valid = cube != 255``; ``clean = arrays.where(valid,cube,0)``;
        ``count = arrays.count_nonzero(valid,axis=[2,3])``; replace zero counts
        with one before division; then let the Agent define activity, run
        length, censoring, median and IQR. The tool itself computes none of
        those statistics and never claims a complete lifecycle.

        At stride 8, twenty 49-frame events fit in one call. At full resolution
        the value budget normally fits only one event, so use stride 2/4/8 for
        a broad historical comparison and disclose the chosen stride.
        """
        return self._call(
            "statistics.event_histories_raw",
            {
                "event_ids": event_ids,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "spatial_stride": spatial_stride,
                "max_total_values": max_total_values,
            },
            statistics.load_event_history_raw,
        )

    def similar_event_candidates(
        self,
        reference_event_id: str,
        field: str = "data_max",
        candidate_limit: int = 10,
        same_event_type: bool = True,
        local_hdf5_only: bool = True,
    ) -> dict[str, Any]:
        """Rank metadata-nearest candidate events without calculating duration.

        ``field``, candidate count, event-type restriction, and whether to keep
        only locally readable HDF5 rows are Agent-selected. The default local
        filter avoids returning candidates that cannot support frame analysis.
        """
        return self._call(
            "statistics.similar_event_candidates",
            {
                "reference_event_id": reference_event_id,
                "field": field,
                "candidate_limit": candidate_limit,
                "same_event_type": same_event_type,
                "local_hdf5_only": local_hdf5_only,
            },
            statistics.find_similar_event_candidates,
        )

    def rank(
        self,
        records: dict[str, Any] | list[dict[str, Any]],
        field: str,
        limit: int = 10,
        descending: bool = True,
    ) -> dict[str, Any]:
        """Rank query rows by a selected field with an agent-selected limit."""
        return self._call(
            "statistics.rank",
            {
                "records": records,
                "field": field,
                "limit": limit,
                "descending": descending,
            },
            statistics.rank_sevir_records,
        )

    def aggregate_catalog(
        self,
        year: int | None = None,
        event_type: str = "",
        field: str = "data_max",
        operations: list[str] | None = None,
        quantiles: list[float] | None = None,
    ) -> dict[str, Any]:
        """Aggregate the complete filtered catalog using agent-selected operations."""
        return self._call(
            "statistics.aggregate_catalog",
            {
                "year": year,
                "event_type": event_type,
                "field": field,
                "operations": operations,
                "quantiles": quantiles,
            },
            statistics.aggregate_sevir_catalog,
        )

    def rank_catalog(
        self,
        year: int | None = None,
        event_type: str = "",
        field: str = "data_max",
        limit: int = 10,
        descending: bool = True,
    ) -> dict[str, Any]:
        """Rank the complete filtered catalog with an agent-selected limit."""
        return self._call(
            "statistics.rank_catalog",
            {
                "year": year,
                "event_type": event_type,
                "field": field,
                "limit": limit,
                "descending": descending,
            },
            statistics.rank_sevir_catalog,
        )

    def group_catalog(
        self,
        group_by: str,
        field: str = "data_max",
        year: int | None = None,
        event_type: str = "",
        operations: list[str] | None = None,
        quantiles: list[float] | None = None,
        group_limit: int = 50,
    ) -> dict[str, Any]:
        """Group catalog rows using agent-selected fields, metrics and output size."""
        return self._call(
            "statistics.group_catalog",
            {
                "group_by": group_by,
                "field": field,
                "year": year,
                "event_type": event_type,
                "operations": operations,
                "quantiles": quantiles,
                "group_limit": group_limit,
            },
            statistics.group_sevir_catalog,
        )

    def compare_groups(
        self,
        group_by: str,
        group_values: list[str],
        field: str = "data_max",
        year: int | None = None,
        operations: list[str] | None = None,
        quantiles: list[float] | None = None,
    ) -> dict[str, Any]:
        """Compare exact groups with statistics explicitly selected by the agent."""
        return self._call(
            "statistics.compare_groups",
            {
                "group_by": group_by,
                "group_values": group_values,
                "field": field,
                "year": year,
                "operations": operations,
                "quantiles": quantiles,
            },
            statistics.compare_sevir_catalog_groups,
        )

    def historical(self, year: int | None = None, event_type: str = "") -> dict[str, Any]:
        """Summarize catalog rows; results are not per-frame rainfall trends."""
        arguments = {"year": year, "event_type": event_type}
        return self._call(
            "statistics.historical", arguments, statistics.historical_sevir_statistics
        )

    def compare_event(
        self,
        event_id: Any,
        threshold: float = 133.0,
    ) -> dict[str, Any]:
        """Compare an event with history and summarize observed high-VIL duration.

        The percentile uses catalog data_max. Duration follows catalog metadata
        to the official HDF5 row, falling back to saved observation frames.
        """
        event_id = resolve_event_id(event_id)
        return self._call(
            "statistics.compare_event",
            {
                "event_id": event_id,
                "threshold": threshold,
            },
            statistics.compare_event_with_history,
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
        """Compare CasCast, held-out truth, and same-region historical events."""
        arguments = {
            "event_id": event_id,
            "month_window": month_window,
            "day_window": day_window,
            "threshold": threshold,
            "history_frame_count": history_frame_count,
            "candidate_limit": candidate_limit,
            "region": region,
        }
        return self._call(
            "statistics.compare_cascast_history",
            arguments,
            statistics.compare_cascast_with_historical_region,
        )

    def compare_observed_event_history(
        self,
        event_id: str,
        day_window: int = 15,
        threshold: float = 133.0,
        candidate_limit: int | None = None,
        display_limit: int = 20,
    ) -> dict[str, Any]:
        """Compare any observed SEVIR event with same-region events in other years."""
        event_id = resolve_event_id(event_id)
        return self._call(
            "statistics.compare_observed_event_history",
            {
                "event_id": event_id,
                "day_window": day_window,
                "threshold": threshold,
                "candidate_limit": candidate_limit,
                "display_limit": display_limit,
            },
            statistics.compare_observed_event_with_historical_window,
        )

    def demo_cases(self, limit: int = 20) -> dict[str, Any]:
        """List data-driven CasCast demo/regression cases."""
        return self._call("statistics.demo_cases", {"limit": limit}, list_demo_cases)

    def compare_storm_types(
        self, event_types: list[str] | None = None, year: int | None = None
    ) -> dict[str, Any]:
        """Compare catalog data_max distributions and seasonality by storm type.

        Units are encoded VIL catalog values. Counts are catalog records, not
        unique disasters. Missing data_max values are excluded from quantiles.
        """
        requested = {item.strip().lower() for item in (event_types or []) if item.strip()}
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in catalog._catalog_vil_records():
            if year is not None and record["year"] != int(year):
                continue
            label = str(record["event_type"] or "Unlabeled")
            if requested and label.lower() not in requested:
                continue
            groups.setdefault(label, []).append(record)

        summaries: list[dict[str, Any]] = []
        for label, records in groups.items():
            values = np.asarray(
                [item["data_max"] for item in records if item["data_max"] is not None],
                dtype=np.float64,
            )
            months: dict[str, int] = {}
            for item in records:
                if item["month"] is not None:
                    key = f"{int(item['month']):02d}"
                    months[key] = months.get(key, 0) + 1
            summaries.append(
                {
                    "event_type": label,
                    "record_count": len(records),
                    "data_max_valid_count": int(values.size),
                    "data_max": {
                        "mean": round(float(values.mean()), 4) if values.size else None,
                        "p10": round(float(np.quantile(values, 0.10)), 4) if values.size else None,
                        "p25": round(float(np.quantile(values, 0.25)), 4) if values.size else None,
                        "p50": round(float(np.quantile(values, 0.50)), 4) if values.size else None,
                        "p75": round(float(np.quantile(values, 0.75)), 4) if values.size else None,
                        "p90": round(float(np.quantile(values, 0.90)), 4) if values.size else None,
                        "p95": round(float(np.quantile(values, 0.95)), 4) if values.size else None,
                    },
                    "counts_by_month": dict(sorted(months.items())),
                }
            )
        summaries.sort(key=lambda item: item["record_count"], reverse=True)
        result = {
            "status": "ok",
            "task": "storm_type_catalog_comparison",
            "year": year,
            "storm_type_count": len(summaries),
            "groups": summaries,
            "interpretation_limit": (
                "Catalog metadata comparison only; record counts are not disaster counts, "
                "and data_max is not a frame-by-frame duration metric."
            ),
        }
        self._recorder(
            "statistics.compare_storm_types",
            {"event_types": event_types, "year": year},
            result,
        )
        return result
