from __future__ import annotations

from typing import Any

import numpy as np

from .backends import statistics
from .repositories import catalog

from .base import WorldComponent, resolve_event_id


class SevirStatistics(WorldComponent):
    """Catalog and optional HDF5 statistics with explicit metadata boundaries."""

    def historical(self, year: int | None = None, event_type: str = "") -> dict[str, Any]:
        """Summarize catalog rows; results are not per-frame rainfall trends."""
        arguments = {"year": year, "event_type": event_type}
        return self._call(
            "statistics.historical", arguments, statistics.historical_sevir_statistics
        )

    def compare_event(self, event_id: Any) -> dict[str, Any]:
        """Compare an event ID or loaded ds with its historical reference group."""
        event_id = resolve_event_id(event_id)
        return self._call(
            "statistics.compare_event",
            {"event_id": event_id},
            statistics.compare_event_with_history,
        )

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
