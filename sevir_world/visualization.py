from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .base import WorldComponent


ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "data" / "artifacts"


class Visualizer(WorldComponent):
    """Create reproducible PNG charts through approved SEVIRWorld analyses.

    Artifacts are written only below the configured artifact directory. Methods
    return an opaque artifact ID and an HTTP URL; generated code cannot choose a
    filesystem path.
    """

    def __init__(self, recorder, metrics, statistics, uncertainty) -> None:
        super().__init__(recorder)
        self._metrics = metrics
        self._statistics = statistics
        self._uncertainty = uncertainty

    @staticmethod
    def _save(figure, chart_type: str, data: dict[str, Any]) -> dict[str, Any]:
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        artifact_id = uuid.uuid4().hex
        path = ARTIFACT_ROOT / f"{artifact_id}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return {
            "status": "ok",
            "task": "sevir_chart",
            "chart_type": chart_type,
            "artifact_id": artifact_id,
            "image_url": f"/api/artifacts/{artifact_id}.png",
            "data": data,
        }

    def lead_time_metrics(
        self, event_id: str, threshold: float = 133.0, metric: str = "csi"
    ) -> dict[str, Any]:
        """Plot per-lead CSI, POD, FAR, or HSS for CasCast and Persistence."""
        if metric not in {"csi", "pod", "far", "hss"}:
            raise ValueError("metric must be csi, pod, far, or hss")
        evaluation = self._metrics.evaluate_event(
            event_id, thresholds=[threshold], fss_scales=[1]
        )
        threshold_key = str(int(threshold) if float(threshold).is_integer() else threshold)
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        styles = {
            "cascast_ensemble_mean": ("CasCast ensemble mean", "#1976d2"),
            "persistence": ("Persistence", "#f28b55"),
        }
        chart_data = {}
        for model, (label, color) in styles.items():
            rows = evaluation["models"][model][threshold_key]["per_lead"]
            x = [row["lead_minutes"] for row in rows]
            y = [row[metric] for row in rows]
            chart_data[model] = [{"lead_minutes": a, metric: b} for a, b in zip(x, y)]
            axis.plot(x, y, marker="o", linewidth=2, label=label, color=color)
        axis.set(title=f"{metric.upper()} by lead time, VIL >= {threshold:g}", xlabel="Lead time (min)", ylabel=metric.upper())
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
        result = self._save(figure, "lead_time_metrics", chart_data)
        self._recorder(
            "visualizer.lead_time_metrics",
            {"event_id": event_id, "threshold": threshold, "metric": metric},
            result,
        )
        return result

    def storm_tracks(
        self, event_id: str, source: str = "ensemble_mean", threshold: float = 133.0
    ) -> dict[str, Any]:
        """Plot up to ten connected-object tracks on the 384x384 SEVIR grid."""
        tracking = self._uncertainty.track_objects(event_id, source=source, threshold=threshold)
        figure, axis = plt.subplots(figsize=(6.2, 6.2))
        data = []
        for track in tracking["tracks"][:10]:
            points = [row["centroid"] for row in track["observations"]]
            if not points:
                continue
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            axis.plot(x, y, marker="o", linewidth=2, label=track["track_id"])
            data.append({"track_id": track["track_id"], "centroids": points})
        axis.set(
            title=f"Storm-object tracks, {source}, VIL >= {threshold:g}",
            xlabel="Grid column (px)",
            ylabel="Grid row (px)",
            xlim=(0, 383),
            ylim=(383, 0),
        )
        axis.grid(alpha=0.2)
        if data:
            axis.legend(frameon=False, ncol=2, fontsize=8)
        result = self._save(figure, "storm_tracks", {"event_id": event_id, "tracks": data})
        self._recorder(
            "visualizer.storm_tracks",
            {"event_id": event_id, "source": source, "threshold": threshold},
            result,
        )
        return result

    def storm_type_comparison(
        self, event_types: list[str] | None = None, year: int | None = None
    ) -> dict[str, Any]:
        """Plot catalog data_max median and p95 by storm type."""
        comparison = self._statistics.compare_storm_types(event_types=event_types, year=year)
        groups = [item for item in comparison["groups"] if item["data_max"]["p50"] is not None][:12]
        labels = [item["event_type"] for item in groups]
        medians = [item["data_max"]["p50"] for item in groups]
        p95 = [item["data_max"]["p95"] for item in groups]
        figure, axis = plt.subplots(figsize=(8.2, 4.6))
        positions = range(len(groups))
        axis.bar(positions, p95, color="#9cc8c0", label="p95")
        axis.bar(positions, medians, color="#1976d2", label="p50")
        axis.set_xticks(list(positions), labels, rotation=35, ha="right")
        axis.set(title="Catalog data_max by storm type", ylabel="Encoded VIL data_max")
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.2)
        result = self._save(
            figure,
            "storm_type_comparison",
            {"groups": [{"event_type": label, "p50": median, "p95": high} for label, median, high in zip(labels, medians, p95)]},
        )
        self._recorder(
            "visualizer.storm_type_comparison",
            {"event_types": event_types, "year": year},
            result,
        )
        return result

    def ensemble_consensus(self, event_id: str, threshold: float = 133.0) -> dict[str, Any]:
        """Plot cross-member consensus-object centroids and support fractions."""
        matching = self._uncertainty.match_ensemble_objects(event_id, threshold=threshold)
        objects = matching["consensus_objects"][:15]
        figure, axis = plt.subplots(figsize=(6.2, 6.2))
        for item in objects:
            x, y = item["mean_centroid"]
            support = item["member_support_fraction"]
            axis.scatter(x, y, s=80 + support * 500, alpha=0.65, color="#1976d2")
            axis.text(x + 4, y, f"{item['consensus_id']} {support:.0%}", fontsize=8)
        axis.set(
            title=f"Cross-ensemble object support, VIL >= {threshold:g}",
            xlabel="Grid column (px)",
            ylabel="Grid row (px)",
            xlim=(0, 383),
            ylim=(383, 0),
        )
        axis.grid(alpha=0.2)
        result = self._save(
            figure,
            "ensemble_consensus",
            {"event_id": event_id, "consensus_objects": objects},
        )
        self._recorder(
            "visualizer.ensemble_consensus",
            {"event_id": event_id, "threshold": threshold},
            result,
        )
        return result
