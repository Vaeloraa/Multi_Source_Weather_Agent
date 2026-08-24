from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..domain.vil_palette import cascast_vil_cmap, colorize_vil, mask_missing_vil
from .base import WorldComponent


_DEFAULT_ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "data" / "artifacts"
_FALLBACK_ARTIFACT_ROOT = Path(tempfile.gettempdir()) / "sevir_world_artifacts"


def _artifact_root() -> Path:
    candidate = Path(os.getenv("SEVIR_ARTIFACT_DIR", str(_DEFAULT_ARTIFACT_ROOT)))
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.write_bytes(b"ok")
        probe.unlink(missing_ok=True)
        return candidate
    except (OSError, PermissionError):
        _FALLBACK_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        return _FALLBACK_ARTIFACT_ROOT


ARTIFACT_ROOT = _artifact_root()
_ARTIFACT_WRITE_LOCK = threading.Lock()
_RENDER_CACHE_LOCK = threading.Lock()
_RENDER_CACHE: dict[str, dict[str, Any]] = {}
_RENDERER_VERSION = "fast-raster-2"


def _render_cache_key(spec: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    def visit(value: Any) -> None:
        if isinstance(value, np.ndarray):
            digest.update(b"ndarray:")
            digest.update(str(value.shape).encode())
            digest.update(str(value.dtype).encode())
            digest.update(value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode("utf-8")); visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value: visit(item)
        else:
            digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
    digest.update(_RENDERER_VERSION.encode("ascii")); visit(spec)
    return digest.hexdigest()
_FAST_FONT_LOCK = threading.Lock()
_FAST_FONT: ImageFont.ImageFont | ImageFont.FreeTypeFont | None = None


def _fast_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Reuse the raster font across renders; font discovery is surprisingly costly."""
    global _FAST_FONT
    if _FAST_FONT is not None:
        return _FAST_FONT
    with _FAST_FONT_LOCK:
        if _FAST_FONT is not None:
            return _FAST_FONT
        for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
            try:
                if Path(candidate).is_file():
                    _FAST_FONT = ImageFont.truetype(candidate, 16)
                    break
            except (OSError, ValueError):
                continue
        if _FAST_FONT is None:
            _FAST_FONT = ImageFont.load_default()
        return _FAST_FONT


def _configure_plot_fonts() -> None:
    """Use an installed CJK font so Chinese Agent labels render as text."""

    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            return
        except (OSError, RuntimeError, ValueError):
            continue


_configure_plot_fonts()


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
        self._render_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _save(
        figure,
        chart_type: str,
        data: dict[str, Any],
        title: str | None = None,
        description: str = "",
    ) -> dict[str, Any]:
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        dpi = max(90, int(os.getenv("SEVIR_VISUALIZER_DPI", "120")))
        figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        payload = buffer.getvalue()
        verified = len(payload) > 8 and payload.startswith(b"\x89PNG\r\n\x1a\n")
        if not verified:
            raise RuntimeError("visualizer did not produce a valid PNG artifact")
        # Identical renders use the same opaque ID. This makes repeated model
        # repair attempts converge on one artifact instead of showing duplicate
        # image sets to the user. The API contract requires 32 hex characters.
        artifact_id = hashlib.sha256(payload).hexdigest()[:32]
        path = ARTIFACT_ROOT / f"{artifact_id}.png"
        if not path.exists():
            with _ARTIFACT_WRITE_LOCK:
                if not path.exists():
                    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                    try:
                        temporary.write_bytes(payload)
                        temporary.replace(path)
                    except PermissionError:
                        # Some managed Windows temp folders deny sibling temp
                        # files even though the final artifact is writable.
                        # Keep the normal atomic path and degrade only there.
                        path.write_bytes(payload)
                    finally:
                        temporary.unlink(missing_ok=True)
        image_url = f"/api/artifacts/{artifact_id}.png"
        return {
            "status": "ok",
            "task": "sevir_chart",
            "chart_type": chart_type,
            "artifact_id": artifact_id,
            "url": image_url,
            "image_url": image_url,
            "mime_type": "image/png",
            "title": str(title or chart_type.replace("_", " ").title()),
            "description": str(description),
            "size_bytes": len(payload),
            "verified": True,
            "data": data,
        }

    @staticmethod
    def _save_bytes(
        payload: bytes,
        chart_type: str,
        data: dict[str, Any],
        title: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Persist a pre-rendered artifact using the same verified contract."""
        if len(payload) <= 8 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("visualizer did not produce a valid PNG artifact")
        artifact_id = hashlib.sha256(payload).hexdigest()[:32]
        path = ARTIFACT_ROOT / f"{artifact_id}.png"
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with _ARTIFACT_WRITE_LOCK:
                if not path.exists():
                    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                    try:
                        temporary.write_bytes(payload)
                        temporary.replace(path)
                    except PermissionError:
                        path.write_bytes(payload)
                    finally:
                        temporary.unlink(missing_ok=True)
        image_url = f"/api/artifacts/{artifact_id}.png"
        return {
            "status": "ok",
            "task": "sevir_chart",
            "chart_type": chart_type,
            "artifact_id": artifact_id,
            "url": image_url,
            "image_url": image_url,
            "mime_type": "image/png",
            "title": str(title),
            "description": str(description),
            "size_bytes": len(payload),
            "verified": True,
            "data": data,
        }

    @staticmethod
    def _fast_raster_panels(
        panels: list[dict[str, Any]],
        chart_type: str,
        title: str,
        overlays: list[dict[str, Any]] | None = None,
        annotations: dict[str, Any] | None = None,
    ) -> bytes:
        """Render raster panels without constructing a matplotlib Figure."""
        panel_images: list[Image.Image] = []
        font = _fast_font()
        for index, panel in enumerate(panels):
            if not isinstance(panel, dict):
                raise ValueError("each panel must be an object")
            # Accept the two common semantic aliases produced by chart agents
            # while keeping one canonical renderer contract internally.
            values = np.asarray(panel.get("values", panel.get("image")), dtype=np.float32)
            if panel.get("cmap") == "cascast_vil":
                rgb = colorize_vil(values)
            else:
                finite = values[np.isfinite(values)]
                vmin = float(panel.get("vmin", finite.min()))
                vmax = float(panel.get("vmax", finite.max()))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                normalized = np.clip((np.nan_to_num(values, nan=vmin) - vmin) / (vmax - vmin), 0.0, 1.0)
                cmap = plt.get_cmap(str(panel.get("cmap", "viridis")))
                rgb = (cmap(normalized)[..., :3] * 255.0).astype(np.uint8)
                rgb[~np.isfinite(values)] = 0
            source_height, source_width = values.shape
            image = Image.fromarray(rgb, mode="RGB").resize((384, 384), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 384, 27), fill=(0, 0, 0))
            draw.text((8, 5), str(panel.get("label", panel.get("title", "panel")))[:60], fill="white", font=font)
            if chart_type == "overlay":
                for overlay in overlays or []:
                    overlay_values = np.asarray(overlay["values"], dtype=np.float32)
                    level = float(overlay.get("level", 0.5))
                    mask = np.isfinite(overlay_values) & (overlay_values >= level)
                    if np.any(mask):
                        resized_mask = np.asarray(
                            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                                (384, 384), Image.Resampling.NEAREST
                            )
                        ) > 0
                        interior = (
                            np.roll(resized_mask, 1, axis=0)
                            & np.roll(resized_mask, -1, axis=0)
                            & np.roll(resized_mask, 1, axis=1)
                            & np.roll(resized_mask, -1, axis=1)
                        )
                        outline = resized_mask & ~interior
                        rendered = np.asarray(image).copy()
                        rendered[outline] = (211, 47, 47)
                        image = Image.fromarray(rendered, mode="RGB")
                        draw = ImageDraw.Draw(image)
                annotation_spec = annotations if isinstance(annotations, dict) else {}
                for box in annotation_spec.get("boxes", []):
                    if not isinstance(box, dict):
                        continue
                    left = float(box.get("left", 0)) * 384.0 / source_width
                    top = float(box.get("top", 0)) * 384.0 / source_height
                    right = (float(box.get("right", 0)) + 1.0) * 384.0 / source_width - 1.0
                    bottom = (float(box.get("bottom", 0)) + 1.0) * 384.0 / source_height - 1.0
                    color = str(box.get("color", "#d32f2f"))
                    draw.rectangle((left, top, right, bottom), outline=color, width=max(1, int(box.get("width", 3))))
                for point in annotation_spec.get("points", []):
                    if not isinstance(point, dict):
                        continue
                    column = float(point.get("column", point.get("x", 0)))
                    row = float(point.get("row", point.get("y", 0)))
                    x = (column + 0.5) * 384.0 / source_width
                    y = (row + 0.5) * 384.0 / source_height
                    radius = max(2.0, float(point.get("radius", 6)))
                    color = str(point.get("color", "#d32f2f"))
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=max(1, int(point.get("width", 3))))
            panel_images.append(image)
        columns = min(3, len(panel_images))
        rows = (len(panel_images) + columns - 1) // columns
        canvas = Image.new("RGB", (384 * columns, 414 * rows + 36), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), str(title)[:120], fill="black", font=font)
        for index, image in enumerate(panel_images):
            x = (index % columns) * 384
            y = 36 + (index // columns) * 414
            canvas.paste(image, (x, y))
        output = io.BytesIO()
        # The fast path is latency-sensitive; exhaustive PNG optimization costs
        # more time than drawing a single 384x384 annotation.
        canvas.save(output, format="PNG", compress_level=3)
        return output.getvalue()

    def grid_panels(
        self,
        panels: list[dict[str, Any]],
        title: str = "Agent-selected spatial comparison",
    ) -> dict[str, Any]:
        """Render 1-6 Agent-selected 2-D grids with explicit, comparable scales.

        Each panel is ``{"label": str, "values": 2-D matrix}`` and may also
        specify ``cmap``, ``vmin``, ``vmax``, and ``colorbar_label``. Use
        ``cmap='cascast_vil'`` for encoded VIL panels so the plot matches the
        CasCast/SEVIR palette; other derived quantities should choose their own
        appropriate scale. This is a
        generic renderer: the generated program decides what each matrix means
        (member, ensemble mean, truth, spread, error, agreement, or a novel
        derived field). It does not calculate or select scientific metrics.
        Panels with the same colormap and colorbar label automatically share a
        scale unless the Agent explicitly supplies ``vmin`` or ``vmax``. This
        prevents identical values from appearing as different colors while
        keeping unlike quantities (for example VIL and ensemble spread) apart.
        """
        if not 1 <= len(panels) <= 6:
            raise ValueError("panels must contain between 1 and 6 grids")
        clean: list[dict[str, Any]] = []
        for index, panel in enumerate(panels):
            if not isinstance(panel, dict):
                raise ValueError("each panel must be an object")
            values = np.asarray(panel.get("values", panel.get("image")), dtype=np.float32)
            if values.ndim != 2 or values.size > 384 * 384:
                raise ValueError("each panel must be a 2-D grid no larger than 384x384")
            finite = values[np.isfinite(values)]
            if not finite.size:
                raise ValueError("each panel must contain at least one finite value")
            vmin = float(panel.get("vmin", finite.min()))
            vmax = float(panel.get("vmax", finite.max()))
            if vmax <= vmin:
                vmax = vmin + 1.0
            clean.append({
                "label": str(panel.get("label", panel.get("title", f"panel {index + 1}"))),
                "values": values,
                "cmap": str(panel.get("cmap", "turbo")),
                "vmin": vmin,
                "vmax": vmax,
                "colorbar_label": str(panel.get("colorbar_label", "value")),
                "explicit_scale": "vmin" in panel or "vmax" in panel,
            })
        scale_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for panel in clean:
            if not panel["explicit_scale"]:
                key = (panel["cmap"], panel["colorbar_label"])
                scale_groups.setdefault(key, []).append(panel)
        for group in scale_groups.values():
            shared_vmin = min(item["vmin"] for item in group)
            shared_vmax = max(item["vmax"] for item in group)
            if shared_vmax <= shared_vmin:
                shared_vmax = shared_vmin + 1.0
            for panel in group:
                panel["vmin"] = shared_vmin
                panel["vmax"] = shared_vmax
        columns = min(3, len(clean))
        rows = (len(clean) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 4.0 * rows), squeeze=False)
        for index, axis in enumerate(axes.flat):
            if index >= len(clean):
                axis.axis("off")
                continue
            panel = clean[index]
            if panel["cmap"] == "cascast_vil":
                cmap, norm = cascast_vil_cmap()
                image = axis.imshow(mask_missing_vil(panel["values"]), cmap=cmap, norm=norm, origin="upper")
            else:
                image = axis.imshow(
                    panel["values"], cmap=panel["cmap"], vmin=panel["vmin"],
                    vmax=panel["vmax"], origin="upper"
                )
            axis.set_title(panel["label"])
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03, label=panel["colorbar_label"])
        figure.suptitle(str(title))
        figure.tight_layout()
        metadata = [{
            "label": item["label"], "shape": list(item["values"].shape),
            "cmap": item["cmap"], "vmin": item["vmin"], "vmax": item["vmax"],
            "colorbar_label": item["colorbar_label"],
            "scale_mode": "explicit" if item["explicit_scale"] else "shared_by_unit",
        } for item in clean]
        arguments = {
            "title": title,
            "labels": [item["label"] for item in clean],
            "panel_count": len(clean),
        }
        try:
            result = self._save(
                figure,
                "agent_grid_panels",
                {"panels": metadata},
                title=title,
                description=(
                    "Agent-selected spatial grids; comparable panels share a color scale by unit, "
                    "while unlike quantities use separately labeled scales."
                ),
            )
        except Exception as error:
            # Preserve artifact-directory/render failures in the public tool
            # trace.  Previously a PermissionError escaped before the recorder
            # call, making the run look as if visualizer.grid_panels had never
            # been attempted.
            plt.close(figure)
            self._recorder("visualizer.grid_panels", arguments, error=error)
            raise
        self._recorder("visualizer.grid_panels", arguments, result)
        return result

    def distribution(
        self,
        series: list[dict[str, Any]],
        title: str = "Agent-selected distributions",
        bins: int = 30,
    ) -> dict[str, Any]:
        """Plot 1-6 Agent-selected numeric distributions without choosing fields."""
        if not 1 <= len(series) <= 6:
            raise ValueError("series must contain between 1 and 6 distributions")
        bins = int(bins)
        if not 5 <= bins <= 100:
            raise ValueError("bins must be between 5 and 100")
        figure, axis = plt.subplots(figsize=(8.2, 4.8))
        metadata = []
        for index, item in enumerate(series):
            if not isinstance(item, dict):
                raise ValueError("each distribution must be an object")
            values = np.asarray(item["values"], dtype=np.float32).reshape(-1)
            values = values[np.isfinite(values)]
            if not 1 <= values.size <= 500_000:
                raise ValueError("each distribution must contain 1 to 500,000 finite values")
            label = str(item.get("label", f"series {index + 1}"))
            axis.hist(values, bins=bins, density=True, alpha=0.38, label=label)
            metadata.append({"label": label, "count": int(values.size), "minimum": float(values.min()), "maximum": float(values.max())})
        axis.set_title(str(title))
        axis.set_xlabel("Value")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        result = self._save(figure, "agent_distributions", {"series": metadata, "bins": bins}, title=title)
        self._recorder("visualizer.distribution", {"title": title, "series_labels": [item["label"] for item in metadata], "bins": bins}, result)
        return result

    def scatter(
        self,
        x: list[float],
        y: list[float],
        x_label: str,
        y_label: str,
        title: str = "Agent-selected comparison",
    ) -> dict[str, Any]:
        """Plot an Agent-selected paired comparison with an identity reference."""
        x_values = np.asarray(x, dtype=np.float32).reshape(-1)
        y_values = np.asarray(y, dtype=np.float32).reshape(-1)
        if x_values.size != y_values.size or not 1 <= x_values.size <= 500_000:
            raise ValueError("x and y must have the same length between 1 and 500,000")
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[valid]
        y_values = y_values[valid]
        if not x_values.size:
            raise ValueError("scatter has no finite pairs")
        if x_values.size > 50_000:
            step = max(1, x_values.size // 50_000)
            plot_x = x_values[::step]
            plot_y = y_values[::step]
        else:
            plot_x, plot_y = x_values, y_values
        lower = float(min(plot_x.min(), plot_y.min()))
        upper = float(max(plot_x.max(), plot_y.max()))
        figure, axis = plt.subplots(figsize=(6.2, 6.0))
        axis.scatter(plot_x, plot_y, s=6, alpha=0.25, color="#1976d2")
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#c62828", linewidth=1.5)
        axis.set(title=str(title), xlabel=str(x_label), ylabel=str(y_label))
        axis.grid(alpha=0.2)
        result = self._save(
            figure, "agent_scatter",
            {"pair_count": int(x_values.size), "plotted_count": int(plot_x.size), "x_label": str(x_label), "y_label": str(y_label)},
            title=title,
        )
        self._recorder("visualizer.scatter", {"title": title, "x_label": x_label, "y_label": y_label, "pair_count": int(x_values.size)}, result)
        return result


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

    def forecast_trends(
        self,
        lead_minutes: list[int],
        series: list[dict[str, Any]],
        title: str = "Agent-selected forecast trends",
    ) -> dict[str, Any]:
        """Plot agent-selected forecast time series on separate aligned panels.

        ``series`` is a list of ``{"label": str, "values": list[number]}``
        dictionaries. This method only renders values selected and calculated by
        generated code; it does not choose fields, thresholds, or statistics.
        """
        x = [int(value) for value in lead_minutes]
        if not 1 <= len(x) <= 72:
            raise ValueError("lead_minutes must contain between 1 and 72 values")
        if not 1 <= len(series) <= 6:
            raise ValueError("series must contain between 1 and 6 plotted fields")
        clean_series: list[dict[str, Any]] = []
        for index, item in enumerate(series):
            if not isinstance(item, dict):
                raise ValueError("each plotted series must be an object")
            label = str(item.get("label", f"series {index + 1}"))
            values = [float(value) for value in item["values"]]
            if len(values) != len(x):
                raise ValueError("every plotted series must match lead_minutes length")
            clean_series.append({"label": label, "values": values})
        figure, axes = plt.subplots(
            len(clean_series),
            1,
            figsize=(8.4, 2.35 * len(clean_series) + 1.0),
            sharex=True,
            squeeze=False,
        )
        colors = ["#1976d2", "#ef6c00", "#2e7d32", "#7b1fa2", "#c62828", "#00838f"]
        for index, item in enumerate(clean_series):
            axis = axes[index][0]
            axis.plot(x, item["values"], color=colors[index], linewidth=2, marker="o", markersize=3)
            axis.set_ylabel(item["label"])
            axis.grid(alpha=0.22)
        axes[0][0].set_title(str(title))
        axes[-1][0].set_xlabel("Forecast lead time (minutes)")
        figure.tight_layout()
        result = self._save(
            figure,
            "agent_forecast_trends",
            {"lead_minutes": x, "series": clean_series},
        )
        self._recorder(
            "visualizer.forecast_trends",
            {"lead_minutes": x, "series_labels": [item["label"] for item in clean_series]},
            result,
        )
        return result

    def plan_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Validate and enqueue intent without drawing inside the Agent loop.

        The HTTP layer consumes the returned ``render_spec`` after the textual
        answer is complete.  This keeps optional PNG work out of the analysis
        budget while preserving the exact data and chart semantics selected by
        generated code.
        """
        if not isinstance(spec, dict) and callable(getattr(spec, "to_dict", None)):
            spec = spec.to_dict()
        if not isinstance(spec, dict):
            raise ValueError("spec must be a dictionary or VisualizationSpec")
        requested_chart_type = str(spec.get("chart_type", "")).strip().lower()
        chart_type = requested_chart_type or (
            "raster_map" if isinstance(spec.get("panels"), list) else "line"
        )
        chart_type = {
            "map": "raster_map",
            "raster": "raster_map",
            "spatial": "raster_map",
            "spatial_focus": "raster_map",
            "panel": "small_multiples",
            "panels": "small_multiples",
            "timeseries": "line",
            "time_series": "line",
            "ensemble": "ensemble_range_line",
            "ensemble_range": "ensemble_range_line",
            "members": "small_multiples",
            "difference": "difference_map",
            "diff": "difference_map",
        }.get(chart_type, chart_type)
        allowed = {
            "raster_map", "bar", "grouped_bar", "line", "ensemble_range_line",
            "heatmap", "difference_map", "small_multiples", "overlay",
            "trajectory", "box",
        }
        if chart_type not in allowed:
            raise ValueError(f"unsupported chart_type: {chart_type}")
        spatial_types = {"raster_map", "heatmap", "difference_map", "small_multiples", "overlay"}
        if chart_type in spatial_types:
            panels = spec.get("panels")
            if not isinstance(panels, list) or not 1 <= len(panels) <= (12 if chart_type == "small_multiples" else 6):
                raise ValueError("render plan panels must contain bounded panel objects")
            for index, panel in enumerate(panels):
                if not isinstance(panel, dict) or "values" not in panel:
                    raise ValueError(f"render plan panels[{index}] requires values")
                values = np.asarray(panel["values"])
                if values.ndim != 2 or values.size > 384 * 384:
                    raise ValueError(f"render plan panels[{index}].values must be a bounded 2-D grid")
        elif chart_type == "trajectory":
            tracks = spec.get("tracks", spec.get("series"))
            if not isinstance(tracks, list) or not tracks:
                raise ValueError("render plan trajectory requires tracks")
        else:
            series = spec.get("series")
            if not isinstance(series, list) or not series:
                raise ValueError("render plan line/bar charts require series")
            for index, item in enumerate(series):
                if not isinstance(item, dict) or "values" not in item:
                    raise ValueError(f"render plan series[{index}] requires values")
        planned = dict(spec)
        planned["chart_type"] = chart_type
        planned.setdefault("title", chart_type.replace("_", " ").title())
        result = {
            "status": "planned",
            "task": "sevir_render_plan",
            "chart_type": chart_type,
            "render_spec": planned,
            "render_deferred": True,
        }
        self._recorder("visualizer.plan_spec", {"spec": planned}, result)
        return result

    def render_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Render an Agent-produced chart specification.

        This is deliberately a presentation-only endpoint: generated code
        selects and computes the values, while this method validates bounded
        payloads and draws one of the approved chart types.  One-dimensional
        charts use ``series``; spatial charts use ``panels``.  Spatial panel
        values are rendered with the CasCast VIL palette when ``cmap`` is
        ``"cascast_vil"``.  Supported chart types are ``raster_map``,
        ``bar``, ``grouped_bar``, ``line``, ``ensemble_range_line``,
        ``heatmap``, ``difference_map``, ``small_multiples``, ``overlay``,
        and ``trajectory`` (plus the existing ``box`` chart).  A trajectory
        uses ``tracks=[{"label": ..., "x": [...], "y": [...]}, ...]``;
        an ensemble range can use top-level ``lower``/``upper`` arrays.
        """
        if not isinstance(spec, dict) and callable(getattr(spec, "to_dict", None)):
            spec = spec.to_dict()
        if not isinstance(spec, dict):
            raise ValueError("chart spec must be an object or VisualizationSpec")
        requested_chart_type = str(spec.get("chart_type", "")).strip().lower()
        chart_type = requested_chart_type or (
            "raster_map" if isinstance(spec.get("panels"), list) else "line"
        )
        chart_type = {
            "map": "raster_map",
            "raster": "raster_map",
            "spatial": "raster_map",
            "spatial_focus": "raster_map",
            "panel": "small_multiples",
            "panels": "small_multiples",
            "timeseries": "line",
            "time_series": "line",
            "ensemble": "ensemble_range_line",
            "ensemble_range": "ensemble_range_line",
            "members": "small_multiples",
            "difference": "difference_map",
            "diff": "difference_map",
        }.get(chart_type, chart_type)
        supported = {
            "raster_map", "bar", "grouped_bar", "line", "ensemble_range_line",
            "heatmap", "difference_map", "small_multiples", "overlay", "trajectory",
            "box",
        }
        if chart_type not in supported:
            raise ValueError("unsupported chart_type; use raster_map, bar, grouped_bar, line, ensemble_range_line, heatmap, difference_map, small_multiples, overlay, trajectory, or box")
        title = str(spec.get("title", "Agent-selected chart"))[:200]
        labels = [str(value)[:80] for value in spec.get("labels", [])]
        render_key = _render_cache_key(spec)
        replay_arguments = {
            "render_key": render_key,
            "chart_type": chart_type,
            "title": title,
        }
        lookup = getattr(self._recorder, "lookup", None)
        replayed = lookup("visualizer.render_spec", replay_arguments) if callable(lookup) else None
        if replayed is not None:
            try:
                self._recorder(
                    "visualizer.render_spec", replay_arguments, replayed, cache_hit=True
                )
            except TypeError:
                self._recorder("visualizer.render_spec", replay_arguments, replayed)
            return replayed
        with _RENDER_CACHE_LOCK:
            cached_result = _RENDER_CACHE.get(render_key)
        if cached_result is not None:
            cached = dict(cached_result)
            cached["cache"] = {"hit": True, "key": render_key}
            try:
                self._recorder("visualizer.render_spec", replay_arguments, cached, cache_hit=True)
            except TypeError:
                self._recorder("visualizer.render_spec", replay_arguments, cached)
            return cached

        def _finish(rendered: dict[str, Any]) -> dict[str, Any]:
            rendered["cache"] = {"hit": False, "key": render_key}
            with _RENDER_CACHE_LOCK:
                _RENDER_CACHE[render_key] = dict(rendered)
            return rendered

        def _matrix(value: Any, name: str) -> np.ndarray:
            array = np.asarray(value, dtype=np.float32)
            if array.ndim != 2 or array.size > 384 * 384:
                raise ValueError(f"{name} must be a 2-D grid no larger than 384x384")
            if not np.isfinite(array).any():
                raise ValueError(f"{name} must contain a finite value")
            return array

        def _panels() -> list[dict[str, Any]]:
            raw = spec.get("panels", [])
            panel_limit = 12 if chart_type == "small_multiples" else 6
            if not isinstance(raw, list) or not 1 <= len(raw) <= panel_limit:
                raise ValueError(f"panels must contain between 1 and {panel_limit} items")
            result = []
            for index, item in enumerate(raw):
                if not isinstance(item, dict):
                    raise ValueError("each panel must be an object")
                clean_item = {
                    "label": str(item.get("label", f"panel {index + 1}"))[:80],
                    "values": _matrix(item.get("values"), f"panels[{index}].values"),
                    "cmap": str(item.get("cmap", "coolwarm" if chart_type == "difference_map" else ("cascast_vil" if chart_type in {"raster_map", "small_multiples", "overlay"} else "viridis"))),
                    "colorbar_label": str(item.get("colorbar_label", spec.get("colorbar_label", "Value")))[:80],
                    **({"vmin": float(item["vmin"])} if "vmin" in item else {}),
                    **({"vmax": float(item["vmax"])} if "vmax" in item else {}),
                }
                if chart_type == "difference_map" and "vmin" not in clean_item and "vmax" not in clean_item:
                    finite = clean_item["values"][np.isfinite(clean_item["values"])]
                    limit = max(1.0, float(np.max(np.abs(finite))))
                    clean_item.update({"vmin": -limit, "vmax": limit})
                result.append(clean_item)
            shapes = {tuple(item["values"].shape) for item in result}
            if len(shapes) != 1:
                raise ValueError("all panels must have the same grid shape")
            return result

        # The ten public chart names map to two compact input grammars.  This
        # keeps the Agent free to choose fields without multiplying tools.
        if chart_type in {"raster_map", "heatmap", "difference_map", "small_multiples", "overlay"}:
            panels = _panels()
            if chart_type in {"raster_map", "heatmap", "difference_map", "overlay"} or len(panels) >= 3:
                payload = self._fast_raster_panels(
                    panels,
                    chart_type,
                    title,
                    [item for item in spec.get("overlays", []) if isinstance(item, dict)],
                    spec.get("annotations") if isinstance(spec.get("annotations"), dict) else None,
                )
                result = self._save_bytes(
                    payload,
                    "agent_chart_spec",
                    {
                        "chart_type": chart_type,
                        "panels": [{"label": p["label"], "shape": list(p["values"].shape), "cmap": p["cmap"]} for p in panels],
                        "annotations": spec.get("annotations", {}),
                        "renderer": "fast_raster",
                    },
                    title=title,
                    description=str(spec.get("description", "")),
                )
                self._recorder("visualizer.render_spec", replay_arguments, result)
                return _finish(result)
            columns = min(4 if chart_type == "small_multiples" else 3, len(panels))
            rows = (len(panels) + columns - 1) // columns
            figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 4.1 * rows), squeeze=False)
            images = []
            for index, axis in enumerate(axes.flat):
                if index >= len(panels):
                    axis.axis("off")
                    continue
                panel = panels[index]
                values = panel["values"]
                if panel["cmap"] == "cascast_vil":
                    cmap, norm = cascast_vil_cmap()
                    image = axis.imshow(mask_missing_vil(values), cmap=cmap, norm=norm, origin="upper")
                else:
                    image = axis.imshow(values, cmap=panel["cmap"], vmin=panel.get("vmin"), vmax=panel.get("vmax"), origin="upper")
                images.append(image)
                axis.set_title(panel["label"])
                axis.set_xticks([])
                axis.set_yticks([])
                if chart_type == "overlay":
                    for overlay in spec.get("overlays", []):
                        if not isinstance(overlay, dict):
                            raise ValueError("each overlay must be an object")
                        overlay_values = _matrix(overlay.get("values"), "overlay.values")
                        level = float(overlay.get("level", 0.5))
                        axis.contour(overlay_values, levels=[level], colors=[str(overlay.get("color", "#d32f2f"))], linewidths=1.6)
                if chart_type == "difference_map":
                    axis.axhline(0, color="none")
            if chart_type == "difference_map" and len(panels) == 1:
                figure.axes[0].set_title(panels[0]["label"] + " (difference)")
            figure.suptitle(title)
            if images:
                figure.colorbar(images[0], ax=list(axes.flat), fraction=0.025, pad=0.02, label=panels[0]["colorbar_label"])
            figure.tight_layout()
            result = self._save(
                figure, "agent_chart_spec",
                {"chart_type": chart_type, "panels": [{"label": p["label"], "shape": list(p["values"].shape), "cmap": p["cmap"]} for p in panels]},
                title=title,
            )
            self._recorder("visualizer.render_spec", replay_arguments, result)
            return _finish(result)

        if chart_type == "trajectory":
            raw_tracks = spec.get("tracks", spec.get("series", []))
            if not isinstance(raw_tracks, list) or not 1 <= len(raw_tracks) <= 12:
                raise ValueError("tracks must contain between 1 and 12 items")
            figure, axis = plt.subplots(figsize=(7.2, 6.2))
            metadata = []
            for index, track in enumerate(raw_tracks):
                if not isinstance(track, dict):
                    raise ValueError("each trajectory must be an object")
                xs = [float(value) for value in track.get("x", [])]
                ys = [float(value) for value in track.get("y", [])]
                if len(xs) != len(ys) or not 1 <= len(xs) <= 72:
                    raise ValueError("trajectory x and y must have the same length between 1 and 72")
                label = str(track.get("label", f"track {index + 1}"))[:80]
                axis.plot(xs, ys, marker="o", markersize=3, linewidth=2, label=label)
                metadata.append({"label": label, "point_count": len(xs)})
            axis.set_title(title)
            axis.set_xlabel(str(spec.get("x_label", "Grid column / longitude")))
            axis.set_ylabel(str(spec.get("y_label", "Grid row / latitude")))
            if bool(spec.get("invert_y", True)):
                axis.invert_yaxis()
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
            figure.tight_layout()
            result = self._save(figure, "agent_chart_spec", {"chart_type": chart_type, "tracks": metadata}, title=title)
            self._recorder("visualizer.render_spec", replay_arguments, result)
            return _finish(result)

        series = spec.get("series", [])
        if not isinstance(series, list) or not 1 <= len(series) <= 12:
            raise ValueError("series must contain between 1 and 12 items")
        cleaned: list[dict[str, Any]] = []
        for item in series:
            if not isinstance(item, dict):
                raise ValueError("every series item must be an object")
            values = [float(value) for value in item.get("values", [])]
            if not 1 <= len(values) <= 72:
                raise ValueError("each series must contain between 1 and 72 values")
            cleaned_item = {"label": str(item.get("label", "series"))[:80], "values": values}
            for bound in ("lower", "upper"):
                if bound in item:
                    bound_values = [float(value) for value in item[bound]]
                    if len(bound_values) != len(values):
                        raise ValueError(f"{bound} must match series length")
                    cleaned_item[bound] = bound_values
            cleaned.append(cleaned_item)
        x_values = [float(value) for value in spec.get("x", list(range(len(cleaned[0]["values"]))))]
        if len(x_values) != len(cleaned[0]["values"]):
            raise ValueError("x length must match series length")
        if any(len(item["values"]) != len(x_values) for item in cleaned):
            raise ValueError("all series must have the same length")
        figure, axis = plt.subplots(figsize=(8.4, 4.8))
        if chart_type in {"line", "ensemble_range_line"}:
            for item in cleaned:
                axis.plot(x_values, item["values"], marker="o", markersize=3, linewidth=2, label=item["label"])
                if chart_type == "ensemble_range_line" and "lower" in item and "upper" in item:
                    lower = np.asarray(item["lower"], dtype=np.float32)
                    upper = np.asarray(item["upper"], dtype=np.float32)
                    if lower.size != len(x_values) or upper.size != len(x_values):
                        raise ValueError("ensemble range lower/upper must match x length")
                    axis.fill_between(x_values, lower, upper, alpha=0.18)
            if chart_type == "ensemble_range_line":
                lower = spec.get("lower")
                upper = spec.get("upper")
                if lower is not None and upper is not None:
                    if len(lower) != len(x_values) or len(upper) != len(x_values):
                        raise ValueError("ensemble range lower/upper must match x length")
                    axis.fill_between(x_values, lower, upper, color="#1976d2", alpha=0.16, label="ensemble range")
            axis.set_xlabel(str(spec.get("x_label", "Lead/time")))
            axis.legend(frameon=False)
        elif chart_type in {"bar", "grouped_bar"}:
            positions = np.arange(len(x_values), dtype=np.float32)
            width = 0.8 / max(1, len(cleaned))
            for index, item in enumerate(cleaned):
                axis.bar(positions + index * width, item["values"], width=width, label=item["label"])
            axis.set_xticks(positions + width * (len(cleaned) - 1) / 2, labels or [str(value) for value in x_values])
            axis.legend(frameon=False)
        else:
            axis.boxplot([item["values"] for item in cleaned], labels=[item["label"] for item in cleaned], showmeans=True)
        axis.set_title(title)
        axis.set_ylabel(str(spec.get("y_label", "Value")))
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        result = self._save(
            figure,
            "agent_chart_spec",
            {"chart_type": chart_type, "x": x_values, "series": cleaned, "labels": labels},
            title=title,
        )
        self._recorder("visualizer.render_spec", replay_arguments, result)
        return _finish(result)

    def time_series(
        self,
        time_minutes: list[int],
        series: list[dict[str, Any]],
        title: str = "Agent-selected time series",
        data_kind: str = "historical_observation",
    ) -> dict[str, Any]:
        """Plot observed/verified series without labelling them as forecasts."""
        kind = str(data_kind).strip().lower()
        kind = {"observed": "historical_observation", "observation": "historical_observation", "historical": "historical_observation", "truth": "ground_truth"}.get(kind, kind)
        if kind not in {"historical_observation", "ground_truth", "analysis"}:
            raise ValueError("data_kind must be historical_observation, ground_truth, or analysis")
        x = [int(value) for value in time_minutes]
        if not 1 <= len(x) <= 72:
            raise ValueError("time_minutes must contain between 1 and 72 values")
        if not 1 <= len(series) <= 6:
            raise ValueError("series must contain between 1 and 6 plotted fields")
        clean_series: list[dict[str, Any]] = []
        for index, item in enumerate(series):
            if not isinstance(item, dict):
                raise ValueError("each plotted series must be an object")
            label = str(item.get("label", f"series {index + 1}"))
            values = [float(value) for value in item["values"]]
            if len(values) != len(x):
                raise ValueError("every plotted series must match time_minutes length")
            clean_series.append({"label": label, "values": values})
        figure, axes = plt.subplots(
            len(clean_series), 1,
            figsize=(8.4, 2.35 * len(clean_series) + 1.0),
            sharex=True, squeeze=False,
        )
        colors = ["#1976d2", "#ef6c00", "#2e7d32", "#7b1fa2", "#c62828", "#00838f"]
        for index, item in enumerate(clean_series):
            axis = axes[index][0]
            axis.plot(x, item["values"], color=colors[index], linewidth=2, marker="o", markersize=3)
            axis.set_ylabel(item["label"])
            axis.grid(alpha=0.22)
        axes[0][0].set_title(str(title))
        axes[-1][0].set_xlabel("Relative observation time (minutes)")
        figure.tight_layout()
        result = self._save(
            figure,
            "agent_time_series",
            {"time_minutes": x, "series": clean_series, "data_kind": kind},
        )
        self._recorder(
            "visualizer.time_series",
            {
                "time_minutes": x,
                "series_labels": [item["label"] for item in clean_series],
                "data_kind": kind,
            },
            result,
        )
        return result

    def forecast_montage(
        self,
        frames: list[list[list[int]]],
        labels: list[str],
        title: str = "Agent-selected forecast snapshots",
    ) -> dict[str, Any]:
        """Render up to six agent-selected raw forecast frames as a PNG montage."""
        return self._render_forecast_montage(frames, labels, title, record_call=True)

    def _render_forecast_montage(
        self,
        frames: list[list[list[int]]],
        labels: list[str],
        title: str,
        *,
        record_call: bool,
    ) -> dict[str, Any]:
        """Render one batch, optionally exposing it as a public tool call."""
        if not 1 <= len(frames) <= 18:
            raise ValueError("frames must contain between 1 and 18 snapshots")
        if len(labels) != len(frames):
            raise ValueError("labels length must match frames length")
        arrays = [np.asarray(frame, dtype=np.float32) for frame in frames]
        if any(array.ndim != 2 for array in arrays):
            raise ValueError("each forecast snapshot must be a 2-D pixel matrix")
        render_hash = hashlib.sha256()
        render_hash.update("forecast_montage\0".encode("utf-8"))
        render_hash.update(json.dumps([str(label) for label in labels], ensure_ascii=True).encode("utf-8"))
        for array in arrays:
            render_hash.update(str(array.shape).encode("ascii"))
            render_hash.update(array.tobytes())
        render_key = render_hash.hexdigest()
        if render_key in self._render_cache:
            cached = dict(self._render_cache[render_key])
            if record_call:
                self._recorder(
                    "visualizer.forecast_montage",
                    {"labels": [str(label) for label in labels], "frame_count": len(arrays)},
                    cached,
                    cache_hit=True,
                )
            return cached
        payload = self._fast_raster_panels(
            [{"label": str(label), "values": array, "cmap": "cascast_vil"} for label, array in zip(labels, arrays)],
            "raster_map",
            title,
        )
        result = self._save_bytes(
            payload,
            "agent_forecast_montage",
            {"labels": [str(label) for label in labels], "frame_count": len(arrays), "renderer": "fast_raster"},
            title=title,
        )
        self._render_cache[render_key] = dict(result)
        if record_call:
            self._recorder(
                "visualizer.forecast_montage",
                {"labels": [str(label) for label in labels], "frame_count": len(arrays)},
                result,
            )
        return result

    def frame_montage(
        self,
        frames: list[list[list[int]]],
        labels: list[str],
        title: str = "Agent-selected observation snapshots",
        data_kind: str = "historical_observation",
    ) -> dict[str, Any]:
        """Render observed/verified 2-D frames without forecast semantics."""
        kind = str(data_kind).strip().lower()
        kind = {"observed": "historical_observation", "observation": "historical_observation", "historical": "historical_observation", "truth": "ground_truth"}.get(kind, kind)
        if kind not in {"historical_observation", "ground_truth", "analysis"}:
            raise ValueError("data_kind must be historical_observation, ground_truth, or analysis")
        if not 1 <= len(frames) <= 6:
            raise ValueError("frames must contain between 1 and 6 snapshots")
        if len(labels) != len(frames):
            raise ValueError("labels length must match frames length")
        arrays = [np.asarray(frame, dtype=np.float32) for frame in frames]
        if any(array.ndim != 2 for array in arrays):
            raise ValueError("each snapshot must be a 2-D pixel matrix")
        columns = min(3, len(arrays))
        rows = (len(arrays) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.8 * rows), squeeze=False)
        image = None
        for index, axis in enumerate(axes.flat):
            if index >= len(arrays):
                axis.axis("off")
                continue
            cmap, norm = cascast_vil_cmap()
            image = axis.imshow(mask_missing_vil(arrays[index]), cmap=cmap, norm=norm, origin="upper")
            axis.set_title(str(labels[index]))
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(str(title))
        if image is not None:
            figure.colorbar(image, ax=list(axes.flat), fraction=0.025, pad=0.02, label="Encoded VIL (0-255)")
        result = self._save(
            figure,
            "agent_frame_montage",
            {
                "labels": [str(label) for label in labels],
                "frame_count": len(arrays),
                "data_kind": kind,
            },
        )
        self._recorder(
            "visualizer.frame_montage",
            {
                "labels": [str(label) for label in labels],
                "frame_count": len(arrays),
                "data_kind": kind,
            },
            result,
        )
        return result

    def forecast_montage_sequence(
        self,
        frames: list[list[list[int]]],
        labels: list[str],
        title: str = "Agent-selected complete forecast sequence",
    ) -> dict[str, Any]:
        """Render every requested forecast frame in continuous batches.

        A single PNG montage is intentionally limited to six panels for
        readability. This method never samples or drops frames: it splits the
        complete input into consecutive batches and returns every artifact.
        """
        if not 1 <= len(frames) <= 72:
            raise ValueError("frames must contain between 1 and 72 snapshots")
        if len(labels) != len(frames):
            raise ValueError("labels length must match frames length")
        artifacts: list[dict[str, Any]] = []
        batch_size = 6
        for start in range(0, len(frames), batch_size):
            end = min(start + batch_size, len(frames))
            batch_title = f"{title} ({start + 1}-{end}/{len(frames)})"
            artifacts.append(
                self._render_forecast_montage(
                    frames[start:end],
                    labels[start:end],
                    batch_title,
                    record_call=False,
                )
            )
        result = {
            "status": "ok",
            "task": "sevir_chart_sequence",
            "chart_type": "agent_forecast_montage_sequence",
            "artifact_count": len(artifacts),
            "frame_count": len(frames),
            "labels": [str(label) for label in labels],
            "artifacts": artifacts,
            "complete_sequence": True,
            "batch_size": batch_size,
            "interpretation_limit": "All requested frames are rendered in order; no temporal sampling was applied.",
        }
        self._recorder(
            "visualizer.forecast_montage_sequence",
            {"labels": [str(label) for label in labels], "frame_count": len(frames), "batch_size": batch_size},
            result,
        )
        return result
