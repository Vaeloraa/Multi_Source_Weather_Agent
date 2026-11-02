from __future__ import annotations

import io
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .styles import colorize


ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT_ROOT = ROOT / "data" / "artifacts"
_FALLBACK_ARTIFACT_ROOT = Path(tempfile.gettempdir()) / "weather_agent_artifacts"


def _artifact_root() -> Path:
    candidate = Path(os.getenv("WEATHER_AGENT_ARTIFACT_DIR", str(_FALLBACK_ARTIFACT_ROOT)))
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.write_bytes(b"ok")
        probe.unlink(missing_ok=True)
        return candidate
    except OSError:
        _FALLBACK_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        return _FALLBACK_ARTIFACT_ROOT


ARTIFACT_ROOT = _artifact_root()
os.environ["WEATHER_AGENT_ARTIFACT_DIR"] = str(ARTIFACT_ROOT)


def artifact_read_roots() -> tuple[Path, ...]:
    return tuple(dict.fromkeys((
        ARTIFACT_ROOT, _DEFAULT_ARTIFACT_ROOT, _FALLBACK_ARTIFACT_ROOT,
    )))


def figure_path(filename: str | Path | None = None) -> Path:
    """Resolve every generated figure into the managed artifact directory."""
    root = Path(os.environ["WEATHER_AGENT_ARTIFACT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    name = Path(filename).name if filename else f"{uuid.uuid4().hex}.png"
    if not name.lower().endswith(".png"):
        name += ".png"
    return root / name


def plot_field(data: Any, title: str = "", cmap: str = "viridis", filename: str | None = None) -> str:
    """Plot a numeric 2-D field, or the last frame of a 3-D cube, as a managed PNG."""
    import matplotlib.pyplot as plt
    from .styles import matplotlib_style, mask_missing

    array = np.asarray(data)
    if array.dtype.kind not in "fiu":
        array = array.astype(np.float32)
    if array.ndim == 3:
        array = array[-1]
    if array.ndim != 2:
        raise ValueError("plot_field expects a 2-D field or a 3-D cube")
    if array.size == 0 or not np.isfinite(array).any():
        raise ValueError("plot_field received an empty or all-missing field")
    figure, axis = plt.subplots(figsize=(6, 5))
    try:
        if cmap == "encoded_vil":
            colors, norm = matplotlib_style(cmap)
            missing = getattr(data, "attrs", {}).get("missing_value")
            image = axis.imshow(mask_missing(array, missing), cmap=colors, norm=norm, origin="upper")
        else:
            image = axis.imshow(array, cmap=cmap, origin="upper")
        figure.colorbar(image, ax=axis)
        if title:
            axis.set_title(title)
        path = figure_path(filename)
        figure.savefig(path, bbox_inches="tight")
        return str(path)
    finally:
        plt.close(figure)


def session_artifacts(value: Any) -> list[dict[str, Any]]:
    """Return detached artifact records that are valid for session replay."""

    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def render_png(
    frame: np.ndarray,
    *,
    variable_metadata: dict[str, Any] | None = None,
    style: str | None = None,
) -> bytes:
    """Render any 2-D field using explicit metadata rather than source checks."""
    metadata = variable_metadata or {}
    output = io.BytesIO()
    Image.fromarray(
        colorize(
            frame,
            style=style,
            missing_value=metadata.get("missing_value"),
        ),
        mode="RGB",
    ).save(
        output, format="PNG", optimize=True
    )
    return output.getvalue()
