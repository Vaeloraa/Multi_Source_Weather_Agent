"""Standalone Aurora worker used by the Aurora cache-miss path.

This file intentionally avoids package-relative imports because it is launched
by the separately managed Python environment that owns Aurora and PyTorch.
"""
from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
from typing import Any

def _prepare(
    values: Any, history_index: int, analysis_index: int, flip_lat: bool
) -> Any:
    import numpy as np
    import torch

    selected = values[[history_index, analysis_index]]
    if flip_lat:
        selected = selected[..., ::-1, :]
    return torch.from_numpy(np.ascontiguousarray(selected[None]))


def _static(values: Any, flip_lat: bool) -> Any:
    import numpy as np
    import torch

    array = np.asarray(values)
    while array.ndim > 2:
        array = array[0]
    if flip_lat:
        array = array[::-1, :]
    return torch.from_numpy(np.ascontiguousarray(array))


def _package_version() -> str:
    try:
        return importlib_metadata.version("microsoft-aurora")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--pressure", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=6)
    parser.add_argument("--analysis-time", default="")
    parser.add_argument("--leads", default="6")
    parser.add_argument("--cache-request-key", default="")
    parser.add_argument("--model-version", default="")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    # Parse arguments before importing the separately managed scientific
    # stack.  Besides making ``--help`` cheap, this prevents the web
    # service's vendored NumPy from being imported before the worker has
    # entered its isolated Aurora environment.
    import numpy as np
    import torch
    import xarray as xr
    from aurora import AuroraPretrained, Batch, Metadata, rollout

    if not str(args.model_version or "").strip():
        raise ValueError("model-version is required for versioned Aurora output")

    with (
        xr.open_dataset(args.surface) as surface,
        xr.open_dataset(args.pressure) as atmos,
        xr.open_dataset(args.static) as static,
    ):
        height = int(surface.sizes["latitude"])
        width = int(surface.sizes["longitude"])
        if height % 4 or width % 4:
            raise ValueError(
                "Aurora input grid must be divisible by patch size 4; "
                f"got {height}x{width}"
            )
        valid_times = surface.valid_time.values.astype("datetime64[s]")
        if args.analysis_time:
            target_time = np.datetime64(str(args.analysis_time).replace("Z", ""), "s")
            exact = np.nonzero(valid_times == target_time)[0]
            if exact.size != 1:
                raise ValueError(
                    "Aurora analysis time must exactly match one ERA5 valid_time"
                )
            index = int(exact[0])
        else:
            index = int(args.index)
            if index < 0 or index >= surface.sizes["valid_time"]:
                raise ValueError("index is outside the ERA5 time dimension")
            target_time = valid_times[index]
        history_time = target_time - np.timedelta64(6, "h")
        history_matches = np.nonzero(valid_times == history_time)[0]
        if history_matches.size != 1:
            raise ValueError(
                "ERA5 input must contain the exact analysis time minus six hours"
            )
        history_index = int(history_matches[0])

        pressure_times = atmos.valid_time.values.astype("datetime64[s]")
        pressure_analysis_matches = np.nonzero(pressure_times == target_time)[0]
        pressure_history_matches = np.nonzero(pressure_times == history_time)[0]
        if (
            pressure_analysis_matches.size != 1
            or pressure_history_matches.size != 1
        ):
            raise ValueError("surface and pressure ERA5 times are not aligned")
        pressure_index = int(pressure_analysis_matches[0])
        pressure_history_index = int(pressure_history_matches[0])

        latitude = surface.latitude.values
        longitude = surface.longitude.values
        if (
            not np.allclose(atmos.latitude.values, latitude)
            or not np.allclose(atmos.longitude.values, longitude)
            or not np.allclose(static.latitude.values, latitude)
            or not np.allclose(static.longitude.values, longitude)
        ):
            raise ValueError("surface, pressure and static ERA5 grids are not aligned")
        flip_lat = bool(
            latitude.size > 1 and float(latitude[0]) < float(latitude[-1])
        )
        levels = tuple(int(value) for value in atmos.pressure_level.values)
        batch = Batch(
            surf_vars={
                "2t": _prepare(surface.t2m.values, history_index, index, flip_lat),
                "10u": _prepare(surface.u10.values, history_index, index, flip_lat),
                "10v": _prepare(surface.v10.values, history_index, index, flip_lat),
                "msl": _prepare(surface.msl.values, history_index, index, flip_lat),
            },
            static_vars={
                "z": _static(static.z.values, flip_lat),
                "slt": _static(static.slt.values, flip_lat),
                "lsm": _static(static.lsm.values, flip_lat),
            },
            atmos_vars={
                "t": _prepare(atmos.t.values, pressure_history_index, pressure_index, flip_lat),
                "u": _prepare(atmos.u.values, pressure_history_index, pressure_index, flip_lat),
                "v": _prepare(atmos.v.values, pressure_history_index, pressure_index, flip_lat),
                "q": _prepare(atmos.q.values, pressure_history_index, pressure_index, flip_lat),
                "z": _prepare(atmos.z.values, pressure_history_index, pressure_index, flip_lat),
            },
            metadata=Metadata(
                lat=torch.from_numpy(
                    np.ascontiguousarray(latitude[::-1] if flip_lat else latitude)
                ),
                lon=torch.from_numpy(np.ascontiguousarray(longitude % 360.0)),
                time=(
                    surface.valid_time.values.astype("datetime64[s]").tolist()[index],
                ),
                atmos_levels=levels,
            ),
        )

        requested_leads = sorted(
            {int(value) for value in str(args.leads).split(",") if value.strip()}
        )
        if not requested_leads or any(
            value <= 0 or value % 6 for value in requested_leads
        ):
            raise ValueError("Aurora leads must be positive multiples of six hours")

        model = AuroraPretrained()
        model.load_checkpoint()
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        arrays: dict[str, np.ndarray] = {}
        requested_set = set(requested_leads)
        with torch.inference_mode():
            for step, prediction in enumerate(
                rollout(model, batch, steps=max(requested_leads) // 6), start=1
            ):
                lead = step * 6
                if lead not in requested_set:
                    continue
                surface_arrays = {
                    name: value.detach().cpu().numpy()
                    for name, value in prediction.surf_vars.items()
                }
                atmos_arrays = {
                    name: value.detach().cpu().numpy()
                    for name, value in prediction.atmos_vars.items()
                }
                for name, value in surface_arrays.items():
                    arrays[f"lead_{lead:03d}_{name}"] = value
                for name, value in atmos_arrays.items():
                    arrays[f"lead_{lead:03d}_atmos_{name}"] = value

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
        summary = {
            "status": "ok",
            "model": "AuroraPretrained",
            "model_version": args.model_version,
            "aurora_package_version": _package_version(),
            "cache_request_key": args.cache_request_key,
            "cache_identity_status": "versioned",
            "event_input_time": str(surface.valid_time.values[index]),
            "device": str(device),
            "pressure_levels": list(levels),
            "input_grid_shape": [height, width],
            "output": str(args.output),
            "completed_leads_hours": requested_leads,
            "surface_output_shapes": {
                name: list(value.shape)
                for name, value in arrays.items()
                if "atmos_" not in name
            },
            "atmos_output_shapes": {
                name: list(value.shape)
                for name, value in arrays.items()
                if "atmos_" in name
            },
            "interpretation_limit": (
                "This is a large-scale environmental forecast; it is not a "
                "local VIL forecast."
            ),
        }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
