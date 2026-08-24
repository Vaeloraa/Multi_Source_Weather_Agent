"""Run a four-step Aurora rollout on the aligned 48x96 ERA5 case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from aurora import AuroraPretrained, Batch, Metadata
from aurora.rollout import rollout


def _prepare(values: np.ndarray, index: int, flip_lat: bool) -> torch.Tensor:
    selected = values[[index - 6, index]]
    if flip_lat:
        selected = selected[..., ::-1, :]
    return torch.from_numpy(np.ascontiguousarray(selected[None]))


def _static(values: np.ndarray, flip_lat: bool) -> torch.Tensor:
    array = np.asarray(values)
    while array.ndim > 2:
        array = array[0]
    if flip_lat:
        array = array[::-1, :]
    return torch.from_numpy(np.ascontiguousarray(array))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--pressure", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=18)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 4:
        raise ValueError("steps must be between 1 and 4")

    with xr.open_dataset(args.surface) as surface, xr.open_dataset(args.pressure) as atmos, xr.open_dataset(args.static) as static:
        height = int(surface.sizes["latitude"])
        width = int(surface.sizes["longitude"])
        if height % 4 or width % 4:
            raise ValueError(f"Aurora input grid must be divisible by patch size 4; got {height}x{width}.")
        i = int(args.index)
        if i < 6 or i + args.steps * 6 >= surface.sizes["valid_time"]:
            raise ValueError("input index and steps exceed the available ERA5 validation window")
        lat = surface.latitude.values
        lon = surface.longitude.values
        flip_lat = bool(lat.size > 1 and float(lat[0]) < float(lat[-1]))
        levels = tuple(int(v) for v in atmos.pressure_level.values)
        batch = Batch(
            surf_vars={
                "2t": _prepare(surface.t2m.values, i, flip_lat),
                "10u": _prepare(surface.u10.values, i, flip_lat),
                "10v": _prepare(surface.v10.values, i, flip_lat),
                "msl": _prepare(surface.msl.values, i, flip_lat),
            },
            static_vars={"z": _static(static.z.values, flip_lat), "slt": _static(static.slt.values, flip_lat), "lsm": _static(static.lsm.values, flip_lat)},
            atmos_vars={
                "t": _prepare(atmos.t.values, i, flip_lat),
                "u": _prepare(atmos.u.values, i, flip_lat),
                "v": _prepare(atmos.v.values, i, flip_lat),
                "q": _prepare(atmos.q.values, i, flip_lat),
                "z": _prepare(atmos.z.values, i, flip_lat),
            },
            metadata=Metadata(
                lat=torch.from_numpy(np.ascontiguousarray(lat[::-1] if flip_lat else lat)),
                lon=torch.from_numpy(np.ascontiguousarray(lon % 360.0)),
                time=(surface.valid_time.values.astype("datetime64[s]").tolist()[i],),
                atmos_levels=levels,
            ),
        )
        model = AuroraPretrained()
        model.load_checkpoint()
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictions = []
        with torch.inference_mode():
            for prediction in rollout(model, batch.to(device), steps=args.steps):
                predictions.append({
                    "surf": {name: value.detach().cpu().numpy() for name, value in prediction.surf_vars.items()},
                    "atmos": {name: value.detach().cpu().numpy() for name, value in prediction.atmos_vars.items()},
                })
        if len(predictions) != args.steps:
            raise RuntimeError(f"Aurora returned {len(predictions)} steps, expected {args.steps}")
        arrays: dict[str, np.ndarray] = {}
        for name in ("2t", "10u", "10v", "msl"):
            arrays[name] = np.concatenate([p["surf"][name] for p in predictions], axis=1)
        for name in ("t", "u", "v", "q", "z"):
            arrays[f"atmos_{name}"] = np.concatenate([p["atmos"][name] for p in predictions], axis=1)
        lead_hours = np.arange(1, args.steps + 1, dtype=np.int32) * 6
        valid_times = [str(surface.valid_time.values[i])[:19] for _ in range(1)]
        output = {**arrays, "lead_hours": lead_hours}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **output)
        summary = {
            "status": "ok",
            "model": "AuroraPretrained",
            "input_latest_time": str(surface.valid_time.values[i]),
            "lead_hours": lead_hours.tolist(),
            "forecast_times": [str(surface.valid_time.values[i + int(hours)]) for hours in lead_hours],
            "device": str(device),
            "input_grid_shape": [height, width],
            "pressure_levels": list(levels),
            "output": str(args.output),
            "surface_output_shapes": {name: list(value.shape) for name, value in arrays.items() if not name.startswith("atmos_")},
            "atmos_output_shapes": {name: list(value.shape) for name, value in arrays.items() if name.startswith("atmos_")},
            "interpretation_limit": "Large-scale environmental forecast; not a local VIL forecast.",
        }
    args.output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
