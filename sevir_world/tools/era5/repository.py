"""Retrieve request-bound ERA5 fields without choosing an analysis strategy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import h5py
import numpy as np
import xarray as xr

from .backend import fetch_era5_live, normalize_live_request
from .cds import data_paths


# Common file aliases for compatibility/documentation, not supported-variable
# allowlists. Other names are resolved from the retrieved source file.
SURFACE_FIELDS: dict[str, tuple[str, str]] = {
    "2m_temperature": ("t2m", "K"),
    "10m_u_component_of_wind": ("u10", "m s-1"),
    "10m_v_component_of_wind": ("v10", "m s-1"),
    "mean_sea_level_pressure": ("msl", "Pa"),
    "total_precipitation": ("tp", "m"),
    "2m_dewpoint_temperature": ("d2m", "K"),
}
PRESSURE_FIELDS: dict[str, tuple[str, str]] = {
    "temperature": ("t", "K"),
    "u_component_of_wind": ("u", "m s-1"),
    "v_component_of_wind": ("v", "m s-1"),
    "specific_humidity": ("q", "kg kg-1"),
    "geopotential": ("z", "m2 s-2"),
}


def _timestamp(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _selected(values: list[str], supported: dict[str, tuple[str, str]], name: str) -> list[str]:
    # ``supported`` is a mapping of common NetCDF aliases, never an allowlist.
    if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"ERA5 {name} must be a list of non-empty CDS variable names")
    selected = [value.strip() for value in values]
    return list(dict.fromkeys(selected))


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _storage_name(source: h5py.File, public_name: str, aliases: dict[str, tuple[str, str]], *, allow_single: bool = False) -> str | None:
    candidates = [public_name, aliases.get(public_name, ("", ""))[0]]
    for candidate in candidates:
        if candidate in source:
            return candidate
    data_names = []
    for name, variable in source.items():
        if not isinstance(variable, h5py.Dataset) or variable.ndim < 3:
            continue
        data_names.append(name)
        for key in ("standard_name", "long_name", "GRIB_cfVarName", "GRIB_shortName"):
            label = _text(variable.attrs.get(key, "")).lower().replace(" ", "_")
            if label == public_name.lower():
                return name
    # Acquisition isolates variables, so source-specific short names need no
    # hardcoded mapping. Never guess among multiple meteorological variables.
    return data_names[0] if allow_single and len(data_names) == 1 else None


def _values(variable: h5py.Dataset, position: Any) -> np.ndarray:
    values = np.asarray(variable[position], dtype=np.float32)
    for key in ("_FillValue", "missing_value"):
        if key in variable.attrs:
            values = np.where(values == variable.attrs[key], np.nan, values)
    return values * variable.attrs.get("scale_factor", 1) + variable.attrs.get("add_offset", 0)


class VariableMappingError(RuntimeError):
    """A batch source uses short names that cannot be mapped unambiguously."""

    def __init__(self, names: list[str]):
        self.names = names
        super().__init__("ERA5 source variable names could not be resolved: " + ", ".join(names))


def _read_group(paths: str | list[str], names: list[str], aliases: dict[str, tuple[str, str]],
                expected_times: list[int], levels: list[int], file_variables: dict[str, str | list[str]]) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    samples: dict[str, dict[int, np.ndarray]] = {name: {} for name in names}
    units: dict[str, str] = {}
    latitude = longitude = None
    expanded = [(path, file_variables.get(origin)) for origin in ([paths] if isinstance(paths, str) else paths) for path in data_paths([origin])]
    resolved = set()
    for path, selected in expanded:
        with h5py.File(path, "r") as source:
            lat = np.asarray(source["latitude"], dtype=np.float32)
            lon = np.asarray(source["longitude"], dtype=np.float32)
            if latitude is not None and (not np.array_equal(latitude, lat) or not np.array_equal(longitude, lon)):
                raise RuntimeError("ERA5 variables use different grids; use era5.fields for each native grid")
            latitude, longitude = lat, lon
            time_axis = source["valid_time"] if "valid_time" in source else source["time"]
            timestamps = np.asarray(time_axis, dtype=np.int64)
            time_units = _text(time_axis.attrs.get("units", ""))
            if "since" in time_units:
                decoded = xr.coding.times.decode_cf_datetime(timestamps, time_units)
                timestamps = np.asarray(decoded, dtype="datetime64[s]").astype(np.int64)
            index = {int(value): position for position, value in enumerate(timestamps)}
            level_positions = []
            if levels:
                axis = source["pressure_level"] if "pressure_level" in source else source["level"]
                stored = [int(value) for value in np.asarray(axis).tolist()]
                if set(levels) - set(stored):
                    raise RuntimeError(f"ERA5 pressure file is missing levels: {sorted(set(levels) - set(stored))}")
                level_positions = [stored.index(value) for value in levels]
            selected_names = [selected] if isinstance(selected, str) else selected or names
            # Only a single-variable request permits a sole-field fallback.
            # Never guess among the variables of a batch response.
            identified = {name: _storage_name(source, name, aliases, allow_single=len(selected_names) == 1) for name in selected_names}
            for name, storage in identified.items():
                if storage is None:
                    continue
                resolved.add(name)
                variable = source[storage]
                units[name] = _text(variable.attrs.get("units", aliases.get(name, ("", ""))[1]))
                for stamp in expected_times:
                    if stamp in index:
                        position = index[stamp]
                        samples[name][stamp] = np.stack([_values(variable, (position, level)) for level in level_positions]) if levels else _values(variable, position)
    unresolved = [name for name in names if name not in resolved]
    if unresolved:
        raise VariableMappingError(unresolved)
    for name, values in samples.items():
        if set(expected_times) - values.keys():
            raise RuntimeError(f"ERA5 variable {name} does not cover the requested time window")
    return {name: np.stack([values[stamp] for stamp in expected_times]) for name, values in samples.items()}, units, latitude, longitude


def regional_era5_fields(
    reference_time_utc: str,
    latitude: float | None,
    longitude: float | None,
    region: dict[str, float] | None,
    history_before_hours: int,
    history_after_hours: int,
    data_policy: str,
    surface_variables: list[str],
    pressure_variables: list[str],
    pressure_levels: list[int],
) -> dict[str, Any]:
    """Return only the ERA5 variables and levels explicitly selected by the Agent."""
    surface_names = _selected(surface_variables, SURFACE_FIELDS, "surface variables")
    pressure_names = _selected(pressure_variables, PRESSURE_FIELDS, "pressure variables")
    levels = sorted({int(value) for value in pressure_levels}, reverse=True)
    if not surface_names and not pressure_names:
        raise ValueError("select at least one ERA5 variable")
    if pressure_names and not levels:
        raise ValueError("pressure_levels are required for pressure variables")
    if set(surface_names) & set(pressure_names):
        raise ValueError("variables shared by surface and pressure groups require separate calls to era5.fields")

    request = normalize_live_request(
        analysis_time_utc=reference_time_utc,
        latitude=latitude,
        longitude=longitude,
        region=region,
        variables=surface_names,
        pressure_variables=pressure_names,
        pressure_levels=levels,
        forecast_leads=[],
        history_before_hours=history_before_hours,
        history_after_hours=history_after_hours,
        event_id="",
        model_version="era5-data-selection-v1",
        model_contract="data_selection",
    )
    request.update({
        "surface_variables": surface_names,
        "pressure_variables": pressure_names,
        "pressure_levels_hpa": levels,
        "aurora_pressure_levels_hpa": levels,
        "access_mode": (
            "cache_only" if str(data_policy).strip().lower() == "cached" else "cache_then_remote"
        ),
    })
    fetched = fetch_era5_live(request, cache_only=request["access_mode"] == "cache_only")
    if fetched.get("status") != "ok":
        return {
            "status": "era5_unavailable",
            "reason": fetched.get("reason") or fetched.get("error_code"),
            "era5_temporality": fetched.get("era5_temporality") or request.get("era5_temporality"),
        }
    temporality = dict(request.get("era5_temporality") or {})
    if temporality.get("availability_status") == "not_yet_available":
        # A successful source response overrides our conservative lag estimate;
        # it does not prove which revision (ERA5 or ERA5T) the source supplied.
        temporality.update(product="ERA5/ERA5T", data_class="reanalysis", availability_status="available",
                           provisional=None, classification_basis="successful_source_retrieval",
                           label_zh="ERA5/ERA5T reanalysis returned by the data source")

    files = fetched.get("files") if isinstance(fetched.get("files"), dict) else {}
    base = _timestamp(str(request["era5_time_utc"]))
    offsets = list(range(-int(history_before_hours), int(history_after_hours) + 1))
    expected_times = [base + offset * 3600 for offset in offsets]
    fields: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None
    valid_labels = [
        datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        for value in expected_times
    ]

    available_levels: list[int] = []
    for group, names, aliases, selected_levels in (
        ("surface", surface_names, SURFACE_FIELDS, []),
        ("pressure", pressure_names, PRESSURE_FIELDS, levels),
    ):
        if not names:
            continue
        if not files.get(group):
            return {"status": "era5_unavailable", "reason": f"era5_{group}_file_missing"}
        group_files = [files[group]] if isinstance(files[group], str) else list(files[group])
        file_variables = dict(fetched.get("file_variables", {}).get(group, {}))
        try:
            values, source_units, lat, lon = _read_group(
                group_files, names, aliases, expected_times, selected_levels, file_variables,
            )
        except VariableMappingError as error:
            # Rare source-specific names need isolated retrieval to establish
            # their identity. All fields already resolved from the batch stay.
            for name in error.names:
                single_request = normalize_live_request(
                    analysis_time_utc=reference_time_utc, latitude=latitude, longitude=longitude, region=region,
                    variables=[name] if group == "surface" else [],
                    pressure_variables=[name] if group == "pressure" else [],
                    pressure_levels=selected_levels, forecast_leads=[],
                    history_before_hours=history_before_hours, history_after_hours=history_after_hours,
                    model_version="era5-data-selection-v1", model_contract="data_selection",
                )
                single = fetch_era5_live(single_request, cache_only=request["access_mode"] == "cache_only")
                if single.get("status") != "ok":
                    raise RuntimeError(f"ERA5 variable {name} unavailable: {single.get('reason') or single.get('error_code')}")
                new_files = single["files"][group]
                new_files = [new_files] if isinstance(new_files, str) else new_files
                group_files.extend(new_files)
                file_variables.update({path: [name] for path in new_files})
                fetched["cache_hit"] = bool(fetched.get("cache_hit")) and bool(single.get("cache_hit"))
            files[group] = group_files
            values, source_units, lat, lon = _read_group(
                group_files, names, aliases, expected_times, selected_levels, file_variables,
            )
        if latitudes is not None and (not np.array_equal(latitudes, lat) or not np.array_equal(longitudes, lon)):
            raise RuntimeError("ERA5 surface and pressure grids do not match; use era5.fields for separate grids")
        latitudes, longitudes = lat, lon
        fields.update(values)
        units.update(source_units)
        if group == "pressure":
            available_levels = levels

    return {
        "status": "ok",
        "offset_hours": offsets,
        "valid_time_utc": valid_labels,
        "latitude": latitudes,
        "longitude": longitudes,
        "pressure_levels": available_levels,
        "fields": fields,
        "units": units,
        "surface_variables": surface_names,
        "pressure_variables": pressure_names,
        "cache_hit": bool(fetched.get("cache_hit", False)),
        "reference_time_utc": str(request["era5_time_utc"]),
        "era5_temporality": temporality,
        "access_mode": str(request.get("access_mode") or "cache_then_remote"),
        "provenance": {"source": "CDS ERA5 reanalysis", "files": files},
    }
