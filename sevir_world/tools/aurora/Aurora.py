"""Public Aurora tool for request-bound regional environment fields."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np
import xarray as xr

from ...capabilities.base import BoundCapability
from ...capabilities.contracts import ParameterSpec, ProductMetadata, tool_parameters
from .._shared import require_utc_text
from ..geospatial.contracts import BoundingBox, GeoPoint


_SURFACE_VARIABLES = {
    "t2m_c": "temperature_2m_c",
    "u10_m_s": "u_wind_10m_m_s",
    "v10_m_s": "v_wind_10m_m_s",
    "mslp_hpa": "mean_sea_level_pressure_hpa",
}
_PRESSURE_VARIABLES = {
    "temperature_c": "air_temperature_c",
    "u_m_s": "u_wind_m_s",
    "v_m_s": "v_wind_m_s",
    "specific_humidity_g_kg": "specific_humidity_g_kg",
    "geopotential_height_m": "geopotential_height_m",
}
_PUBLIC_SURFACE_VARIABLES = frozenset(_SURFACE_VARIABLES.values())
_PUBLIC_PRESSURE_VARIABLES = frozenset(_PRESSURE_VARIABLES.values())


def _valid_times(initialization_time_utc: str, lead_hours: Sequence[int]) -> tuple[str, ...]:
    """Build explicit UTC valid-time labels for Aurora leads.

    Parameters
    ----------
    initialization_time_utc : str
        Forecast initialization time.
    lead_hours : sequence of int
        Relative forecast leads.
    Returns
    -------
    tuple of str
        ISO-8601 UTC labels.
    """

    parsed = datetime.fromisoformat(initialization_time_utc.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return tuple(
        (parsed.astimezone(timezone.utc) + timedelta(hours=int(lead))).isoformat().replace("+00:00", "Z")
        for lead in lead_hours
    )


def _environment_dataset(raw: dict[str, Any], requested_leads: tuple[int, ...], initialization: str) -> xr.Dataset:
    """Convert ERA5 and Aurora raw mappings into one standard Dataset.

    Parameters
    ----------
    raw : dict
        Backend ERA5/Aurora payload.
    selected_leads : tuple of int
        Positive Aurora leads to include.
    initialization : str
        Bound Aurora initialization time.
    Returns
    -------
    xarray.Dataset
        Combined analysis and forecast fields.
    """

    latitude = np.asarray(raw["latitude"], dtype=np.float32)
    longitude = np.asarray(raw["longitude"], dtype=np.float32)
    available_fields = {
        int(key): value
        for key, value in (raw.get("aurora", {}).get("fields", {}) or {}).items()
    }
    states: dict[int, dict[str, Any]] = {0: dict(raw.get("era5") or {})}
    states.update({lead: dict(available_fields.get(lead) or {}) for lead in requested_leads})
    leads = (0, *requested_leads)
    coords: dict[str, Any] = {
        "lead_hour": np.asarray(leads, dtype=np.int32),
        "valid_time_utc": ("lead_hour", np.asarray(_valid_times(initialization, leads), dtype=str)),
        "source": ("lead_hour", np.asarray([
            "ERA5" if lead == 0 else "Aurora" if lead in available_fields else "unavailable"
            for lead in leads
        ], dtype=str)),
        "latitude": latitude,
        "longitude": longitude,
    }
    pressure_levels = sorted({int(level) for state in states.values() for level in (state.get("pressure_levels") or {})})
    if pressure_levels:
        coords["pressure_level_hpa"] = np.asarray(pressure_levels, dtype=np.int32)
    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for source, name in _SURFACE_VARIABLES.items():
        values = [np.asarray((states[lead].get("surface") or {}).get(source, np.full((latitude.size, longitude.size), np.nan)), dtype=np.float32) for lead in leads]
        variables[name] = (("lead_hour", "latitude", "longitude"), np.stack(values))
    for source, name in (_PRESSURE_VARIABLES.items() if pressure_levels else ()):
        values = []
        for lead in leads:
            by_level = states[lead].get("pressure_levels") or {}
            values.append(np.stack([
                np.asarray((by_level.get(str(level)) or {}).get(source, np.full((latitude.size, longitude.size), np.nan)), dtype=np.float32)
                for level in pressure_levels
            ]))
        if pressure_levels:
            variables[name] = (("lead_hour", "pressure_level_hpa", "latitude", "longitude"), np.stack(values))
    dataset = xr.Dataset(data_vars=variables, coords=coords, attrs={
        "forecast_id": str(raw.get("forecast_id") or ""),
        "initialization_time_utc": initialization,
        "cache_hit": bool(raw.get("cache_hit", False)),
        "source": "ERA5 and Aurora",
        "requested_region": raw.get("requested_region"),
        "effective_region": raw.get("effective_region"),
        "requested_leads_hours": requested_leads,
        "available_leads_hours": tuple(sorted(available_fields)),
        "missing_leads_hours": tuple(lead for lead in requested_leads if lead not in available_fields),
    })
    # Public regional coordinates are monotonic increasing so the natural
    # ``sel(latitude=slice(south, north))`` form generated by users and models
    # cannot silently produce an empty array when the source ERA5 latitude is
    # stored north-to-south.
    dataset = dataset.sortby("latitude").sortby("longitude")
    dataset.attrs["coordinate_order"] = "latitude_ascending_longitude_ascending"
    sorted_latitude = np.asarray(dataset.latitude.values, dtype=np.float64)
    sorted_longitude = np.asarray(dataset.longitude.values, dtype=np.float64)
    dataset.attrs["spatial_metadata"] = {
        "grid_type": "latlon_grid", "crs": "EPSG:4326",
        "shape": [int(sorted_latitude.size), int(sorted_longitude.size)],
        "latitude": sorted_latitude.tolist(), "longitude": sorted_longitude.tolist(),
        "bounds": {
            "south": float(sorted_latitude.min()), "west": float(sorted_longitude.min()),
            "north": float(sorted_latitude.max()), "east": float(sorted_longitude.max()),
        },
        "axis_order": ["latitude", "longitude"],
    }
    # These units follow the explicit conversions in Environment.regional_environment_fields.
    units = {
        "temperature_2m_c": "degC", "air_temperature_c": "degC",
        "u_wind_10m_m_s": "m s-1", "v_wind_10m_m_s": "m s-1",
        "u_wind_m_s": "m s-1", "v_wind_m_s": "m s-1",
        "mean_sea_level_pressure_hpa": "hPa", "specific_humidity_g_kg": "g kg-1",
        "geopotential_height_m": "m",
    }
    segments = tuple({
        "coordinate": "lead_hour", "value": lead,
        "valid_time_utc": _valid_times(initialization, (lead,))[0],
        "source": "ERA5" if lead == 0 else "Aurora" if lead in available_fields else "unavailable",
        "role": "reanalysis" if lead == 0 else "forecast" if lead in available_fields else "unavailable",
    } for lead in leads)
    dataset.coords["product_role"] = ("lead_hour", [segment["role"] for segment in segments])
    for name in dataset.data_vars:
        dataset[name].attrs.update(units=units[name], product_role="mixed")
    dataset.attrs.update(model_id="aurora", model_version=str(raw.get("model_version") or ""))
    dataset.attrs["product_metadata"] = ProductMetadata(
        model_id="aurora", model_version=dataset.attrs["model_version"],
        input_identity={"initialization_time_utc": initialization, "region": raw.get("effective_region") or raw.get("requested_region")},
        initialization_time_utc=initialization,
        variables={name: {
            "role": "mixed", "units": field.attrs["units"], "dimensions": list(field.dims),
            "role_coordinate": "product_role",
        } for name, field in dataset.data_vars.items()},
        segments=segments, cache_identity={"forecast_id": str(raw.get("forecast_id") or "")},
        limitations=(
            "Lead zero is ERA5 reanalysis; missing forecast leads contain no usable predictions.",
            "not_local_vil",
            "not_official_warning",
        ),
    ).to_dict()
    return dataset


def _select_fields(
    dataset: xr.Dataset,
    surface_variables: Sequence[str] | None,
    pressure_variables: Sequence[str] | None,
    pressure_levels: Sequence[int] | None,
) -> xr.Dataset:
    """Select and validate the Agent's explicit Aurora output fields.

    Parameters
    ----------
    dataset : xarray.Dataset
        Complete validated Aurora environment result.
    surface_variables, pressure_variables : sequence of str, optional
        Public output variable names to retain. ``None`` keeps every variable
        in that group; an empty sequence keeps none.
    pressure_levels : sequence of int, optional
        Pressure levels to retain when pressure variables are selected.

    Returns
    -------
    xarray.Dataset
        Dataset containing only the explicitly selected fields and levels.
    """

    surfaces = (
        sorted(name for name in _PUBLIC_SURFACE_VARIABLES if name in dataset)
        if surface_variables is None else list(dict.fromkeys(map(str, surface_variables)))
    )
    pressures = (
        sorted(name for name in _PUBLIC_PRESSURE_VARIABLES if name in dataset)
        if pressure_variables is None else list(dict.fromkeys(map(str, pressure_variables)))
    )
    unknown_surface = sorted(set(surfaces) - _PUBLIC_SURFACE_VARIABLES)
    unknown_pressure = sorted(set(pressures) - _PUBLIC_PRESSURE_VARIABLES)
    if unknown_surface:
        raise ValueError(
            "unsupported Aurora surface variables: " + ", ".join(unknown_surface)
            + "; supported names: " + ", ".join(sorted(_PUBLIC_SURFACE_VARIABLES))
        )
    if unknown_pressure:
        raise ValueError(
            "unsupported Aurora pressure variables: " + ", ".join(unknown_pressure)
            + "; supported names: " + ", ".join(sorted(_PUBLIC_PRESSURE_VARIABLES))
        )
    if not surfaces and not pressures:
        raise ValueError("select at least one Aurora output variable")
    if pressure_levels is not None and not pressures:
        raise ValueError("pressure_levels require pressure_variables")
    selected = dataset[[*surfaces, *pressures]]
    if pressure_levels is not None:
        levels = tuple(dict.fromkeys(int(item) for item in pressure_levels))
        if not levels:
            raise ValueError("pressure_levels must not be empty")
        available = set(int(item) for item in selected.pressure_level_hpa.values)
        missing = sorted(set(levels) - available)
        if missing:
            raise ValueError(
                "unsupported Aurora pressure levels: "
                + ", ".join(str(item) for item in missing)
            )
        selected = selected.sel(pressure_level_hpa=list(levels))
    return selected


class Aurora(BoundCapability):
    """Request-bound regional ERA5/Aurora fields as one xarray Dataset.

    Lead hour 0 is the ERA5 analysis and positive leads are Aurora forecasts.
    """

    @tool_parameters(
        history_before_hours=ParameterSpec(minimum=6, maximum=72),
        history_after_hours=ParameterSpec(minimum=0, maximum=72),
    )
    def regional_fields(
        self,
        initialization_time_utc: str,
        region: Any | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        forecast_leads_hours: Sequence[int] = (6, 12, 18, 24),
        history_before_hours: int = 6,
        history_after_hours: int = 24,
        surface_variables: Sequence[str] | None = None,
        pressure_variables: Sequence[str] | None = None,
        pressure_levels: Sequence[int] | None = None,
    ) -> xr.Dataset:
        """Return regional environment and Aurora fields for explicit inputs.

        Parameters
        ----------
        initialization_time_utc : str
            Timezone-aware ISO-8601 model initialization time.
        region : BoundingBox or dict, optional
            WGS84 mapping with exactly south, west, north, east, or BoundingBox.
            Pass named_region.bbox or geospatial.region_bounds(named_region)
            for a Region/GeoJSON polygon. Exclusive with latitude/longitude.
        latitude, longitude : float, optional
            Explicit WGS84 point. Both values are required together.
        forecast_leads_hours : sequence of int
            Positive model lead hours.
        history_before_hours, history_after_hours : int
            ERA5 context window used to initialize and describe the forecast:
            6--72 hours before and 0--72 hours after initialization. The
            minimum input history applies to Aurora, not ERA5-only reads.
        surface_variables, pressure_variables : sequence of str, optional
            Public Aurora output fields to retain. ``None`` keeps all fields
            from that group; an empty sequence keeps none.
        pressure_levels : sequence of int, optional
            Pressure levels to retain for selected pressure variables.

        Returns
        -------
        xarray.Dataset
            Variables use dimensions ``lead_hour, latitude, longitude`` or
            ``lead_hour, pressure_level_hpa, latitude, longitude``. The
            ``lead_hour=0`` slice is ERA5; positive leads are Aurora. Coordinates
            include ``valid_time_utc``, ``source``, ``latitude``, ``longitude``
            and optionally ``pressure_level_hpa``. ``product_role(lead_hour)``
            distinguishes reanalysis, forecast and unavailable leads. Dataset
            attributes include ``forecast_id``, ``initialization_time_utc``,
            ``cache_hit`` and ``product_metadata`` with model identity, physical
            units and per-lead provenance. Unknown model versions remain empty.

            Exact surface variables (degrees C, m/s, or hPa according to the
            suffix) are ``temperature_2m_c``, ``u_wind_10m_m_s``,
            ``v_wind_10m_m_s`` and ``mean_sea_level_pressure_hpa``. Exact
            pressure-level variables are ``air_temperature_c``,
            ``u_wind_m_s``, ``v_wind_m_s``, ``specific_humidity_g_kg`` and
            ``geopotential_height_m``. All model pressure levels are exposed;
            derived values and verification scores are left to Analytics.

        Raises
        ------
        ValueError
            If time, spatial target, leads or history windows are invalid.
        RuntimeError
            If cached-only fields are absent or required ERA5/model assets and
            live inference are unavailable. Do not substitute ERA5 positive
            offsets and label them as Aurora forecasts.

        Examples
        --------
        >>> environment = aurora.regional_fields(
        ...     "2019-06-27T00:00:00Z",
        ...     latitude=47.5, longitude=-100.5,
        ...     forecast_leads_hours=(6, 12))
        >>> current = environment["temperature_2m_c"].sel(lead_hour=0)
        >>> future = environment["temperature_2m_c"].sel(lead_hour=6)
        >>> upper_u = environment["u_wind_m_s"].sel(lead_hour=6, pressure_level_hpa=850)
        >>> local = environment.where(
        ...     (environment.latitude >= 46.5) & (environment.latitude <= 48.5)
        ...     & (environment.longitude >= -101.0) & (environment.longitude <= -98.0),
        ...     drop=True,
        ... )
        """
        from .Environment import regional_environment_fields

        initialization = require_utc_text(
            initialization_time_utc, "initialization_time_utc"
        )
        requested = tuple(dict.fromkeys(int(item) for item in forecast_leads_hours))
        if not requested or any(item <= 0 for item in requested):
            raise ValueError("forecast_leads_hours must contain positive hours")
        before = int(history_before_hours)
        after = int(history_after_hours)
        if not 6 <= before <= 72 or not 0 <= after <= 72:
            raise ValueError(
                "Aurora history windows must satisfy 6<=before<=72 and 0<=after<=72 hours; "
                f"received before={before}, after={after}"
            )
        forecast_box = BoundingBox.from_value(region)
        if forecast_box is not None and (latitude is not None or longitude is not None):
            raise ValueError("supply either region or latitude/longitude, not both")
        if (latitude is None) != (longitude is None):
            raise ValueError("latitude and longitude must be supplied together")
        point = (
            GeoPoint(float(latitude), float(longitude))
            if latitude is not None and longitude is not None else None
        )
        if forecast_box is None and point is None:
            raise ValueError("region or latitude/longitude is required")
        forecast_region = forecast_box.as_dict() if forecast_box else None
        arguments = {
            "event_id": self.request.assets.event_id or None,
            "initialization_time_utc": initialization,
            # The explicit forecast box fully determines a regional field.
            # Do not fragment the capability cache with a redundant geocoder
            # centre point that may be present or absent across equivalent
            # requests.
            "latitude": point.latitude if point else None,
            "longitude": point.longitude if point else None,
            "region": forecast_region,
            "forecast_leads": list(requested),
            "history_before_hours": before,
            "history_after_hours": after,
            "data_policy": self.request.policy.mode,
        }
        selected_surface = None if surface_variables is None else tuple(surface_variables)
        selected_pressure = None if pressure_variables is None else tuple(pressure_variables)
        selected_levels = None if pressure_levels is None else tuple(int(item) for item in pressure_levels)
        invocation_arguments = {
            **arguments,
            "output_surface_variables": selected_surface,
            "output_pressure_variables": selected_pressure,
            "output_pressure_levels": selected_levels,
        }

        def load_selected_environment(**values: Any) -> dict[str, Any]:
            """Call the backend after removing public output selectors.

            Parameters
            ----------
            **values : Any
                Validated backend arguments plus output-only selectors.

            Returns
            -------
            dict
                Raw regional ERA5/Aurora backend result.
            """

            values.pop("output_surface_variables", None)
            values.pop("output_pressure_variables", None)
            values.pop("output_pressure_levels", None)
            return regional_environment_fields(**values)

        return self._invoke(
            "aurora.regional_fields", invocation_arguments, load_selected_environment,
            lambda raw: _select_fields(
                _environment_dataset(raw, requested, initialization),
                selected_surface, selected_pressure, selected_levels,
            ),
            public_arguments={
                "initialization_time_utc": initialization,
                "region": forecast_region,
                "latitude": point.latitude if point else None,
                "longitude": point.longitude if point else None,
                "forecast_leads_hours": list(requested),
                "history_before_hours": before,
                "history_after_hours": after,
                "surface_variables": selected_surface,
                "pressure_variables": selected_pressure,
                "pressure_levels": selected_levels,
            },
        )
