from __future__ import annotations

"""Public Aurora tool for request-bound regional environment fields."""

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np
import xarray as xr

from ...capabilities.base import BoundCapability
from .._shared import utc_text


_SURFACE_VARIABLES = {
    "t2m_c": "temperature_2m_c",
    "u10_m_s": "u_wind_10m_m_s",
    "v10_m_s": "v_wind_10m_m_s",
    "wind10_m_s": "wind_speed_10m_m_s",
    "mslp_hpa": "mean_sea_level_pressure_hpa",
}
_PRESSURE_VARIABLES = {
    "temperature_c": "air_temperature_c",
    "u_m_s": "u_wind_m_s",
    "v_m_s": "v_wind_m_s",
    "wind_m_s": "wind_speed_m_s",
    "specific_humidity_g_kg": "specific_humidity_g_kg",
    "geopotential_height_m": "geopotential_height_m",
}


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
    for source, name in _PRESSURE_VARIABLES.items():
        values = []
        for lead in leads:
            by_level = states[lead].get("pressure_levels") or {}
            values.append(np.stack([
                np.asarray((by_level.get(str(level)) or {}).get(source, np.full((latitude.size, longitude.size), np.nan)), dtype=np.float32)
                for level in pressure_levels
            ]))
        if pressure_levels:
            variables[name] = (("lead_hour", "pressure_level_hpa", "latitude", "longitude"), np.stack(values))
    verification = {
        int(key): value for key, value in (raw.get("era5_verification") or {}).items()
    }
    for source, name in _SURFACE_VARIABLES.items():
        values = [np.full((latitude.size, longitude.size), np.nan, dtype=np.float32)]
        values.extend([
            np.asarray((verification.get(lead, {}).get("surface") or {}).get(
                source, np.full((latitude.size, longitude.size), np.nan)
            ), dtype=np.float32)
            for lead in requested_leads
        ])
        variables[f"verification_{name}"] = (("lead_hour", "latitude", "longitude"), np.stack(values))
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
        "verification_available_leads_hours": tuple(sorted(verification)),
    })
    for name in dataset.data_vars:
        dataset[name].attrs["units"] = "encoded field value; see variable suffix"
    return dataset


class Aurora(BoundCapability):
    """Request-bound regional ERA5/Aurora fields as one xarray Dataset.

    Lead hour 0 is the ERA5 analysis and positive leads are Aurora forecasts.
    """

    def regional_fields(self) -> xr.Dataset:
        """Return the bound regional environment and Aurora forecast fields.

        Parameters
        ----------
        This method has no forecast or region arguments: initialization,
        requested region and positive lead hours are fixed when the request is
        constructed. Lead hour 0 is the initializing ERA5 analysis.

        Returns
        -------
        xarray.Dataset
            Variables use dimensions ``lead_hour, latitude, longitude`` or
            ``lead_hour, pressure_level_hpa, latitude, longitude``. The
            ``lead_hour=0`` slice is ERA5; positive leads are Aurora. Coordinates
            include ``valid_time_utc``, ``source``, ``latitude``, ``longitude``
            and optionally ``pressure_level_hpa``. Dataset attributes include
            ``forecast_id``, ``initialization_time_utc`` and ``cache_hit``.

            Exact surface variables (degrees C, m/s, or hPa according to the
            suffix) are ``temperature_2m_c``, ``u_wind_10m_m_s``,
            ``v_wind_10m_m_s``, ``wind_speed_10m_m_s`` and
            ``mean_sea_level_pressure_hpa``. Exact pressure-level variables are
            ``air_temperature_c``, ``u_wind_m_s``, ``v_wind_m_s``,
            ``wind_speed_m_s``, ``specific_humidity_g_kg`` and
            ``geopotential_height_m``. Verification surface fields use the
            same names prefixed by ``verification_``. Never substitute ERA5
            source abbreviations such as ``t2m`` or ``u10`` for these names.

        Examples
        --------
        >>> environment = aurora.regional_fields()
        >>> current = environment["temperature_2m_c"].sel(lead_hour=0)
        >>> future = environment["temperature_2m_c"].sel(lead_hour=6)
        >>> upper_wind = environment["wind_speed_m_s"].sel(lead_hour=6, pressure_level_hpa=850)
        >>> local = environment.where(
        ...     (environment.latitude >= 46.5) & (environment.latitude <= 48.5)
        ...     & (environment.longitude >= -101.0) & (environment.longitude <= -98.0),
        ...     drop=True,
        ... )
        """
        from .Environment import regional_environment_fields

        initialization = utc_text(self.request.forecast.initialization_time_utc)
        if not initialization:
            raise ValueError("this request has no bound Aurora initialization time")
        requested = tuple(int(item) for item in self.request.forecast.lead_hours)
        bound = tuple(int(item) for item in self.request.forecast.lead_hours)
        arguments = {
            "event_id": self.request.assets.event_id or None,
            "initialization_time_utc": initialization,
            "latitude": self.request.target.latitude,
            "longitude": self.request.target.longitude,
            "region": self.request.forecast.region.as_dict() if self.request.forecast.region else None,
            "forecast_leads": list(bound),
            "history_before_hours": self.request.forecast.history_before_hours,
            "history_after_hours": self.request.forecast.history_after_hours,
            "data_policy": self.request.policy.mode,
        }
        return self._invoke(
            "aurora.regional_fields", arguments, regional_environment_fields,
            lambda raw: _environment_dataset(raw, requested, initialization),
            public_arguments={},
        )
