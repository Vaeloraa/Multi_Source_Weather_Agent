"""Request-bound ERA5/ERA5T reanalysis fields without Aurora inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from ...capabilities.base import BoundCapability
from ...capabilities.contracts import ParameterSpec, tool_parameters
from .repository import (
    regional_era5_fields as _regional_era5_fields,
)
from .cds import list_datasets, describe_dataset, retrieve_data, open_fields
from .._shared import require_utc_text
from ..geospatial.contracts import BoundingBox, GeoPoint


def _dataset(raw: dict[str, Any]) -> xr.Dataset:
    """Convert raw ERA5 mappings into an xarray dataset.

    Parameters
    ----------
    raw : dict
        Raw ERA5 backend result.
    Returns
    -------
    xarray.Dataset
        Standard regional ERA5 fields.
    """
    coords: dict[str, Any] = {
        "time": np.arange(len(raw["offset_hours"]), dtype=np.int32),
        "offset_hour": (
            "time", np.asarray(raw["offset_hours"], dtype=np.int32)
        ),
        "valid_time_utc": (
            "time", np.asarray(raw["valid_time_utc"], dtype=str)
        ),
        "latitude": np.asarray(raw["latitude"], dtype=np.float32),
        "longitude": np.asarray(raw["longitude"], dtype=np.float32),
    }
    if raw["pressure_levels"]:
        coords["pressure_level_hpa"] = np.asarray(raw["pressure_levels"], dtype=np.int32)
    surface_names = set(raw["surface_variables"])
    variables: dict[str, Any] = {}
    for name, value in raw["fields"].items():
        dimensions = (
            ("time", "latitude", "longitude")
            if name in surface_names
            else ("time", "pressure_level_hpa", "latitude", "longitude")
        )
        variables[name] = (dimensions, value, {"units": raw["units"][name]})
    dataset = xr.Dataset(variables, coords=coords, attrs={
        "source": "ERA5/ERA5T reanalysis",
        "reference_time_utc": raw["reference_time_utc"],
        "cache_hit": bool(raw.get("cache_hit", False)),
        "data_temporality": str(
            raw.get("era5_temporality", {}).get("data_class") or "reanalysis"
        ),
        "era5_product": str(
            raw.get("era5_temporality", {}).get("product") or "ERA5"
        ),
        "provisional": raw.get("era5_temporality", {}).get("provisional"),
        "is_real_time": False,
        "access_mode": str(raw.get("access_mode") or "cache_then_remote"),
        "provenance": str(raw.get("provenance") or {}),
    })
    latitude = np.asarray(raw["latitude"], dtype=np.float64)
    longitude = np.asarray(raw["longitude"], dtype=np.float64)
    dataset.attrs["spatial_metadata"] = {
        "grid_type": "latlon_grid", "crs": "EPSG:4326",
        "shape": [int(latitude.size), int(longitude.size)],
        "latitude": latitude.tolist(), "longitude": longitude.tolist(),
        "bounds": {
            "south": float(np.min(latitude)), "west": float(np.min(longitude)),
            "north": float(np.max(latitude)), "east": float(np.max(longitude)),
        },
        "axis_order": ["latitude", "longitude"],
    }
    return dataset


class Era5(BoundCapability):
    """Discover and read source-native CDS/ERA5 products and hourly regional fields."""

    def datasets(self, query: str = "era5") -> dict:
        """Discover datasets from the CDS catalogue, including ERA5-Land,
        monthly means, daily statistics, time series and ERA5 complete.

        query is matched against source IDs and titles; an empty string lists
        the entire CDS catalogue. This is live discovery, not a fixed list.
        Cached policy requires previously cached catalogue pages.
        Returns a dict with datasets (id, title, description, extent, license,
        links) and source_url. This does not retrieve meteorological arrays.
        """
        return self._invoke("era5.datasets", {"query": query, "data_policy": self.request.policy.mode},
                            list_datasets, lambda raw: raw, public_arguments={"query": query})

    def describe(self, dataset: str) -> dict:
        """Read one CDS dataset's metadata and complete parameter form.

        dataset is its exact CDS ID, e.g. reanalysis-era5-land or
        reanalysis-era5-complete. Returns collection (source coverage,
        description, license, links), form (native parameter names/options),
        and source_url. Use these choices to construct fields/retrieve
        selections. No local dataset or weather-variable allowlist applies.
        """
        return self._invoke("era5.describe", {"dataset": dataset, "data_policy": self.request.policy.mode},
                            describe_dataset, lambda raw: raw, public_arguments={"dataset": dataset})

    @tool_parameters(split_by=ParameterSpec(allowed=("month", "day", "none")))
    def retrieve(self, dataset: str, selection: dict[str, Any], split_by: str = "month") -> dict:
        """Download native CDS data with resumable per-part caching.

        dataset is any CDS dataset ID. selection is the dataset's native API
        request mapping: variable/product_type/year/month/day/time/area/grid,
        ensemble selectors or MARS date/param/levtype/levelist/stream/type,
        as appropriate. Read describe(dataset) or its official API example.
        All source selectors and file formats are forwarded unchanged.
        Omitting area uses the source's default extent (normally global).
        No project time-span, variable, pressure-level or grid-size cap applies.
        Source availability, licences, supported selections and quotas apply.

        split_by='month' partitions year/month lists or ISO date/to/date
        ranges; 'day' uses smaller daily requests; 'none' forwards the exact
        mapping, including arbitrary MARS date expressions. Completed parts
        survive failures and are reused when the same request is retried.
        Returns dict: dataset, selection, files (native paths), parts (exact
        request selections and cache_hit), cache_hit and source_url. Native
        ZIP/GRIB files are preserved. Use fields() for xarray data.
        """
        public = {"dataset": dataset, "selection": selection, "split_by": split_by}
        return self._invoke("era5.retrieve", {**public, "data_policy": self.request.policy.mode},
                            retrieve_data, lambda raw: raw, public_arguments=public)

    @tool_parameters(split_by=ParameterSpec(allowed=("month", "day", "none")))
    def fields(self, dataset: str, selection: dict[str, Any], split_by: str = "month") -> xr.Dataset:
        """Read any CDS product into an xarray Dataset with native axes/units.

        Uses the same source-native selection and split_by rules as retrieve.
        Supports NetCDF, multi-file ZIP and GRIB (requires cfgrib/ecCodes).
        Preserves every returned variable, coordinate, ensemble member, model
        level and source unit; there is no fixed output schema. Inspect
        data_vars, coords, dims and variable.attrs before analysis. Chunks are
        combined by coordinates; incompatible source grids should be requested
        separately or accessed as native files through retrieve().
        Arrays remain lazy on disk: select dates/levels/regions before loading
        large global data, and call close() when finished. No Aurora padding,
        minimum grid dimensions or 72-hour time-window limit applies.

        Examples
        --------
        >>> data = era5.fields('reanalysis-era5-land', {
        ...     'variable': ['total_precipitation'], 'year': ['2019'],
        ...     'month': ['01', '02'], 'day': ['01'], 'time': ['00:00'],
        ...     'data_format': 'netcdf', 'download_format': 'unarchived'})
        """
        public = {"dataset": dataset, "selection": selection, "split_by": split_by}
        return self._invoke("era5.fields", {**public, "data_policy": self.request.policy.mode},
                            retrieve_data, open_fields, public_arguments=public)

    @tool_parameters(
        hours_before=ParameterSpec(minimum=0),
        hours_after=ParameterSpec(minimum=0),
    )
    def regional_fields(
        self,
        reference_time_utc: str,
        surface_variables: list[str],
        pressure_variables: list[str] | None = None,
        pressure_levels: list[int] | None = None,
        region: Any | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        hours_before: int = 6,
        hours_after: int = 0,
    ) -> xr.Dataset:
        """Return ERA5 fields for an explicit time window and WGS84 target.

        The Agent explicitly selects source variables, pressure levels and time
        extent. This data capability retrieves and validates those fields; it
        does not derive wind speed, statistics, plots, or scientific conclusions.

        Parameters
        ----------
        reference_time_utc : str
            Timezone-aware ISO-8601 reference time. A local clock time without
            an offset is rejected rather than guessed as UTC.
        surface_variables : list[str]
            Any single-level CDS source names, including precipitation and
            dewpoint. Inspect describe('reanalysis-era5-single-levels') for
            current choices. Pass [] for pressure-only data.
        pressure_variables : list[str], optional
            Any pressure-level CDS source names. These are not Aurora output
            names. For other datasets/product types use fields().
        pressure_levels : list[int], optional
            Required when pressure variables are selected.
        region : BoundingBox or dict, optional
            Explicit WGS84 bounds: exactly south, west, north, east, in degrees.
            A Region or GeoJSON polygon is not accepted: pass region.bbox or
            geospatial.region_bounds(region). Exclusive with latitude/longitude.
            No spatial arguments selects the globe; global/polar bounds are
            accepted without Aurora padding. Uses a regular 0.25-degree grid;
            fields() supports source-native or explicitly selected resolutions.
        latitude, longitude : float, optional
            Explicit WGS84 point. Both values are required together.
        hours_before, hours_after : int, optional
            Non-negative whole hours with no project upper limit.
            Both zero selects one reference hour. A UTC day
            uses midnight with hours_before=0, hours_after=23 (24 hourly
            samples). This ERA5-only read needs no Aurora input history.

        Returns
        -------
        xarray.Dataset
            Only the requested source variables. Each field carries its source
            unit and dimensions; dataset attributes carry provenance and time.
            Surface fields have dimensions (time, latitude, longitude); pressure
            fields have (time, pressure_level_hpa, latitude, longitude).
            time is an integer index, offset_hour(time) contains relative hours,
            and valid_time_utc(time) contains ISO UTC strings. There is no
            valid_time field. latitude/longitude are 1-D WGS84 degree coordinates;
            pressure_level_hpa exists only for pressure fields. Use the actual
            coordinate ordering. Temperature is K and pressure is Pa; do not
            assume the converted Aurora units or its lead_hour dimension.

        Raises
        ------
        ValueError
            If time, spatial target, variables, levels or windows are invalid.
        RuntimeError
            If cached-only data are absent, required surface/pressure files are
            missing, or the requested time is outside supported ERA5/ERA5T
            availability. Do not reinterpret this as a forecast or silently
            substitute Aurora output.

        Examples
        --------
        >>> fields = era5.regional_fields(
        ...     "2019-06-27T03:00:00Z", ["2m_temperature"],
        ...     latitude=47.5, longitude=-100.5)
        >>> fields = era5.regional_fields(
        ...     "2019-06-27T03:00:00Z", [],
        ...     pressure_variables=["temperature"], pressure_levels=[850, 500],
        ...     latitude=47.5, longitude=-100.5, hours_before=24)
        >>> analysis = fields["2m_temperature"].where(
        ...     fields.offset_hour == 0, drop=True)
        """
        reference_time = require_utc_text(reference_time_utc, "reference_time_utc")
        before = int(hours_before)
        after = int(hours_after)
        if before < 0:
            raise ValueError(f"hours_before must be non-negative; received {hours_before!r}")
        if after < 0:
            raise ValueError(f"hours_after must be non-negative; received {hours_after!r}")
        bbox = BoundingBox.from_value(region)
        if bbox is not None and (latitude is not None or longitude is not None):
            raise ValueError("supply either region or latitude/longitude, not both")
        if (latitude is None) != (longitude is None):
            raise ValueError("latitude and longitude must be supplied together")
        point = (
            GeoPoint(float(latitude), float(longitude))
            if latitude is not None and longitude is not None else None
        )
        arguments = {
            "reference_time_utc": reference_time,
            "latitude": point.latitude if point else None,
            "longitude": point.longitude if point else None,
            "region": bbox.as_dict() if bbox else None,
            "history_before_hours": before,
            "history_after_hours": after,
            "data_policy": self.request.policy.mode,
            "surface_variables": list(surface_variables),
            "pressure_variables": list(pressure_variables or []),
            "pressure_levels": [int(value) for value in (pressure_levels or [])],
        }
        return self._invoke(
            "era5.regional_fields", arguments, _regional_era5_fields, _dataset,
            public_arguments={
                "surface_variables": list(surface_variables),
                "pressure_variables": list(pressure_variables or []),
                "pressure_levels": [int(value) for value in (pressure_levels or [])],
                "reference_time_utc": reference_time,
                "region": bbox.as_dict() if bbox else None,
                "latitude": point.latitude if point else None,
                "longitude": point.longitude if point else None,
                "hours_before": before,
                "hours_after": after,
            },
        )
