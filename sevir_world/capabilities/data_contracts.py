"""Model-visible data contracts, read without importing weather backends.

Variable enumerations reference the same literal declarations used by providers.
Array schemas describe the public converters, not upstream file representations.
Tests exercise those converters against these declarations.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .dataframe_schema import PUBLIC_DATAFRAME_SCHEMAS, SCHEMAS


SOURCE_ROOT = Path(__file__).resolve().parents[1]
EnumValue = str | int | float | bool | None


@dataclass(frozen=True)
class SourceEnum:
    path: str
    symbol: str
    values: bool = False
    prefix: tuple[EnumValue, ...] = ()
    column: int | None = None

    def names(self) -> tuple[EnumValue, ...]:
        """Read a literal allowlist without loading a provider or model."""
        tree = ast.parse((SOURCE_ROOT / self.path).read_text(encoding="utf-8"))
        for node in tree.body:
            names = ([part.id for part in node.targets if isinstance(part, ast.Name)]
                     if isinstance(node, ast.Assign)
                     else [node.target.id] if isinstance(node, ast.AnnAssign)
                     and isinstance(node.target, ast.Name) else [])
            if self.symbol in names:
                value = ast.literal_eval(node.value)
                items = value.values() if self.values else value
                if self.column is not None:
                    items = (item[self.column] for item in items)
                return tuple(dict.fromkeys((*self.prefix, *items)))
        raise ValueError(f"missing contract enum {self.path}:{self.symbol}")


@dataclass(frozen=True)
class VariableGroup:
    names: tuple[str, ...] | SourceEnum
    dimensions: tuple[str, ...]

    def variables(self) -> tuple[str, ...]:
        return self.names.names() if isinstance(self.names, SourceEnum) else self.names


@dataclass(frozen=True)
class DataContract:
    result: str
    variables: tuple[VariableGroup, ...] = ()
    # Coordinate name, its dimensions, and its meaning/type. Optional axes are
    # identified in notes; these tuples never imply that a missing field exists.
    coordinates: tuple[tuple[str, tuple[str, ...], str], ...] = ()
    enums: tuple[tuple[str, tuple[EnumValue, ...] | SourceEnum], ...] = ()
    region: str = "none"
    notes: str = ""
    table: str = ""
    brief: str = ""
    # Long non-grid selectors (for example analogue signature feature names)
    # stay in the paginated archive; weather-variable enums are always inline.
    detail_enums: tuple[tuple[str, tuple[EnumValue, ...] | SourceEnum], ...] = ()


ERA5_SURFACE = SourceEnum("tools/era5/repository.py", "SURFACE_FIELDS")
ERA5_PRESSURE = SourceEnum("tools/era5/repository.py", "PRESSURE_FIELDS")
AURORA_SURFACE = SourceEnum("tools/aurora/Aurora.py", "_SURFACE_VARIABLES", values=True)
AURORA_PRESSURE = SourceEnum("tools/aurora/Aurora.py", "_PRESSURE_VARIABLES", values=True)
SATELLITE_CHANNELS = SourceEnum("tools/sevir/channels.py", "SUPPORTED_CHANNELS")
ALL_CHANNELS = SourceEnum("tools/sevir/channels.py", "SUPPORTED_CHANNELS", prefix=("vil",))
AURORA_LEVELS = SourceEnum("tools/era5/backend.py", "DEFAULT_AURORA_LEVELS")
ANALOGUE_FEATURES = SourceEnum("tools/retrieval/signature.py", "_FEATURES", column=0)
ANALOGUE_GROUPS = SourceEnum("tools/retrieval/signature.py", "_FEATURES", column=1)

BBOX_INPUT = (
    "BoundingBox or mapping with exactly south, west, north, east in WGS84 degrees; "
    "-90<=south<north<=90, -180<=west<east<=180. Region/GeoJSON are not accepted here: "
    "pass named_region.bbox or geospatial.region_bounds(named_region)."
)
GEOMETRY_INPUT = (
    "Region, BoundingBox, mapping containing south/west/north/east, GeoJSON Polygon/Feature, "
    "or sequence of (longitude, latitude) vertices; WGS84 degrees. Polygon uses its exterior ring; "
    "holes/MultiPolygon are not supported. Pass polygon.coordinates for a Polygon object."
)
REGION_TYPES = {"bbox": BBOX_INPUT, "geometry": GEOMETRY_INPUT}

OBS_COORDINATES = (
    ("time", ("time",), "datetime64[ns], UTC"),
    ("time_utc", ("time",), "ISO-8601 UTC strings"),
    ("minute_offset", ("time",), "integer minutes relative to event anchor"),
    ("row", ("row",), "zero-based pixel index"),
    ("column", ("column",), "zero-based pixel index"),
)
OBS_VARIABLE = VariableGroup(("vil",), ("time", "row", "column"))
GRID_NOTE = "Pixel row/column are not latitude/longitude; use attrs.spatial_metadata and geospatial methods."
TIME_MATCH = (("time_match", ("exact", "nearest")),)
WEATHER_REGION = "Supply region or both latitude/longitude, never both spatial forms."


METHOD_DATA_CONTRACTS: dict[str, DataContract] = {
    "knowledge.search": DataContract(
        "DataFrame", table="knowledge.search",
        enums=(("tools", ("sevir", "cascast", "era5", "aurora")),),
        notes="tools=None searches all approved documents; [] searches none. No gridded variables/coordinates/dimensions.",
        brief="tools=None searches all approved documents; [] searches none; table has no raster axes.",
    ),
    "knowledge.read": DataContract(
        "dict", notes="One source passage: chunk_id, doc_id, revision, title, version, tools, text, section, locator, source_url, citation, url. No gridded variables/coordinates/dimensions.",
    ),
    "knowledge.describe": DataContract(
        "dict", notes="Fields status, document_count, chunk_count, identity, documents. Index coverage/model identity and document versions; no gridded variables/coordinates/dimensions.",
    ),
    "sevir.event": DataContract(
        "Event", notes="Metadata record, not pixels: id, subject_key, event_anchor_time_utc, frame_times_utc, minute_offsets, south/west/north/east and spatial_metadata. No xarray coordinates/dimensions.",
    ),
    "sevir.search": DataContract(
        "DataFrame", table="sevir.search", enums=TIME_MATCH, region="geometry",
        notes="VIL catalog only; event_type is a catalog label, not a channel enum. Region filters footprint bounding boxes, not pixels. No raster coordinates/dimensions. nearest requires positive tolerance_seconds; exact forbids it. "
              "start_time_utc/end_time_utc/observation_time_utc require an explicit timezone: Z or UTC offset (e.g. 2017-09-12T00:00:00Z); resolve local timezone first.",
        brief="VIL catalog; region matches bounding boxes. nearest needs tolerance_seconds; exact forbids it. No raster axes. "
              "start_time_utc/end_time_utc/observation_time_utc require an explicit timezone: Z or UTC offset (e.g. 2017-09-12T00:00:00Z); resolve local timezone first.",
    ),
    "sevir.observations": DataContract(
        "DataArray", variables=(OBS_VARIABLE,), coordinates=OBS_COORDINATES,
        enums=TIME_MATCH, region="geometry",
        notes="encoded_vil; source missing code 255 becomes NaN. Region masks pixels without cropping. " + GRID_NOTE,
        brief="encoded_vil; time=datetime64 UTC; region masks without cropping; row/column are pixels. Absolute time and relative windows are exclusive.",
    ),
    "sevir.channel_observations": DataContract(
        "DataArray named by channel",
        variables=(VariableGroup(SATELLITE_CHANNELS, ("time", "row", "column")),),
        coordinates=OBS_COORDINATES, enums=(("channel", SATELLITE_CHANNELS),),
        notes="One requested channel per return. IR: scaled_brightness_temperature; VIS: scaled_reflectance; LGHT: flash_extent_density. Missing source values become NaN. " + GRID_NOTE,
        brief="One requested channel; time=datetime64 UTC; units depend on channel; row/column are pixels.",
    ),
    "sevir.multimodal_observations": DataContract(
        "MultimodalObservations (channel -> DataArray)",
        variables=(VariableGroup(ALL_CHANNELS, ("time", "row", "column")),),
        coordinates=OBS_COORDINATES,
        enums=(("channels", ALL_CHANNELS),),
        notes="Only available channels are keys. available_channels/unavailable_channels give explicit availability. Each channel has its own grid/units; do not assume aligned pixels. " + GRID_NOTE,
        brief="Per-channel grids/units; inspect available_channels and unavailable_channels; time=datetime64 UTC.",
    ),
    "sevir.observation_batch": DataContract(
        "Dataset", variables=(VariableGroup(("vil",), ("event", "minute_offset", "row", "column")),),
        coordinates=(
            ("event_id", ("event",), "catalog event ID"),
            ("subject_key", ("event",), "exact catalog row key"),
            ("minute_offset", ("minute_offset",), "relative integer minutes"),
            ("time", ("event", "minute_offset"), "datetime64[ns], UTC"),
            ("time_utc", ("event", "minute_offset"), "ISO-8601 UTC strings"),
            ("row", ("row",), "pixel index"), ("column", ("column",), "pixel index"),
        ),
        notes="encoded_vil; event is a dimension without an index coordinate; align by minute_offset, not absolute UTC. Empty Dataset if none load; check loaded_subject_keys/unavailable_subject_keys. Pixel grids belong to separate events, not one regional mosaic.",
        brief="encoded_vil; event has no index coordinate; time=datetime64 UTC; empty Dataset if none load.",
    ),
    "sevir.find_analogues": DataContract(
        "DataFrame", table="sevir.find_analogues", region="bbox",
        enums=(("feature_weights groups", ANALOGUE_GROUPS),),
        detail_enums=(("feature_weights keys", ANALOGUE_FEATURES),),
        notes=WEATHER_REGION + " No raster coordinates/dimensions. feature_weights overrides exact feature names or groups with non-negative weights; group weights apply to their member features.",
    ),
    "sevir.match_upload": DataContract(
        "UploadMatch", notes="Metadata fields upload_id, subject_key, event_id, confidence, method; no raster variables/coordinates/dimensions. An unmatched event_id can be None.",
    ),
    "geospatial.resolve_place": DataContract(
        "Place", notes="Record fields query, status, name, latitude, longitude, candidates DataFrame (geospatial.place.candidate). No bbox: use boundary(name, level=...) for a named Region. No raster variables/coordinates/dimensions. country_codes uses comma-separated country codes, not a fixed local enum.",
        table="geospatial.resolve_place.candidates",
        brief="Place fields query,status,name,latitude,longitude,candidates; no bbox. country_codes is comma-separated. No raster axes.",
    ),
    "geospatial.point_to_location": DataContract(
        "AdministrativeArea", notes="Record with latitude/longitude and country/state/county identifiers; no raster variables/coordinates/dimensions.",
    ),
    "geospatial.boundary": DataContract(
        "Region", enums=(("level", ("state", "county")),),
        notes="level=None searches both levels and can raise ambiguity. Specify level='state' for a US state. Fields bbox:BoundingBox, polygon:Polygon, name, level, properties; no raster coordinates/dimensions. polygon is the largest exterior ring of the matched boundary. Pass result.bbox to ERA5/Aurora.",
        brief="level=None can be ambiguous; use level='state' for states; result.bbox is accepted by ERA5/Aurora.",
    ),
    "geospatial.region_bounds": DataContract(
        "BoundingBox", region="geometry",
        notes="Fields south, west, north, east; as_dict() returns the strict bbox mapping. No gridded variables/coordinates/dimensions.",
    ),
    "geospatial.pixel_to_latlon": DataContract(
        "GeoPoint", notes="Zero-based row/column -> latitude, longitude and optional projected_x/projected_y/crs. Input requires spatial_metadata or latitude/longitude coordinates; no returned array axes.",
    ),
    "geospatial.latlon_to_pixel": DataContract(
        "GridPoint", notes="WGS84 latitude/longitude -> zero-based row, column and actual cell-centre latitude/longitude. Input requires spatial_metadata or lat/lon coordinates; no returned array axes.",
    ),
    "geospatial.mask": DataContract(
        "read-only bool ndarray", region="geometry",
        notes="Shape (grid_rows, grid_columns): axis 0 is row/latitude, axis 1 column/longitude, aligned with the target grid. Plain NumPy array: no named coordinates. No weather-variable selector.",
        brief="2-D bool shape (grid_rows,grid_columns); no named coordinates; aligns with input spatial grid.",
    ),
    "geospatial.select_region": DataContract(
        "same xarray type as data_ref", region="geometry",
        notes="Preserves input variable names and non-spatial axes/coordinates. Masks and crops row,column or latitude,longitude to a contiguous window; updates spatial_metadata/spatial_selection. No fixed weather-variable enum; inherit input schema.",
        brief="Variables/coords/dims inherit input; crops spatial row,column or latitude,longitude; other axes retained.",
    ),
    "cascast.available_forecasts": DataContract(
        "DataFrame", table="cascast.available_forecasts",
        notes="Discovery only; no forecast arrays. Pass subject_key for an event selected this turn; binding conflicts are rejected. cache_status=miss_live_can_run permits forecast under live policy.",
    ),
    "cascast.forecast": DataContract(
        "Dataset", variables=(
            VariableGroup(("history_vil",), ("history_minute", "row", "column")),
            VariableGroup(("forecast_mean_vil", "ground_truth_vil"), ("lead_minute", "row", "column")),
            VariableGroup(("forecast_member_vil",), ("member", "lead_minute", "row", "column")),
        ),
        coordinates=(
            ("history_minute", ("history_minute",), "relative minutes"),
            ("history_time_utc", ("history_minute",), "ISO-8601 UTC strings"),
            ("lead_minute", ("lead_minute",), "forecast lead minutes"),
            ("valid_time_utc", ("lead_minute",), "ISO-8601 UTC strings"),
            ("member", ("member",), "zero-based member index"),
            ("member_seed", ("member",), "integer seed"),
            ("row", ("row",), "pixel index"), ("column", ("column",), "pixel index"),
        ),
        notes="All variables encoded_vil. Pass subject_key from sevir.search for a subject selected this turn; conflicting bound assets are rejected. ground_truth_vil exists only when requested and available; inspect ground_truth_available. No region argument; use geospatial.select_region on output. " + GRID_NOTE,
        brief="encoded_vil; ground_truth_vil optional; UTC coords are strings; row/column are pixels.",
    ),
    "era5.datasets": DataContract(
        "dict", notes="datasets lists source IDs, titles, descriptions, extents, licences and links. Live CDS catalogue discovery, not a fixed enum. query defaults to era5; empty query lists the whole CDS catalogue. No weather arrays.",
        brief="datasets: source metadata; query='' lists all CDS, default='era5'; no fixed enum.",
    ),
    "era5.describe": DataContract(
        "dict", notes="collection contains source metadata/coverage/licence; form contains the source's complete request parameters/options; source_url identifies the primary source. dataset is an exact CDS ID, not a local enum. No weather arrays.",
        brief="collection metadata and full native request form/options, with source_url.",
    ),
    "era5.retrieve": DataContract(
        "dict", notes="Native CDS/MARS selection mapping, no variable/dataset/time-span/grid allowlist. Returns dataset, selection, files (native paths), parts (exact selections, path, request_key, cache_hit), cache_hit, source_url. split_by=month/day/none; completed parts are reusable. Omitting area uses source default extent. Source availability/licences/quotas apply. No invented grid schema.",
        brief="Native CDS/MARS selection -> files, parts, source_url; split_by=month/day/none; resumable cache.",
    ),
    "era5.fields": DataContract(
        "Dataset", notes="Native CDS/MARS selection for any CDS dataset ID. Variables, coordinates, dimensions, level/member axes and units inherit the source, with no fixed schema. Inspect data_vars/coords/dims/attrs. Supports NetCDF/ZIP; GRIB requires cfgrib/ecCodes. Lazy on-disk arrays: select before loading and close when done. split_by=month/day/none. Omitting area uses source default extent; area/grid are passed unchanged. No project time-span limit or Aurora grid padding. Source availability/licences/quotas apply.",
        brief="Native lazy Dataset; inspect data_vars/coords/dims/units; no fixed schema. split_by=month/day/none.",
    ),
    "era5.regional_fields": DataContract(
        "Dataset", variables=(
            VariableGroup(ERA5_SURFACE, ("time", "latitude", "longitude")),
            VariableGroup(ERA5_PRESSURE, ("time", "pressure_level_hpa", "latitude", "longitude")),
        ),
        coordinates=(
            ("time", ("time",), "integer index, NOT timestamps"),
            ("offset_hour", ("time",), "integer hours relative to reference_time_utc"),
            ("valid_time_utc", ("time",), "ISO-8601 UTC strings; no valid_time field"),
            ("latitude", ("latitude",), "WGS84 degrees north"),
            ("longitude", ("longitude",), "WGS84 degrees east"),
            ("pressure_level_hpa", ("pressure_level_hpa",), "hPa, only with pressure fields"),
        ),
        region="bbox",
        notes="Variable names above are common examples, not an enum/allowlist. All CDS hourly source variables (including precipitation/dewpoint) may be selected; describe() exposes full choices. Only requested variables return; [] selects no fields in that group. pressure_levels required for pressure fields: positive hPa values covered by the source. Units come from variable.attrs['units']. Supply region or both latitude/longitude; omit all spatial arguments for global data. No Aurora padding. Non-negative hours_before/hours_after have no project upper bound; both endpoints included (0/0=one hour, 0/23=one UTC day). time is an integer index; read valid_time_utc for timestamps. For other products/native grids/member axes use fields(dataset, selection).",
        brief="Source names above are examples, not an enum; all hourly variables accepted. time=integer index, valid_time_utc=UTC strings; pressure fields require pressure_levels; []=no fields. No project window cap; omit spatial arguments=global. Use fields() for other datasets/native axes.",
    ),
    "aurora.regional_fields": DataContract(
        "Dataset", variables=(
            VariableGroup(AURORA_SURFACE, ("lead_hour", "latitude", "longitude")),
            VariableGroup(AURORA_PRESSURE, ("lead_hour", "pressure_level_hpa", "latitude", "longitude")),
        ),
        coordinates=(
            ("lead_hour", ("lead_hour",), "0=ERA5 reanalysis; positive=Aurora forecast"),
            ("valid_time_utc", ("lead_hour",), "ISO-8601 UTC strings"),
            ("source", ("lead_hour",), "ERA5/Aurora/unavailable"),
            ("product_role", ("lead_hour",), "reanalysis/forecast/unavailable"),
            ("latitude", ("latitude",), "WGS84 degrees north, ascending"),
            ("longitude", ("longitude",), "WGS84 degrees east, ascending"),
            ("pressure_level_hpa", ("pressure_level_hpa",), "available hPa levels, optional"),
        ),
        enums=(("surface_variables", AURORA_SURFACE), ("pressure_variables", AURORA_PRESSURE),
               ("pressure_levels", AURORA_LEVELS)), region="bbox",
        notes=WEATHER_REGION + " Selectors use PUBLIC output names, not ERA5 source codes. None keeps all available in group; [] keeps none. Only selected fields/levels return; unavailable leads are NaN, not predictions. Units follow suffix/attrs (degC, m s-1, hPa, g kg-1, m).",
        brief="Public output names; None=all in group, []=none; pressure axis optional; lead 0=ERA5, positive=Aurora. " + WEATHER_REGION,
    ),
}


def contract_source_paths(selected: tuple[str, ...]) -> set[Path]:
    """Include enum providers in documentation-cache invalidation."""
    paths = {Path(__file__), Path(__file__).with_name("dataframe_schema.py")}
    for operation, contract in METHOD_DATA_CONTRACTS.items():
        if operation.split(".")[0] not in selected:
            continue
        enums = [value for _, value in (*contract.enums, *contract.detail_enums)]
        enums.extend(group.names for group in contract.variables)
        paths.update(SOURCE_ROOT / value.path for value in enums if isinstance(value, SourceEnum))
    return paths


def render_data_contract(operation: str, *, compact: bool = False) -> str:
    """Render critical fields in both the selected index and full archive."""
    contract = METHOD_DATA_CONTRACTS.get(operation)
    if contract is None:
        return ""  # Third-party registered tools can retain source docstrings.
    parts = []
    for parameter, value in (*contract.enums, *(contract.detail_enums if not compact else ())):
        names = value.names() if isinstance(value, SourceEnum) else value
        parts.append(f"{parameter} enum=[{', '.join(map(str, names))}]")
    if contract.region != "none":
        parts.append("region: " + (contract.region if compact else REGION_TYPES[contract.region]))
    elif not compact:
        parts.append("region: no region parameter")
    parts.append("returns " + contract.result)
    if contract.table:
        schema = SCHEMAS[PUBLIC_DATAFRAME_SCHEMAS[contract.table]]
        parts.append(f"columns ({schema.id}@{schema.version}): " + ", ".join(column.name for column in schema.columns))
    for group in contract.variables:
        # Enum names have already been printed above; reference their selector
        # in the compact index instead of repeating the full same allowlist.
        selector = next((name for name, source in contract.enums if source == group.names), "")
        names = selector if compact and selector else f"[{', '.join(group.variables())}]"
        parts.append(f"variables {names} dims=({','.join(group.dimensions)})")
    if contract.coordinates:
        parts.append("coords: " + "; ".join(
            f"{name}({','.join(dims)})" + ("=" + meaning if not compact else "")
            for name, dims, meaning in contract.coordinates
        ))
    note = contract.brief if compact else contract.notes
    if note:
        parts.append(note)
    elif not contract.variables and not contract.coordinates:
        parts.append("axes: n/a" if compact else "no gridded variables/coordinates/dimensions")
    return "Data contract: " + "; ".join(parts)
