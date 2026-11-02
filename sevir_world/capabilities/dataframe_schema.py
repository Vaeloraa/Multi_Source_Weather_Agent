from __future__ import annotations

"""Versioned schemas for every public pandas DataFrame capability result.

Backend payloads may keep source-native names.  Public tables cross one
normalization boundary here and expose canonical columns plus schema identity
in ``DataFrame.attrs``.  Legacy aliases are accepted only on input and are
never emitted.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd


DATAFRAME_CONTRACT_VERSION = "dataframe-contract-v1"


class DataFrameSchemaError(ValueError):
    """Raised when a table cannot satisfy its declared public schema."""


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str = "object"
    aliases: tuple[str, ...] = ()
    default: Any = None
    description: str = ""


@dataclass(frozen=True)
class DataFrameSchema:
    id: str
    version: str
    columns: tuple[ColumnSpec, ...]
    allow_extra: bool = False


def _c(
    name: str,
    dtype: str = "object",
    aliases: Iterable[str] = (),
    default: Any = None,
    description: str = "",
) -> ColumnSpec:
    return ColumnSpec(name, dtype, tuple(aliases), default, description)


SEVIR_CATALOG_VIL = DataFrameSchema(
    "sevir.catalog.vil", "1.1.0", (
        _c("subject_key", "lower_string"),
        _c("event_id", "upper_string", ("id", "sevir_id")),
        _c("event_anchor_time_utc", "string", ("time_utc",)),
        _c("observation_time_utc", "string", description="Actual matched frame time in UTC; distinct from the event anchor"),
        _c("time_match_kind", "string", description="exact or nearest; empty for anchor-only searches"),
        _c("time_delta_seconds", "Float64", description="Actual frame time minus requested time in seconds"),
        _c("event_type", "string", default="Unlabeled", description="NOAA hazard/event label; never a sensor channel"),
        _c("channel", "lower_string", ("img_type",), "vil", "SEVIR physical/product channel"),
        _c("south", "Float64", ("llcrnrlat",)),
        _c("west", "Float64", ("llcrnrlon",)),
        _c("north", "Float64", ("urcrnrlat",)),
        _c("east", "Float64", ("urcrnrlon",)),
        _c("distance_km", "Float64", ("distance_to_footprint_km",)),
        _c("min_vil", "Float64", ("data_min",), description="catalog-reported minimum encoded VIL"),
        _c("max_vil", "Float64", ("data_max", "peak_vil"), description="catalog-reported maximum encoded VIL"),
        _c("missing_fraction", "fraction", ("pct_missing",)),
        _c("observation_status", "string"),
        _c("history_span_minutes", "Float64"),
        _c("history_expected_frames", "Int64"),
        _c("history_available_frames", "Int64"),
        _c("history_is_continuous", "boolean"),
    ),
)


SEVIR_ANALOGUE = DataFrameSchema(
    "sevir.analogue.result", "1.0.0", (
        _c("rank", "Int64"), _c("subject_key", "lower_string"),
        _c("event_id", "upper_string", ("id",)),
        _c("event_anchor_time_utc", "string", ("time_utc",)),
        _c("event_type", "string", default="Unlabeled"),
        _c("channel", "lower_string", ("img_type",), "vil"),
        _c("distance", "Float64", ("similarity_distance",)),
        _c("distance_squared", "Float64"),
        _c("stage1_recall_distance", "Float64"),
        _c("matched_feature_count", "Int64"),
        _c("feature_contributions"), _c("group_contributions"),
        _c("largest_differences"), _c("geographic_distance_km", "Float64"),
        _c("context"), _c("quality"), _c("signature_id", "string"),
    ),
)

GEO_PLACE_CANDIDATE = DataFrameSchema(
    "geospatial.place.candidate", "1.0.0", (
        _c("name", "string", ("display_name",)),
        _c("latitude", "Float64"), _c("longitude", "Float64"),
        _c("category", "string"), _c("place_type", "string", ("type",)),
    ),
)

KNOWLEDGE_SEARCH = DataFrameSchema(
    "knowledge.search.result", "2.0.0", (
        _c("rank", "Int64"), _c("chunk_id", "Int64"), _c("doc_id", "string"),
        _c("revision", "Int64"), _c("title", "string"), _c("version", "string"),
        _c("tools"), _c("text", "string"), _c("section", "string"), _c("locator"),
        _c("source_url", "string"), _c("citation", "string"), _c("url", "string"),
        _c("dense_score", "Float64"), _c("bm25_score", "Float64"),
        _c("rrf_score", "Float64"), _c("rerank_score", "Float64"),
    ),
)

CASCAST_AVAILABILITY = DataFrameSchema(
    "cascast.forecast.availability", "1.0.0", (
        _c("subject_key", "lower_string"), _c("upload_id", "string"),
        _c("horizon_hours", "Float64"), _c("ensemble_members", "Int64"),
        _c("ground_truth_available", "boolean"),
        _c("cache_status", "string"), _c("source", "string"),
        _c("action_required", "string"),
    ),
)


SCHEMAS = {
    schema.id: schema for schema in (
        SEVIR_CATALOG_VIL, SEVIR_ANALOGUE,
        GEO_PLACE_CANDIDATE,
        KNOWLEDGE_SEARCH, CASCAST_AVAILABILITY,
    )
}


def validate_schema_registry(
    schemas: Mapping[str, DataFrameSchema] = SCHEMAS,
) -> None:
    """Reject ambiguous or internally inconsistent schema registrations."""

    for registry_id, schema in schemas.items():
        if registry_id != schema.id:
            raise DataFrameSchemaError(
                f"schema registry key {registry_id!r} does not match id {schema.id!r}"
            )
        if not schema.version:
            raise DataFrameSchemaError(f"{schema.id}: schema version is required")
        owners: dict[str, str] = {}
        for spec in schema.columns:
            for name in (spec.name, *spec.aliases):
                owner = owners.get(name)
                if owner is not None:
                    raise DataFrameSchemaError(
                        f"{schema.id}@{schema.version}: column or alias {name!r} "
                        f"is assigned to both {owner!r} and {spec.name!r}"
                    )
                owners[name] = spec.name


validate_schema_registry()

PUBLIC_DATAFRAME_SCHEMAS = {
    "sevir.search": SEVIR_CATALOG_VIL.id,
    "sevir.find_analogues": SEVIR_ANALOGUE.id,
    "geospatial.resolve_place.candidates": GEO_PLACE_CANDIDATE.id,
    "knowledge.search": KNOWLEDGE_SEARCH.id,
    "cascast.available_forecasts": CASCAST_AVAILABILITY.id,
}


def _missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def _normalise_scalar(value: Any, dtype: str) -> Any:
    if _missing(value):
        return pd.NA
    if dtype in {"string", "lower_string", "upper_string"}:
        text = str(value).strip()
        if dtype == "lower_string":
            return text.lower()
        if dtype == "upper_string":
            return text.upper()
        return text
    if dtype in {"Float64", "fraction"}:
        number = float(value)
        if dtype == "fraction" and number > 1.0:
            number /= 100.0
        if dtype == "fraction" and not 0.0 <= number <= 1.0:
            raise DataFrameSchemaError(
                f"fraction value must be between 0 and 1 (or 0 and 100 percent), got {value!r}"
            )
        return number
    if dtype == "Int64":
        return int(value)
    if dtype == "boolean":
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
        return bool(value)
    return value


def _normalise_series(series: pd.Series, spec: ColumnSpec) -> pd.Series:
    result = series.map(lambda value: _normalise_scalar(value, spec.dtype))
    if spec.dtype in {"string", "lower_string", "upper_string"}:
        return result.astype("string")
    if spec.dtype == "Float64" or spec.dtype == "fraction":
        return pd.to_numeric(result, errors="coerce").astype("Float64")
    if spec.dtype == "Int64":
        return pd.to_numeric(result, errors="coerce").astype("Int64")
    if spec.dtype == "boolean":
        return result.astype("boolean")
    return result.astype("object")


def _series_conflicts(left: pd.Series, right: pd.Series) -> bool:
    if len(left) != len(right):
        return True
    for first, second in zip(left.tolist(), right.tolist()):
        if _missing(first) or _missing(second):
            continue
        try:
            if first != second:
                return True
        except (TypeError, ValueError):
            if repr(first) != repr(second):
                return True
    return False


def normalize_dataframe(
    value: pd.DataFrame | Iterable[Mapping[str, Any]],
    schema: str | DataFrameSchema,
    *,
    attrs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return a canonical, typed table and reject conflicting aliases."""

    contract = SCHEMAS[schema] if isinstance(schema, str) else schema
    frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(list(value))
    inherited_attrs = dict(getattr(value, "attrs", {}))
    output: dict[str, pd.Series] = {}
    consumed_aliases: list[str] = []
    known = {item.name for item in contract.columns}
    known.update(alias for item in contract.columns for alias in item.aliases)
    for spec in contract.columns:
        candidates = [name for name in (spec.name, *spec.aliases) if name in frame.columns]
        if not candidates:
            fill = spec.default if spec.default is not None else pd.NA
            output[spec.name] = _normalise_series(
                pd.Series([fill] * len(frame), index=frame.index), spec
            )
            continue
        normalized = [_normalise_series(frame[name], spec) for name in candidates]
        chosen = normalized[0]
        for name, candidate in zip(candidates[1:], normalized[1:]):
            if _series_conflicts(chosen, candidate):
                raise DataFrameSchemaError(
                    f"{contract.id}@{contract.version}: conflicting columns "
                    f"{candidates[0]!r} and {name!r} for {spec.name!r}"
                )
            chosen = chosen.combine_first(candidate)
        output[spec.name] = chosen
        consumed_aliases.extend(name for name in candidates if name != spec.name)
    if contract.allow_extra:
        for name in frame.columns:
            if name not in known:
                output[str(name)] = frame[name].copy()
    result = pd.DataFrame(output, index=frame.index)
    result.attrs.update(inherited_attrs)
    result.attrs.update(dict(attrs or {}))
    result.attrs.update({
        "dataframe_contract_version": DATAFRAME_CONTRACT_VERSION,
        "dataframe_schema_id": contract.id,
        "dataframe_schema_version": contract.version,
        "legacy_aliases_consumed": sorted(set(consumed_aliases)),
        "column_semantics": {
            item.name: item.description
            for item in contract.columns if item.description
        },
    })
    return result


def dataframe_schema_metadata(value: Any) -> dict[str, Any]:
    """Return public schema identity for diagnostics and worker summaries."""

    attrs = getattr(value, "attrs", {})
    if not isinstance(attrs, Mapping):
        return {}
    return {
        key: attrs[key]
        for key in (
            "dataframe_contract_version", "dataframe_schema_id",
            "dataframe_schema_version",
        )
        if attrs.get(key)
    }
