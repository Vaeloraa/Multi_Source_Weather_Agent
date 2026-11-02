from __future__ import annotations

"""Serializable contracts shared by deterministic analytics operators."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence


ANALYSIS_REQUEST_SCHEMA_VERSION = "analysis-request-v1"
ANALYSIS_RESULT_SCHEMA_VERSION = "analysis-result-v1"

REPRESENTATIONS = {
    "grid_cube",
    "point_events",
    "object_tracks",
    "ensemble_cube",
    "table",
}
RESULT_STATUSES = {"ok", "partial", "error"}


class AnalysisContractError(ValueError):
    """An analytics request or result violates the scientific contract."""


def _text(value: Any, field_name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise AnalysisContractError(f"{field_name} must be non-empty")
    return result


def _tuple_text(values: Sequence[Any] | None) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in (values or ()) if str(item).strip())


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): item for key, item in dict(value or {}).items()}


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, set):
        return sorted(_canonical(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class DataRef:
    """Small identity/provenance record; large arrays stay behind data_handle."""

    subject_key: str
    source: str
    modality: str
    variable: str
    profile_id: str
    representation: str
    units: str
    data_handle: str
    dimensions: tuple[str, ...] = ()
    coordinates: Mapping[str, Any] = field(default_factory=dict)
    time_start_utc: str = ""
    time_end_utc: str = ""
    spatial_metadata: Mapping[str, Any] = field(default_factory=dict)
    missing_value: float | int | None = None
    quality_flags: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "subject_key", "source", "modality", "variable", "profile_id",
            "representation", "units", "data_handle",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.representation not in REPRESENTATIONS:
            raise AnalysisContractError(
                f"unsupported representation {self.representation!r}"
            )
        object.__setattr__(self, "dimensions", _tuple_text(self.dimensions))
        object.__setattr__(self, "quality_flags", _tuple_text(self.quality_flags))
        object.__setattr__(self, "coordinates", _mapping(self.coordinates))
        object.__setattr__(self, "spatial_metadata", _mapping(self.spatial_metadata))
        object.__setattr__(self, "provenance", _mapping(self.provenance))
        if not self.provenance:
            raise AnalysisContractError("DataRef.provenance must identify the source")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DataRef":
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return _canonical(asdict(self))


@dataclass(frozen=True, slots=True)
class VariableProfile:
    profile_id: str
    version: str
    canonical_representation: str
    measure_type: str
    units: str
    allowed_operations: tuple[str, ...]
    required_normalization: tuple[str, ...] = ()
    semantic_limits: tuple[str, ...] = ()
    valid_range: tuple[float, float] | None = None
    missing_policy: str = "source_specific"
    time_semantics: str = "UTC"
    spatial_semantics: str = "source_defined"

    def __post_init__(self) -> None:
        for name in (
            "profile_id", "version", "canonical_representation",
            "measure_type", "units", "missing_policy", "time_semantics",
            "spatial_semantics",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.canonical_representation not in REPRESENTATIONS:
            raise AnalysisContractError(
                f"unsupported profile representation {self.canonical_representation!r}"
            )
        operations = _tuple_text(self.allowed_operations)
        if not operations:
            raise AnalysisContractError("VariableProfile.allowed_operations is empty")
        object.__setattr__(self, "allowed_operations", operations)
        object.__setattr__(
            self, "required_normalization", _tuple_text(self.required_normalization)
        )
        object.__setattr__(self, "semantic_limits", _tuple_text(self.semantic_limits))

    def to_dict(self) -> dict[str, Any]:
        return _canonical(asdict(self))


@dataclass(frozen=True, slots=True)
class CohortSpec:
    name: str
    region: Mapping[str, Any] = field(default_factory=dict)
    months: tuple[int, ...] = ()
    years: tuple[int, ...] = ()
    modalities: tuple[str, ...] = ()
    time_basis: str = "UTC"
    required_coverage: float = 0.0
    normalization: tuple[str, ...] = ()
    filters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "cohort.name"))
        object.__setattr__(self, "region", _mapping(self.region))
        object.__setattr__(self, "months", tuple(int(item) for item in self.months))
        object.__setattr__(self, "years", tuple(int(item) for item in self.years))
        object.__setattr__(self, "modalities", _tuple_text(self.modalities))
        object.__setattr__(self, "normalization", _tuple_text(self.normalization))
        object.__setattr__(self, "filters", _mapping(self.filters))
        if any(item < 1 or item > 12 for item in self.months):
            raise AnalysisContractError("cohort months must be in 1..12")
        if not 0.0 <= float(self.required_coverage) <= 1.0:
            raise AnalysisContractError("required_coverage must be in [0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CohortSpec":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    op: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    output_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", _text(self.op, "operator.op"))
        object.__setattr__(self, "parameters", _mapping(self.parameters))
        object.__setattr__(
            self, "output_name", str(self.output_name or self.op).strip()
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OperatorSpec":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    inputs: tuple[DataRef, ...]
    operators: tuple[OperatorSpec, ...]
    cohorts: tuple[CohortSpec, ...] = ()
    requested_outputs: tuple[str, ...] = ("values", "uncertainty", "provenance")
    failure_policy: str = "partial"
    schema_version: str = ANALYSIS_REQUEST_SCHEMA_VERSION
    request_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != ANALYSIS_REQUEST_SCHEMA_VERSION:
            raise AnalysisContractError(
                f"unsupported analysis request schema {self.schema_version!r}"
            )
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "operators", tuple(self.operators))
        object.__setattr__(self, "cohorts", tuple(self.cohorts))
        object.__setattr__(
            self, "requested_outputs", _tuple_text(self.requested_outputs)
        )
        object.__setattr__(
            self, "failure_policy", _text(self.failure_policy, "failure_policy")
        )
        if not self.inputs:
            raise AnalysisContractError("AnalysisRequest.inputs is empty")
        if not self.operators:
            raise AnalysisContractError("AnalysisRequest.operators is empty")
        handles = [item.data_handle for item in self.inputs]
        if len(handles) != len(set(handles)):
            raise AnalysisContractError("AnalysisRequest input handles must be unique")
        if not self.request_id:
            canonical = self.to_dict(include_request_id=False)
            encoded = json.dumps(
                canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            object.__setattr__(self, "request_id", hashlib.sha256(encoded).hexdigest())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AnalysisRequest":
        raw = dict(value)
        raw["inputs"] = tuple(
            item if isinstance(item, DataRef) else DataRef.from_mapping(item)
            for item in raw.get("inputs", ())
        )
        raw["operators"] = tuple(
            item if isinstance(item, OperatorSpec) else OperatorSpec.from_mapping(item)
            for item in raw.get("operators", ())
        )
        raw["cohorts"] = tuple(
            item if isinstance(item, CohortSpec) else CohortSpec.from_mapping(item)
            for item in raw.get("cohorts", ())
        )
        return cls(**raw)

    def to_dict(self, *, include_request_id: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "inputs": [item.to_dict() for item in self.inputs],
            "cohorts": [_canonical(asdict(item)) for item in self.cohorts],
            "operators": [_canonical(asdict(item)) for item in self.operators],
            "requested_outputs": list(self.requested_outputs),
            "failure_policy": self.failure_policy,
        }
        if include_request_id:
            value["request_id"] = self.request_id
        return value


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    status: str
    method: str
    request_id: str
    parameters: Mapping[str, Any]
    sample: Mapping[str, Any]
    values: Mapping[str, Any]
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    provenance: tuple[Mapping[str, Any], ...] = ()
    artifacts: tuple[Mapping[str, Any], ...] = ()
    limitations: tuple[str, ...] = ()
    failed_work: tuple[str, ...] = ()
    schema_version: str = ANALYSIS_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ANALYSIS_RESULT_SCHEMA_VERSION:
            raise AnalysisContractError(
                f"unsupported analysis result schema {self.schema_version!r}"
            )
        status = str(self.status).strip().lower()
        if status not in RESULT_STATUSES:
            raise AnalysisContractError(f"unsupported result status {status!r}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "method", _text(self.method, "method"))
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(self, "parameters", _mapping(self.parameters))
        object.__setattr__(self, "sample", _mapping(self.sample))
        object.__setattr__(self, "values", _mapping(self.values))
        object.__setattr__(self, "uncertainty", _mapping(self.uncertainty))
        object.__setattr__(self, "warnings", _tuple_text(self.warnings))
        object.__setattr__(self, "limitations", _tuple_text(self.limitations))
        object.__setattr__(self, "failed_work", _tuple_text(self.failed_work))
        object.__setattr__(self, "provenance", tuple(_mapping(item) for item in self.provenance))
        object.__setattr__(self, "artifacts", tuple(_mapping(item) for item in self.artifacts))
        for key in ("input_count", "valid_count", "excluded_count"):
            if key not in self.sample:
                raise AnalysisContractError(f"AnalysisResult.sample lacks {key}")
            if int(self.sample[key]) < 0:
                raise AnalysisContractError(f"AnalysisResult.sample.{key} is negative")
        if int(self.sample["valid_count"]) + int(self.sample["excluded_count"]) > int(
            self.sample["input_count"]
        ):
            raise AnalysisContractError("valid + excluded exceeds input_count")
        if status in {"ok", "partial"} and not self.provenance:
            raise AnalysisContractError("successful results require provenance")
        if status == "partial" and not (self.warnings or self.failed_work or self.limitations):
            raise AnalysisContractError("partial results must explain the incomplete work")

    def to_dict(self) -> dict[str, Any]:
        return _canonical(asdict(self))
