from __future__ import annotations

"""Serializable contracts for the historical analogue pipeline."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence


EVENT_SIGNATURE_SCHEMA_VERSION = "sevir-event-signature-v1"
ANALOGUE_INDEX_SCHEMA_VERSION = "sevir-analogue-index-v1"
ANALOGUE_QUERY_SCHEMA_VERSION = "sevir-analogue-query-v1"
ANALOGUE_RESULT_SCHEMA_VERSION = "sevir-analogue-result-v1"


class AnalogueContractError(ValueError):
    """Raised when a retrieval value violates a versioned contract."""


def _text(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise AnalogueContractError(f"{name} must be non-empty")
    return result


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EventSignature:
    """One deterministic, future-safe representation of an observed event."""

    subject_key: str
    event_id: str
    feature_names: tuple[str, ...]
    features: tuple[float | None, ...]
    feature_units: tuple[str, ...]
    feature_groups: tuple[str, ...]
    frame_count: int
    window_start_minute: int
    window_end_minute: int
    threshold: float
    source: str
    source_fingerprint: Mapping[str, Any]
    event_anchor_time_utc: str = ""
    event_type: str = "Unlabeled"
    context: Mapping[str, Any] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)
    extraction: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EVENT_SIGNATURE_SCHEMA_VERSION
    signature_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != EVENT_SIGNATURE_SCHEMA_VERSION:
            raise AnalogueContractError(
                f"unsupported signature schema {self.schema_version!r}"
            )
        for name in ("subject_key", "event_id", "source"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        names = tuple(str(item) for item in self.feature_names)
        values = tuple(None if item is None else float(item) for item in self.features)
        units = tuple(str(item) for item in self.feature_units)
        groups = tuple(str(item) for item in self.feature_groups)
        if not names or len({*names}) != len(names):
            raise AnalogueContractError("feature_names must be non-empty and unique")
        if not (len(names) == len(values) == len(units) == len(groups)):
            raise AnalogueContractError("signature feature fields have different lengths")
        if int(self.frame_count) < 2:
            raise AnalogueContractError("signature requires at least two past frames")
        if int(self.window_end_minute) > 0:
            raise AnalogueContractError("signature window must not include future frames")
        if not self.source_fingerprint:
            raise AnalogueContractError("source_fingerprint is required")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "features", values)
        object.__setattr__(self, "feature_units", units)
        object.__setattr__(self, "feature_groups", groups)
        object.__setattr__(self, "context", _clean(self.context))
        object.__setattr__(self, "quality", _clean(self.quality))
        object.__setattr__(self, "extraction", _clean(self.extraction))
        object.__setattr__(self, "source_fingerprint", _clean(self.source_fingerprint))
        if not self.signature_id:
            object.__setattr__(
                self,
                "signature_id",
                _digest(self.to_dict(include_signature_id=False)),
            )

    def to_dict(self, *, include_signature_id: bool = True) -> dict[str, Any]:
        value = _clean(asdict(self))
        if not include_signature_id:
            value.pop("signature_id", None)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EventSignature":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class AnalogueQuery:
    """Explicit retrieval scope and weighting for one signature."""

    reference: EventSignature
    top_k: int = 5
    recall_count: int = 40
    candidate_subject_keys: tuple[str, ...] = ()
    exclude_same_event: bool = True
    same_event_type: bool = False
    max_month_distance: int | None = None
    max_geographic_km: float | None = None
    feature_weights: Mapping[str, float] = field(default_factory=dict)
    schema_version: str = ANALOGUE_QUERY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ANALOGUE_QUERY_SCHEMA_VERSION:
            raise AnalogueContractError(
                f"unsupported query schema {self.schema_version!r}"
            )
        if not 1 <= int(self.top_k) <= 100:
            raise AnalogueContractError("top_k must be in 1..100")
        if int(self.recall_count) < int(self.top_k):
            raise AnalogueContractError("recall_count must be at least top_k")
        if self.max_month_distance is not None and not 0 <= int(self.max_month_distance) <= 6:
            raise AnalogueContractError("max_month_distance must be in 0..6")
        if self.max_geographic_km is not None and float(self.max_geographic_km) <= 0:
            raise AnalogueContractError("max_geographic_km must be positive")
        object.__setattr__(self, "candidate_subject_keys", tuple(dict.fromkeys(
            str(item).strip().lower()
            for item in self.candidate_subject_keys
            if str(item).strip()
        )))
        weights = {str(key): float(item) for key, item in self.feature_weights.items()}
        if any(item < 0 for item in weights.values()):
            raise AnalogueContractError("feature weights must be non-negative")
        object.__setattr__(self, "feature_weights", weights)

    def to_dict(self) -> dict[str, Any]:
        value = _clean(asdict(self))
        value["reference"] = self.reference.to_dict()
        return value


@dataclass(frozen=True, slots=True)
class AnalogueResult:
    """Ranked candidates plus scope, provenance and explanations."""

    query_signature_id: str
    index_id: str
    records: tuple[Mapping[str, Any], ...]
    candidate_scope: Mapping[str, Any]
    distance_definition: Mapping[str, Any]
    provenance: tuple[Mapping[str, Any], ...]
    elapsed_ms: float
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    schema_version: str = ANALOGUE_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ANALOGUE_RESULT_SCHEMA_VERSION:
            raise AnalogueContractError(
                f"unsupported result schema {self.schema_version!r}"
            )
        for name in ("query_signature_id", "index_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "records", tuple(_clean(item) for item in self.records))
        object.__setattr__(self, "candidate_scope", _clean(self.candidate_scope))
        object.__setattr__(self, "distance_definition", _clean(self.distance_definition))
        object.__setattr__(self, "provenance", tuple(_clean(item) for item in self.provenance))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "limitations", tuple(str(item) for item in self.limitations))

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AnalogueResult":
        raw = dict(value)
        raw["records"] = tuple(raw.get("records", ()))
        raw["provenance"] = tuple(raw.get("provenance", ()))
        raw["warnings"] = tuple(raw.get("warnings", ()))
        raw["limitations"] = tuple(raw.get("limitations", ()))
        return cls(**raw)


__all__ = [
    "ANALOGUE_INDEX_SCHEMA_VERSION",
    "ANALOGUE_QUERY_SCHEMA_VERSION",
    "ANALOGUE_RESULT_SCHEMA_VERSION",
    "EVENT_SIGNATURE_SCHEMA_VERSION",
    "AnalogueContractError",
    "AnalogueQuery",
    "AnalogueResult",
    "EventSignature",
]
