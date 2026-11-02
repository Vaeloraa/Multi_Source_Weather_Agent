from __future__ import annotations

"""Versioned variable profiles used to reject invalid analytics requests."""

from dataclasses import dataclass
from typing import Iterable

from .contracts import AnalysisContractError, VariableProfile


@dataclass(frozen=True)
class ProfileRegistry:
    profiles: dict[str, VariableProfile]

    def get(self, profile_id: str) -> VariableProfile:
        key = str(profile_id).strip().lower()
        try:
            return self.profiles[key]
        except KeyError as error:
            raise AnalysisContractError(f"unknown VariableProfile {profile_id!r}") from error

    def operation_allowed(self, profile_id: str, operation: str) -> bool:
        profile = self.get(profile_id)
        candidate = str(operation).strip().lower()
        return any(
            candidate == allowed
            or (allowed.endswith(".*") and candidate.startswith(allowed[:-1]))
            for allowed in profile.allowed_operations
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "profile_count": len(self.profiles),
            "profiles": [self.profiles[key].to_dict() for key in sorted(self.profiles)],
        }


def _profile(
    profile_id: str,
    representation: str,
    measure_type: str,
    units: str,
    operations: Iterable[str],
    *,
    normalization: Iterable[str] = (),
    limits: Iterable[str] = (),
) -> VariableProfile:
    return VariableProfile(
        profile_id=profile_id,
        version="1.0",
        canonical_representation=representation,
        measure_type=measure_type,
        units=units,
        allowed_operations=tuple(operations),
        required_normalization=tuple(normalization),
        semantic_limits=tuple(limits),
    )


_COMMON_GRID = (
    "quality.inspect",
    "statistics.describe",
    "temporal.trend",
    "spatial.threshold",
    "objects.describe",
)
_VERIFICATION_GRID = (*_COMMON_GRID, "verification.continuous", "verification.threshold")

DEFAULT_PROFILE_REGISTRY = ProfileRegistry({
    "sevir.vil": _profile(
        "sevir.vil", "grid_cube", "continuous_encoded_field", "encoded_vil",
        (*_VERIFICATION_GRID, "analogue.signature", "retrieval.analogue"),
        limits=(
            "encoded VIL is not millimetres of rainfall",
            "research thresholds are not official warning levels",
        ),
    ),
    "sevir.vis": _profile(
        "sevir.vis", "grid_cube", "continuous_imagery", "source_encoded",
        _COMMON_GRID,
        limits=("visible imagery is unavailable at night",),
    ),
    "sevir.ir069": _profile(
        "sevir.ir069", "grid_cube", "continuous_imagery", "source_encoded",
        _COMMON_GRID,
        limits=("water-vapour imagery does not directly measure precipitation",),
    ),
    "sevir.ir107": _profile(
        "sevir.ir107", "grid_cube", "brightness_temperature", "source_encoded",
        _COMMON_GRID,
        limits=("cold cloud tops do not prove a surface hazard",),
    ),
    "sevir.lght": _profile(
        "sevir.lght", "point_events", "event_record_or_count", "source_defined",
        ("quality.inspect", "statistics.describe"),
        normalization=("observation_duration", "valid_area"),
        limits=("event/group/flash record levels must not be mixed",),
    ),
    "cascast.vil_ensemble": _profile(
        "cascast.vil_ensemble", "ensemble_cube", "ensemble_forecast", "encoded_vil",
        _VERIFICATION_GRID,
        limits=("ensemble agreement is not forecast accuracy",),
    ),
    "aurora.surface": _profile(
        "aurora.surface", "grid_cube", "model_forecast", "variable_specific",
        ("quality.inspect", "statistics.describe", "verification.continuous"),
        limits=("large-scale fields must not be equated with local VIL",),
    ),
    "sevir.analogue_index": _profile(
        "sevir.analogue_index", "table", "versioned_signature_index", "feature_specific",
        ("retrieval.analogue",),
        limits=("the local index is not the complete SEVIR catalog",),
    ),
})
