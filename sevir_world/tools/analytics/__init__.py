"""Agent-facing deterministic scientific analytics capability."""

from .contracts import (
    ANALYSIS_REQUEST_SCHEMA_VERSION,
    ANALYSIS_RESULT_SCHEMA_VERSION,
    AnalysisContractError,
    AnalysisRequest,
    AnalysisResult,
    CohortSpec,
    DataRef,
    OperatorSpec,
    VariableProfile,
)
from .executor import execute_analysis
from .schema import DEFAULT_PROFILE_REGISTRY, ProfileRegistry
from .public import ANALYTICS_CAPABILITY_SCHEMA_VERSION, Analytics

__all__ = [
    "ANALYSIS_REQUEST_SCHEMA_VERSION",
    "ANALYSIS_RESULT_SCHEMA_VERSION",
    "ANALYTICS_CAPABILITY_SCHEMA_VERSION",
    "Analytics",
    "AnalysisContractError",
    "AnalysisRequest",
    "AnalysisResult",
    "CohortSpec",
    "DEFAULT_PROFILE_REGISTRY",
    "DataRef",
    "OperatorSpec",
    "ProfileRegistry",
    "VariableProfile",
    "execute_analysis",
]
