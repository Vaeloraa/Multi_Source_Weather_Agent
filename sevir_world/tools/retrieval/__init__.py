"""SEVIR-specific, past-only historical event retrieval shared by SEVIR tools."""

from .contracts import (
    ANALOGUE_INDEX_SCHEMA_VERSION,
    ANALOGUE_RESULT_SCHEMA_VERSION,
    EVENT_SIGNATURE_SCHEMA_VERSION,
    AnalogueContractError,
    AnalogueQuery,
    AnalogueResult,
    EventSignature,
)
from .index import AnalogueIndex, build_local_sevir_index
from .retriever import retrieve_analogues
from .signature import FEATURE_GROUPS, FEATURE_NAMES, extract_event_signature

__all__ = [
    "ANALOGUE_INDEX_SCHEMA_VERSION",
    "ANALOGUE_RESULT_SCHEMA_VERSION",
    "EVENT_SIGNATURE_SCHEMA_VERSION",
    "AnalogueContractError",
    "AnalogueIndex",
    "AnalogueQuery",
    "AnalogueResult",
    "EventSignature",
    "FEATURE_GROUPS",
    "FEATURE_NAMES",
    "build_local_sevir_index",
    "extract_event_signature",
    "retrieve_analogues",
]
