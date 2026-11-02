"""Parse external execution inputs and select their request-bound assets."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..agent.contracts import DATA_CONTRACT_VERSION, DATA_ID_FIELDS, ExecutionRequest
from ..capabilities.base import RequestBinding
from ..capabilities.registry import select_capabilities


def data_descs_from_payload(
    payload: dict[str, Any] | None,
) -> list[dict[str, Any]] | tuple[dict[str, Any], ...]:
    """Extract descriptions; normalization happens once when building the request."""

    bound = payload or {}
    if bound.get("data_descs"):
        return bound["data_descs"]
    descriptions: list[dict[str, Any]] = []
    upload_id = bound.get("sequence_upload_id") or bound.get("upload_id")
    if upload_id:
        descriptions.append({"source": "upload", "upload_id": upload_id})
    if bound.get("event_id"):
        event = {"source": "sevir", "event_id": bound["event_id"]}
        if bound.get("sevir_subject_key"):
            event["subject_key"] = bound["sevir_subject_key"]
        descriptions.append(event)
    return descriptions


def _normalize_data_descs(
    descriptions: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> tuple[dict[str, Any], ...]:
    if descriptions is None:
        return ()
    if not isinstance(descriptions, (list, tuple)):
        raise ValueError("data_descs must be a list")
    normalized: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(descriptions):
        if not isinstance(raw, dict):
            raise ValueError(f"data_descs[{index}] must be an object")
        item = dict(raw)
        source = str(item.get("source") or "").strip().lower()
        if source not in DATA_ID_FIELDS:
            raise ValueError(f"data_descs[{index}].source must be one of {', '.join(DATA_ID_FIELDS)}")
        item["source"] = source
        key = DATA_ID_FIELDS[source]
        identity = str(item.get(key) or "").strip()
        if not identity:
            raise ValueError(f"data_descs[{index}] requires {key}")
        item[key] = identity if source == "upload" else identity.upper()
        if source == "sevir" and item.get("subject_key"):
            item["subject_key"] = str(item["subject_key"]).strip().lower()
        item.setdefault("contract_version", DATA_CONTRACT_VERSION)
        record_id = str(item.get("subject_key") or item[key]) if source == "sevir" else item[key]
        identity_key = (source, record_id)
        previous = normalized.get(identity_key)
        if previous is not None and previous != item:
            raise ValueError(f"conflicting data_descs for {source} identity {identity_key[1]!r}")
        normalized[identity_key] = item
    # Uploaded data is the primary analysis input; catalog records supplement it.
    return tuple(sorted(normalized.values(), key=lambda item: item["source"] != "upload"))


def _bind_described_assets(
    descriptions: tuple[dict[str, Any], ...], binding: RequestBinding,
) -> RequestBinding:
    uploads = [item for item in descriptions if item["source"] == "upload"]
    events = [item for item in descriptions if item["source"] == "sevir"]
    updates: dict[str, str] = {}
    if uploads and not binding.upload_id:
        updates["upload_id"] = uploads[0]["upload_id"]
    matching = [
        item for item in events
        if (not binding.event_id or item["event_id"] == binding.event_id)
        and (not binding.sevir_subject_key or not item.get("subject_key")
             or item["subject_key"] == binding.sevir_subject_key)
    ]
    if matching:
        selected = matching[0]
        if not binding.event_id:
            updates["event_id"] = selected["event_id"]
        if not binding.sevir_subject_key and selected.get("subject_key"):
            updates["sevir_subject_key"] = selected["subject_key"]
    return replace(binding, **updates) if updates else binding


def build_execution_request(
    data_descs: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    selected_tools: list[str] | tuple[str, ...] | None,
    binding: dict[str, Any] | RequestBinding | None = None,
    evidence_state: dict[str, Any] | None = None,
    *,
    generate_visual: bool = False,
    explicit_inputs: dict[str, Any] | None = None,
    conversation_context: dict[str, Any] | None = None,
) -> ExecutionRequest:
    """Normalize caller input once, authorize tools and bind a coherent identity."""

    descriptions = _normalize_data_descs(data_descs)
    tools = select_capabilities(selected_tools)
    bound = binding if isinstance(binding, RequestBinding) else RequestBinding.from_mapping(binding)
    return ExecutionRequest(
        data_descs=descriptions,
        selected_tools=tools,
        binding=_bind_described_assets(descriptions, bound),
        explicit_inputs={} if explicit_inputs is None else dict(explicit_inputs),
        generate_visual=generate_visual,
        evidence_state={} if evidence_state is None else dict(evidence_state),
        conversation_context={} if conversation_context is None else dict(conversation_context),
    )
