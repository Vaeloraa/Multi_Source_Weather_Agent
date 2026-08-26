from __future__ import annotations

"""Durable structured evidence shared by consecutive Agent requests.

The ledger stores request provenance, real tool observations and Agent-derived
calculation results. Natural-language answers are deliberately excluded: they
may explain evidence, but they are not evidence themselves.
"""

import json
import re
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = "study-evidence-v1"
_FACT_FIELDS = (
    "requested_time_utc",
    "aurora_initialization_time_utc",
    "event_id",
    "sevir_subject_key",
    "upload_id",
    "sequence_upload_id",
    "latitude",
    "longitude",
    "region",
    "aurora_region",
    "forecast_leads_hours",
    "history_before_hours",
    "history_after_hours",
    "data_policy",
)
_DECISION_FIELDS = (
    "aurora_region",
    "focus_window",
    "candidate_subjects",
    "candidate_event_ids",
    "candidate_subject_keys",
    "selected_subject",
    "selected_event_id",
    "selected_subject_key",
    "target_time_utc",
    "target_region",
    "aurora_initialization_time_utc",
    "cascast_forecast_status",
    "cascast_forecast_error",
    "cascast_forecast_subject_key",
    "cascast_forecast_horizon_hours",
    "cascast_forecast_evidence",
)

_FOLLOW_UP_PATTERN = re.compile(
    r"(?:刚才|上述|上一步|前面|继续|接着|该事件|这个事件|所选事件|候选里|"
    r"目标时刻|对应时段|对应时间|这个区域|该区域|这张图|这些结果|"
    r"这周围|周围|附近|周边|周边区域|"
    r"follow[ -]?up|previous|above)",
    flags=re.IGNORECASE,
)


def _bounded(value: Any, depth: int = 0) -> Any:
    """Keep evidence calculable while excluding raw raster payloads."""

    if depth > 7:
        return {"type": type(value).__name__, "omitted": True}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:4_000]
    if isinstance(value, dict):
        return {
            str(key): _bounded(item, depth + 1)
            for key, item in list(value.items())[:80]
            if key not in {"traceback", "code", "raw_data", "values", "frames"}
        }
    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) > 40:
            items = [*items[:30], {"omitted_items": len(items) - 31}, items[-1]]
        return [_bounded(item, depth + 1) for item in items]
    return str(value)[:1_000]


def normalize_study_update(value: Any) -> dict[str, Any]:
    """Validate one Agent-published durable case decision."""

    if not isinstance(value, dict):
        return {}
    update = {
        key: _bounded(value[key])
        for key in _DECISION_FIELDS
        if key in value and value[key] not in (None, "", [], {})
    }
    selected = str(update.get("selected_event_id") or "").strip().upper()
    if selected:
        update["selected_event_id"] = selected
    candidates = update.get("candidate_event_ids")
    if isinstance(candidates, list):
        update["candidate_event_ids"] = list(dict.fromkeys(
            str(item).strip().upper() for item in candidates if str(item).strip()
        ))
    subject_keys = update.get("candidate_subject_keys")
    if isinstance(subject_keys, list):
        update["candidate_subject_keys"] = list(dict.fromkeys(
            str(item).strip().lower() for item in subject_keys if str(item).strip()
        ))
    if update.get("selected_subject_key"):
        update["selected_subject_key"] = str(update["selected_subject_key"]).strip().lower()
    candidate_subjects = update.get("candidate_subjects")
    if isinstance(candidate_subjects, list):
        normalized_subjects: list[dict[str, str]] = []
        seen_subjects: set[str] = set()
        for item in candidate_subjects:
            if not isinstance(item, dict):
                continue
            key = str(item.get("subject_key") or "").strip().lower()
            event = str(item.get("event_id") or "").strip().upper()
            if not key or key in seen_subjects:
                continue
            seen_subjects.add(key)
            normalized_subjects.append({"subject_key": key, "event_id": event})
        update["candidate_subjects"] = normalized_subjects
    selected_subject = update.get("selected_subject")
    if isinstance(selected_subject, dict):
        key = str(selected_subject.get("subject_key") or "").strip().lower()
        event = str(selected_subject.get("event_id") or "").strip().upper()
        update["selected_subject"] = (
            {"subject_key": key, "event_id": event} if key else None
        )
    elif update.get("selected_subject_key"):
        update["selected_subject"] = {
            "subject_key": update["selected_subject_key"],
            "event_id": str(update.get("selected_event_id") or "").upper(),
        }
    if update.get("selected_subject") is None:
        update.pop("selected_subject", None)
    return update


def _updates_from_result(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    updates = [
        normalize_study_update(value),
        normalize_study_update(value.get("study_update")),
    ]
    return [item for item in updates if item]


def build_turn_evidence(
    *,
    turn_id: str,
    request_id: str,
    question: str,
    payload: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any] | None:
    """Build one evidence record from real execution observations."""

    observations = [
        item for item in result.get("observations", []) if isinstance(item, dict)
    ]
    if not observations:
        return None
    tool_evidence: list[dict[str, Any]] = []
    derived_results: list[Any] = []
    updates: list[dict[str, Any]] = []
    for observation_index, observation in enumerate(observations, start=1):
        observed_result = observation.get("result")
        if observed_result is not None and observation.get("status") in {"ok", "partial"}:
            derived_results.append(_bounded(observed_result))
            updates.extend(_updates_from_result(observed_result))
        for call in observation.get("calls", []):
            if not isinstance(call, dict):
                continue
            api = str(call.get("api") or "")
            source, _, operation = api.partition(".")
            tool_evidence.append({
                "observation": observation_index,
                "source": source,
                "operation": operation,
                "status": call.get("status"),
                "cache_hit": bool(call.get("cache_hit", False)),
                "arguments": _bounded(call.get("arguments", {})),
                "return_type": call.get("return_type"),
                "evidence": _bounded(call.get("evidence")),
                "error": _bounded(call.get("error")),
            })
            if api == "sevir.search" and call.get("status") == "ok":
                evidence = call.get("evidence")
                events = None
                if isinstance(evidence, dict):
                    events = evidence.get("records")
                    if not isinstance(events, list):
                        events = evidence.get("events")
                if isinstance(events, list):
                    candidate_ids = [
                        str(item.get("event_id") or item.get("id") or "").strip().upper()
                        for item in events
                        if isinstance(item, dict)
                        and str(item.get("event_id") or item.get("id") or "").strip()
                    ]
                    if candidate_ids:
                        updates.append({"candidate_event_ids": candidate_ids})
                    candidate_keys = [
                        str(item.get("subject_key") or "").strip().lower()
                        for item in events
                        if isinstance(item, dict) and str(item.get("subject_key") or "").strip()
                    ]
                    if candidate_keys:
                        updates.append({"candidate_subject_keys": candidate_keys})
                    candidate_subjects = [
                        {
                            "subject_key": str(item.get("subject_key") or "").strip().lower(),
                            "event_id": str(item.get("event_id") or "").strip().upper(),
                        }
                        for item in events
                        if isinstance(item, dict) and item.get("subject_key")
                    ]
                    if candidate_subjects:
                        updates.append({"candidate_subjects": candidate_subjects})
            if api == "cascast.forecast":
                arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                subject_key = str(
                    payload.get("sevir_subject_key")
                    or payload.get("event_id")
                    or payload.get("upload_id")
                    or ""
                ).strip()
                if call.get("status") == "ok":
                    updates.append({
                        "cascast_forecast_status": "available",
                        "cascast_forecast_subject_key": subject_key,
                        "cascast_forecast_horizon_hours": arguments.get("horizon_hours", 1.0),
                        "cascast_forecast_evidence": call.get("evidence") or {
                            "cache_hit": bool(call.get("cache_hit", False))
                        },
                    })
                else:
                    updates.append({
                        "cascast_forecast_status": "failed",
                        "cascast_forecast_subject_key": subject_key,
                        "cascast_forecast_horizon_hours": arguments.get("horizon_hours", 1.0),
                        "cascast_forecast_error": call.get("error") or "cascast.forecast failed",
                    })
            if api == "aurora.regional_fields" and call.get("status") == "ok":
                arguments = call.get("arguments")
                observed_region = (
                    arguments.get("region") if isinstance(arguments, dict) else None
                )
                if isinstance(observed_region, dict) and all(
                    key in observed_region for key in ("south", "west", "north", "east")
                ):
                    # Keep the verified Aurora evidence domain distinct from
                    # an Agent-selected local target region.
                    updates.append({"aurora_region": observed_region})
    facts = {
        key: _bounded(payload[key])
        for key in _FACT_FIELDS
        if key in payload and payload[key] not in (None, "", [], {})
    }
    if (
        facts.get("aurora_initialization_time_utc")
        and any(
            item.get("source") == "aurora" and item.get("status") == "ok"
            for item in tool_evidence
        )
    ):
        updates.insert(0, {
            "aurora_initialization_time_utc": facts["aurora_initialization_time_utc"]
        })
    if (
        any(
            item.get("source") == "aurora" and item.get("status") == "ok"
            for item in tool_evidence
        )
    ):
        verified_domain = facts.get("aurora_region") or facts.get("region")
        if isinstance(verified_domain, dict):
            updates.insert(0, {"aurora_region": verified_domain})
    merged_update: dict[str, Any] = {}
    for update in updates:
        merged_update.update(update)
    return {
        "record_id": f"{turn_id}:evidence",
        "turn_id": turn_id,
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": str(question)[:2_000],
        "status": result.get("status"),
        "request_facts": facts,
        "tool_evidence": tool_evidence,
        "derived_results": derived_results[-4:],
        "artifacts": _bounded(result.get("artifacts", [])),
        "study_update": merged_update,
    }


def rebuild_study_case(
    session_id: str,
    turns: list[dict[str, Any]],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild the canonical case ledger from persisted turn records."""

    old = previous if isinstance(previous, dict) else {}
    case_id = str(old.get("case_id") or session_id)
    records = [
        dict(item["evidence_record"])
        for item in turns
        if (
            isinstance(item, dict)
            and isinstance(item.get("evidence_record"), dict)
            # Legacy turns predate per-turn case IDs and belong to the original
            # session-scoped case. New scopes must not absorb their evidence.
            and str(item.get("study_case_id") or session_id) == case_id
        )
    ]
    decisions: dict[str, Any] = {}
    for record in records:
        update = normalize_study_update(record.get("study_update"))
        decisions.update(update)
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "revision": len(records),
        "decisions": decisions,
        "records": records,
    }


def should_inherit_study_case_context(
    question: str,
    payload: dict[str, Any],
    state: dict[str, Any],
) -> bool:
    """Return whether this request is a continuation of the active study case.

    A session is only a conversation container. It must not silently turn every
    later, self-contained question into a continuation of an earlier weather
    event. Explicit case IDs or clear follow-up wording may continue a case,
    but an explicit time outside the active case always starts a new scope.
    """

    study_case = state.get("study_case") if isinstance(state, dict) else None
    if not isinstance(study_case, dict) or not study_case.get("records"):
        return False
    decisions = study_case.get("decisions")
    decisions = decisions if isinstance(decisions, dict) else {}

    requested_time = str(payload.get("requested_time_utc") or "").strip()
    anchors = {
        str(decisions.get("target_time_utc") or "").strip(),
        str(decisions.get("aurora_initialization_time_utc") or "").strip(),
    }
    focus_window = decisions.get("focus_window")
    if isinstance(focus_window, dict):
        anchors.update({
            str(focus_window.get("start_utc") or "").strip(),
            str(focus_window.get("end_utc") or "").strip(),
        })
    anchors.discard("")
    # Comparing the year first is deliberate: follow-up windows within one
    # case can use different hours, but a 2017 request can never inherit a
    # 2019 event merely because it shares the same browser session.
    if requested_time and anchors:
        requested_year = requested_time[:4]
        anchor_years = {value[:4] for value in anchors if len(value) >= 4}
        if requested_year and anchor_years and requested_year not in anchor_years:
            return False

    requested_case_id = str(payload.get("study_case_id") or "").strip()
    active_case_id = str(study_case.get("case_id") or "").strip()
    if requested_case_id:
        return requested_case_id == active_case_id
    return bool(_FOLLOW_UP_PATTERN.search(str(question or "")))


def evidence_index(state: dict[str, Any] | None) -> dict[str, Any]:
    """Return a compact index; detailed values remain queryable in the Worker."""

    value = state if isinstance(state, dict) else {}
    records = []
    for record in value.get("records", []):
        if not isinstance(record, dict):
            continue
        calls = [item for item in record.get("tool_evidence", []) if isinstance(item, dict)]
        records.append({
            "record_id": record.get("record_id"),
            "turn_id": record.get("turn_id"),
            "question": str(record.get("question") or "")[:240],
            "status": record.get("status"),
            "sources": sorted({str(item.get("source")) for item in calls if item.get("source")}),
            "operations": sorted({
                f"{item.get('source')}.{item.get('operation')}"
                for item in calls if item.get("source") and item.get("operation")
            }),
            "request_facts": record.get("request_facts", {}),
            "study_update": record.get("study_update", {}),
            "artifact_count": len(record.get("artifacts", [])),
        })
    return {
        "schema_version": value.get("schema_version", SCHEMA_VERSION),
        "case_id": value.get("case_id", ""),
        "revision": int(value.get("revision", len(records)) or 0),
        "decisions": value.get("decisions", {}),
        "records": records[-20:],
    }


def runtime_state(state: dict[str, Any] | None) -> dict[str, Any]:
    """Return a bounded, still-structured ledger suitable for Worker transport."""

    value = state if isinstance(state, dict) else {}
    records: list[dict[str, Any]] = []
    for raw in value.get("records", [])[-20:]:
        if not isinstance(raw, dict):
            continue
        record = {
            "record_id": raw.get("record_id"),
            "turn_id": raw.get("turn_id"),
            "created_at": raw.get("created_at"),
            "question": str(raw.get("question") or "")[:500],
            "status": raw.get("status"),
            "request_facts": _bounded(raw.get("request_facts", {}), 3),
            "tool_evidence": _bounded(raw.get("tool_evidence", []), 3),
            "derived_results": _bounded(raw.get("derived_results", []), 3),
            "artifacts": _bounded(raw.get("artifacts", []), 3),
            "study_update": normalize_study_update(raw.get("study_update")),
        }
        if len(json.dumps(record, ensure_ascii=True, default=str)) > 6_000:
            calls = [item for item in raw.get("tool_evidence", []) if isinstance(item, dict)]
            record["tool_evidence"] = [
                {
                    "source": item.get("source"),
                    "operation": item.get("operation"),
                    "status": item.get("status"),
                    "arguments": _bounded(item.get("arguments", {}), 5),
                    "return_type": item.get("return_type"),
                    "evidence": _bounded(item.get("evidence"), 6),
                }
                for item in calls[-8:]
            ]
            record["derived_results"] = _bounded(
                list(raw.get("derived_results", []))[-2:], 6
            )
        if len(json.dumps(record, ensure_ascii=True, default=str)) > 8_000:
            for call in record.get("tool_evidence", []):
                if isinstance(call, dict) and "evidence" in call:
                    call["evidence"] = {"omitted_from_worker_transport": True}
            record["derived_results"] = _bounded(
                list(raw.get("derived_results", []))[-1:], 7
            )
        records.append(record)
    # Keep the most recent evidence detailed when the complete chain exceeds
    # the worker pipe budget. Older records remain represented by their facts,
    # operations and promoted decisions instead of disappearing entirely.
    for index, oldest in enumerate(list(records)):
        if len(json.dumps(records, ensure_ascii=True, default=str)) <= 42_000:
            break
        records[index] = {
            "record_id": oldest.get("record_id"),
            "turn_id": oldest.get("turn_id"),
            "question": oldest.get("question"),
            "status": oldest.get("status"),
            "request_facts": oldest.get("request_facts", {}),
            "operations": [
                f"{item.get('source')}.{item.get('operation')}"
                for item in oldest.get("tool_evidence", [])
                if isinstance(item, dict) and item.get("source") and item.get("operation")
            ],
            "derived_results": _bounded(
                list(oldest.get("derived_results", []))[-1:], 6
            ),
            "artifacts": _bounded(oldest.get("artifacts", []), 6),
            "study_update": oldest.get("study_update", {}),
            "detail": "tool_details_compacted_for_worker_transport",
        }
    return {
        "schema_version": value.get("schema_version", SCHEMA_VERSION),
        "case_id": value.get("case_id", ""),
        "revision": int(value.get("revision", len(records)) or 0),
        "decisions": _bounded(value.get("decisions", {})),
        "records": records,
    }


class StudyEvidence:
    """Read-only, request-independent view of prior structured evidence."""

    def __init__(self, state: dict[str, Any] | None) -> None:
        self._state = _bounded(state if isinstance(state, dict) else {})

    @property
    def case_id(self) -> str:
        """Return the durable Study Case identity."""

        return str(self._state.get("case_id") or "")

    @property
    def revision(self) -> int:
        """Return the number of persisted evidence-producing turns."""

        return int(self._state.get("revision", 0) or 0)

    @property
    def decisions(self) -> dict[str, Any]:
        """Return validated durable decisions, such as the selected event."""

        return json.loads(
            json.dumps(self._state.get("decisions") or {}, ensure_ascii=False)
        )

    def records(self) -> tuple[dict[str, Any], ...]:
        """Return all prior evidence records in chronological order."""

        return tuple(
            json.loads(json.dumps(item, ensure_ascii=False))
            for item in self._state.get("records", [])
            if isinstance(item, dict)
        )

    def __iter__(self):
        """Iterate over chronological evidence records."""

        return iter(self.records())

    def select(
        self,
        *,
        source: str | None = None,
        operation: str | None = None,
        turn_id: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return records matching a tool source, operation or turn identity."""

        selected = []
        for record in self.records():
            if turn_id is not None and str(record.get("turn_id")) != str(turn_id):
                continue
            calls = [item for item in record.get("tool_evidence", []) if isinstance(item, dict)]
            if source is not None and not any(str(item.get("source")) == source for item in calls):
                continue
            if operation is not None and not any(str(item.get("operation")) == operation for item in calls):
                continue
            selected.append(record)
        return tuple(selected)

    def latest(
        self,
        *,
        source: str | None = None,
        operation: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the newest matching evidence record, or None."""

        matches = self.select(source=source, operation=operation)
        return dict(matches[-1]) if matches else None

    def get(self, record_id: str) -> dict[str, Any]:
        """Return one record by its stable record identity."""

        for record in self.records():
            if str(record.get("record_id")) == str(record_id):
                return record
        raise KeyError(f"unknown study evidence record: {record_id}")

    def get_record(self, record_id: str) -> dict[str, Any]:
        """Return one evidence record by stable identity.

        Parameters
        ----------
        record_id : str
            Identifier returned in a record's ``record_id`` field.

        Returns
        -------
        dict
            The matching structured evidence record.
        """

        return self.get(record_id)


def prompt_context(state: dict[str, Any] | None) -> str:
    """Render the compact evidence index supplied before code generation."""

    index = evidence_index(state)
    if not index.get("records"):
        return ""
    records = [
        {
            "turn_id": item.get("turn_id"),
            "status": item.get("status"),
            "sources": item.get("sources", []),
            "operations": item.get("operations", []),
            "study_update": item.get("study_update", {}),
            "artifact_count": item.get("artifact_count", 0),
        }
        for item in index.get("records", [])
        if isinstance(item, dict)
    ]
    compact = {
        "schema_version": index.get("schema_version"),
        "case_id": index.get("case_id"),
        "revision": index.get("revision"),
        "decisions": index.get("decisions", {}),
        "completed_steps": records[-8:],
    }
    return (
        "Persisted Study Case decision state. Previous answer prose is not part "
        "of the analytical context. Use the injected study object to retrieve "
        "structured evidence records when needed:\n"
        + json.dumps(compact, ensure_ascii=False, default=str)
    )


def apply_study_case_context(payload: dict[str, Any], state: dict[str, Any]) -> None:
    """Fill absent request identities from validated prior case decisions.

    Explicit values in the new request always win. Evidence remains stored in
    the session ledger; only a small set of promoted decisions may become the
    next request's bound identity.
    """

    study_case = state.get("study_case") if isinstance(state, dict) else None
    decisions = study_case.get("decisions") if isinstance(study_case, dict) else None
    if not isinstance(decisions, dict):
        decisions = {}
    if not str(payload.get("event_id") or "").strip():
        selected_subject = (
            decisions.get("selected_subject")
            if isinstance(decisions.get("selected_subject"), dict)
            else {}
        )
        selected = str(
            selected_subject.get("event_id")
            or decisions.get("selected_event_id")
            or ""
        ).strip().upper()
        if selected:
            payload["event_id"] = selected
            subject_key = str(
                selected_subject.get("subject_key")
                or decisions.get("selected_subject_key")
                or ""
            ).strip().lower()
            if subject_key:
                payload["sevir_subject_key"] = subject_key
            payload["event_resolution"] = {
                "status": "resolved",
                "event_id": selected,
                "subject_key": subject_key,
                "source": "study_case_decision",
            }
    if not str(payload.get("requested_time_utc") or "").strip():
        target_time = str(
            decisions.get("target_time_utc")
            or ""
        ).strip()
        if target_time:
            payload["requested_time_utc"] = target_time
    if not str(payload.get("aurora_initialization_time_utc") or "").strip():
        initialization = str(
            decisions.get("aurora_initialization_time_utc")
            or ""
        ).strip()
        if initialization:
            payload["aurora_initialization_time_utc"] = initialization
    if not isinstance(payload.get("region"), dict):
        inherited_region = decisions.get("target_region")
        if isinstance(inherited_region, dict):
            payload["region"] = dict(inherited_region)
    if not isinstance(payload.get("aurora_region"), dict):
        aurora_region = decisions.get("aurora_region")
        if isinstance(aurora_region, dict):
            payload["aurora_region"] = dict(aurora_region)
