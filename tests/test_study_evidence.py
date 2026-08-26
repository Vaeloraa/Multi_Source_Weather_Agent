from __future__ import annotations

from pathlib import Path

from sevir_world.agent import session
from sevir_world.agent.session import context_for_prompt
from sevir_world.agent.study import (
    StudyEvidence,
    apply_study_case_context,
    should_inherit_study_case_context,
    build_turn_evidence,
    runtime_state,
)
from sevir_world.code_execution.client import ExecutionClient
from sevir_world.api.unified_service import bind_request_assets
from sevir_world.api.response_pipeline import _persist_turn
from scripts.run_acceptance_42 import _payload


def _isolated_sessions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(session, "SESSION_ROOT", tmp_path)
    monkeypatch.setattr(session, "LATEST_SESSION_PATH", tmp_path / "_latest.json")
    session._MEMORY.clear()


def _evidence(turn_id: str, selected_event_id: str = "") -> dict:
    study_update = {
        "target_time_utc": "2019-06-27T09:00:00Z",
        "target_region": {"south": 46.0, "west": -103.0, "north": 49.0, "east": -97.0},
    }
    if selected_event_id:
        study_update["selected_event_id"] = selected_event_id
    result = {
        "status": "ok",
        "artifacts": [{"url": f"/api/artifacts/{turn_id}.png"}],
        "observations": [{
            "status": "ok",
            "result": {"regional_peak": 12.5, "study_update": study_update},
            "calls": [{
                "api": "aurora.regional_fields",
                "status": "ok",
                "cache_hit": True,
                "arguments": {"forecast_leads": [6, 12, 18, 24]},
                "return_type": "EnvironmentFields",
                "evidence": {
                    "initialization_time_utc": "2019-06-27T03:00:00Z",
                    "valid_times_utc": ["2019-06-27T09:00:00Z"],
                },
            }],
        }],
    }
    value = build_turn_evidence(
        turn_id=turn_id,
        request_id=f"request-{turn_id}",
        question="regional evidence",
        payload={
            "requested_time_utc": "2019-06-27T03:00:00Z",
            "aurora_initialization_time_utc": "2019-06-27T03:00:00Z",
            "region": {"south": 45.0, "west": -105.0, "north": 50.0, "east": -96.0},
        },
        result=result,
    )
    assert value is not None
    return value


def test_cross_request_evidence_is_structured_and_queryable(monkeypatch, tmp_path: Path) -> None:
    _isolated_sessions(monkeypatch, tmp_path)
    evidence = _evidence("q33")
    saved = session.append_turn("study-chain", {
        "turn_id": "q33",
        "question": "Q33",
        "answer": "This prose must not become evidence.",
        "evidence_record": evidence,
        "study_case_id": "north-dakota-20190627",
    })

    state = saved["study_case"]
    assert state["case_id"] == "north-dakota-20190627"
    assert state["revision"] == 1
    assert "This prose" not in str(state)
    study = StudyEvidence(runtime_state(state))
    record = study.latest(source="aurora", operation="regional_fields")
    assert record is not None
    assert record["derived_results"][0]["regional_peak"] == 12.5
    assert study.decisions["target_time_utc"] == "2019-06-27T09:00:00Z"


def test_study_prompt_does_not_repeat_prior_question_or_answer_prose(
    monkeypatch, tmp_path: Path
) -> None:
    _isolated_sessions(monkeypatch, tmp_path)
    saved = session.append_turn("study-chain", {
        "turn_id": "q33",
        "question": "旧问题：请重复整段天气分析",
        "answer": "旧回答：这段叙述不应被重新注入分析提示",
        "evidence_record": _evidence("q33"),
    })

    prompt_context_text = context_for_prompt(saved)

    assert "旧问题" not in prompt_context_text
    assert "旧回答" not in prompt_context_text
    assert "Persisted Study Case decision state" in prompt_context_text
    assert "completed_steps" in prompt_context_text


def test_promoted_decision_binds_next_request_without_copying_ledger(
    monkeypatch, tmp_path: Path
) -> None:
    _isolated_sessions(monkeypatch, tmp_path)
    evidence = _evidence("q36", "r19062708588472")
    saved = session.append_turn("study-chain", {
        "turn_id": "q36",
        "question": "Q36",
        "answer": "selected",
        "evidence_record": evidence,
    })
    payload: dict = {}
    apply_study_case_context(payload, saved)

    assert payload["event_id"] == "R19062708588472"
    assert payload["requested_time_utc"] == "2019-06-27T09:00:00Z"
    assert payload["region"] == {
        "south": 46.0, "west": -103.0, "north": 49.0, "east": -97.0,
    }
    assert "study_case" not in payload
    assert payload["event_resolution"]["source"] == "study_case_decision"


def test_new_explicit_year_does_not_inherit_old_weather_case(
    monkeypatch, tmp_path: Path
) -> None:
    _isolated_sessions(monkeypatch, tmp_path)
    saved = session.append_turn("study-chain", {
        "turn_id": "q42",
        "question": "2019 case",
        "answer": "old conclusion",
        "evidence_record": _evidence("q42", "R19062708588472"),
    })
    payload = {"requested_time_utc": "2017-06-27T03:00:00Z"}

    inherit = should_inherit_study_case_context(
        "现在是2017年6月27日03:00 UTC，请用Aurora重新判断。",
        payload,
        saved,
    )

    assert inherit is False
    assert "event_id" not in payload
    assert "aurora_initialization_time_utc" not in payload


def test_explicit_matching_case_and_follow_up_keep_study_context(
    monkeypatch, tmp_path: Path
) -> None:
    _isolated_sessions(monkeypatch, tmp_path)
    saved = session.append_turn("study-chain", {
        "turn_id": "q36",
        "question": "select event",
        "answer": "selected",
        "study_case_id": "north-dakota-20190627",
        "evidence_record": _evidence("q36", "R19062708588472"),
    })

    assert should_inherit_study_case_context(
        "请继续验证该事件。",
        {"study_case_id": "north-dakota-20190627"},
        saved,
    )
    assert should_inherit_study_case_context(
        "请用目标时刻之后的观测继续验证。",
        {},
        saved,
    )


def test_switching_case_id_excludes_previous_case_records(
    monkeypatch, tmp_path: Path
) -> None:
    _isolated_sessions(monkeypatch, tmp_path)
    session.append_turn("study-chain", {
        "turn_id": "old",
        "study_case_id": "case-2019",
        "evidence_record": _evidence("old", "R19062708588472"),
    })
    saved = session.append_turn("study-chain", {
        "turn_id": "new",
        "study_case_id": "case-2017",
        "evidence_record": {
            **_evidence("new"),
            "request_facts": {"requested_time_utc": "2017-06-27T03:00:00Z"},
            "study_update": {"aurora_initialization_time_utc": "2017-06-27T03:00:00Z"},
        },
    })

    assert saved["study_case"]["case_id"] == "case-2017"
    assert [item["turn_id"] for item in saved["study_case"]["records"]] == ["new"]
    assert "selected_subject" not in saved["study_case"]["decisions"]


def test_deleting_upstream_turn_rebuilds_decisions(monkeypatch, tmp_path: Path) -> None:
    _isolated_sessions(monkeypatch, tmp_path)
    session.append_turn("study-chain", {
        "turn_id": "q33",
        "question": "Q33",
        "evidence_record": _evidence("q33"),
    })
    session.append_turn("study-chain", {
        "turn_id": "q36",
        "question": "Q36",
        "evidence_record": _evidence("q36", "R19062708588472"),
    })
    session.delete_turn("study-chain", "q36")
    state = session.load_session("study-chain")["study_case"]

    assert state["revision"] == 1
    assert "selected_event_id" not in state["decisions"]
    assert state["decisions"]["target_time_utc"] == "2019-06-27T09:00:00Z"


def test_persistent_worker_reads_study_and_keeps_python_names() -> None:
    state = {
        "schema_version": "study-evidence-v1",
        "case_id": "north-dakota-20190627",
        "revision": 2,
        "decisions": {"selected_event_id": "R19062708588472"},
        "records": [],
    }
    context = {"data_policy": "cached", "_study_case": state}
    client = ExecutionClient(timeout_seconds=10, memory_limit_mb=512)
    try:
        first = client.execute(
            'retained_value = 41\nresult = {"revision": study.revision}',
            context,
            data_descs=[],
            selected_tools=["ds"],
        )
        second = client.execute(
            'result = {"value": retained_value + 1, "case_id": study.case_id, '
            '"event_id": study.decisions["selected_event_id"]}',
            context,
            data_descs=[],
            selected_tools=["ds"],
        )
    finally:
        client.close()

    assert first["status"] == "ok"
    assert first["result"]["revision"] == 2
    assert second["status"] == "ok"
    assert second["result"] == {
        "value": 42,
        "case_id": "north-dakota-20190627",
        "event_id": "R19062708588472",
    }
    assert first["worker_session"]["id"] == second["worker_session"]["id"]
    assert first["worker_session"]["step"] == 1
    assert second["worker_session"]["step"] == 2


def test_study_case_candidate_mention_does_not_prebind_event() -> None:
    payload = {"study_case_id": "north-dakota-20190627"}
    descriptions, _ = bind_request_assets(
        "请判断R19062708588472是否由检索条件筛选出来",
        payload,
    )

    assert payload.get("event_id", "") == ""
    assert descriptions == []


def test_catalog_search_candidates_are_promoted_from_real_tool_evidence() -> None:
    record = build_turn_evidence(
        turn_id="q35",
        request_id="request-q35",
        question="search the focus window",
        payload={},
        result={
            "status": "ok",
            "observations": [{
                "status": "ok",
                "result": {"matched": 2},
                "calls": [{
                    "api": "sevir.search",
                    "status": "ok",
                    "arguments": {"start_time_utc": "2019-06-27T08:00:00Z"},
                    "return_type": "CatalogSearchResult",
                    "evidence": {
                        "events": [
                            {"id": "R19062708588472"},
                            {"id": "R19062709000000"},
                        ]
                    },
                }],
            }],
        },
    )

    assert record is not None
    assert record["study_update"]["candidate_event_ids"] == [
        "R19062708588472", "R19062709000000",
    ]


def test_response_boundary_persists_evidence_and_returns_case_index(
    monkeypatch, tmp_path: Path
) -> None:
    _isolated_sessions(monkeypatch, tmp_path)
    result = {
        "session_id": "study-chain",
        "request_id": "request-q34",
        "status": "ok",
        "answer": "narrative only",
        "observations": [{
            "status": "ok",
            "result": {
                "focus_window": "first six-hour signal",
                "study_update": {
                    "target_time_utc": "2019-06-27T09:00:00Z",
                    "target_region": {
                        "south": 46.0, "west": -103.0,
                        "north": 49.0, "east": -97.0,
                    },
                },
            },
            "calls": [],
        }],
        "artifacts": [],
    }
    _persist_turn(
        result,
        {"session_id": "study-chain", "study_case_id": "north-dakota-20190627"},
        "Q34",
    )

    assert result["study_case"]["case_id"] == "north-dakota-20190627"
    assert result["study_case"]["revision"] == 1
    assert result["study_case"]["decisions"]["target_time_utc"] == "2019-06-27T09:00:00Z"
    stored = session.load_session("study-chain")
    assert stored["turns"][0]["answer"] == "narrative only"
    assert "narrative only" not in str(stored["study_case"])


def test_study_acceptance_payload_only_seeds_the_first_regional_stage() -> None:
    ids = {"single": "", "sequence": "", "invalid_sequence": ""}
    session_id = "north-dakota-20190627"
    q33 = _payload(33, "event_environment", "regional discovery", ids, session_id)
    q34 = _payload(34, "event_environment", "focus window", ids, session_id)
    q36 = _payload(36, "sequence_environment", "select event", ids, session_id)

    assert q33["aurora_region"]
    assert "region" not in q33
    assert q33["aurora_initialization_time_utc"] == "2019-06-27T03:00:00Z"
    assert "region" not in q34
    assert "requested_time_utc" not in q34
    assert q36["event_id"] == ""
    assert q33["study_case_id"] == q34["study_case_id"] == q36["study_case_id"]


def test_verified_aurora_domain_is_distinct_from_local_target() -> None:
    broad_region = {
        "south": 41.3, "west": -111.98, "north": 53.05, "east": -88.23,
    }
    record = build_turn_evidence(
        turn_id="q33",
        request_id="request-q33",
        question="regional discovery",
        payload={
            "aurora_initialization_time_utc": "2019-06-27T03:00:00Z",
            "region": broad_region,
        },
        result={
            "status": "ok",
            "observations": [{
                "status": "ok",
                "result": {"summary": "computed"},
                "calls": [{
                    "api": "aurora.regional_fields",
                    "status": "ok",
                    "arguments": {"region": broad_region},
                    "return_type": "Dataset",
                }],
            }],
            "artifacts": [],
        },
    )
    assert record is not None
    assert record["study_update"]["aurora_region"] == broad_region

    local_region = {
        "south": 46.0, "west": -103.0, "north": 49.0, "east": -97.0,
    }
    following: dict[str, object] = {}
    apply_study_case_context(following, {
        "study_case": {
            "decisions": {
                "aurora_region": broad_region,
                "target_region": local_region,
            }
        }
    })
    assert following["region"] == local_region
