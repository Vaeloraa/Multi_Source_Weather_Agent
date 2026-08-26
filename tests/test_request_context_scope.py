from __future__ import annotations

from sevir_world.agent.study import should_inherit_study_case_context
from sevir_world.data_access.request_context import enrich_request_context


def test_explicit_aurora_now_time_becomes_initialization() -> None:
    payload: dict = {}
    result = enrich_request_context(
        "现在是2017年6月27日03:00 UTC，请用Aurora判断未来6到24小时。",
        payload,
    )

    assert result["status"] == "unresolved"
    assert payload["requested_time_utc"] == "2017-06-27T03:00:00Z"
    assert payload["aurora_initialization_time_utc"] == "2017-06-27T03:00:00Z"
    assert payload["aurora_initialization_time_source"] == "requested_time_for_aurora"


def test_legacy_2019_session_rejects_cross_year_context() -> None:
    state = {"study_case": {
        "case_id": "legacy-2019",
        "records": [{"record_id": "q42:evidence"}],
        "decisions": {
            "aurora_initialization_time_utc": "2019-06-27T03:00:00Z",
            "focus_window": {
                "start_utc": "2019-06-27T09:00:00Z",
                "end_utc": "2019-06-28T03:00:00Z",
            },
            "selected_subject": {
                "event_id": "R19062708588472",
                "subject_key": "sevir:feb5f0f6599442f9ee04f71a",
            },
        },
    }}
    payload: dict = {}
    question = (
        "现在是2017年6月27日03:00 UTC。请用Aurora判断北达科他州及周边"
        "未来6到24小时是否存在值得追踪的显著天气风险。"
    )
    enrich_request_context(question, payload)

    assert should_inherit_study_case_context(question, payload, state) is False
    assert "event_id" not in payload
    assert payload["aurora_initialization_time_utc"].startswith("2017-")
