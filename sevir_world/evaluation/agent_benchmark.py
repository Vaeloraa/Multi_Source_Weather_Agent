from __future__ import annotations

from typing import Any


AUTONOMY_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "S01",
        "question": "2019年SEVIR VIL catalog的data_max中位数是多少？",
        "expected_statistics": ["median"],
        "requires_refinement": False,
    },
    {
        "id": "S02",
        "question": "2019年冰雹事件data_max的典型水平和离散程度如何？",
        "expected_statistics": ["median", "std"],
        "requires_refinement": False,
    },
    {
        "id": "S03",
        "question": "比较Hail和Tornado的data_max分布，避免只看极端值。",
        "expected_statistics": ["median", "quantile"],
        "requires_refinement": True,
    },
    {
        "id": "S04",
        "question": "按月份判断data_max是否存在季节性。",
        "expected_statistics": ["median"],
        "requires_refinement": True,
    },
    {
        "id": "S05",
        "question": "列出data_max最高的5条记录。",
        "expected_statistics": ["ranking"],
        "requires_refinement": False,
    },
    {
        "id": "S06",
        "question": "用13帧输入运行CasCast，并说明未来60分钟强回波。",
        "expected_statistics": [],
        "requires_refinement": False,
    },
    {
        "id": "S07",
        "question": "集合成员对风暴对象的位置和持续时间分歧多大？",
        "expected_statistics": ["spread", "duration"],
        "requires_refinement": True,
    },
    {
        "id": "S08",
        "question": "先判断证据是否充分，不够时追加一个稳健统计量。",
        "expected_statistics": ["median", "quantile"],
        "requires_refinement": True,
    },
    {
        "id": "S09",
        "question": "刚才那个事件与2019年Hail相比如何？",
        "expected_statistics": ["median"],
        "requires_refinement": True,
        "requires_session": True,
    },
    {
        "id": "S10",
        "question": "分析一张上传的单帧VIL图，不要声称完成未来预测。",
        "expected_statistics": [],
        "requires_refinement": False,
        "ground_truth_available": False,
    },
)


def _normalized(values: Any) -> set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    aliases = {
        "p25": "quantile",
        "p50": "median",
        "p75": "quantile",
        "p90": "quantile",
        "p95": "quantile",
        "rank": "ranking",
    }
    return {aliases.get(str(value).lower(), str(value).lower()) for value in values}


def evaluate_result(result: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """Score observable execution behavior without requiring a runtime plan schema."""
    trace = result.get("agent_trace") or []
    execute_steps = [item for item in trace if item.get("action", "execute") == "execute"]
    generated_programs = [item.get("code", "") for item in execute_steps if item.get("code")]
    expected = _normalized(scenario.get("expected_statistics", []))
    generated_text = "\n".join(generated_programs).lower()
    statistic_tokens = {
        "median": ("median", "quantile", "p50"),
        "std": ("std", "standard_deviation", "variance"),
        "quantile": ("quantile", "percentile", "p25", "p75", "p90", "p95"),
        "ranking": ("rank", "sorted", "top"),
        "spread": ("spread", "std", "variance", "range"),
        "duration": ("duration", "consecutive", "active_minutes"),
    }
    observed_statistics = {
        name for name, tokens in statistic_tokens.items()
        if any(token in generated_text for token in tokens)
    }
    route_mode = str(result.get("route_mode", ""))
    checks = {
        "completed": result.get("status") == "ok",
        "no_fixed_fallback": "fallback" not in route_mode and route_mode != "local",
        "generated_code": bool(generated_programs),
        "no_runtime_plan_schema": all(
            key not in result
            for key in ("analysis_plan", "selected_statistics", "metric_definitions")
        ),
        "question_driven_calculation": expected.issubset(observed_statistics),
        "observed_execution": bool(result.get("observations") or result.get("code_result")),
    }
    if scenario.get("requires_refinement"):
        checks["agent_refinement"] = (
            len(generated_programs) >= 2 or result.get("termination_reason") == "finished"
        )
    if scenario.get("requires_session"):
        checks["session_preserved"] = bool(result.get("session_id")) and int(
            result.get("session_turn_count", 0)
        ) >= 2
    if scenario.get("ground_truth_available") is False:
        payload = result.get("tool_result", {})
        checks["no_metrics_without_ground_truth"] = (
            payload.get("ground_truth_available") is False
            and payload.get("metrics_available") is False
        )
    passed = sum(bool(value) for value in checks.values())
    return {
        "scenario_id": scenario["id"],
        "mode": result.get("mode", route_mode or "unknown"),
        "score": round(passed / len(checks), 4) if checks else 0.0,
        "passed": all(checks.values()),
        "checks": checks,
    }


def summarize_results(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate benchmark outcomes by execution mode."""
    modes: dict[str, list[dict[str, Any]]] = {}
    for evaluation in evaluations:
        modes.setdefault(str(evaluation.get("mode", "unknown")), []).append(evaluation)
    mode_summary = {}
    for mode, rows in modes.items():
        mode_summary[mode] = {
            "scenario_count": len(rows),
            "pass_count": sum(bool(row.get("passed")) for row in rows),
            "mean_score": round(sum(float(row.get("score", 0)) for row in rows) / len(rows), 4),
        }
    return {
        "scenario_count": len(evaluations),
        "pass_count": sum(bool(row.get("passed")) for row in evaluations),
        "modes": mode_summary,
        "evaluations": evaluations,
    }
