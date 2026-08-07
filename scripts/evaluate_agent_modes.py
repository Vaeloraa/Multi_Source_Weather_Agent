from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "data" / "direct_evaluation_cases.json"
DEFAULT_OUTPUT = ROOT / "data" / "agent_evaluations"


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body[:2000]}") from error


def score(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    runs = response.get("tool_runs") if isinstance(response.get("tool_runs"), list) else []
    tool_names = [str(run.get("tool_name", "")) for run in runs]
    expected = [str(value) for value in case.get("expected_api_fragments", [])]
    matched = {
        fragment: any(fragment in tool_name for tool_name in tool_names)
        for fragment in expected
    }
    trace = response.get("direct_trace") if isinstance(response.get("direct_trace"), list) else []
    return {
        "request_succeeded": response.get("status") == "ok",
        "tool_call_count": len(runs),
        "distinct_tool_count": len(set(tool_names)),
        "minimum_tool_calls_met": len(runs) >= int(case.get("minimum_tool_calls", 1)),
        "expected_tools": matched,
        "expected_tools_met": all(matched.values()),
        "execution_grounded": bool(response.get("code_result"))
        if response.get("mode") == "direct"
        else bool(response.get("tool_result")),
        "repair_count": max(len(trace) - 1, 0),
        "tool_names": tool_names,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in sorted({row["mode"] for row in rows}):
        selected = [row for row in rows if row["mode"] == mode]
        count = len(selected) or 1
        summary[mode] = {
            "case_count": len(selected),
            "request_success_rate": sum(row["score"]["request_succeeded"] for row in selected) / count,
            "multi_tool_success_rate": sum(
                row["score"]["minimum_tool_calls_met"] for row in selected
            )
            / count,
            "expected_tool_success_rate": sum(
                row["score"]["expected_tools_met"] for row in selected
            )
            / count,
            "execution_grounding_rate": sum(
                row["score"]["execution_grounded"] for row in selected
            )
            / count,
            "mean_tool_calls": sum(row["score"]["tool_call_count"] for row in selected) / count,
            "mean_elapsed_ms": sum(row["elapsed_ms"] for row in selected) / count,
            "repair_rate": sum(row["score"]["repair_count"] > 0 for row in selected) / count,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed API routing and Direct code agent.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--provider", default="qwen")
    parser.add_argument("--model", default="")
    parser.add_argument("--modes", default="api,direct")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=180.0)
    arguments = parser.parse_args()

    cases = json.loads(arguments.cases.read_text(encoding="utf-8"))
    modes = [value.strip() for value in arguments.modes.split(",") if value.strip()]
    rows = []
    for case in cases:
        for mode in modes:
            payload = {
                "mode": mode,
                "provider": arguments.provider,
                "model": arguments.model,
                "task": "auto",
                "data_source": "cascast",
                "weather_model": "cascast",
                "event_id": case["event_id"],
                "question": case["question"],
            }
            started = time.perf_counter()
            try:
                response = post_json(f"{arguments.base_url.rstrip('/')}/api/run", payload, arguments.timeout)
                error = None
            except Exception as exception:
                response = {"status": "error", "error": str(exception), "mode": mode}
                error = str(exception)
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            rows.append(
                {
                    "case_id": case["id"],
                    "mode": mode,
                    "elapsed_ms": elapsed_ms,
                    "score": score(case, response),
                    "error": error,
                    "response": response,
                }
            )
            print(f"{case['id']}: {mode}: {response.get('status')} ({elapsed_ms} ms)")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": arguments.base_url,
        "provider": arguments.provider,
        "model": arguments.model or "provider_default",
        "summary": summarize(rows),
        "runs": rows,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = arguments.output / f"agent-mode-evaluation-{stamp}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
