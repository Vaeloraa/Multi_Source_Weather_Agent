from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sevir_world.evaluation.agent_benchmark import (  # noqa: E402
    AUTONOMY_SCENARIOS,
    evaluate_result,
    summarize_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score exported local/direct/reflective task results."
    )
    parser.add_argument("input", type=Path, help="JSON file containing a results list")
    parser.add_argument("--output", type=Path, help="Optional report JSON path")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("input must be a list or an object with a results list")
    scenarios = {item["id"]: item for item in AUTONOMY_SCENARIOS}
    evaluations = []
    for row in rows:
        scenario_id = str(row.get("scenario_id", ""))
        if scenario_id not in scenarios:
            raise ValueError(f"unknown scenario_id: {scenario_id}")
        evaluations.append(evaluate_result(dict(row.get("result", {})), scenarios[scenario_id]))
    report = summarize_results(evaluations)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
