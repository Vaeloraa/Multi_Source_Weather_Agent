from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_acceptance_42 import post

QUESTION_FILE = ROOT / "data" / "evaluation" / "resident_north_dakota_story_questions.json"

def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8024"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "evaluation" / f"resident-story-batch-{time.strftime('%Y%m%d-%H%M%S')}"
    question_set = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))
    questions = question_set["questions"]
    if len(sys.argv) > 3:
        selected = {int(value) for value in sys.argv[3].split(",") if value.strip()}
        questions = [item for item in questions if int(item["number"]) in selected]
    scenario = question_set["scenario"]
    demo_case = json.loads(
        (ROOT / "data" / "demo_cases" / "R19062708588472.json").read_text(encoding="utf-8")
    )
    session_id = sys.argv[4] if len(sys.argv) > 4 else f"resident-story-{uuid.uuid4()}"
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    results = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.is_file()
        else []
    )
    for item in questions:
        number = int(item["number"])
        question = str(item["prompt"])
        payload = {
            "mode": "unified", "task": "auto", "data_source": "environment",
            "data_policy": "live", "requested_time_utc": scenario["request_time_utc"],
            "aurora_initialization_time_utc": scenario["request_time_utc"],
            "place_name": scenario["resident_region"],
            "aurora_region": demo_case["regional_domain"],
            "region": demo_case["regional_domain"],
            "forecast_leads_hours": [6, 12, 18, 24], "session_id": session_id,
            "study_case_id": session_id, "study_stage": f"Q{number:02d}",
            "request_id": uuid.uuid4().hex, "trace": True,
            "visual_artifact_required": bool(item.get("visual_artifact_required", False)),
            "question": f"【居民故事批测Q{number:02d}】{question}",
        }
        started = time.perf_counter()
        try:
            body = post(base_url, "/api/run", payload, timeout=905)
            result = {"number": number, "question": question, "status": body.get("status"),
                      "answer": body.get("answer", ""), "elapsed_seconds": round(time.perf_counter()-started, 2),
                      "termination_reason": body.get("termination_reason"),
                      "request_budget": body.get("request_budget"), "artifacts": body.get("artifacts", []),
                      "error": body.get("error")}
        except Exception as exc:
            result = {"number": number, "question": question, "status": "request_error",
                      "elapsed_seconds": round(time.perf_counter()-started, 2), "error": repr(exc)}
        results.append(result)
        print(f"Q{number:02d} | {result['status']} | {result['elapsed_seconds']}s", flush=True)
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"session_id": session_id, "question_file": str(QUESTION_FILE), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
