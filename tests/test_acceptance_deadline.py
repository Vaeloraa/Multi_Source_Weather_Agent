from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_acceptance_42 as acceptance  # noqa: E402


class AcceptanceDeadlineTests(unittest.TestCase):
    def tearDown(self) -> None:
        acceptance.STOP_ON_DEADLINE.clear()

    def test_timeout_stops_later_questions_and_writes_partial_diagnosis(self) -> None:
        acceptance.STOP_ON_DEADLINE.clear()
        with patch.object(acceptance, "post", side_effect=TimeoutError("timed out")):
            result = acceptance.run_question(
                "http://127.0.0.1:1",
                1,
                "single",
                "现在外面安全吗？",
                {"single": "x"},
                "deadline-test",
                timeout=30,
            )

        self.assertEqual(result["status"], "deadline_exceeded")
        self.assertTrue(result["diagnosis_required"])
        self.assertTrue(acceptance.STOP_ON_DEADLINE.is_set())

        skipped = acceptance.run_question(
            "http://127.0.0.1:1",
            2,
            "single",
            "下一题不应启动。",
            {"single": "x"},
            "deadline-test-2",
            timeout=30,
        )
        self.assertEqual(skipped["status"], "not_started")

        output = ROOT / "tests" / "_acceptance_deadline_output"
        with patch.object(Path, "write_text") as write_text:
            stopped = acceptance.stop_after_deadline(
                output,
                {"single": "x"},
                [result],
                started_at="2026-08-24T00:00:00",
            )
        manifest = json.loads(write_text.call_args.args[0])

        self.assertTrue(stopped)
        self.assertTrue(manifest["stopped_early"])
        self.assertTrue(manifest["diagnosis_required"])
        self.assertEqual(manifest["stop_reason"], "request_exceeded_30s_budget")

    def test_unified_agent_failure_is_not_an_agent_pass(self) -> None:
        for termination in (
            "consecutive_execution_errors", "max_steps_without_solution",
        ):
            status, failure_class = acceptance._result_status({
                "status": "partial",
                "termination_reason": termination,
                "answer": "缓存观测显示回波增强，出行前请复核当地预警。",
                "technical_details": {"agent_error": {"message": "presentation timeout"}},
            })
            self.assertEqual(status, "failed")
            self.assertEqual(failure_class, "agent_or_protocol_failure")

    def test_acceptance_payload_has_no_case_specific_execution_profile(self) -> None:
        ids = {"single": "", "sequence": "", "invalid_sequence": ""}
        for number in (27, *range(33, 43)):
            kind, prompt = acceptance.QUESTIONS[number]
            payload = acceptance._payload(number, kind, prompt, ids, "study-case")
            self.assertNotIn("execution_profile", payload)


if __name__ == "__main__":
    unittest.main()
