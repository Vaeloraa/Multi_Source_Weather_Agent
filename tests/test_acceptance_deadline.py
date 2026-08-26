from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_acceptance_42 as acceptance  # noqa: E402
from sevir_world.agent.visual_contract import visual_artifact_required


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
        self.assertEqual(manifest["stop_reason"], "request_exceeded_90s_budget")

    def test_http_transport_has_only_completion_grace_beyond_agent_budget(self) -> None:
        body = {
            "status": "ok",
            "termination_reason": "agent_completed",
            "answer": "完成",
            "artifacts": [{"url": "/api/artifacts/test.png"}],
            "request_budget": {"deadline_reached": False},
        }
        with patch.object(acceptance, "post", return_value=body) as post:
            result = acceptance.run_question(
                "http://127.0.0.1:1",
                1,
                "single",
                "现在外面安全吗？",
                {"single": "x"},
                "deadline-grace-test",
                timeout=acceptance.ACCEPTANCE_TIMEOUT_SECONDS,
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            post.call_args.kwargs["timeout"],
            acceptance.ACCEPTANCE_TIMEOUT_SECONDS
            + acceptance.TRANSPORT_COMPLETION_GRACE_SECONDS,
        )
        self.assertEqual(acceptance.ACCEPTANCE_TIMEOUT_SECONDS, 90)

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

    def test_visual_artifact_is_required_only_by_positive_question_intent(self) -> None:
        self.assertTrue(visual_artifact_required("请生成事件位置图和数量统计图。"))
        self.assertTrue(visual_artifact_required("请通过证据链图说明数据依赖。"))
        self.assertTrue(visual_artifact_required("只重新计算和绘制受到影响的结果。"))
        self.assertFalse(visual_artifact_required("请列出Springfield候选并要求确认。"))
        self.assertFalse(
            visual_artifact_required("不得绘制没有时间序列支持的确定性轨迹。")
        )

        completed_without_figure = {
            "status": "ok",
            "termination_reason": "agent_completed",
            "answer": "候选地点已经列出，请确认具体州。",
            "artifacts": [],
        }
        self.assertEqual(
            acceptance._result_status(
                completed_without_figure, visual_required=False
            ),
            ("passed", ""),
        )
        self.assertEqual(
            acceptance._result_status(
                completed_without_figure, visual_required=True
            ),
            ("failed", "missing_visual_artifact"),
        )

    def test_current_42_questions_have_question_level_visual_contracts(self) -> None:
        question_text = (ROOT / "ques.md").read_text(encoding="utf-8")
        questions = {
            int(number): prompt.strip()
            for number, prompt in re.findall(
                r"(?ms)^(\d+)\.\s*(.*?)(?=^\d+\.\s|\Z)", question_text
            )
        }

        self.assertEqual(set(questions), set(range(1, 43)))
        optional = {
            number
            for number, prompt in questions.items()
            if not visual_artifact_required(prompt)
        }
        self.assertEqual(optional, {6, 27, 32, *range(33, 43)})

    def test_study_case_question_is_explicitly_forced_to_draw(self) -> None:
        body = {
            "status": "ok",
            "termination_reason": "agent_completed",
            "answer": "区域环境信号及其证据边界已经说明。",
            "artifacts": [],
            "request_budget": {"deadline_reached": False},
        }
        with patch.object(acceptance, "post", return_value=body):
            result = acceptance.run_question(
                "http://127.0.0.1:1",
                33,
                "catalog",
                "请用Aurora做区域判断，并说明证据边界。",
                {},
                "study-case",
                timeout=30,
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_class"], "missing_visual_artifact")
        self.assertTrue(result["visual_artifact_required"])

    def test_acceptance_payload_has_no_case_specific_execution_profile(self) -> None:
        ids = {"single": "", "sequence": "", "invalid_sequence": ""}
        for number in (27, *range(33, 43)):
            kind, prompt = acceptance.QUESTIONS[number]
            payload = acceptance._payload(number, kind, prompt, ids, "study-case")
            self.assertNotIn("execution_profile", payload)
            self.assertEqual(
                payload.get("visual_artifact_required"),
                True if number >= 33 else None,
            )


if __name__ == "__main__":
    unittest.main()
