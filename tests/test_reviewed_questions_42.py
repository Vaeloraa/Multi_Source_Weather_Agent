from __future__ import annotations

import json
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUESTION_SET = ROOT / "data" / "evaluation" / "resident_questions_42.json"


class ReviewedQuestionSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(QUESTION_SET.read_text(encoding="utf-8"))
        cls.questions = cls.payload["questions"]
        cls.by_number = {int(item["number"]): item for item in cls.questions}

    def test_has_exactly_42_unique_consecutive_questions(self) -> None:
        self.assertEqual(len(self.questions), 42)
        self.assertEqual(sorted(self.by_number), list(range(1, 43)))

    def test_dependencies_only_point_to_earlier_questions(self) -> None:
        for question in self.questions:
            number = int(question["number"])
            for dependency in question["depends_on"]:
                self.assertIn(int(dependency), self.by_number)
                self.assertLess(int(dependency), number)

    def test_prompts_do_not_restore_the_old_metric_checklists(self) -> None:
        prompts = "\n".join(str(item["prompt"]) for item in self.questions)
        for fixed_metric in (
            "MAE", "RMSE", "SSIM", "FSS", "CSI", "POD", "FAR",
            "P90", "P95", "P99", "Otsu", "spread–skill",
        ):
            self.assertNotIn(fixed_metric, prompts)

    def test_study_case_does_not_prebind_a_local_event(self) -> None:
        study = self.payload["study_case"]
        self.assertIsNone(study["selected_event_id"])
        self.assertEqual(study["requested_time_utc"], "2019-06-27T03:00:00Z")
        self.assertEqual(study["aurora_initialization_time_utc"], "2019-06-27T03:00:00Z")
        self.assertIsNone(study["selected_event_anchor_time_utc"])
        self.assertEqual(study["user_region"], "North Dakota, United States")
        integrated = "\n".join(self.by_number[n]["prompt"] for n in range(33, 43))
        self.assertIn("North Dakota", integrated)
        self.assertIn("北达科他州", self.by_number[33]["prompt"])
        self.assertNotRegex(integrated, r"R\d{6,}")
        self.assertNotIn("R19062708588472", "\n".join(str(item["prompt"]) for item in self.questions))
        self.assertNotIn("Chatsworth", integrated)
        self.assertNotIn("查茨沃思", integrated)
        self.assertNotIn("Springfield", integrated)

    def test_integrated_chain_covers_teacher_workflow(self) -> None:
        integrated = "\n".join(self.by_number[n]["prompt"] for n in range(33, 43))
        for required in (
            "6到24小时", "时空窗口", "前一小时", "之后60分钟",
            "实际匹配到的局地天气事件", "实际SEVIR观测", "独立验证", "完整Study Case",
            "显著天气", "风险区域", "风险类型",
        ):
            self.assertIn(required, integrated)
        self.assertIn("2019年6月27日03:00 UTC", self.by_number[33]["prompt"])
        self.assertIn("选定最早一个", self.by_number[34]["prompt"])
        self.assertNotIn("哪些区域和时段最值得", self.by_number[34]["prompt"])
        self.assertIn("局地观测时空窗口", self.by_number[34]["prompt"])
        self.assertIn("目标天气形势", self.by_number[39]["prompt"])
        self.assertEqual(self.by_number[33]["payload_kind"], "event_environment")
        for number in range(36, 43):
            self.assertEqual(
                self.by_number[number]["payload_kind"],
                "sequence_environment",
            )

    def test_acceptance_policy_is_goal_based_not_reference_answer_based(self) -> None:
        policy = self.payload["review_policy"]
        self.assertIn("回答了居民实际关心的问题", policy["pass_when"])
        self.assertIn("没有列出全部可计算指标", policy["do_not_rewrite_for"])

    def test_no_image_and_input_validation_cases_are_present(self) -> None:
        self.assertIn("没有天气图片", self.by_number[25]["prompt"])
        self.assertEqual(self.by_number[31]["payload_kind"], "invalid_sequence")
        self.assertIn("不要勉强运行模型", self.by_number[31]["prompt"])


if __name__ == "__main__":
    unittest.main()
