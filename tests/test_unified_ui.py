from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UnifiedUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        cls.js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

    def test_ui_exposes_one_agent_execution_flow(self) -> None:
        self.assertIn("理解问题 → 按需读取数据 → 计算/绘图 → 根据证据回答", self.js)
        self.assertIn("agent_request_budget_seconds || 90", self.js)
        self.assertIn("统一 Agent", self.js)
        for retired in (
            "Reflective", "reflective_trace", "Direct 快速分析",
            "direct_trace", "reviewed-acceptance", "精选答案",
        ):
            self.assertNotIn(retired, self.js)

    def test_logs_are_human_readable_and_do_not_show_raw_json(self) -> None:
        self.assertIn("调用语言模型决定下一步", self.js)
        self.assertIn("执行代码、读取观察并继续分析", self.js)
        self.assertNotIn("raw-json", self.html)
        self.assertNotIn("JSON.stringify(data, null, 2)", self.js)

    def test_image_upload_is_optional(self) -> None:
        self.assertIn('id="question"', self.html)
        self.assertIn("/api/run", self.js)
        self.assertNotIn("每次必须上传", self.html + self.js)

    def test_turn_presentation_receives_response_and_tool_runs_explicitly(self) -> None:
        self.assertIn(
            "function attachTurnPresentation(anchorMessage, data, runs)",
            self.js,
        )
        self.assertIn(
            "attachTurnPresentation(anchorMessage, data, runs)",
            self.js,
        )
        self.assertNotIn("function attachTurnPresentation(anchorMessage) {", self.js)

    def test_presentation_failure_is_not_reported_as_agent_failure(self) -> None:
        self.assertIn("Agent回答已返回，但页面展示失败", self.js)
        self.assertIn("Agent回答已成功返回，但部分页面展示失败", self.js)


if __name__ == "__main__":
    unittest.main()
