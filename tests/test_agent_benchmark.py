from __future__ import annotations

import unittest

from sevir_world.evaluation.agent_benchmark import AUTONOMY_SCENARIOS, evaluate_result


class AgentBenchmarkTests(unittest.TestCase):
    def test_unified_agent_result_passes_single_statistic_scenario(self) -> None:
        result = {
            "status": "ok",
            "route_mode": "unified_agent",
            "observations": [{"result": {"median": 235}}],
            "agent_trace": [{
                "action": "execute",
                "code": "result = statistics.aggregate([], 'x', operations=['median'])",
            }],
        }
        evaluation = evaluate_result(result, AUTONOMY_SCENARIOS[0])
        self.assertTrue(evaluation["passed"])

    def test_fixed_answer_without_execution_is_detected(self) -> None:
        result = {
            "status": "ok",
            "route_mode": "unified_agent",
            "observations": [],
            "agent_trace": [],
        }
        evaluation = evaluate_result(result, AUTONOMY_SCENARIOS[0])
        self.assertFalse(evaluation["checks"]["generated_code"])
        self.assertFalse(evaluation["passed"])


if __name__ == "__main__":
    unittest.main()
