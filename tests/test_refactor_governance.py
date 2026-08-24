from __future__ import annotations

from copy import deepcopy
import unittest

from scripts import refactor_gate


class RefactorGovernanceTests(unittest.TestCase):
    def test_every_registered_counter_probe_returns_a_probe(self) -> None:
        for criterion_id, counter_probe in refactor_gate.PROBES.items():
            with self.subTest(criterion_id=criterion_id):
                result = counter_probe()
                self.assertIsInstance(result, refactor_gate.Probe)
                self.assertEqual(result.criterion_id, criterion_id)

    def test_current_state_contains_only_two_accepted_phases(self) -> None:
        state = refactor_gate.load_state()
        report = refactor_gate.collect_report(state)

        self.assertTrue(report["governance_valid"], report["errors"])
        self.assertEqual(report["active_phase"], "phase_2")
        self.assertEqual(report["accepted_phases"], ["phase_1", "phase_2"])
        self.assertEqual(
            [item["status"] for item in report["phases"]],
            ["accepted", "accepted"],
        )
        self.assertEqual(report["phases"][0]["accepted"], 5)

    def test_phase_cannot_be_accepted_with_unaccepted_criteria(self) -> None:
        state = deepcopy(refactor_gate.load_state())
        state["phases"][1]["status"] = "accepted"
        state["phases"][1]["acceptance"][0]["status"] = "implemented"

        errors = refactor_gate.validate_state(state)

        self.assertTrue(
            any("阶段 accepted 但仍有未 accepted" in item for item in errors),
            errors,
        )

    def test_verified_criterion_requires_every_declared_evidence_type(self) -> None:
        state = deepcopy(refactor_gate.load_state())
        criterion = state["phases"][0]["acceptance"][0]
        criterion["status"] = "verified"
        criterion["evidence"] = [
            {
                "type": "targeted_test",
                "reference": "example",
                "result": "pass",
                "recorded_at": "2026-08-24T00:00:00Z",
                "notes": "只提供一种证据，用来验证缺失证据检查。",
            }
        ]

        errors = refactor_gate.validate_state(state)

        self.assertTrue(
            any("P1-01: 缺少通过证据" in item for item in errors),
            errors,
        )

    def test_later_phase_cannot_start_before_previous_phase_acceptance(self) -> None:
        state = deepcopy(refactor_gate.load_state())
        state["phases"][0]["status"] = "not_started"
        state["phases"][1]["status"] = "in_progress"
        state["active_phase"] = "phase_2"

        errors = refactor_gate.validate_state(state)

        self.assertTrue(
            any("phase_2: 前一阶段未验收" in item for item in errors),
            errors,
        )

    def test_phase_claim_accepts_the_two_remaining_phases(self) -> None:
        state = refactor_gate.load_state()
        report = refactor_gate.collect_report(state)

        self.assertEqual(refactor_gate.phase_claim_errors(state, report, "phase_1"), [])
        self.assertEqual(refactor_gate.phase_claim_errors(state, report, "phase_2"), [])

    def test_evidence_requires_auditable_reference_time_and_notes(self) -> None:
        state = deepcopy(refactor_gate.load_state())
        criterion = state["phases"][0]["acceptance"][0]
        criterion["status"] = "implemented"
        criterion["evidence"] = [{"type": "source_diff", "result": "pass"}]

        errors = refactor_gate.validate_state(state)

        self.assertTrue(
            any("证据缺少 reference, recorded_at, notes" in item for item in errors),
            errors,
        )

    def test_known_baseline_cannot_be_marked_resolved_before_acceptance(self) -> None:
        state = deepcopy(refactor_gate.load_state())
        state["phases"][1]["acceptance"][1]["status"] = "implemented"

        errors = refactor_gate.validate_state(state)

        self.assertTrue(
            any("B-02: 标为已解决" in item for item in errors),
            errors,
        )

    def test_counter_probes_support_formally_accepted_phase_two(self) -> None:
        report = refactor_gate.collect_report(refactor_gate.load_state())
        probes = {item["criterion_id"]: item for item in report["probes"]}

        # Source probes and persisted evidence must agree after final acceptance.
        for criterion_id in (
            "P1-01", "P1-02", "P1-03", "P1-04", "P1-05",
            "P2-01", "P2-02", "P2-03", "P2-04", "P2-05",
        ):
            self.assertTrue(probes[criterion_id]["ok"], probes[criterion_id])
        self.assertEqual(report["accepted_phases"], ["phase_1", "phase_2"])
        self.assertEqual(report["active_phase"], "phase_2")
        self.assertEqual(report["phases"][1]["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
