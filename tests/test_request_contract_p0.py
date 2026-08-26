from __future__ import annotations

import unittest

from sevir_world.agent.contracts import ExecutionRequest
from sevir_world.agent.tool_docs import selected_api_reference
from sevir_world.agent.unified import _request_context_for_prompt
from sevir_world.capabilities.base import RequestBinding


class RequestContractP0Tests(unittest.TestCase):
    def test_event_identity_cannot_split_between_ds_and_capabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "binding.event_id"):
            ExecutionRequest.from_payload(
                [{"source": "sevir", "event_id": "R1"}],
                ["ds", "sevir"],
                RequestBinding(event_id="R2"),
            )

    def test_upload_identity_cannot_split_between_ds_and_capabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "binding.upload_id"):
            ExecutionRequest.from_payload(
                [{"source": "upload", "upload_id": "upload-a"}],
                ["ds", "cascast"],
                RequestBinding(upload_id="upload-b"),
            )

    def test_data_identity_fills_an_empty_typed_binding(self) -> None:
        request = ExecutionRequest.from_payload(
            [
                {"source": "upload", "upload_id": "upload-a"},
                {"source": "sevir", "event_id": "r19062708588472"},
            ],
            ["ds", "sevir", "cascast"],
            RequestBinding(),
        )
        self.assertEqual(request.binding.upload_id, "upload-a")
        self.assertEqual(request.binding.event_id, "R19062708588472")

    def test_prompt_is_derived_from_the_execution_request(self) -> None:
        request = ExecutionRequest.from_payload(
            [{"source": "sevir", "event_id": "R1"}],
            ["ds", "sevir"],
            RequestBinding(event_id="R1", forecast_leads_hours=(3, 6)),
        )
        visible = _request_context_for_prompt(request)
        self.assertEqual(visible["assets"]["event_id"], "R1")
        self.assertEqual(visible["forecast"]["lead_hours"], (3, 6))
        self.assertEqual(visible["policy"]["mode"], "cached")
        self.assertEqual(visible["data_assets"], [
            {"source": "sevir", "event_id": "R1", "representation": "handle"}
        ])
        self.assertNotIn("sequence_upload_id", visible["assets"])

    def test_agent_reference_derives_request_binding_from_python_source(self) -> None:
        reference = selected_api_reference(("cascast", "aurora"))
        self.assertIn("cascast.request", reference)
        self.assertIn("RequestBinding(event_id: str = ''", reference)
        self.assertIn("forecast_leads_hours: tuple[int, ...] = (6, 12, 18, 24)", reference)
        self.assertIn("data_policy: str = 'cached'", reference)


if __name__ == "__main__":
    unittest.main()
