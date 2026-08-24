import unittest
import inspect

import numpy as np

from sevir_world.agent.schema import (
    ContractDict, RESULT_CONTRACTS, contractize, describe_dataset, describe_value,
    normalize_tool_result, register_result_contract,
)
from sevir_world.agent.contracts import ExecutionRequest, data_descs_for_event
from sevir_world.agent.runtime import DatasetRecord, ToolInterface
from sevir_world.tools.base import WorldComponent


class SchemaContractTests(unittest.TestCase):
    def test_dataset_record_values_attribute_is_not_dict_values_method(self):
        values = np.zeros((2, 2), dtype=np.float32)
        record = DatasetRecord({"values": values, "role": "local_observation"})
        self.assertIs(record.values, values)
        self.assertIs(record["values"], values)

    def test_explicit_event_does_not_displace_uploaded_pixels(self):
        descriptions = data_descs_for_event(
            "R19080121067677",
            [{"source": "upload", "upload_id": "u1"}],
        )
        self.assertEqual(descriptions[0]["source"], "upload")
        self.assertEqual(descriptions[1]["source"], "sevir")
        self.assertNotIn("foundation_case", [item["source"] for item in descriptions])

    def test_local_event_analysis_binds_an_observation_sequence(self):
        descriptions = data_descs_for_event(
            "R19062708588472",
            [{"source": "sevir", "event_id": "R19062708588472"}],
            include_observations=True,
            observation_window_minutes=(-60, 0),
        )
        self.assertEqual(len(descriptions), 1)
        self.assertEqual(descriptions[0]["representation"], "observation_sequence")
        self.assertEqual(descriptions[0]["observation_window_minutes"], [-60, 0])

    def test_execution_request_keeps_local_observation_primary(self):
        request = ExecutionRequest.from_payload([
            {"source": "foundation_case", "event_id": "e"},
            {"source": "upload", "upload_id": "u"},
        ], ["ds"])
        self.assertEqual(request.data_descs[0]["source"], "upload")

    def test_unavailable_tool_result_is_explicitly_unavailable(self):
        result = normalize_tool_result("foundation.fetch_era5", {
            "status": "dependency_missing",
            "error_code": "cdsapi_not_installed",
        })
        self.assertEqual(result["result_status"], "dependency_missing")
        self.assertFalse(result["available"])

    def test_contract_objects_support_attribute_access(self):
        value = contractize({"forecast": {"frames": [1]}, "lead_minutes": [5]})
        self.assertIsInstance(value, ContractDict)
        self.assertEqual(value.forecast.frames, [1])

    def test_tool_interface_preserves_real_adapter_results(self):
        class DirectAdapter:
            def render(self):
                return {"status": "ok", "artifact_id": "a1", "image_url": "/a1.png"}

        tool = ToolInterface("visualizer", DirectAdapter(), ("render",))
        result = tool.render()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["artifact_id"], "a1")
        self.assertEqual(result["image_url"], "/a1.png")

    def test_tool_interface_supports_on_demand_introspection(self):
        class DirectAdapter:
            def render(self, title: str, width: int = 640):
                """Render one requested figure."""
                return {"title": title, "width": width}

        tool = ToolInterface("visualizer", DirectAdapter(), ("render",))
        self.assertIn("render", dir(tool))
        self.assertEqual(
            str(inspect.signature(tool.render)),
            "(title: str, width: int = 640)",
        )
        self.assertIn("requested figure", inspect.getdoc(tool.render))

    def test_dataset_description_reports_real_dimensions_and_dtype(self):
        values = np.zeros((13, 384, 384), dtype=np.float32)
        values.setflags(write=False)
        description = describe_dataset([{
            "source": "upload",
            "upload_id": "u1",
            "kind": "sequence",
            "shape": [13, 384, 384],
            "values": values,
            "relative_minutes": [-60, 0],
        }])
        record = description["records"][0]
        self.assertEqual(record["fields"]["values"]["shape"], [13, 384, 384])
        self.assertEqual(record["fields"]["values"]["dimensions"], ["frame", "row", "column"])
        self.assertTrue(record["fields"]["values"]["read_only"])

    def test_forecast_legacy_result_gets_canonical_paths_without_losing_fields(self):
        frames = np.zeros((12, 384, 384), dtype=np.float32)
        result = normalize_tool_result("forecaster.rolling_raw", {
            "forecast_frames": frames,
            "forecast_relative_minutes": [5, 10],
            "lead_minutes": 60,
        })
        self.assertIs(result["forecast"]["frames"], frames)
        self.assertEqual(result["forecast"]["lead_minutes"], [5, 10])
        self.assertEqual(result["forecast"]["horizon_minutes"], 60)
        self.assertIn("forecast_frames", result)

    def test_rolling_ensemble_is_available_at_the_documented_data_root(self):
        members = [
            {"frames": np.zeros((12, 8, 8), dtype=np.float32), "lead_minutes": [5, 10]}
            for _ in range(3)
        ]
        result = normalize_tool_result("forecaster.rolling_raw", {
            "status": "ok",
            "forecast_frames": np.zeros((12, 8, 8), dtype=np.float32),
            "forecast_relative_minutes": [5, 10],
            "lead_minutes": 60,
            "ensemble": {"members": members, "member_count": 3},
        })
        self.assertEqual(len(result.data.members), 3)
        self.assertIs(result.data.members[0].frames, members[0]["frames"])
        self.assertEqual(result.data.member_count, 3)
        self.assertEqual(result.data.relative_minutes, [5, 10])
        self.assertNotIn("lead_minutes", result.data)
        self.assertEqual(result.data.horizon_minutes, 60)
        self.assertEqual(result.data_kind, "ensemble_forecast")

    def test_nested_schema_describes_available_keys_after_tool_call(self):
        schema = describe_value({
            "ground_truth": {
                "frames": np.zeros((2, 4, 4), dtype=np.uint8),
                "lead_minutes": [5, 10],
            }
        })
        fields = schema["fields"]["ground_truth"]["fields"]
        self.assertEqual(fields["frames"]["shape"], [2, 4, 4])
        self.assertEqual(fields["lead_minutes"]["type"], "array")

    def test_world_component_normalizes_once_and_reuses_cached_payload(self):
        calls = []
        component = WorldComponent(lambda *args, **kwargs: calls.append((args, kwargs)))
        backend_calls = []

        def backend(**kwargs):
            backend_calls.append(kwargs)
            return {"records": [{"frames": np.zeros((1, 2, 2))}], "ensemble_members": 1}

        first = component._call("uncertainty.uploaded_raw", {"upload_id": "u1"}, backend)
        second = component._call("uncertainty.uploaded_raw", {"upload_id": "u1"}, backend)
        self.assertIs(first, second)
        self.assertEqual(len(backend_calls), 1)
        self.assertEqual(first["ensemble"]["member_count"], 1)
        self.assertEqual(first["schema_version"], "sevir-contract-2")

    def test_all_mapping_results_have_one_canonical_data_root(self):
        result = normalize_tool_result("data.match_uploaded_event", {
            "status": "ok", "event_id": "R19062708588472", "confidence": 0.9,
        })
        self.assertEqual(result.data.event_id, "R19062708588472")
        self.assertEqual(result.data.confidence, 0.9)
        self.assertTrue(result.available)

    def test_observation_and_foundation_results_share_data_root(self):
        frames = np.zeros((13, 8, 8), dtype=np.float32)
        observation = normalize_tool_result("data.load_hdf5", {
            "frames": frames, "relative_minutes": [-60, 0], "missing_value": 255,
        })
        self.assertIs(observation.data.values, frames)
        self.assertEqual(observation.data.relative_minutes, [-60, 0])

        latitude = np.array([1.0], dtype=np.float32)
        foundation = normalize_tool_result("foundation.region_fields", {
            "latitude": latitude, "longitude": np.array([2.0]),
            "era5": {"surface": {}}, "aurora": {"lead_hours": [6, 12]},
        })
        self.assertIs(foundation.data.latitude, latitude)
        self.assertEqual(foundation.data.aurora.lead_hours, [6, 12])

    def test_new_model_can_register_shape_contract_without_executor_changes(self):
        register_result_contract(
            "test-foundation-output",
            lambda value: "latent_grid" in value,
            lambda value: value.setdefault("output", {"grid": value["latent_grid"]}),
        )
        try:
            result = normalize_tool_result("new_model.predict", {"latent_grid": [1, 2]})
            self.assertEqual(result["output"]["grid"], [1, 2])
            self.assertIn("test-foundation-output", result["contracts_applied"])
        finally:
            RESULT_CONTRACTS.pop()


if __name__ == "__main__":
    unittest.main()
