from __future__ import annotations

import ast
import inspect
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import sevir_world
import xarray as xr

from sevir_world.agent.contracts import data_descs_from_context
from sevir_world.agent.runtime import _TOOL_IMPORTS
from sevir_world.capabilities import Aurora, CasCast, Geolocator, Sevir
from sevir_world.capabilities.base import RequestBinding
from sevir_world.capabilities.models import Event, PixelLocation
from sevir_world.api.unified_service import bind_request_assets
import sevir_world.api.unified_service as unified_service


EVENT_ID = "R19062708588472"


def _recorder(*_args, **_kwargs) -> None:
    return None


def test_only_four_request_bound_capabilities_are_active() -> None:
    assert set(_TOOL_IMPORTS) == {"sevir", "geolocator", "cascast", "aurora"}
    assert {value[0] for value in _TOOL_IMPORTS.values()} == {
        "sevir_world.tools.sevir.Sevir",
        "sevir_world.tools.geolocator.Geolocator",
        "sevir_world.tools.cascast.CasCast",
        "sevir_world.tools.aurora.Aurora",
    }
    binding = RequestBinding(event_id=EVENT_ID, upload_id="upload-1")
    for capability in (Sevir, Geolocator, CasCast, Aurora):
        assert capability(_recorder, binding).request is binding


def test_public_method_contract_is_signature_return_and_example() -> None:
    for capability in (Sevir, Geolocator, CasCast, Aurora):
        for name, method in inspect.getmembers(capability, inspect.isfunction):
            if name.startswith("_"):
                continue
            signature = inspect.signature(method)
            assert signature.return_annotation is not inspect.Signature.empty
            doc = inspect.getdoc(method) or ""
            assert "Returns\n-------" in doc
            assert "Examples\n--------" in doc

    assert "event_id" not in inspect.signature(CasCast.forecast).parameters
    assert not hasattr(CasCast, "forecast_upload")


def test_every_tool_function_has_a_complete_source_docstring() -> None:
    """Keep implementation documentation complete without exposing it to the Agent."""
    tool_root = Path(__file__).resolve().parents[1] / "sevir_world" / "tools"
    failures: list[str] = []
    for path in sorted(tool_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            label = f"{path.relative_to(tool_root)}:{node.lineno}:{node.name}"
            doc = ast.get_docstring(node) or ""
            arguments = [
                item.arg
                for item in (*node.args.args, *node.args.kwonlyargs)
                if item.arg not in {"self", "cls"}
            ]
            returns = ast.unparse(node.returns) if node.returns is not None else ""
            if not doc:
                failures.append(f"{label} missing docstring")
            if arguments and "Parameters\n----------" not in doc:
                failures.append(f"{label} missing Parameters section")
            if returns and returns != "None" and "Returns\n-------" not in doc:
                failures.append(f"{label} missing Returns section")
    assert not failures, "\n".join(failures)


def test_agent_reference_exposes_only_public_capability_contracts() -> None:
    import sevir_world.agent.tool_docs as tool_docs
    from sevir_world.agent.tool_docs import selected_api_reference

    reference = selected_api_reference(("sevir", "geolocator", "cascast", "aurora"))
    for internal_name in (
        "_load_boundary_kmz",
        "_projection_context",
        "_load_vil_array",
        "_prepare_prediction_data",
        "regional_environment_fields",
    ):
        assert internal_name not in reference
    for public_name in (
        "sevir.event",
        "geolocator.pixel",
        "cascast.forecast",
        "aurora.regional_fields",
    ):
        assert public_name in reference
    generator_source = inspect.getsource(tool_docs.class_api_docs)
    assert "doc_limit" not in generator_source
    assert "[:600]" not in generator_source


def test_upload_and_explicit_event_are_both_bound() -> None:
    descriptions = data_descs_from_context({
        "upload_id": "upload-1",
        "event_id": EVENT_ID,
    })
    assert descriptions == [
        {"source": "upload", "upload_id": "upload-1", "contract_version": "sevir-data-v3"},
        {"source": "sevir", "event_id": EVENT_ID, "contract_version": "sevir-data-v3"},
    ]


def test_sevir_and_geolocator_return_direct_python_values() -> None:
    binding = RequestBinding(event_id=EVENT_ID)
    sevir = Sevir(_recorder, binding)
    event = sevir.event()
    observations = sevir.observations(minutes_before=60, minutes_after=0)
    pixel = Geolocator(_recorder, binding).pixel(100, 100)
    assert isinstance(event, Event)
    assert isinstance(observations, xr.DataArray)
    assert isinstance(pixel, PixelLocation)
    assert not isinstance(event, dict)
    assert observations.shape == (13, 384, 384)
    assert np.isfinite(observations.values).all()
    assert isinstance(pixel.latitude, float)


def test_cascast_preserves_the_exact_model_input_history() -> None:
    forecast = CasCast(_recorder, RequestBinding(event_id=EVENT_ID)).forecast(
        horizon_hours=1.0,
        ensemble_members=10,
    )
    assert isinstance(forecast, xr.Dataset)
    assert forecast["history_vil"].shape == (13, 384, 384)
    assert forecast.coords["history_minute"].values.tolist() == list(range(-60, 5, 5))
    assert forecast["forecast_mean_vil"].shape == (12, 384, 384)


def test_upload_binding_resolves_event_before_agent_construction() -> None:
    payload = {"upload_id": "upload-1", "data_source": "upload"}
    with patch(
        "sevir_world.api.unified_service.uploaded_event_match",
        return_value={"event_match": {"event_id": EVENT_ID, "status": "matched"}},
    ):
        bind_request_assets("请分析这张图", payload)
    assert payload["event_id"] == EVENT_ID
    assert payload["event_resolution"]["source"] == "bound_upload_exact_match"


def test_cached_event_does_not_promote_evidence_into_request_time_or_domain() -> None:
    payload = {"event_id": EVENT_ID, "data_policy": "cached"}
    bind_request_assets("分析 R190", payload)
    assert not payload.get("requested_time_utc")
    assert not payload.get("aurora_initialization_time_utc")
    assert payload.get("latitude") is None
    assert payload.get("longitude") is None
    assert payload.get("region") is None


def test_regional_request_exposes_all_lazy_capabilities_without_event_binding() -> None:
    payload = {
        "data_source": "environment",
        "data_policy": "cached",
        "requested_time_utc": "2019-06-27T03:00:00Z",
        "aurora_initialization_time_utc": "2019-06-27T03:00:00Z",
        "region": {"north": 53.05, "west": -111.98, "south": 41.3, "east": -88.23},
    }
    bind_request_assets("分析北达科他州未来6到24小时的大范围环境", payload)
    assert payload.get("event_id", "") == ""
    assert payload["requested_time_utc"] == "2019-06-27T03:00:00Z"
    assert payload["aurora_initialization_time_utc"] == "2019-06-27T03:00:00Z"
    assert payload["selected_tools"] == ["ds", "sevir", "geolocator", "cascast", "aurora"]


def test_return_field_names_are_spelled_as_documented() -> None:
    from sevir_world.capabilities.models import Location, Place, RegionCoverage, UploadMatch

    assert {field.name for field in __import__("dataclasses").fields(UploadMatch)} == {
        "upload_id", "subject_key", "event_id", "confidence", "method",
    }
    assert {field.name for field in __import__("dataclasses").fields(Location)} == {
        "subject_key", "event_id", "latitude", "longitude", "south", "west", "north", "east",
        "row_count", "column_count", "crs",
    }
    assert __import__("typing").get_type_hints(Place)["candidates"] is pd.DataFrame
    assert __import__("typing").get_type_hints(RegionCoverage)["areas"] is pd.DataFrame
    assert "row_count" in (inspect.getdoc(Geolocator.event) or "")
    assert "event_id`` is None" in (inspect.getdoc(Sevir.match_upload) or "")


def test_generated_reference_derives_omitted_fields_from_value_classes() -> None:
    from sevir_world.agent.tool_docs import selected_api_reference

    reference = selected_api_reference(("sevir", "geolocator", "cascast", "aurora"))
    assert "geolocator.pixel" in reference
    assert "PixelLocation.projected_x" in reference
    assert "cascast.forecast" in reference
    assert "forecast_member_vil" in reference
    assert "aurora.regional_fields" in reference
    assert "lead_hour" in reference


def test_agent_admission_serializes_requests_before_budget() -> None:
    active = 0
    maximum = 0
    lock = threading.Lock()
    original_slots = unified_service._AGENT_REQUEST_SLOTS
    original_runner = unified_service._execute_unified_request_held

    def fake_runner(*_args, **_kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return {"status": "ok"}

    unified_service._AGENT_REQUEST_SLOTS = threading.BoundedSemaphore(1)
    unified_service._execute_unified_request_held = fake_runner
    try:
        results: list[dict] = []

        def invoke() -> None:
            results.append(unified_service.execute_unified_request(
                "qwen", "deepseek-v3.2", "test", {"_cancelled": lambda: False},
                providers={"qwen": {}}, complete=lambda *_args: {},
            ))

        first = threading.Thread(target=invoke)
        second = threading.Thread(target=invoke)
        first.start(); second.start(); first.join(); second.join()
        assert maximum == 1
        assert any(item["request_admission"]["queue_wait_ms"] > 0 for item in results)
    finally:
        unified_service._AGENT_REQUEST_SLOTS = original_slots
        unified_service._execute_unified_request_held = original_runner


def test_old_architecture_is_absent_from_active_tree() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "sevir_world/agent/schema.py",
        "sevir_world/data_access/tool_contracts.py",
        "sevir_world/data_access/foundation.py",
        "sevir_world/data_access/foundation_cache.py",
        "sevir_world/data_access/foundation_validation.py",
        "sevir_world/api/tool_gateway.py",
        "legacy_architecture",
    ):
        assert not (root / relative).exists()
    assert not (root / "sevir_world/capabilities/weather.py").exists()
    for relative in (
        "sevir_world/tools/sevir/Sevir.py",
        "sevir_world/tools/geolocator/Geolocator.py",
        "sevir_world/tools/cascast/CasCast.py",
        "sevir_world/tools/aurora/Aurora.py",
    ):
        assert (root / relative).is_file()


class CapabilityContractTests(unittest.TestCase):
    def test_active_capability_count(self) -> None:
        test_only_four_request_bound_capabilities_are_active()

    def test_complete_method_contracts(self) -> None:
        test_public_method_contract_is_signature_return_and_example()

    def test_complete_implementation_docstrings(self) -> None:
        test_every_tool_function_has_a_complete_source_docstring()

    def test_internal_helpers_are_not_agent_contracts(self) -> None:
        test_agent_reference_exposes_only_public_capability_contracts()

    def test_combined_request_binding(self) -> None:
        test_upload_and_explicit_event_are_both_bound()

    def test_direct_domain_values(self) -> None:
        test_sevir_and_geolocator_return_direct_python_values()

    def test_cascast_input_history(self) -> None:
        test_cascast_preserves_the_exact_model_input_history()

    def test_upload_event_prebinding(self) -> None:
        test_upload_binding_resolves_event_before_agent_construction()

    def test_return_field_names(self) -> None:
        test_return_field_names_are_spelled_as_documented()

    def test_agent_admission(self) -> None:
        test_agent_admission_serializes_requests_before_budget()

    def test_no_old_architecture(self) -> None:
        test_old_architecture_is_absent_from_active_tree()
