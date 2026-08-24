from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import numpy as np

from sevir_world.agent.contracts import data_descs_from_context
from sevir_world.agent.runtime import _TOOL_IMPORTS
from sevir_world.capabilities import Aurora, CasCast, Geolocator, Sevir
from sevir_world.capabilities.base import RequestBinding
from sevir_world.capabilities.models import Event, ObservationCube, PixelLocation


EVENT_ID = "R19062708588472"


def _recorder(*_args, **_kwargs) -> None:
    return None


def test_only_four_request_bound_capabilities_are_active() -> None:
    assert set(_TOOL_IMPORTS) == {"sevir", "geolocator", "cascast", "aurora"}
    assert {value[0] for value in _TOOL_IMPORTS.values()} == {
        "sevir_world.capabilities.weather"
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
            assert "Returns:" in doc
            assert "Example:" in doc


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
    observations = sevir.observations(end_frame=2)
    pixel = Geolocator(_recorder, binding).pixel(100, 100)
    assert isinstance(event, Event)
    assert isinstance(observations, ObservationCube)
    assert isinstance(pixel, PixelLocation)
    assert not isinstance(event, dict)
    assert observations.values.shape == (2, 384, 384)
    assert np.isfinite(observations.values[observations.values != 255]).all()
    assert isinstance(pixel.latitude, float)


def test_old_architecture_is_absent_from_active_tree() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "sevir_world/tools",
        "sevir_world/agent/schema.py",
        "sevir_world/data_access/tool_contracts.py",
        "sevir_world/api/tool_gateway.py",
        "legacy_architecture",
    ):
        assert not (root / relative).exists()


class CapabilityContractTests(unittest.TestCase):
    def test_active_capability_count(self) -> None:
        test_only_four_request_bound_capabilities_are_active()

    def test_complete_method_contracts(self) -> None:
        test_public_method_contract_is_signature_return_and_example()

    def test_combined_request_binding(self) -> None:
        test_upload_and_explicit_event_are_both_bound()

    def test_direct_domain_values(self) -> None:
        test_sevir_and_geolocator_return_direct_python_values()

    def test_no_old_architecture(self) -> None:
        test_old_architecture_is_absent_from_active_tree()
