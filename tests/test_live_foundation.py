from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "sevir_world" / "data_access" / "live_foundation.py"


def _module():
    spec = importlib.util.spec_from_file_location("live_foundation_under_test", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_request_normalizes_time_location_and_signature():
    module = _module()
    request = module.normalize_live_request("2019-08-01T21:37:00Z", 34.75, -84.75)
    assert request["data_policy"] == "live"
    assert request["era5_time_utc"] == "2019-08-01T21:00:00Z"
    assert request["time_offset_minutes"] == 37
    assert request["area_north_west_south_east"] == [40.75, -96.75, 29.0, -73.0]
    assert len(request["request_key"]) == 20
    assert request["forecast_leads_hours"] == [6, 12, 18, 24]


def test_live_request_requires_explicit_time_and_location():
    module = _module()
    try:
        module.normalize_live_request("", None, None)
    except ValueError as error:
        assert "analysis_time_utc" in str(error)
    else:
        raise AssertionError("missing time must be rejected")

    try:
        module.normalize_live_request("2019-08-01T21:00:00Z", None, None)
    except ValueError as error:
        assert "latitude/longitude" in str(error)
    else:
        raise AssertionError("missing location must be rejected")
