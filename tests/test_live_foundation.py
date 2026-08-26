from __future__ import annotations

from pathlib import Path

from sevir_world.data_access import live_foundation


def _module():
    return live_foundation


def test_live_request_normalizes_time_location_and_signature():
    module = _module()
    request = module.normalize_live_request("2019-08-01T21:37:00Z", 34.75, -84.75)
    assert request["data_policy"] == "live"
    assert request["era5_time_utc"] == "2019-08-01T21:00:00Z"
    assert request["time_offset_minutes"] == 37
    assert request["area_north_west_south_east"] == [40.75, -96.75, 29.0, -73.0]
    assert len(request["request_key"]) == 32
    assert request["forecast_leads_hours"] == [6, 12, 18, 24]


def test_aurora_forecast_identity_does_not_depend_on_sevir_event():
    module = _module()
    region = {"north": 53.05, "west": -111.98, "south": 41.3, "east": -88.23}
    plain = module.normalize_live_request(
        "2019-06-27T03:00:00Z", region=region, event_id="",
        history_before_hours=6, history_after_hours=24,
    )
    linked_case = module.normalize_live_request(
        "2019-06-27T03:00:00Z", region=region, event_id="R19062708588472",
        history_before_hours=24, history_after_hours=24,
    )
    assert plain["aurora_request_key"] == linked_case["aurora_request_key"]
    assert plain["artifact_dir"] == linked_case["artifact_dir"]
    assert "R19062708588472" not in plain["artifact_dir"]
    assert "event_id" not in plain["aurora_forecast_identity"]


def test_prepared_0300_aurora_cache_is_found_without_event_binding():
    module = _module()
    region = {"north": 53.05, "west": -111.98, "south": 41.3, "east": -88.23}
    request = module.normalize_live_request(
        "2019-06-27T03:00:00Z", region=region, event_id="",
        history_before_hours=6, history_after_hours=24,
    )
    era5 = module.fetch_era5_live(request, cache_only=True)
    assert era5["status"] == "ok"
    assert era5["cache_hit"] is True
    assert era5["cache_compatibility"] == "regional_forecast_identity"
    forecast = module.run_aurora_live(
        request, artifact_dir=era5["artifact_dir"], cache_only=True,
    )
    assert forecast["status"] == "ok"
    assert forecast["cache_hit"] is True
    assert Path(forecast["output"]).name == "aurora_forecast_aligned_0300.npz"


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
