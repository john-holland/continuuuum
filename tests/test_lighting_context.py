from __future__ import annotations

from lighting_context import LightingContextService, WeatherSnapshot, WeatherAdapter


class _StubWeather(WeatherAdapter):
    name = "stub-weather"

    def get_snapshot(self, *, lat: float, lon: float, when_utc):
        return WeatherSnapshot(
            provider=self.name,
            cloud_cover_pct=65.0,
            visibility_m=8000.0,
            precipitation_mm=0.1,
            wind_speed_mps=5.0,
        )


def test_infers_sun_direction_from_surface_shadow_metadata():
    svc = LightingContextService(weather_adapters={"stub-weather": _StubWeather()}, default_weather_adapter="stub-weather")
    result = svc.compute(
        lat=47.62,
        lon=-122.35,
        metadata={
            "capture_datetime_utc": "2026-02-21T18:30:00Z",
            "shadow_vector": [1.0, 0.0, 0.0],
            "sun_plane_normal": [0.0, 1.0, 0.0],
        },
    )
    assert result["sun_direction_source"] == "inferred_surface"
    assert "sun_visibility" in result
    assert isinstance(result["sun_visibility"], bool)
    assert result["inferred_sun_direction_vector"] is not None
    assert result["inferred_sun_direction_confidence"] > 0.0
    assert "sun_direction_vector_world" in result
    assert "moon_visibility" in result
    assert isinstance(result["moon_visibility"], bool)
    assert "moon_azimuth_deg" in result
    assert "moon_elevation_deg" in result


def test_metadata_direction_takes_priority_over_inference():
    svc = LightingContextService(weather_adapters={"stub-weather": _StubWeather()}, default_weather_adapter="stub-weather")
    result = svc.compute(
        lat=34.05,
        lon=-118.24,
        metadata={
            "capture_datetime_utc": "2026-07-03T12:00:00Z",
            "sun_direction_vector_world": [0.0, 0.8, 0.2],
            "sun_direction_confidence": 0.95,
            "shadow_vector": [1.0, 0.0, 0.0],
            "sun_plane_normal": [0.0, 1.0, 0.0],
        },
    )
    assert result["sun_direction_source"] == "metadata"
    assert isinstance(result["sun_visibility"], bool)
    assert result["sun_direction_confidence"] == 0.95
    assert result["inferred_sun_direction_vector"] is not None
    assert "moon_visibility" in result


def test_rollout_flags_can_disable_surface_inference():
    svc = LightingContextService(weather_adapters={"stub-weather": _StubWeather()}, default_weather_adapter="stub-weather")
    result = svc.compute(
        lat=40.0,
        lon=-74.0,
        metadata={
            "capture_datetime_utc": "2026-02-21T18:30:00Z",
            "shadow_vector": [1.0, 0.0, 0.0],
            "sun_plane_normal": [0.0, 1.0, 0.0],
            "lighting_rollout_flags": {"enable_surface_inference": False},
        },
    )
    assert result["sun_direction_source"] == "calculated"
    assert isinstance(result["sun_visibility"], bool)
    assert result["inferred_sun_direction_vector"] is None
    assert result["lighting_rollout_flags"]["enable_surface_inference"] is False
    assert "moon_visibility" in result


def test_query_lighting_falls_back_to_compute_without_ephemeris():
    svc = LightingContextService(weather_adapters={"stub-weather": _StubWeather()}, default_weather_adapter="stub-weather")
    result = svc.query_lighting(
        observer_body_id="earth",
        lat=47.62,
        lon=-122.35,
        datetime_utc="2026-02-21T12:00:00Z",
        include_eclipses=True,
    )
    assert "sun_azimuth_deg" in result
    assert "sun_elevation_deg" in result
    assert "sun_visibility" in result
    assert "moon_azimuth_deg" in result
    assert result.get("light_sources") is None or isinstance(result.get("light_sources"), list)
    assert "aggregate_direction" not in result or result.get("aggregate_direction") is not None


def test_query_lighting_preserves_earth_moon_compatibility():
    svc = LightingContextService(weather_adapters={"stub-weather": _StubWeather()}, default_weather_adapter="stub-weather")
    result = svc.query_lighting(lat=40.0, lon=-74.0, datetime_utc="2026-07-03T18:00:00Z")
    assert result["sun_azimuth_deg"] is not None
    assert result["sun_elevation_deg"] is not None
    assert result["moon_azimuth_deg"] is None or isinstance(result["moon_azimuth_deg"], (int, float))
    assert "lighting_validity_score" in result
    assert result["weather_snapshot"]["provider"] == "stub-weather"


def test_rollout_flags_can_disable_lunar_context():
    svc = LightingContextService(weather_adapters={"stub-weather": _StubWeather()}, default_weather_adapter="stub-weather")
    result = svc.compute(
        lat=40.0,
        lon=-74.0,
        metadata={
            "capture_datetime_utc": "2026-02-21T18:30:00Z",
            "lighting_rollout_flags": {"enable_lunar_context": False},
        },
    )
    assert result["moon_visibility"] is False
    assert result["moon_azimuth_deg"] is None
    assert result["moon_elevation_deg"] is None
    assert result["lighting_rollout_flags"]["enable_lunar_context"] is False
