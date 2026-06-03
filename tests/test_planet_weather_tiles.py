import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve_library import app  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTINUUM_PLANET_WEATHER", str(tmp_path / "weather"))
    return app.test_client()


def test_weather_tiles_get_defaults(client):
    r = client.get("/api/planet/weather_tiles?planet_id=test&lat=10&lon=20")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert "cloud_base_m" in data
    assert "cloud_top_m" in data
    assert "altitude_band_mask" in data


def test_weather_tiles_post_and_get(client):
    payload = {
        "cloud_base_m": 1500.0,
        "cloud_top_m": 4000.0,
        "cloud_cover": 0.7,
        "pressure_scale_height": 9000.0,
        "altitude_band_mask": 8,
    }
    r = client.post(
        "/api/planet/weather_tiles?planet_id=test&lat=1.5&lon=2.5",
        json=payload,
    )
    assert r.status_code == 201
    r2 = client.get("/api/planet/weather_tiles?planet_id=test&lat=1.5&lon=2.5")
    assert r2.status_code == 200
    data = json.loads(r2.data)
    assert data["cloud_base_m"] == 1500.0
    assert data["cloud_top_m"] == 4000.0
