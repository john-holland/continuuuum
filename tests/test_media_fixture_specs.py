import json
from pathlib import Path


def test_fixture_specs_valid_shape():
    specs_path = Path(__file__).resolve().parent / "media_fixture_specs.json"
    assert specs_path.is_file(), "media_fixture_specs.json must exist"

    data = json.loads(specs_path.read_text(encoding="utf-8"))
    assert data.get("version") == 1
    profiles = data.get("profiles")
    assert isinstance(profiles, list) and profiles

    required = {"id", "description", "output_file", "max_size_mb"}
    for profile in profiles:
        assert required.issubset(profile.keys())
        assert profile["max_size_mb"] > 0


def test_fixture_profiles_cover_audio_video_and_image():
    specs_path = Path(__file__).resolve().parent / "media_fixture_specs.json"
    data = json.loads(specs_path.read_text(encoding="utf-8"))
    ids = {p["id"] for p in data["profiles"]}
    assert {"audio_heavy_short", "video_heavy_short", "image_source"}.issubset(ids)
