import json
from pathlib import Path


def test_media_parity_matrix_has_required_shape():
    matrix_path = Path(__file__).resolve().parent.parent / "library" / "media_parity_matrix.json"
    assert matrix_path.is_file(), "media_parity_matrix.json must exist"

    data = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert data.get("source") == "video_storage_tool"
    features = data.get("features")
    assert isinstance(features, list) and features, "features list must be non-empty"

    required_keys = {"id", "category", "feature", "current_behavior", "usc_owner", "continuum_owner", "target_api", "status"}
    for feature in features:
        assert required_keys.issubset(feature.keys()), f"missing keys in feature: {feature}"


def test_media_parity_matrix_has_no_unmapped_rows():
    matrix_path = Path(__file__).resolve().parent.parent / "library" / "media_parity_matrix.json"
    data = json.loads(matrix_path.read_text(encoding="utf-8"))
    unmapped = [f for f in data.get("features", []) if f.get("status") == "Not mapped"]
    assert not unmapped, f"Found unmapped features: {[f['id'] for f in unmapped]}"


def test_media_parity_matrix_includes_fixture_strategy():
    matrix_path = Path(__file__).resolve().parent.parent / "library" / "media_parity_matrix.json"
    data = json.loads(matrix_path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    ids = {f["id"] for f in features}
    assert "tests-bitexact" in ids
