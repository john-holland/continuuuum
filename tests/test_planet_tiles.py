"""Planet tile API smoke tests."""
from pathlib import Path

import pytest

pytest.importorskip("flask")


def test_planet_tile_path_helper(tmp_path, monkeypatch):
    import serve_library as lib

    monkeypatch.setattr(lib, "_PLANET_TILES_DIR", tmp_path)
    p = lib._planet_tile_path("earth", 0, 1, 2, 3)
    assert p == tmp_path / "earth" / "0" / "1" / "2_3.bin"
