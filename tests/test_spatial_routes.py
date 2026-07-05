"""Smoke tests for /api/spatial/* routes."""

import os
import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["CONTINUUM_DB_PATH"] = _tmp.name
os.environ.pop("CONTINUUM_API_KEY", None)

conn = sqlite3.connect(_tmp.name)
conn.executescript(
    """
    CREATE TABLE episodes (id TEXT PRIMARY KEY, title TEXT);
    CREATE TABLE vocabulary_render_masks (
        id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL DEFAULT 'default',
        asset_synonym TEXT NOT NULL, episode_id TEXT);
    CREATE TABLE vocabulary_render_mask_buckets (
        id TEXT PRIMARY KEY, mask_id TEXT NOT NULL, bucket_id TEXT NOT NULL);
    CREATE TABLE episode_assets (
        id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, usc_asset_id TEXT NOT NULL,
        asset_type TEXT NOT NULL, role TEXT, causality_leaf_id TEXT);
    """
)
ep = str(uuid.uuid4())
conn.execute("INSERT INTO episodes (id, title) VALUES (?, ?)", (ep, "Test"))
mid = str(uuid.uuid4())
conn.execute(
    "INSERT INTO vocabulary_render_masks (id, tenant_id, asset_synonym, episode_id) VALUES (?,?,?,?)",
    (mid, "default", "ladder", ep),
)
conn.execute(
    "INSERT INTO vocabulary_render_mask_buckets (id, mask_id, bucket_id) VALUES (?,?,?)",
    (str(uuid.uuid4()), mid, "Q2.1.3"),
)
conn.execute(
    "INSERT INTO episode_assets (id, episode_id, usc_asset_id, asset_type, role, causality_leaf_id) VALUES (?,?,?,?,?,?)",
    (str(uuid.uuid4()), ep, "42", "document", "scene_prop", "O2.1.7"),
)
conn.commit()
conn.close()

from serve_library import app  # noqa: E402


client = app.test_client()


def test_render_masks_list():
    r = client.get(f"/api/spatial/render-masks?episodeId={ep}")
    assert r.status_code == 200
    data = r.get_json()
    assert len(data["items"]) == 1
    assert data["items"][0]["assetSynonym"] == "ladder"
    assert "Q2.1.3" in data["items"][0]["bucketIds"]


def test_bucket_tree():
    r = client.get(f"/api/spatial/bucket-tree?episodeId={ep}")
    assert r.status_code == 200
    data = r.get_json()
    assert data["bucketCount"] >= 2
    assert "Q2.1.3" in data["bucketIds"]
    assert "O2.1.7" in data["bucketIds"]


def test_resolve_highlight():
    r = client.get(f"/api/spatial/resolve-highlight?token=42&episodeId={ep}")
    assert r.status_code == 200
    data = r.get_json()
    assert data["libraryDocumentId"] == 42
    assert data["causalityLeafId"] == "O2.1.7"
