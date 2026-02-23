"""
Smoke tests for continuum library API (search returns 200, upload then fetch).
Run from continuum repo root: pytest tests/
"""
import os
import sys
import tempfile
from pathlib import Path

# Ensure repo root is on path so serve_library can be imported
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Use a temp DB so we don't touch a real DB; set before importing serve_library
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["CONTINUUM_DB_PATH"] = _tmp.name
os.environ.pop("CONTINUUM_API_KEY", None)

from serve_library import app  # noqa: E402


class _FakeMediaService:
    def __init__(self):
        self.jobs = {"job-1": {"status": "ready", "manifest": {"id": "job-1"}}}
        self.settings = {"stream_cache": {"enabled": False}}
        self.stream_file = _repo_root / "tests" / "fixtures" / "media_stream.bin"
        self.stream_file.parent.mkdir(parents=True, exist_ok=True)
        self.stream_file.write_bytes(b"0123456789")
        self.t2v = {"status": "idle", "message": "", "model_id": ""}

    def store(self, input_path, tenant_id, settings=None):
        return {"id": "job-1", "status": "processing"}

    def list_jobs(self, tenant_id):
        return [{"id": "job-1", "status": "ready"}]

    def get_job_status(self, job_id, tenant_id):
        if job_id != "job-1":
            raise FileNotFoundError(job_id)
        return self.jobs[job_id]

    def retry_store(self, job_id, tenant_id, force_script=False):
        if job_id != "job-1":
            raise FileNotFoundError(job_id)
        return {"id": job_id, "status": "processing"}

    def reconstitute(self, job_id, tenant_id, use_original):
        if job_id != "job-1":
            raise FileNotFoundError(job_id)
        return {"path": str(self.stream_file), "out_path": "reconstituted.mp4"}

    def stream_info(self, job_id, tenant_id, use_original):
        if job_id != "job-1":
            raise FileNotFoundError(job_id)
        return {"content_length": 10, "filename": "reconstituted.mp4", "original": use_original, "path": str(self.stream_file)}

    def open_stream(self, job_id, tenant_id, use_original, byte_range):
        if job_id != "job-1":
            raise FileNotFoundError(job_id)
        total = 10
        if byte_range is None:
            start, end = 0, total - 1
            partial = False
        else:
            start, end = byte_range
            partial = True
        return {
            "path": self.stream_file,
            "start": start,
            "end": end,
            "total": total,
            "content_length": end - start + 1,
            "partial": partial,
        }

    def get_settings(self):
        return self.settings

    def update_settings(self, body):
        self.settings.update(body)
        return {"ok": True}

    def start_t2v_download(self):
        self.t2v = {"status": "downloading", "message": "started", "model_id": "foo/bar"}
        return {"ok": True}

    def get_t2v_download_status(self):
        return self.t2v


def test_search_returns_200():
    with app.test_client() as client:
        r = client.get("/api/library/search")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, list)


def test_upload_then_fetch():
    with app.test_client() as client:
        # Upload a minimal document (no file)
        r = client.post(
            "/api/library/upload",
            data={
                "document_type": "document",
                "type_metadata": "{}",
            },
            headers={"X-Tenant-ID": "default"},
        )
    assert r.status_code == 201
    data = r.get_json()
    assert "id" in data
    doc_id = data["id"]

    with app.test_client() as client:
        r = client.get(f"/api/library/documents/{doc_id}", headers={"X-Tenant-ID": "default"})
    assert r.status_code == 200
    doc = r.get_json()
    assert doc["id"] == doc_id
    assert doc["document_type"] == "document"


def test_media_parity_endpoint_returns_matrix():
    with app.test_client() as client:
        r = client.get("/api/admin/media-parity")
    assert r.status_code == 200
    data = r.get_json()
    assert data["source"] == "video_storage_tool"
    assert isinstance(data.get("features"), list)


def test_media_endpoints_surface(monkeypatch):
    import serve_library as module

    fake = _FakeMediaService()
    monkeypatch.setattr(module, "get_media_service", lambda: fake)

    with app.test_client() as client:
        r = client.get("/api/media/stored", headers={"X-Tenant-ID": "default"})
    assert r.status_code == 200
    assert r.get_json()["ids"] == ["job-1"]

    with app.test_client() as client:
        r = client.get("/api/media/stored/job-1/status", headers={"X-Tenant-ID": "default"})
    assert r.status_code == 200
    assert r.get_json()["status"] == "ready"

    with app.test_client() as client:
        r = client.post("/api/media/reconstitute", json={"stored_id": "job-1"}, headers={"X-Tenant-ID": "default"})
    assert r.status_code == 200
    assert "/api/media/stream/job-1" in r.get_json()["stream_url"]

    with app.test_client() as client:
        r = client.get("/api/media/stream/job-1", headers={"X-Tenant-ID": "default", "Range": "bytes=0-3"})
    assert r.status_code == 206
    assert r.headers["Content-Range"] == "bytes 0-3/10"
    assert r.data == b"0123"

    with app.test_client() as client:
        r = client.get("/api/media/settings")
    assert r.status_code == 200

    with app.test_client() as client:
        r = client.put("/api/media/settings", json={"stream_cache": {"enabled": True}})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_lighting_query_endpoint(monkeypatch):
    """Lighting query returns earth/moon compatible response with optional multi-body fields."""
    import serve_library as module

    class _FakeLightingService:
        def query_lighting(self, *, lat, lon, datetime_utc=None, **kw):
            return {
                "sun_azimuth_deg": 180.0,
                "sun_elevation_deg": 45.0,
                "sun_visibility": True,
                "moon_azimuth_deg": 210.0,
                "moon_elevation_deg": 12.0,
                "moon_visibility": True,
                "capture_datetime_utc": (datetime_utc or "2026-02-21T12:00:00Z").replace("Z", "+00:00"),
                "light_sources": [],
                "aggregate_direction": [0.0, 0.707, 0.707],
            }

    monkeypatch.setattr(module, "get_lighting_context_service", lambda: _FakeLightingService())
    with app.test_client() as client:
        r = client.get(
            "/api/library/lighting/query",
            query_string={"lat": 47.62, "lon": -122.35, "datetime_utc": "2026-02-21T12:00:00Z"},
        )
    assert r.status_code == 200
    data = r.get_json()
    assert "sun_azimuth_deg" in data
    assert "sun_elevation_deg" in data
    assert "sun_visibility" in data
    assert "moon_azimuth_deg" in data
    assert "capture_datetime_utc" in data


def test_lighting_estimate_endpoint(monkeypatch):
    import serve_library as module

    class _FakeLightingService:
        def compute(self, *, lat, lon, altitude_m=None, metadata=None):
            return {
                "sun_azimuth_deg": 180.0,
                "sun_elevation_deg": 45.0,
                "sun_visibility": True,
                "sun_direction_source": "inferred_surface",
                "sun_direction_vector_world": [0.0, 0.707, -0.707],
                "sun_direction_confidence": 0.8,
                "inferred_sun_direction_vector": [0.0, 0.707, -0.707],
                "inferred_sun_direction_confidence": 0.8,
                "moon_azimuth_deg": 210.0,
                "moon_elevation_deg": 12.0,
                "moon_direction_vector_world": [0.2, 0.2, -0.9],
                "moon_illumination_fraction": 0.64,
                "moon_visibility": True,
                "lighting_validity_score": 0.91,
                "lighting_validation_flags": [],
            }

    monkeypatch.setattr(module, "get_lighting_context_service", lambda: _FakeLightingService())
    with app.test_client() as client:
        r = client.post(
            "/api/library/lighting/estimate",
            json={
                "lat": 47.6,
                "lon": -122.3,
                "type_metadata": {"shadow_vector": [1, 0, 0], "sun_plane_normal": [0, 1, 0]},
            },
        )
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["sun_direction_source"] == "inferred_surface"
    assert payload["sun_visibility"] is True
    assert payload["inferred_sun_direction_vector"] is not None
    assert payload["moon_visibility"] is True
    assert payload["moon_azimuth_deg"] == 210.0


def test_document_lighting_recompute_endpoint(monkeypatch):
    import serve_library as module

    class _FakeLightingService:
        def compute(self, *, lat, lon, altitude_m=None, metadata=None):
            return {
                "sun_direction_source": "calculated",
                "sun_azimuth_deg": 201.0,
                "sun_elevation_deg": 18.0,
                "sun_visibility": True,
                "moon_azimuth_deg": 333.0,
                "moon_elevation_deg": -6.0,
                "moon_illumination_fraction": 0.2,
                "moon_visibility": False,
                "lighting_validity_score": 0.81,
            }

    monkeypatch.setattr(module, "get_lighting_context_service", lambda: _FakeLightingService())
    with app.test_client() as client:
        r = client.post(
            "/api/library/upload",
            data={
                "document_type": "image",
                "lat": "37.7749",
                "lon": "-122.4194",
                "type_metadata": "{}",
            },
            headers={"X-Tenant-ID": "default"},
        )
    assert r.status_code == 201
    doc_id = r.get_json()["id"]

    with app.test_client() as client:
        r = client.get(f"/api/library/documents/{doc_id}/lighting?recompute=1", headers={"X-Tenant-ID": "default"})
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["sun_direction_source"] == "calculated"
    assert payload["sun_visibility"] is True
    assert payload["lighting_validity_score"] == 0.81
    assert payload["moon_visibility"] is False
    assert payload["moon_elevation_deg"] == -6.0


def test_lighting_estimate_include_lunar_overrides_rollout(monkeypatch):
    import serve_library as module

    captured = {}

    class _FakeLightingService:
        def compute(self, *, lat, lon, altitude_m=None, metadata=None):
            captured["metadata"] = metadata
            return {"moon_visibility": False}

    monkeypatch.setattr(module, "get_lighting_context_service", lambda: _FakeLightingService())
    with app.test_client() as client:
        r = client.post(
            "/api/library/lighting/estimate",
            json={
                "lat": 47.6,
                "lon": -122.3,
                "include_lunar": False,
                "type_metadata": {},
            },
        )
    assert r.status_code == 200
    assert r.get_json()["moon_visibility"] is False
    assert captured["metadata"]["lighting_rollout_flags"]["enable_lunar_context"] is False
