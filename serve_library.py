"""
Continuum Library HTTP server (SPA + API).
Uses unified_semantic_archiver (unified-semantic-compressor package) for ContinuumDb.
Run: pip install -e ../unified-semantic-compressor && python serve_library.py
"""
from __future__ import annotations

import json
import os
import secrets
import urllib.request
import urllib.parse
from pathlib import Path

from flask import Flask, request, jsonify, send_file, redirect, Response
from lighting_context import LightingContextService

try:
    from unified_semantic_archiver.db import ContinuumDb
except ImportError:
    class ContinuumDb:  # type: ignore[override]
        """
        Lightweight fallback DB used when USC is unavailable.
        Intended for local development/tests of Continuum service surface only.
        """

        def __init__(self, _path: str):
            self._docs: list[dict] = []
            self._next_id = 1

        def library_document_insert(
            self,
            *,
            document_type: str,
            blob_ref: str | None,
            url: str | None,
            type_metadata: dict | None,
            owner_id: str | None,
            tenant_id: str,
            lat: float | None,
            lon: float | None,
            altitude_m: float | None,
        ) -> int:
            doc_id = self._next_id
            self._next_id += 1
            self._docs.append(
                {
                    "id": doc_id,
                    "document_type": document_type,
                    "blob_ref": blob_ref,
                    "url": url,
                    "type_metadata": type_metadata or {},
                    "owner_id": owner_id,
                    "tenant_id": tenant_id,
                    "lat": lat,
                    "lon": lon,
                    "altitude_m": altitude_m,
                }
            )
            return doc_id

        def library_document_get(self, doc_id: int, tenant_id: str = "default"):
            for doc in self._docs:
                if doc["id"] == doc_id and doc.get("tenant_id", "default") == tenant_id:
                    return doc
            return None

        def library_document_search(
            self,
            *,
            document_type: str | None = None,
            q: str | None = None,
            lat: float | None = None,
            lon: float | None = None,
            distance_mi: float | None = None,
            tenant_id: str = "default",
            limit: int = 100,
        ):
            rows = [d for d in self._docs if d.get("tenant_id", "default") == tenant_id]
            if document_type:
                rows = [d for d in rows if d.get("document_type") == document_type]
            if q:
                qq = q.lower()
                rows = [d for d in rows if qq in str(d.get("url") or "").lower() or qq in str(d.get("type_metadata") or "").lower()]
            # lat/lon/distance are ignored in fallback mode
            return rows[:limit]

try:
    from unified_semantic_archiver.media import UscMediaService, MediaServiceUnavailable
except ImportError:
    UscMediaService = None  # type: ignore[assignment]

    class MediaServiceUnavailable(RuntimeError):
        pass

_here = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(_here / "library"), static_url_path="")
LIBRARY_HTML = _here / "library" / "library.html"

DB_PATH = os.environ.get("CONTINUUM_DB_PATH") or str(_here / "continuum.db")
UPLOADS_DIR = Path(os.environ.get("CONTINUUM_LIBRARY_UPLOADS") or str(_here / "library_uploads"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

_db: ContinuumDb | None = None
_media_service = None
_lighting_context_service: LightingContextService | None = None

# Per-tenant API keys: env CONTINUUM_TENANT_KEYS='{"tenant1":"key1"}' and/or file CONTINUUM_TENANT_KEYS_FILE.
# Global CONTINUUM_API_KEY is used when the request tenant has no per-tenant key (backward compatible).
_API_KEY = (os.environ.get("CONTINUUM_API_KEY") or "").strip()
_TENANT_KEYS: dict[str, str] = {}
_TENANT_KEYS_FILE = (os.environ.get("CONTINUUM_TENANT_KEYS_FILE") or "").strip()


def _load_tenant_keys() -> dict[str, str]:
    out: dict[str, str] = {}
    env_json = (os.environ.get("CONTINUUM_TENANT_KEYS") or "").strip()
    if env_json:
        try:
            out.update(json.loads(env_json))
        except json.JSONDecodeError:
            pass
    if _TENANT_KEYS_FILE:
        path = Path(_TENANT_KEYS_FILE)
        if path.is_file():
            try:
                out.update(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                pass
    return out


def _save_tenant_keys(keys: dict[str, str]) -> None:
    if not _TENANT_KEYS_FILE:
        return
    path = Path(_TENANT_KEYS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keys, indent=2))


def _get_tenant_keys() -> dict[str, str]:
    global _TENANT_KEYS
    if not _TENANT_KEYS and (_API_KEY or os.environ.get("CONTINUUM_TENANT_KEYS") or _TENANT_KEYS_FILE):
        _TENANT_KEYS = _load_tenant_keys()
    return _TENANT_KEYS


def get_db() -> ContinuumDb:
    global _db
    if _db is None:
        _db = ContinuumDb(DB_PATH)
    return _db


def get_media_service():
    global _media_service
    if _media_service is None:
        if UscMediaService is None:
            raise MediaServiceUnavailable(
                "USC media service is unavailable. Install unified-semantic-compressor with media support."
            )
        _media_service = UscMediaService(
            storage_root=Path(os.environ.get("CONTINUUM_MEDIA_STORAGE") or (_here / "media_storage")),
            config_path=Path(os.environ.get("CONTINUUM_MEDIA_CONFIG_PATH") or (_here / "media_config.yaml")),
            settings_path=Path(os.environ.get("CONTINUUM_MEDIA_SETTINGS_PATH") or (_here / "media_settings.json")),
        )
    return _media_service


def get_lighting_context_service() -> LightingContextService:
    global _lighting_context_service
    if _lighting_context_service is None:
        try:
            db = get_db()
            if hasattr(db, "ephemeris_sample_list_near_epoch"):
                from ephemeris import create_default_ephemeris_provider
                provider = create_default_ephemeris_provider(db=db, tenant_id="default")
                body_catalog = {"sun": "star", "moon": "moon", "earth": "planet"}
                body_radii = {"earth": 6371.0, "moon": 1737.4, "sun": 695700.0}
                _lighting_context_service = LightingContextService(
                    ephemeris_provider=provider,
                    body_catalog=body_catalog,
                    body_radii_km=body_radii,
                )
            else:
                _lighting_context_service = LightingContextService()
        except Exception:
            _lighting_context_service = LightingContextService()
    return _lighting_context_service


def _as_bool(raw: object, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def row_to_json(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat() if v else None
        else:
            out[k] = v
    return out


def _parse_range(header: str | None, total: int) -> tuple[int, int] | None:
    if not header or not header.strip().lower().startswith("bytes="):
        return None
    try:
        raw = header.strip()[6:]
        start_s, end_s = raw.split("-", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else total - 1
    except Exception:
        return None
    if start > end or start >= total:
        return None
    return start, min(end, total - 1)


def _stream_file(path: Path, *, start: int, end: int):
    chunk_size = 256 * 1024
    remaining = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            data = f.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


def get_tenant_from_request() -> str:
    """Tenant from X-Tenant-ID header or query param 'tenant'; default 'default'."""
    tenant = request.headers.get("X-Tenant-ID") or request.args.get("tenant")
    if tenant is not None:
        tenant = (tenant or "").strip()
    return tenant or "default"


def _key_for_tenant(tenant: str) -> str | None:
    """Return the API key that must be presented for this tenant, or None if no auth required."""
    keys = _get_tenant_keys()
    if tenant in keys and keys[tenant]:
        return keys[tenant]
    if _API_KEY:
        return _API_KEY
    return None


@app.before_request
def optional_api_key():
    """Require X-API-Key or api_key for /api/library when global or per-tenant key is configured."""
    if not request.path.startswith("/api/library"):
        return None
    tenant = get_tenant_from_request()
    required = _key_for_tenant(tenant)
    if not required:
        return None
    provided = (request.headers.get("X-API-Key") or request.args.get("api_key") or "").strip()
    if provided != required:
        return jsonify({"error": "Unauthorized"}), 401
    return None


@app.route("/")
@app.route("/library")
def index():
    if LIBRARY_HTML.exists():
        return send_file(str(LIBRARY_HTML))
    return "Library UI not found (missing library/library.html)", 404


@app.route("/api/library/search")
def search():
    try:
        tenant = get_tenant_from_request()
        document_type = request.args.get("document_type") or None
        q = request.args.get("q") or None
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        distance_mi = request.args.get("distance_mi")
        limit = min(request.args.get("limit", default=100, type=int), 500)
        rows = get_db().library_document_search(
            document_type=document_type,
            q=q,
            lat=lat,
            lon=lon,
            distance_mi=distance_mi,
            tenant_id=tenant,
            limit=limit,
        )
        return jsonify([row_to_json(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/library/upload", methods=["POST"])
def upload():
    try:
        document_type = request.form.get("document_type", "").strip().lower()
        if document_type not in ("video", "document", "audio", "image", "program", "data"):
            return jsonify({"error": "Invalid document_type"}), 400
        lat = request.form.get("lat", type=float)
        lon = request.form.get("lon", type=float)
        altitude_m = request.form.get("altitude_m", type=float)
        url = (request.form.get("url") or "").strip() or None
        type_metadata_raw = request.form.get("type_metadata") or "{}"
        try:
            type_metadata = json.loads(type_metadata_raw)
        except json.JSONDecodeError:
            type_metadata = {}
        if not isinstance(type_metadata, dict):
            type_metadata = {}
        blob_ref = None
        if "file" in request.files and request.files["file"].filename:
            f = request.files["file"]
            safe_name = f"{hash(f.filename) % 2**32:08x}{Path(f.filename).suffix or '.bin'}"
            path = UPLOADS_DIR / safe_name
            f.save(str(path))
            blob_ref = safe_name
        tenant = get_tenant_from_request()
        compute_lighting = _as_bool(request.form.get("compute_lighting_context"), default=False)
        include_lunar = _as_bool(request.form.get("include_lunar"), default=True)
        if "lighting_rollout_flags" not in type_metadata or not isinstance(type_metadata.get("lighting_rollout_flags"), dict):
            type_metadata["lighting_rollout_flags"] = {}
        type_metadata["lighting_rollout_flags"]["enable_lunar_context"] = include_lunar
        if compute_lighting and lat is not None and lon is not None:
            try:
                type_metadata["lighting_context"] = get_lighting_context_service().compute(
                    lat=float(lat),
                    lon=float(lon),
                    altitude_m=altitude_m,
                    metadata=type_metadata,
                )
            except Exception:
                # Upload should not fail when external weather lookup fails.
                pass
        doc_id = get_db().library_document_insert(
            document_type=document_type,
            blob_ref=blob_ref,
            url=url,
            type_metadata=type_metadata,
            owner_id=None,
            tenant_id=tenant,
            lat=lat,
            lon=lon,
            altitude_m=altitude_m,
        )
        return jsonify({"id": doc_id, "url": url or (f"/api/library/documents/{doc_id}/download" if doc_id else None)}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/library/documents/<int:doc_id>")
def get_document(doc_id: int):
    try:
        tenant = get_tenant_from_request()
        row = get_db().library_document_get(doc_id, tenant_id=tenant)
        if not row:
            return jsonify({"error": "Not found"}), 404
        return jsonify(row_to_json(row))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/library/documents/<int:doc_id>/lighting")
def get_document_lighting(doc_id: int):
    try:
        tenant = get_tenant_from_request()
        row = get_db().library_document_get(doc_id, tenant_id=tenant)
        if not row:
            return jsonify({"error": "Not found"}), 404
        type_metadata = row.get("type_metadata") or {}
        if not isinstance(type_metadata, dict):
            type_metadata = {}
        recompute = _as_bool(request.args.get("recompute"), default=False)
        existing = type_metadata.get("lighting_context")
        if existing and not recompute:
            return jsonify(existing)
        include_lunar = request.args.get("include_lunar")
        if include_lunar is not None:
            rollout = type_metadata.get("lighting_rollout_flags")
            if not isinstance(rollout, dict):
                rollout = {}
            rollout["enable_lunar_context"] = _as_bool(include_lunar, default=True)
            type_metadata["lighting_rollout_flags"] = rollout
        lat = row.get("lat")
        lon = row.get("lon")
        if lat is None or lon is None:
            return jsonify({"error": "lat/lon required for lighting context"}), 400
        result = get_lighting_context_service().compute(
            lat=float(lat),
            lon=float(lon),
            altitude_m=row.get("altitude_m"),
            metadata=type_metadata,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/library/lighting/query", methods=["GET", "POST"])
def lighting_query():
    """
    Generalized N-body lighting query.
    Params: observer_body_id, lat, lon, datetime_utc, sources[] (optional), include_eclipses (optional).
    Preserves earth/moon compatibility.
    """
    try:
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
        else:
            body = dict(request.args)
        if not isinstance(body, dict):
            return jsonify({"error": "JSON object or query params required"}), 400
        observer_body_id = (body.get("observer_body_id") or "earth").strip()
        lat = body.get("lat")
        lon = body.get("lon")
        if lat is None or lon is None:
            return jsonify({"error": "lat and lon required"}), 400
        lat, lon = float(lat), float(lon)
        datetime_utc = body.get("datetime_utc")
        sources_raw = body.get("sources")
        sources = sources_raw if isinstance(sources_raw, list) else None
        include_eclipses = _as_bool(body.get("include_eclipses"), default=True)
        altitude_m = body.get("altitude_m")
        if altitude_m is not None:
            altitude_m = float(altitude_m)
        result = get_lighting_context_service().query_lighting(
            observer_body_id=observer_body_id,
            lat=lat,
            lon=lon,
            altitude_m=altitude_m,
            datetime_utc=datetime_utc,
            sources=sources,
            include_eclipses=include_eclipses,
        )
        eclipses = result.get("eclipses") or []
        if eclipses and hasattr(get_db(), "occlusion_event_insert"):
            when = result.get("capture_datetime_utc", "").replace("Z", "+00:00")[:19]
            if when:
                tenant = get_tenant_from_request()
                for ev in eclipses:
                    try:
                        get_db().occlusion_event_insert(
                            epoch_utc=when,
                            source_body_id=ev.get("source_body_id", ""),
                            target_body_id=ev.get("target_body_id", ""),
                            occluder_body_id=ev.get("occluder_body_id", ""),
                            occlusion_ratio=ev.get("occlusion_ratio"),
                            eclipse_type=ev.get("eclipse_type"),
                            tenant_id=tenant,
                        )
                    except Exception:
                        pass
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/library/lighting/estimate", methods=["POST"])
def estimate_lighting():
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "JSON object required"}), 400
    lat = body.get("lat")
    lon = body.get("lon")
    if lat is None or lon is None:
        return jsonify({"error": "lat and lon required"}), 400
    metadata = body.get("type_metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    include_lunar = body.get("include_lunar")
    if include_lunar is not None:
        rollout = metadata.get("lighting_rollout_flags")
        if not isinstance(rollout, dict):
            rollout = {}
        rollout["enable_lunar_context"] = _as_bool(include_lunar, default=True)
        metadata["lighting_rollout_flags"] = rollout
    try:
        result = get_lighting_context_service().compute(
            lat=float(lat),
            lon=float(lon),
            altitude_m=body.get("altitude_m"),
            metadata=metadata,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/library/documents/<int:doc_id>/download")
def download_document(doc_id: int):
    try:
        tenant = get_tenant_from_request()
        row = get_db().library_document_get(doc_id, tenant_id=tenant)
        if not row:
            return jsonify({"error": "Not found"}), 404
        if row.get("url") and not row.get("blob_ref"):
            return redirect(row["url"], code=302)
        blob_ref = row.get("blob_ref")
        if not blob_ref:
            return jsonify({"error": "No file"}), 404
        path = UPLOADS_DIR / blob_ref
        if not path.is_file():
            return jsonify({"error": "File not found"}), 404
        return send_file(str(path), as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/geocode")
def geocode():
    address = (request.args.get("address") or "").strip()
    if not address:
        return jsonify({"error": "address query required"}), 400
    try:
        url = "https://nominatim.openstreetmap.org/search?q=" + urllib.parse.quote(address) + "&format=json&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "ContinuumLibrary/1.0"})
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read().decode())
        if not data:
            return jsonify({"lat": None, "lon": None})
        return jsonify({"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/media/store", methods=["POST"])
def media_store():
    if "video" not in request.files and "file" not in request.files:
        return jsonify({"error": "No media file"}), 400
    upload = request.files.get("video") or request.files.get("file")
    if not upload or not upload.filename:
        return jsonify({"error": "Empty filename"}), 400

    tenant = get_tenant_from_request()
    temp_name = f"{secrets.token_hex(8)}{Path(upload.filename).suffix.lower()}"
    temp_path = UPLOADS_DIR / temp_name
    upload.save(str(temp_path))
    try:
        result = get_media_service().store(temp_path, tenant_id=tenant)
        return jsonify(result), 202
    except MediaServiceUnavailable as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        temp_path.unlink(missing_ok=True)


@app.route("/api/media/stored", methods=["GET"])
def media_stored_list():
    tenant = get_tenant_from_request()
    try:
        items = get_media_service().list_jobs(tenant_id=tenant)
        return jsonify({"items": items, "ids": [x["id"] for x in items if x["status"] == "ready"]})
    except MediaServiceUnavailable as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/media/stored/<job_id>/status", methods=["GET"])
def media_stored_status(job_id: str):
    tenant = get_tenant_from_request()
    try:
        return jsonify(get_media_service().get_job_status(job_id=job_id, tenant_id=tenant))
    except FileNotFoundError:
        return jsonify({"error": "Not found"}), 404
    except MediaServiceUnavailable as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/media/stored/<job_id>/retry", methods=["POST"])
def media_stored_retry(job_id: str):
    tenant = get_tenant_from_request()
    body = request.get_json(silent=True) or {}
    force_script = bool(body.get("force_script"))
    try:
        result = get_media_service().retry_store(job_id=job_id, tenant_id=tenant, force_script=force_script)
        return jsonify(result), 202
    except FileNotFoundError:
        return jsonify({"error": "Not found"}), 404
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409
    except MediaServiceUnavailable as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/media/reconstitute", methods=["POST"])
def media_reconstitute():
    body = request.get_json(silent=True) or {}
    job_id = body.get("stored_id") or body.get("id")
    use_original = bool(body.get("original", False))
    if not job_id:
        return jsonify({"error": "Missing stored_id"}), 400
    tenant = get_tenant_from_request()
    try:
        get_media_service().reconstitute(job_id=job_id, tenant_id=tenant, use_original=use_original)
        return jsonify(
            {
                "stream_url": f"/api/media/stream/{job_id}?original={1 if use_original else 0}",
                "out_path": "reconstituted_original.mp4" if use_original else "reconstituted.mp4",
            }
        )
    except FileNotFoundError:
        return jsonify({"error": "Not found"}), 404
    except MediaServiceUnavailable as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/media/stream/<job_id>/info", methods=["GET"])
def media_stream_info(job_id: str):
    use_original = request.args.get("original", "0") == "1"
    tenant = get_tenant_from_request()
    try:
        data = get_media_service().stream_info(job_id=job_id, tenant_id=tenant, use_original=use_original)
        return jsonify(
            {
                "content_length": data["content_length"],
                "filename": data["filename"],
                "original": data["original"],
            }
        )
    except FileNotFoundError:
        return jsonify({"error": "Not found"}), 404
    except MediaServiceUnavailable as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/media/stream/<job_id>", methods=["GET"])
def media_stream(job_id: str):
    use_original = request.args.get("original", "0") == "1"
    tenant = get_tenant_from_request()
    try:
        info = get_media_service().stream_info(job_id=job_id, tenant_id=tenant, use_original=use_original)
        total = int(info["content_length"])
        rng = _parse_range(request.headers.get("Range"), total)
        stream = get_media_service().open_stream(
            job_id=job_id,
            tenant_id=tenant,
            use_original=use_original,
            byte_range=rng,
        )
        resp = Response(
            _stream_file(stream["path"], start=stream["start"], end=stream["end"]),
            status=206 if stream["partial"] else 200,
            mimetype="video/mp4",
        )
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Content-Length"] = str(stream["content_length"])
        if stream["partial"]:
            resp.headers["Content-Range"] = f"bytes {stream['start']}-{stream['end']}/{stream['total']}"
        return resp
    except FileNotFoundError:
        return jsonify({"error": "Not found"}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 416
    except MediaServiceUnavailable as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/media/settings", methods=["GET"])
def media_settings_get():
    try:
        return jsonify(get_media_service().get_settings())
    except MediaServiceUnavailable as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/media/settings", methods=["PUT"])
def media_settings_put():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "JSON object required"}), 400
    try:
        return jsonify(get_media_service().update_settings(body))
    except MediaServiceUnavailable as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/media/t2v/download", methods=["POST"])
def media_t2v_download():
    try:
        return jsonify(get_media_service().start_t2v_download()), 202
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409
    except MediaServiceUnavailable as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/media/t2v/download/status", methods=["GET"])
def media_t2v_download_status():
    try:
        return jsonify(get_media_service().get_t2v_download_status())
    except MediaServiceUnavailable as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


_ADMIN_KEY = (os.environ.get("CONTINUUM_ADMIN_KEY") or "").strip()
_MEDIA_PARITY_MATRIX = _here / "library" / "media_parity_matrix.json"


@app.route("/api/admin/tenant-keys", methods=["POST"])
def admin_tenant_keys():
    """Generate a new API key for a tenant. Requires X-Admin-Key or Authorization: Bearer <CONTINUUM_ADMIN_KEY>."""
    if _ADMIN_KEY:
        auth = request.headers.get("X-Admin-Key") or request.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            auth = auth[7:].strip()
        if auth != _ADMIN_KEY:
            return jsonify({"error": "Forbidden"}), 403
    body = request.get_json(silent=True) or {}
    tenant_id = (body.get("tenant_id") or "").strip()
    if not tenant_id:
        return jsonify({"error": "tenant_id required"}), 400
    api_key = secrets.token_urlsafe(32)
    keys = _get_tenant_keys()
    keys[tenant_id] = api_key
    _save_tenant_keys(keys)
    return jsonify({"tenant_id": tenant_id, "api_key": api_key}), 201


@app.route("/api/astral/nasa/register", methods=["POST"])
def nasa_register():
    """Register a NASA kernel/flat-file. Body: file_type, local_path, source_url (optional)."""
    try:
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return jsonify({"error": "JSON object required"}), 400
        file_type = (body.get("file_type") or "").strip().lower()
        local_path = (body.get("local_path") or "").strip()
        if not file_type or not local_path:
            return jsonify({"error": "file_type and local_path required"}), 400
        if file_type not in ("spk", "pck", "lsk", "fk", "horizons"):
            return jsonify({"error": "file_type must be spk, pck, lsk, fk, or horizons"}), 400
        from pathlib import Path
        path = Path(local_path)
        if not path.is_absolute():
            path = UPLOADS_DIR / local_path
        if not path.is_file():
            return jsonify({"error": f"File not found: {path}"}), 404
        try:
            from unified_semantic_archiver.etl.nasa_ingestion import NasaIngestionRunner
            runner = NasaIngestionRunner(get_db(), tenant_id=get_tenant_from_request())
            file_id = runner.register_file(
                file_type=file_type,
                local_path=path,
                source_url=body.get("source_url"),
            )
            row = get_db().nasa_file_get(file_id, get_tenant_from_request())
            return jsonify({"id": file_id, "file": row_to_json(row)}), 201
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/astral/nasa/validate/<int:file_id>")
def nasa_validate(file_id: int):
    """Validate checksum and coverage for a registered NASA file."""
    try:
        from unified_semantic_archiver.etl.nasa_ingestion import NasaIngestionRunner
        runner = NasaIngestionRunner(get_db(), tenant_id=get_tenant_from_request())
        checksum_ok = runner.validate_checksum(file_id)
        coverage = runner.validate_coverage(file_id)
        return jsonify({"checksum_valid": checksum_ok, "coverage": coverage})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/astral/ingest", methods=["POST"])
def astral_ingest():
    """Start or run an ingestion job. Body: source (path), body_id (optional), file_id (optional)."""
    try:
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return jsonify({"error": "JSON object required"}), 400
        source = (body.get("source") or "").strip()
        if not source:
            return jsonify({"error": "source path required"}), 400
        from pathlib import Path
        path = Path(source)
        if not path.is_absolute():
            path = UPLOADS_DIR / source
        if not path.is_file():
            return jsonify({"error": f"File not found: {path}"}), 404
        tenant = get_tenant_from_request()
        job_id = get_db().ingestion_job_insert(
            job_type="horizons",
            source=str(path.resolve()),
            payload_json={"source": str(path.resolve()), "body_id": body.get("body_id", "earth"), "file_id": body.get("file_id")},
            tenant_id=tenant,
        )
        run_sync = _as_bool(body.get("run_sync"), default=False)
        if run_sync:
            from unified_semantic_archiver.etl.nasa_ingestion import NasaIngestionRunner
            runner = NasaIngestionRunner(get_db(), tenant_id=tenant)
            result = runner.run_ingestion_job(job_id, body_id=body.get("body_id", "earth"))
            return jsonify({
                "job_id": job_id,
                "status": result.status,
                "samples_inserted": result.samples_inserted,
                "error": result.error_text,
            }), 202
        return jsonify({"job_id": job_id, "status": "pending"}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/astral/ingest/<int:job_id>/status")
def astral_ingest_status(job_id: int):
    """Get ingestion job status."""
    try:
        job = get_db().ingestion_job_get(job_id, get_tenant_from_request())
        if not job:
            return jsonify({"error": "Not found"}), 404
        return jsonify(row_to_json(job))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/media-parity", methods=["GET"])
def media_parity_matrix():
    """
    Return USC+Continuum media parity matrix sourced from video_storage_tool inventory.
    This is an admin tracking endpoint for feature-complete retirement work.
    """
    if not _MEDIA_PARITY_MATRIX.is_file():
        return jsonify({"error": "media parity matrix not found"}), 404
    try:
        with open(_MEDIA_PARITY_MATRIX, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return jsonify(payload)
    except (OSError, json.JSONDecodeError) as e:
        return jsonify({"error": str(e)}), 500


def main():
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")


if __name__ == "__main__":
    main()
