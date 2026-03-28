# USC + Continuum Media Interface Contract

This contract defines how media features from `video_storage_tool` map to USC (core) and Continuum (service/API).

## Ownership

- **USC owns core media primitives**
  - ingest/dehydrate
  - diff compute/apply
  - reconstitution
  - codec/loss policy
  - quality verification primitives
  - stream cache policy implementation

- **Continuum owns service/API surface**
  - HTTP endpoints
  - tenant scoping and auth
  - stream transport and Range responses
  - settings/UI exposure

## Required USC callable surface

```python
class UscMediaService(Protocol):
    def store(self, input_path: Path, tenant_id: str, settings: dict) -> dict: ...
    def list_jobs(self, tenant_id: str) -> list[dict]: ...
    def get_job_status(self, job_id: str, tenant_id: str) -> dict: ...
    def retry_store(self, job_id: str, tenant_id: str, force_script: bool = False) -> dict: ...
    def reconstitute(self, job_id: str, tenant_id: str, use_original: bool) -> dict: ...
    def stream_info(self, job_id: str, tenant_id: str, use_original: bool) -> dict: ...
    def open_stream(self, job_id: str, tenant_id: str, use_original: bool, byte_range: tuple[int, int] | None): ...
    def get_settings(self) -> dict: ...
    def update_settings(self, updates: dict) -> dict: ...
    def start_t2v_download(self) -> dict: ...
    def get_t2v_download_status(self) -> dict: ...
```

## Required Continuum endpoints

- `POST /api/media/store`
- `GET /api/media/stored`
- `GET /api/media/stored/<job_id>/status`
- `POST /api/media/stored/<job_id>/retry`
- `POST /api/media/reconstitute`
- `GET /api/media/stream/<job_id>/info`
- `GET /api/media/stream/<job_id>`
- `GET /api/media/settings`
- `PUT /api/media/settings`
- `POST /api/media/t2v/download`
- `GET /api/media/t2v/download/status`

## Feature completeness rule

A feature is complete only when all are true:

1. USC primitive exists and is covered by unit/integration tests.
2. Continuum endpoint exposes that primitive with tenant-aware behavior.
3. Parity matrix status reaches `Parity-tested`.
