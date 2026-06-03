# Continuum

Continuum library server: upload, search, and view documents with location and type-specific metadata. Uses **USC** (unified-semantic-compressor; Python package name `unified_semantic_archiver`) for the continuum DB and schema.

## Repository map

Sibling repos (assumed layout under `C:/Users/John/`):

| Repo | Role | README |
|------|------|--------|
| **Drawer 2** | Unity game systems, System Drawer, narrative/4D/pathfinding | [../Drawer 2/README.md](../Drawer%202/README.md) |
| **continuum** (this repo) | Library web UI + Flask API (upload, search, media parity) | [README.md](README.md) |
| **unified-semantic-compressor** | USC core: DB schema, compressors, ETL, CLI | [../unified-semantic-compressor/README.md](../unified-semantic-compressor/README.md) |

Unity ↔ Python bridge: [../Drawer 2/Scripts/CONTINUUM_AND_COMPRESSOR.md](../Drawer%202/Scripts/CONTINUUM_AND_COMPRESSOR.md)

## Services

| Service | Entry point | Purpose |
|---------|-------------|---------|
| **Library server** | `python serve_library.py` | Web UI + `/api/library/*` upload, search, map |
| **Entropy ring API** | Same server (when USC entropy schema enabled) | Randomness and credit accounting — see [Entropy API](#entropy-api-entropythief-ring) |
| **Media parity API** | Same server | Store, reconstitute, stream, T2V — see [Media Parity](#media-parity-usc--continuum) |
| **Planet tiles API** | Same server | `GET/POST /api/planet/tiles`, `GET /api/planet/gpx`, `POST /api/planet/google/shapes` |

Depends on **USC** (`unified_semantic_archiver`) for DB schema, compressors, and media primitives.

## Documentation index

When adding a new `.md` in this repo, append it here.

### Root policy

- [SECURITY.md](SECURITY.md) — Security posture and API key usage
- [TERMS_OF_SERVICE.md](TERMS_OF_SERVICE.md) — Terms draft
- [TENANT.md](TENANT.md) — Tenant scoping
- [DEPENDENCIES.md](DEPENDENCIES.md) — Version expectations

### Architecture / compliance

- [docs/ENTROPYTHIEF_RING_ARCHITECTURE.md](docs/ENTROPYTHIEF_RING_ARCHITECTURE.md) — Entropythief ring topology
- [docs/ENTROPY_CLAIMS_AND_EVIDENCE_POLICY.md](docs/ENTROPY_CLAIMS_AND_EVIDENCE_POLICY.md) — Entropy claims and evidence policy
- [docs/ENTROPY_LOWER_BOUND_PROOF.md](docs/ENTROPY_LOWER_BOUND_PROOF.md) — Entropy proof appendix
- [docs/LEGAL_COMPLIANCE_CHECKLIST.md](docs/LEGAL_COMPLIANCE_CHECKLIST.md) — Practical compliance checklist

### Media / USC contract

- [docs/USC_CONTINUUM_MEDIA_INTERFACE_CONTRACT.md](docs/USC_CONTINUUM_MEDIA_INTERFACE_CONTRACT.md) — Media interface contract
- [docs/VIDEO_STORAGE_TOOL_RETIREMENT_CHECKLIST.md](docs/VIDEO_STORAGE_TOOL_RETIREMENT_CHECKLIST.md) — video_storage_tool retirement checklist

### Dev / CI

- [docs/CONDA_BUILD_MANAGEMENT.md](docs/CONDA_BUILD_MANAGEMENT.md) — Conda environments
- [docs/CI_GENERATOR.md](docs/CI_GENERATOR.md) — CI generator

### WebGL editor stub

- [library/continuum_editor_webgl/README.md](library/continuum_editor_webgl/README.md) — WebGL editor stub

### Sibling repositories

- [Drawer 2 documentation index](../Drawer%202/README.md#documentation-index)
- [Unity ↔ Python bridge](../Drawer%202/Scripts/CONTINUUM_AND_COMPRESSOR.md)
- [unified-semantic-compressor documentation index](../unified-semantic-compressor/README.md#documentation-index)

## Getting started (full stack)

1. **Install USC** (unified-semantic-compressor): `pip install -e /path/to/unified-semantic-compressor`
2. **Install continuum deps:** from this repo, `pip install -r requirements.txt`
3. **Init DB:** `python -m unified_semantic_archiver init --db ./continuum.db`
4. **Run continuum server:** `python serve_library.py` → http://localhost:5050
5. **(Optional) Unity (Drawer 2):** Open the project, go to Window → Continuum → Continuum Library, set Base URL to http://localhost:5050 (and DB path for Explorer). See your Drawer 2 repo’s `Scripts/CONTINUUM_AND_COMPRESSOR.md`.
6. **(Optional) Cave (log-view-machine):** Set `CONTINUUM_LIBRARY_URL=http://localhost:5050` so the Cave server proxies `/api/continuum/library/*` to continuum.

## Install

```bash
# From continuum repo root; install the compressor first (sibling or clone)
pip install -e ../unified-semantic-compressor
pip install -r requirements.txt
```

**Conda:** For multiple Python versions (e.g. continuum + USC with torch), see [docs/CONDA_BUILD_MANAGEMENT.md](docs/CONDA_BUILD_MANAGEMENT.md). Use `conda env create -f conda-environment.yml` then `pip install -e /path/to/unified-semantic-compressor`.

If the compressor is elsewhere, install it and ensure `unified_semantic_archiver` is on your Python path, then `pip install -r requirements.txt`. See [DEPENDENCIES.md](DEPENDENCIES.md) for version expectations.

## Run

```bash
python serve_library.py
```

Open http://localhost:5050. Optional env:

- `PORT` — default 5050
- `CONTINUUM_DB_PATH` — path to continuum.db (default: ./continuum.db in repo root)
- `CONTINUUM_LIBRARY_UPLOADS` — uploads directory (default: ./library_uploads)
- `FLASK_DEBUG=1` — enable debug mode
- `CONTINUUM_API_KEY` — if set, require this value in `X-API-Key` or `api_key` for `/api/library/*` (see [SECURITY.md](SECURITY.md))

## Schema

Continuum does not define its own schema. All tables (including `library_documents`) live in **USC** (`unified_semantic_archiver/db/schema.sql`). The continuum app uses USC’s `ContinuumDb` and the same DB file. To add or change tables, extend USC’s schema and migrations; see the USC repo’s `unified_semantic_archiver/db/SCHEMA_OWNERSHIP.md`.

## First run

Initialize the continuum DB (creates schema including library_documents):

```bash
python -m unified_semantic_archiver init --db ./continuum.db
```

## Tests

From the continuum repo root, run: `pytest tests/` (requires `pip install pytest`). Smoke tests cover search (200) and upload-then-fetch.

## Entropy API (Entropythief Ring)

When USC is installed with the entropy schema, the server exposes an entropy ring API for randomness and credit accounting:

- `POST /api/entropy/nodes/register` — Register a petal node
- `GET /api/entropy/center` — Authoritative center guess
- `POST /api/entropy/guess` — Submit guess from petal
- `GET /api/entropy/ring` — Ring topology
- `GET /api/entropy/nodes` — Active + warehouse nodes
- `POST /api/entropy/orchestrator/fit` — Fit ring
- `POST /api/entropy/orchestrator/run-round` — Run round, award credits
- `GET /api/entropy/random` — Request random bytes (spend credits)
- `GET /api/entropy/credits` — Credit balance

Run a petal node:

```bash
python -m entropy.entropythief_node --continuum-url http://localhost:5050 --probe-target 8.8.8.8:53
```

See [docs/ENTROPYTHIEF_RING_ARCHITECTURE.md](docs/ENTROPYTHIEF_RING_ARCHITECTURE.md) for topology and coinstroids integration.

## Policy and Compliance

See [Documentation index](#documentation-index) for policy, security, entropy, and compliance docs.

## Media Parity (USC + Continuum)

To retire `video_storage_tool`, Continuum tracks feature parity in `library/media_parity_matrix.json`. Contract and retirement docs are in the [Documentation index](#documentation-index) under **Media / USC contract**.

Run parity matrix checks with:

```bash
pytest tests/test_media_parity_matrix.py tests/test_serve_library.py -v
```

Media parity endpoints exposed by Continuum:

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

`/api/media/settings` now also exposes USC minimization controls under `minimization.*`, including adapter cohorts and experimental cairn/hyperplane settings:

- `minimization.pipeline.adapter_set`
- `minimization.experiments.enabled|cohort_key|cohorts`
- `minimization.adapter_requirements.*` (per-adapter dependencies and fallback)
- `minimization.cairn.*`
- `minimization.codec.residual_*` (`cairn_residual_v1`)
- `minimization.hyperplane.*`
- `transcript.policy` and `audio_captioning.*` for speech/SFX coverage behavior

Small deterministic media fixture profiles are defined in:

- `tests/media_fixture_specs.json`
- `tests/fixtures/README.md`

Generate fixtures (optional, for media parity execution) with:

```bash
python tests/generate_media_fixtures.py
```

## Unity

Point the Continuum Library window (Window → Continuum → Continuum Library) Base URL to this server (e.g. http://localhost:5050). The Continuum Explorer can use the same continuum.db path with the Python CLI.
