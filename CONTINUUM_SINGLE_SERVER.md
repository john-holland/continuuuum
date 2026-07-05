# Continuum Single-Server Architecture

Target: one Flask process (`serve_library.py`) on port **5050** hosting the USC library, spatial map, lemma library, and episodic hub.

**Current phase:** cross-link configuration is documented here; route migration from Drawer 2 `continuum_api` is a follow-up PR.

---

## Target URL map

| Path | UI / API |
|------|----------|
| `/` | Hub landing (library SPA today; episodic hub after merge) |
| `/library` | USC geo map + spatial d3 map |
| `/lemma-library` | Lemma browser (from Drawer 2 static tree) |
| `/ui` | Episodic content hub |
| `/story-board` | Agile story Kanban + work-order links |
| `/project-calendar` | Production calendar + narrative overlay |
| `/sql-viewer` | SQL recipes (stories, work orders, causality) |
| `/api/library/*` | USC document search/upload |
| `/api/spatial/*` | Render masks, episode assets, bucket trees |
| `/api/thesaurus/*` | Lemma library, XLIFF, localization |
| `/api/episodes/*` | Episodes, drafts, reviews |
| `/api/stories/*` | Agile stories, assignees, work orders |
| `/api/work-orders/*` | Work order CRUD + causality generate |
| `/api/production/*` | Proxy to resaurce budget/schedule |
| `/api/chat/*` | Proxy to resaurce persistent chat |
| `/api/calendar/*` | Calendar sync subscriptions |

---

## Cross-link config matrix

Until merge, two servers may run on different ports. After merge, use **relative URLs** and remove localStorage base overrides.

| Setting | Current (dual-server) | Merged (same-origin) |
|---------|----------------------|----------------------|
| `lemmaApiBase` | [library.html](library/library.html) `#lemma-api-base`, `localStorage.lemmaApiBase`, default `http://127.0.0.1:5050` | `/lemma-library` |
| `continuumLibraryBase` | [lemma-library.js](../Drawer%202/Scripts/continuum_api/static/lemma-library/lemma-library.js) `?libraryBase=`, `localStorage.continuumLibraryBase` | `/library` or `/` |
| Unity API | [continuum_api_url.txt](../Drawer%202/Scripts/continuum_api_url.txt) | Single base URL |
| Cave adapter | [cave_adapter.py](../Drawer%202/Scripts/continuum_api/cave_adapter.py) `CAVE_BASE_URL=http://localhost:3000` | `/api/library/*` on merged server |
| Spatial deep link | `?view=spatial&highlight={token}&episodeId=` | Same |
| Lemma → map link | `{libraryBase}?highlight={prefab-id}&view=spatial` | `/library?highlight=…&view=spatial` |

### Highlight resolution chain

1. **Library doc id** — numeric `library_documents.id`
2. **Episode asset** — `episode_assets.usc_asset_id` or `causality_leaf_id` matches token
3. **Spatial bucket** — `vocabulary_render_mask_buckets.bucket_id` or parsed Q/O/S leaf id

API: `GET /api/spatial/resolve-highlight?token=&episodeId=`

---

## Environment variables

See [.env.example](.env.example). Key vars:

- `PORT=5050`
- `CONTINUUM_DB_PATH` — shared `continuum.db` (Drawer 2 episodic + USC schemas)
- `CONTINUUM_API_KEY` — optional library API auth
- `CONTINUUM_DEEPLINK_PATH` — Unity deeplink file (Drawer 2 `continuum_api` today)
- `RESAURCE_CAVE_URL` — resaurce Cave for production budget/schedule + chat proxy (default `http://127.0.0.1:3456`)

**Agile PM (Drawer 2 `continuum_api` today):**

- `/story-board`, `/project-calendar` — static SPAs with `continuum-nav` Chat toggle
- `Scripts/AGILE_PRODUCTION.md` — schema order, workflow, calendar cron
- resaurce `chat-remote/` Module Federation bundle (`REACT_APP_RES_AURCE_CHAT_URL` / `http://127.0.0.1:3457/remote/chat/remoteEntry.js`)

**Deprecated (transitional dual-server):**

- `CONTINUUM_LEMMA_API_BASE` — lemma library when library SPA runs on a different host
- `CONTINUUM_LIBRARY_BASE` — USC library when lemma SPA runs on a different host

---

## Port collision

Both `serve_library.py` (continuum repo) and `python -m continuum_api.server` (Drawer 2) default to **5050**. Run only one on that port, or set `PORT` / `--port` on the other.

| Service | Default | Notes |
|---------|---------|-------|
| serve_library | 5050 | Geo + spatial map, `/api/library/*`, `/api/spatial/*` |
| continuum_api | 5050 | Lemma library, XLIFF, episodic `/api/*` |
| script-output Vite | 5174 | Proxies `/api` → 5050 |

---

## Future migration checklist

- [ ] Extract `serve_library.py` route groups into blueprints (`library_routes`, `media_routes`, `spatial_routes`)
- [ ] Import Drawer 2 `register_lemma_routes`, `register_localization_routes` into merged app
- [ ] Mount static trees: `continuum/library/`, `Scripts/continuum_api/static/`
- [ ] Unified nav: library ↔ lemma-library ↔ episodic hub
- [ ] Remove `lemmaApiBase` / `continuumLibraryBase` localStorage from SPAs
- [ ] Point Unity `ContinuumLibraryWindow` default URL to merged server (not `:3000` Cave)
- [ ] Update [EPISODIC_PORT_BROKER_AND_CAVE.md](../Drawer%202/Scripts/EPISODIC_PORT_BROKER_AND_CAVE.md) and both READMEs

---

## Dual-server dev (today)

| Task | Server | URL |
|------|--------|-----|
| Spatial map + USC library | `python serve_library.py` | `http://127.0.0.1:5050/library` |
| Lemma library + XLIFF | `python -m continuum_api.server` | `http://127.0.0.1:5050/lemma-library` (different process — use alternate port for one) |

Set `#lemma-api-base` in library toolbar to Drawer 2 API base when running split (e.g. library on 5050, continuum_api on 5051).
