# Continuum Library WebGL editor

Unity WebGL build output for the **Continuum Library** runtime UI (`ContinuumLibraryWebController`), produced by the **Drawer 2** Unity project.

## Canonical URL (with Flask `serve_library.py`)

Static hosting uses Flask `static_folder` (`library/`). After building, open:

`http://<host>:<port>/library/continuum_editor_webgl/index.html`

Optional tenant/API overrides (examples):

`/library/continuum_editor_webgl/index.html?tenant=mytenant`

`/library/continuum_editor_webgl/index.html?apiBase=http://127.0.0.1:5050&tenant=default`

Same-origin deployment can omit `apiBase`; the bootstrap uses the page origin on WebGL.

Shorter entry (redirect):

`/continuum_editor/`

## Build

From this repo root (see `scripts/build_continuum_webgl.sh` or `.ps1`):

- Set **`DRAWER2_PROJECT`** to the Unity project folder that contains `Assets/` (e.g. `.../Drawer 2`).
- Set **`UNITY_EDITOR_PATH`** to the Unity Editor executable matching `ProjectSettings/ProjectVersion.txt` (e.g. `6000.3.2f1`).
- Optional: **`CONTINUUM_WEBGL_BASE_HREF`** — passed through to the Unity post-step that injects `<base href>` into `index.html` (default `/library/continuum_editor_webgl/`).

Or from Unity: **Continuum → Build WebGL Library Editor** (writes to sibling `../continuum/library/continuum_editor_webgl` when the repos live side by side).

## Security

Avoid long-lived API keys in query strings on shared hosts; prefer same-origin sessions or short-lived tokens.

## CI (optional)

GitHub Actions requires a Unity-licensed workflow (for example [game-ci](https://game.ci)) and a checkout of the Drawer 2 repo alongside continuum. There is no default job in `continuum-ci`: keep builds local via `scripts/build_continuum_webgl.sh` / `.ps1` unless you wire secrets and a runner yourself.
