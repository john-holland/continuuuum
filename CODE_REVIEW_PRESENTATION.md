# Code Review Presentation Guide

A structured walkthrough for presenting and reviewing major subsystems across **Drawer 2**, **continuuuum**, **resaurce**, and **Saurce**. Use this as an agenda, demo script, and reviewer checklist.

---

## Suggested agenda (105–135 min)

| Block | Topic | Time | Primary repo |
|-------|--------|------|--------------|
| 1 | Planetary | 15–20 min | Drawer 2 + continuuuum |
| 2 | Roads | 10–15 min | Drawer 2 |
| 3 | Space pathing | 10–15 min | Drawer 2 |
| 4 | Lemma / library / spatial maps | 20–25 min | continuuuum + Drawer 2 |
| 5 | Multiplayer, weather merge, sync alignment | 20–25 min | Drawer 2 |
| 6 | Video to cloud (CloudBake) | 15–20 min | Drawer 2 |
| 7 | Resaurce legal + Saurce investment | 10–15 min | resaurce + Saurce |

---

## 1. Planetary

### What to explain

Planetary is the planet-scale simulation and rendering stack in Unity. Height comes from a **multi-source planar stack**; composition flows through **SdfMax**; the shell is indexed by a **pole-aware manifold grid**. Continuuuum is the HTTP backend for tile and weather streaming—not the simulation engine.

### Architecture (talk track)

```
Continuuuum HTTP  ──►  PlanetMeshStreamingService / PlanetaryWeatherStreamingService
                              │
PlanetaryPlanarBase (height sources) ──► PlanetBody (orchestrator)
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   PlanetRenderer      SdfMax composition    PlanetShellManifoldGrid
   (nadir chunks)      (mantle/lava/crust)   + physics bridges
          │                   │                   │
          └──── PlanetarySdfLodRenderer (horizon LOD handoff) ──┘
```

### Key files

| Area | Path |
|------|------|
| Architecture doc | `Drawer 2/Assets/Planetary/docs/PlanetaryArchitecture.md` |
| Orchestrator | `Drawer 2/Assets/Planetary/Runtime/PlanetBody.cs` |
| Height sources | `Drawer 2/Assets/Planetary/Runtime/PlanetaryPlanarBase.cs`, `Runtime/Sources/*` |
| Shell grid | `Drawer 2/Assets/Planetary/Runtime/Bridges/PlanetShellManifoldGrid.cs` |
| Continuuuum tiles | `Drawer 2/Assets/Planetary/Runtime/PlanetMeshStreamingService.cs` |
| Continuuuum weather | `Drawer 2/Assets/Planetary/Runtime/Weather/PlanetaryWeatherStreamingService.cs` |
| Service wiring | `Drawer 2/Assets/Planetary/Runtime/PlanetServiceWizardComponent.cs` |
| Backend routes | `continuuuum/serve_library.py` (`/api/planet/tiles`, `/weather_tiles`, `/composition`) |

### Demo ideas

1. Show `PlanetBody` in scene → `RebuildAll()` pipeline (planar → SDF → chunks → LOD).
2. Trace one height sample: world position → lat/lon → `IPlanetaryPlanarSource` stack (last non-zero wins).
3. If Continuuuum is running: hit `GET /api/planet/tiles?planet_id=earth&face=0&lod=0&x=0&y=0` and show Unity cache behavior.

### Review questions

- Is **shell grid pole/seam** behavior correct for new changes? (`PlanetShellManifoldGridTests.cs` is the gold standard.)
- Is **planar source ordering** documented and stable when adding providers?
- **Cross-face path adjacency** in `PlanetPathingBackend` — partially implemented; flag before relying on it.
- **Coverage fraction** in `PlanetMeshStreamingService` — currently a placeholder (PosX-only).
- **Continuuuum contract**: tile binary format and weather JSON field names (snake_case HTTP vs PascalCase C#).
- **Security**: POST routes write to disk; auth/path traversal if exposed beyond localhost.

### Test coverage snapshot

- **Strong:** shell grid, spatiotemporal charts, weather time travel, horizon LOD bands, volleyball stitch.
- **Gaps:** streaming HTTP integration, physics bridges, tectonics/volcano, relativity pathing, `RebuildAll()` E2E.

---

## 2. Roads

### What to explain

Roads is spline-based road authoring, mesh/SDF baking, and a **travel registry**. Roads integrate with hierarchical pathfinding: baked corridors become **`RoadCorridorMarker`** volumes that constrain **Drive** mode.

### Architecture (talk track)

```
Narrative placement ──► RoadLayoutPlacementSolver ──► HierarchicalPathingSolver (Drive)
        │
Hand / 4D splines ──► RoadSpline3D / RoadSpline4D
        │
RoadMeshBaker ──► ribbon mesh OR SDF corridor + SpatialVolumeProvider + RoadCorridorMarker
        │
RoadNetwork (nearest segment, waypoint snap) + feature applicators (erosion, weather, physics)
```

### Key files

| Area | Path |
|------|------|
| Splines | `Drawer 2/Assets/Roads/Runtime/RoadSplineBase.cs`, `RoadSpline3D.cs`, `RoadSpline4D.cs` |
| Bake pipeline | `Drawer 2/Assets/Roads/Runtime/RoadMeshBaker.cs`, `RoadCorridorSurfaceMesher.cs` |
| Path-driven placement | `Drawer 2/Assets/Roads/Runtime/Placement/RoadLayoutPlacementSolver.cs` |
| Network registry | `Drawer 2/Assets/Roads/Runtime/RoadNetwork.cs` |
| Weather hook | `Drawer 2/Assets/Roads/Runtime/RoadWeatherIntegration.cs` (rebakes planetary composition) |
| Corridor marker | `Drawer 2/Assets/HierarchicalPathFinding/HierarchicalPathingSolver.cs` (bottom of file — coupling note) |

### Demo ideas

1. Hand-place a road → bake → show corridor marker bounds in scene.
2. `RoadLayoutPlacementSolver`: narrative instruction → Drive-mode `FindPath()` → control points on spline.
3. Compare **ribbon bake** vs **SDF corridor bake** and what each registers with `SpatialVolumeProvider`.

### Review questions

- **`RoadCorridorMarker.IsOnRoadCorridor`** calls `FindObjectsByType` per edge check — hot path during A*.
- **`RoadCorridorMarker` lives in HierarchicalPathFinding assembly** — architectural coupling.
- Drive mode only applies corridor logic on **2D XZ grid**; verify behavior when backend is 3D/octree.
- **`RoadLayoutPlacementSolver`** mutates solver mode/flags — concurrent queries could interfere.
- End-to-end test gap: bake → marker → Drive `FindPath()` (main contract untested in Roads.Tests).

### Test coverage snapshot

- **Roads.Tests:** spline math, hand-authored placement only.
- **Locomotion.Tests:** broader pathfinding coverage that roads depend on.

---

## 3. Space pathing

### What to explain

There is no standalone “Space Pathing” module. Space travel is **`PhysicalPathingMedium.Space`** routed through a **solver registry**. Default stubs return straight segments; **Planetary bootstrap** replaces Space with great-circle / curved-spacetime solvers. **Separately**, spaceship scenes can optionally wrap hierarchical pathing with **`RadiationAwarePathingSolver`** to bias routes away from radiation fields on `PhysicsManifold` (not part of the Space registry).

### Architecture (talk track)

```
Locomotion (PathfindingNode, GenericTraversibilityPlannerSolver)
        │
PhysicalPathingSolverRegistry
        │
   Ground ──► delegates to hierarchical FindPath
   Air    ──► temporarily sets Fly mode
   Water  ──► straight-line stub
   Space  ──► straight-line stub OR Planetary override
        │
PlanetarySystemBootstrap registers CurvedSpacetimeSd2PathingSolver
PlanetShellManifoldGrid may register PlanetShellPathingSolver (last registration wins)

Optional (spaceship / vehicle scenes — not Space registry):
RadiationPathingOptions ──► RadiationAwarePathingSolver
        │                         │
        │                         ├─► HierarchicalPathingSolver.FindPath (base polyline)
        │                         └─► PhysicsManifold.SampleRadiation (path cost)
        FastMoverRegistry (ship ports; configured, lightly wired)
```

**Travel rule:** Space allows **Fly only**; Walk and Drive are blocked (`PhysicalMediumVolumeRules`).

**Radiation layer:** Post-processes a hierarchical base path with lateral offset passes; cost = `radiationVsTimeAlpha × ∫radiation·ds + (1−α) × path length`. Toggle `ignoreRadiation` / `ignoreTime` on the solver. Planet gameplay manifolds (e.g. `PhysicalManifold`) override `SampleRadiation`.

### Key files

| Area | Path |
|------|------|
| Registry + stubs | `Drawer 2/Assets/HierarchicalPathFinding/PhysicalPathingSolverRegistry.cs` |
| Medium volumes | `Drawer 2/Assets/HierarchicalPathFinding/PhysicalMediumVolume.cs`, `PhysicalMediumVolumeIndex.cs` |
| Space solver (Planetary) | `Drawer 2/Assets/Planetary/Runtime/Pathing/CurvedSpacetimeSd2PathingSolver.cs` |
| Shell alternative | `Drawer 2/Assets/Planetary/Runtime/Pathing/PlanetShellPathingSolver.cs` |
| Cross-face backend | `Drawer 2/Assets/Planetary/Runtime/Pathing/PlanetPathingBackend.cs` |
| Bootstrap | `Drawer 2/Assets/Planetary/Runtime/PlanetarySystemBootstrap.cs` |
| Hierarchical core | `Drawer 2/Assets/HierarchicalPathFinding/HierarchicalPathingSolver.cs` |
| Spatial volumes doc | `Drawer 2/Assets/HierarchicalPathFinding/docs/SpatialVolumeProvider.md` |
| Locomotion integration | `Drawer 2/Assets/locomotion/nodes/PathfindingNode.cs`, `travel/GenericTraversibilityPlannerSolver.cs` |
| Radiation-aware (optional) | `Drawer 2/Assets/locomotion/Spaceship/RadiationAwarePathingSolver.cs`, `RadiationPathingOptions.cs` |
| Radiation field hook | `Drawer 2/Assets/Weather/PhysicsManifold.cs` (`SampleRadiation`); `Planetary/Runtime/Relativity/PhysicalManifold.cs` |
| Fast mover registry | `Drawer 2/Assets/locomotion/Spaceship/FastMoverRegistry.cs` |

### Demo ideas

1. Scene **without** Planetary: Space path = straight line (stub).
2. Scene **with** Planetary bootstrap: show slerp great-circle path on planet shell.
3. Walk through **Walk / Fly / Drive** vs **Ground / Air / Water / Space** — two orthogonal axes.
4. **Radiation (optional):** attach `RadiationPathingOptions`, assign `radiationManifold`, compare base `HierarchicalPathingSolver` path vs radiation-biased offset path with `ignoreRadiation` on/off.

### Review questions

- **Two Space implementations** — bootstrap vs shell grid registration order matters.
- **Radiation vs Space** — radiation solver is **not** registered on `PhysicalPathingSolverRegistry`; BT/path nodes do not call it unless a spaceship component does.
- **`PlanetPathingBackend.TryCrossFace`** — incomplete cube-face neighbor logic.
- **`HierarchicalPathingQuadTree.cs`** — empty stub; dead code or WIP?
- **`PhysicalMediumVolume`** marked stub — volumes invalidate rebuild but may not block by medium yet.
- **Radiation tests** — only smoke test with `ignoreRadiation=true`; no test for manifold-driven reroute.
- No dedicated tests for Space solvers or planet shell pathing.

---

## 4. Lemma handling, library, and spatial maps

### What to explain

Two Flask processes share **`continuuuum.db`** today (merge planned). **continuuuum** hosts the USC geo map + **d3 spatial treemap**; **Drawer 2 `continuuuum_api`** hosts lemma/thesaurus, episodic APIs, and proxies library routes when configured.

Two spatial concepts—keep them separate in the presentation:

| Concept | Purpose | UI |
|---------|---------|-----|
| **Bucket treemap** (Q/O/S IDs) | Vocabulary render masks, episode asset causality | `library.html` Spatial tab |
| **`spatial_4d` volumes** | Lemma composition bindings (Back/Pause/Forward gateways) | Lemma library + Unity Explorer |

### Architecture (talk track)

```
continuuuum (serve_library.py)          Drawer 2 (continuuuum_api/server.py)
  /library  geo + spatial d3 map         /lemma-library  lemma browser
  /api/library/*  USC documents          /api/thesaurus/*  lemmas, composition
  /api/spatial/*  bucket trees           library_routes.py → proxy to continuuuum
         │                                        │
         └──────────── continuuuum.db ──────────────┘
              vocabulary_render_masks, episode_assets, thesaurus_entries, spatial_4d
```

See also: `continuuuum/CONTINUUUUM_SINGLE_SERVER.md` for merge target and cross-link matrix.

### Key files

| Area | Path |
|------|------|
| Library SPA | `continuuuum/library/library.html` |
| Library server | `continuuuum/serve_library.py` |
| Spatial API | `continuuuum/spatial_routes.py`, `spatial_bucket_parser.py` |
| Lemma routes | `Drawer 2/Scripts/continuuuum_api/lemma_routes.py` |
| Merge / import | `Drawer 2/Scripts/continuuuum_api/lemma_merge.py`, `lemma_import.py` |
| Composition + spatial | `Drawer 2/Scripts/continuuuum_api/lemma_composition_spatial.py` |
| Prompt expansion | `Drawer 2/Scripts/continuuuum_api/lemma_prompt.py` |
| Lemma SPA | `Drawer 2/Scripts/continuuuum_api/static/lemma-library/lemma-library.js` |
| Library proxy | `Drawer 2/Scripts/continuuuum_api/library_routes.py` |
| DB schemas | `Drawer 2/Scripts/continuuuum_episodes_schema.sql`, `continuuuum_spatial_4d_schema.sql` |
| Unity vocabulary | `Drawer 2/Assets/Continuuuum/VocabularyRenderMask.cs`, Editor lemma windows |

### Bucket ID formats (BedogaGenerator)

| Prefix | Kind | Example |
|--------|------|---------|
| `Q*` | Quad tree | `Q2.1.3` |
| `O*` | Oct tree | `O2.1.7` |
| `S{n}.O*` | 4D slice + oct | `S3.O2.1.7` |

### Spatial API endpoints

| Endpoint | Role |
|----------|------|
| `GET /api/spatial/render-masks` | Masks + bucket IDs (filter by tenant, episode, synonym) |
| `GET /api/spatial/episode-assets` | USC assets + `causalityLeafId` for an episode |
| `GET /api/spatial/bucket-tree` | Nested d3 tree (`treeType`: quad \| oct \| 4d) |
| `GET /api/spatial/resolve-highlight` | Token → library doc / bucket for deep links |

### Instructions: displaying spatial maps (demo script)

**Prerequisites**

1. Apply schemas to `continuuuum.db`: USC library schema + `continuuuum_episodes_schema.sql` (and `continuuuum_spatial_4d_schema.sql` if using 4D lemma bindings).
2. Populate data: `vocabulary_render_masks`, `vocabulary_render_mask_buckets`, and/or `episode_assets` for a known **episode UUID**.
3. Copy env from `continuuuum/.env.example` if needed (`PORT`, `CONTINUUUUM_DB_PATH`).

**Start servers**

```bash
# Terminal 1 — library + spatial (continuuuum repo)
cd c:\Users\John\continuuuum
pip install -e ..\unified-semantic-compressor
python serve_library.py
# Default: http://127.0.0.1:5050

# Terminal 2 — lemma library (optional, use alternate port if 5050 taken)
cd "c:\Users\John\Drawer 2\Scripts"
set PORT=5051
python -m continuuuum_api.server
```

**Port note:** Both default to **5050**. Run only one on that port, or set `PORT` on the other. See `CONTINUUUUM_SINGLE_SERVER.md`.

**View the spatial map**

1. Open `http://127.0.0.1:5050/library`.
2. Click the **Spatial** tab (Geo tab is Leaflet USC documents).
3. Enter an **Episode ID** (UUID with mask/asset data).
4. Optional: set **Synonym** filter (client-side tree prune).
5. Choose **Tree type**: All / Quad / Oct / 4D.
6. Click **Load spatial** → calls `GET /api/spatial/bucket-tree`.

**What renders**

- d3 **partition treemap**: hierarchy from `build_bucket_tree()` in `spatial_bucket_parser.py`.
- Cells colored by first `maskSynonym`; click opens bucket detail panel.
- Zoom/pan (scale 1–40). USC asset chips can jump back to Geo tab.

**Deep links**

```
/library?view=spatial&episodeId={uuid}&highlight={token}
```

- `view=spatial` — open Spatial tab on load.
- `episodeId` — pre-fill episode filter.
- `highlight` — resolved via `/api/spatial/resolve-highlight` (library doc id → episode asset → bucket id).

**Lemma ↔ map cross-link**

- In library toolbar, set **Lemma API URL** to Drawer 2 API (e.g. `http://127.0.0.1:5051` when library is on 5050).
- **Look up lemma** → `/api/thesaurus/entries` → opens `/lemma-library#entry/...`.
- From lemma library: `{libraryBase}?highlight={prefab-id|bucketId}&view=spatial` (via `localStorage.continuuuumLibraryBase` or `?libraryBase=`).

**Highlight resolution chain** (explain to reviewers)

1. Numeric token → `library_documents.id`
2. Episode asset match on `usc_asset_id` or `causality_leaf_id`
3. Direct bucket ID if parseable (Q/O/S format)

### Review questions

- Spatial API is **read-only** — no write paths for masks yet.
- **Two spatial models** (bucket treemap vs `spatial_4d`) — different APIs and UIs; easy to conflate.
- **`causality_leaf_id`** must align with bucket IDs for highlight chain.
- **Port 5050 collision** and `lemmaApiBase` defaults when servers run separately.
- **`library_routes.py` proxy** fallback path is machine-specific — verify in your environment.
- **Single-server merge** checklist in `CONTINUUUUM_SINGLE_SERVER.md` — blueprints partially started (`spatial_routes`).

### Test coverage snapshot

- `continuuuum/tests/test_spatial_bucket_parser.py`, `test_spatial_routes.py`
- `Drawer 2/Scripts/tests/test_lemma_composition.py`, `thesaurus/tests/test_lemma_import.py`

---

## 5. Multiplayer, logarithmic weather report falloff, group merge, synchronous alignment

### What to explain

These are related but distinct layers:

1. **Multiplayer** — dual transport (TCP tree stream + UDP lockstep decisions), domain bridges for weather/society/narrative.
2. **Weather report falloff** — client hyperplane regression payloads merged on the server with **confidence**, **shell overlap weights**, and **timeout decay** (not one single “log” function—clarify which layer you mean).
3. **Group calculation merge** — hyperplane layers combined via **log-sum-exp** (softmax) in spherical regression.
4. **Synchronous alignment** — `WeatherWorkQueue` ties merge to **frame index**; menu spec sync via `MainMenuNetworkRequirementsSync`.

### Multiplayer architecture

```
ClientOrchestrator ◄──TCP──► ServerOrchestrator  (TreeStream: LOD, snapshots, narrative rewind, weather eggs)
        ◄──UDP──►                              (DecisionChannel: lockstep, ownership)
                │
    WeatherLodNetworkBridge / SocietyLodNetworkBridge / NarrativeTimeTravelNetworkBridge
```

| Mode | Behavior |
|------|----------|
| SinglePlayer | Loopback bootstrap |
| Authoritative P2P | Peer tree transfer |
| Classic Lockstep | UDP decisions validated |

### Key files — networking

| Area | Path |
|------|------|
| Architecture doc | `Drawer 2/Assets/SystemDrawer/docs/NetworkingArchitecture.md` |
| Design theory | `Drawer 2/.cursor/plans/networking_architecture.md` (associative/monoid merge) |
| Server / client | `Drawer 2/Assets/SystemDrawer/Networking/ServerOrchestrator.cs`, `ClientOrchestrator.cs` |
| Transports | `TcpTreeStreamChannel.cs`, `UdpDecisionChannel.cs` |
| Weather bridge | `Drawer 2/Assets/SystemDrawer/Networking/WeatherLodNetworkBridge.cs` |
| Society bridge | `Drawer 2/Assets/Continuuuum/Society/SocietyLodNetworkBridge.cs` |
| Narrative rewind | `NarrativeTimeTravelNetworkBridge.cs` |
| Menu sync | `MainMenuNetworkRequirementsSync.cs`, `MainMenuNetworkRequirements.cs` |
| Tests | `Drawer 2/Assets/SystemDrawer/Networking/Tests/` |

### Key files — weather merge / falloff

| Mechanism | Path | What it does |
|-----------|------|--------------|
| Client reports | `Drawer 2/Assets/Weather/Lod/WeatherEggPayloads.cs` | `regressionPayload`, `confidence`, `timeoutOrder` |
| Layer grouping (log-sum-exp) | `Drawer 2/Assets/Weather/Lod/SphericalHyperplaneRegression.cs` | Softmax over hyperplane layers |
| Shell / overlap falloff | `Drawer 2/Assets/Weather/Lod/WeatherEggBounds.cs` | `ShellWeight()`, `OverlapGradientWeight()` |
| Server/client weight | `Drawer 2/Assets/Weather/Executor/WeatherKalmanMerge.cs` | `ServerClientWeight()` — exponential halving on timeout × confidence |
| Merge orchestration | `WeatherGradientEggMerger.cs`, `WeatherExecutorService.cs` | Merge client payloads + overlapping eggs |
| Frame sync | `Drawer 2/Assets/Weather/Executor/WeatherWorkQueue.cs` | Dequeue due by `frameIndex`; timeout bumps `timeoutOrder` |
| Circuit breaker | `Drawer 2/Assets/Weather/Lod/WeatherDiffCircuitBreaker.cs` | Fold sparse diff → regression under byte budget |
| Society mirror (Python) | `Drawer 2/Scripts/continuuuum_api/building_flywheel.py`, `society_merge.py` | Shared `merge_cell` / weight semantics |

### Demo ideas

1. Walk **NetworkingArchitecture.md** mode matrix: which trees are server-authoritative vs peer-transferable.
2. Trace one weather egg: client push → TCP → server queue → `WeatherGradientEggMerger` → manifold paint.
3. Show **three falloff concepts** side by side:
   - Log-sum-exp in `SphericalHyperplaneRegression.Evaluate()`
   - Egg shell weight in `WeatherEggBounds`
   - Timeout confidence in `WeatherKalmanMerge.ServerClientWeight()`
4. Compare C# `Pow(0.5, timeoutOrder)` vs Python `1/(1 + timeout_order*0.1)` in `building_flywheel.py` — parity question for society vs weather.

### Review questions

- **Spectator path**: TCP-only, no UDP decisions — verify tree policies.
- **Cross-asmdef boundary**: Weather registers via `WeatherNetworkSink` to avoid cycles.
- **Frame-index vs wall-clock** in `WeatherWorkQueue` vs client push interval (~100 ms default).
- **Two blend paths**: merger `definitionLevel * weight` vs client `serverBlend` recovery.
- **Society merge** (`society_merge.py`) — same weight vocabulary, different domain; is that intentional?
- **Associative merge design** in networking plan — implemented vs aspirational for weather aggregation order.

### Test coverage snapshot

- Networking: lobby, menu sync, causality audit, weather payload tests in `SystemDrawer/Networking/Tests/`.
- Weather: `WeatherEggMergeTests.cs` (Kalman merge + shell weight).
- Gaps: multi-client egg merge E2E, P2P ownership transfer.

**Related:** Section 6 (Video to Cloud) uses the same `WeatherKalmanMerge`, executor egg zones, and manifold paint path—CloudBake is the offline/authoring counterpart to live multiplayer weather merge.

---

## 6. Video to cloud (CloudBake)

### What to explain

**Video to cloud** turns reference footage (sky timelapse, flight video, stored diff clip) into **weather manifold cloud geometry**. The pipeline has two halves:

1. **Python preprocessing** — sample video frames, compute per-frame **sky color gradients** (`top=# mid=# bottom=#`), emit a **timeline JSON** for Unity or downstream tooling.
2. **Unity CloudBake** — perspective-raycast from a viewer camera, iteratively fit a **half-shell sphere stack** to reference colors, paint into `WeatherPhysicsManifold`, optionally **advect** between frames via the weather executor.

The broader **`video_storage_tool`** package (store → script → T2V → reconstitute) shares frame sampling and gradient extraction; CloudBake is the weather-specific export path.

### Architecture (talk track)

```
Video file (.mp4 / .ogv)
        │
        ├─► video_storage_tool (optional full pipeline)
        │     audio.aac + script.txt + resultant.mp4
        │     script.txt includes [Visual description] + [Color gradient]
        │
        └─► cloud_bake_from_video.py
              _describe_video_frames → _compute_color_gradient (per frame)
              cloud_bake_timeline.json  { frames: [{ frameIndex, gradient, advection, convexion, … }] }
                        │
                        ▼
Unity: Window → Weather → Cloud Perspective Bake
        CloudPerspectiveRaycaster (camera rays → cloud layer hits)
        CloudPerspectiveBakeSolver (iterative sphere fit + loss)
        CloudHalfShellBuilder → PaintIntoManifold(WeatherPhysicsManifold)
        optional: CloudBakeIntegration + WeatherExecutorService (advection between frames)
```

### Key files

| Area | Path |
|------|------|
| Python CLI | `Drawer 2/Scripts/video_storage_tool/cloud_bake_from_video.py` |
| Frame + gradient extraction | `Drawer 2/Scripts/video_storage_tool/video_to_script.py` (`_describe_video_frames`, `_compute_color_gradient`) |
| Video storage pipeline | `Drawer 2/Scripts/video_storage_tool/README.md`, `__main__.py`, `server.py` |
| JSON schema | `Drawer 2/Scripts/continuuuum_api/data/cloud_bake_schema.json` |
| Unity editor window | `Drawer 2/Assets/Weather/Editor/CloudPerspectiveBakeWindow.cs` |
| Bake solver | `Drawer 2/Assets/Weather/CloudBake/CloudPerspectiveBakeSolver.cs` |
| Raycasting | `Drawer 2/Assets/Weather/CloudBake/CloudPerspectiveRaycaster.cs` |
| Half-shell geometry | `Drawer 2/Assets/Weather/CloudBake/CloudHalfShellBuilder.cs`, `CloudHalfShellStack.cs`, `CloudHalfShellConvexion.cs` |
| Executor integration | `Drawer 2/Assets/Weather/CloudBake/CloudBakeIntegration.cs`, `CloudBakeSession.cs` |
| Advection descriptors | `Drawer 2/Assets/Weather/CloudBake/CloudAdvectionDescriptorBuilder.cs` |
| Iteration noise / Kalman | `Drawer 2/Assets/Weather/CloudBake/FresnelNoiseSchedule.cs` |
| Runtime cloud lock | `Drawer 2/Assets/Weather/Cloud.cs` (respects `CloudBakeSession`) |
| Shader hooks | `Drawer 2/Assets/Weather/WeatherShaderLibrary.cs` (`SetupCloudBakeLighting`) |

### Gradient format (shared contract)

Python and Unity both use the same string format:

```
top=#87ceeb mid=#bfd9f2 bottom=#8c9099
```

Python computes bands by sampling image thirds (`video_to_script._compute_color_gradient`). Unity parses via `CloudGradientBands.Parse()` and uses bands in the bake **loss function** (`GradientLoss`).

Timeline frame record (from Python CLI or Unity video bake):

```json
{
  "frameIndex": 0,
  "allowFloatAway": false,
  "anchorHash": 0,
  "gradient": "top=#aabbcc mid=#ddeeff bottom=#001122",
  "advection": { "mode": "AnchorReset" },
  "convexion": { "bias": 0.0, "size": 0.5 }
}
```

**Advection modes** (`cloud_bake_schema.json`):

| Mode | Meaning |
|------|---------|
| `AnchorReset` | Each frame re-anchors manifold cells (default) |
| `WeatherSolverAdvection` | Float-away: weather executor tick between frames (`allowFloatAway`) |

### Bake solver (Unity talk track)

`CloudPerspectiveBakeSolver.Bake()` loop per iteration:

1. Build or warm-start half-shell sphere stack from ray columns.
2. **Loss** = weighted sum of gradient, pixel opacity, density, shadow (TOD), physics (moisture when advecting).
3. **Gradient step** — adjust sphere density toward column target opacity.
4. **FresnelNoiseSchedule.PerturbSphere** — power-law sigma decay + view-dependent noise.
5. **Anchor reset** or **advection step** depending on `allowFloatAway`.
6. **KalmanBlendManifold** — blend painted cells using `FresnelNoiseSchedule.KalmanBlendWeight` (same family as live weather merge).

Video sequence mode (`CloudPerspectiveBakeWindow.RunBakeVideo`): warm-starts scalars between frames when `allowFloatAway` or `warmStartScalarsOnly`; writes `cloud_bake_timeline.json` at repo root.

### Instructions: demo script

**A. Python — extract timeline from video**

```bash
cd "c:\Users\John\Drawer 2\Scripts"
pip install -r video_storage_tool/requirements.txt
python video_storage_tool/cloud_bake_from_video.py path/to/sky_timelapse.mp4 -o cloud_bake_timeline.json --max-frames 10
```

Options:

- `--allow-float-away` — set advection mode to `WeatherSolverAdvection`
- `--convexion-bias`, `--convexion-size` — half-shell convexion (Forward ← → Back)

Requires **ffmpeg** on PATH for frame extraction (same as `video_storage_tool`).

**B. Unity — bake from photo or video**

1. Open a scene with `WeatherPhysicsManifold`, `Cloud`, `Wind`, `Water`, and optionally `WeatherExecutorService`.
2. **Window → Weather → Cloud Perspective Bake**.
3. Set **Viewer** camera (or Bounds / WorldPoint).
4. **Photo bake:** assign **Photo Texture** or paste **Gradient** text; click **Bake Photo** → **Apply Last Result To Scene**.
5. **Video bake:** set **Video Path**, **Video Frame Count**; toggle **Allow Float Away** / **Warm-start scalars only**; click **Bake Video Sequence** → inspect `cloud_bake_timeline.json`.
6. **Preview Convexion** — gizmo for half-shell bias/size before full bake.

**C. Optional — full video storage pipeline**

```bash
cd "c:\Users\John\Drawer 2\Scripts"
python -m video_storage_tool --input path/to/video.mp4 --out ./stored_clip --t2v-backend stub
# Produces script.txt with [Color gradient] section — source for cloud_bake_from_video reuse
python -m video_storage_tool.server --port 5000
# Upload UI + seekable reconstituted stream
```

### Links to other presentation topics

| Topic | Connection |
|-------|------------|
| **Multiplayer / weather merge** | CloudBake uses `WeatherExecutorService` egg zones and `WeatherKalmanMerge.BlendCells`; live eggs use the same manifold |
| **Log-sum-exp / falloff** | Bake uses Kalman `1/(1+sigma)` from `FresnelNoiseSchedule`, not egg shell falloff—but same executor paint path |
| **Planetary** | Cloud altitude bands sample against scene `Cloud.altitude`; planetary weather streaming is separate (Continuuuum tiles) |
| **Lemma / library** | Stored video artifacts (`script.txt`, gradients) could link to USC library docs via episode assets (future) |

### Review questions

- **Unity video pixel sampling gap:** `CloudPerspectiveTarget.videoPath` / `frameIndex` are set in the editor, but `CloudPerspectiveRaycaster.SampleReferenceColor` reads **texture** or **gradient bands** only—not decoded video frames. Per-frame pixel bake may require importing Python timeline gradients manually today.
- **Two timeline formats:** Python emits `{ "frames": [...] }`; Unity video bake writes a bare JSON array—align before automated import.
- **`allowFloatAway` semantics:** `CloudBakeSession` locks runtime `Cloud` float-away; bake integration registers a `cloud_bake` executor egg—verify cleanup on session end.
- **Warm-start path:** `CloneStackScalars` vs full rebuild when switching video frames—loss convergence behavior.
- **Shared Kalman weight** with live multiplayer merge—intentional parity or authoring-only tuning?
- **ffmpeg / BLIP / CogVideoX** deps in `video_storage_tool`—heavy optional stack; stub backends for CI.

### Test coverage snapshot

| File | Coverage |
|------|----------|
| `Drawer 2/Scripts/tests/test_cloud_bake_gradient.py` | Gradient computation, schema roundtrip, `allowFloatAway` flag |
| `Drawer 2/Scripts/video_storage_tool/tests/test_video_to_script_gradients.py` | Per-frame gradient assertion in script pipeline |
| `Drawer 2/Assets/Weather/Tests/CloudBakeTests.cs` | Anchor capture, session lifecycle, convexion utility, advection descriptor |
| Gaps | End-to-end Python timeline → Unity import; video frame decode in raycaster; executor advection across frame sequence |

---

## 7. Resaurce (legal tracking) and Saurce (investment)

### What to explain

Both are **Cave** SOA hosts: structural routes → domain handlers → Tome state machines → LVM events → inventory UI via Module Federation / RobotCopy. They are separate services; inventory talks to Saurce directly for investment—not through resaurce.

### Resaurce — legal documents

| Path | Role |
|------|------|
| Handlers | `D:/Development/resaurce/src/domains/legal/legalHandlers.js` |
| Tome | `D:/Development/resaurce/tomes/legal/v1/review.yaml` |
| Manifest | `D:/Development/resaurce/cave.manifest.yaml` |
| Routes | `D:/Development/resaurce/src/cave/xstate/resaurceRouteRegistry.js` |
| LVM contracts | `D:/Development/resaurce/contracts/lvm2/resaurce-machines.json` |
| Pact | `D:/Development/resaurce/contracts/pacts/inventory-frontend-resaurce-cave.json` |

**Routes**

| Route | Behavior |
|-------|----------|
| `legal/documents/list` | Seed + tenant catalog (ToS, Privacy, Mission) |
| `legal/document/enqueue` | Sync generation job → in-memory catalog |
| `legal/document/review` | Queue review stub (`rev_*`, status `queued`) |

**Storage:** in-memory only (`generatedCatalog`, `jobsById`) — lost on restart.

### Saurce — investment

| Path | Role |
|------|------|
| Handlers | `D:/Development/Saurce/src/domains/investment/investmentHandlers.js` |
| Store | `D:/Development/Saurce/src/domains/investment/investmentStore.js` |
| Tome | `D:/Development/Saurce/tomes/investment/v1/mode.yaml` |
| Wallet math | `D:/Development/Saurce/src/cave/pythonWalletMath.js` |
| Wallet debit | `D:/Development/Saurce/src/domains/wallet/walletHandlers.js` |
| Routes | `D:/Development/Saurce/src/cave/xstate/saurceRouteRegistry.js` |
| Pact | `D:/Development/Saurce/contracts/pacts/inventory-frontend-saurce-cave.json` |

**Eligibility rules**

| Hold type | Investable when |
|-----------|-----------------|
| `shipping` / `shipping_2x` | Risky mode enabled + anti-collateral deposited |
| `additional` | Always (3rd× holds) |
| `insurance` | After item shipped/delivered |

**Enable flow (`investment/mode/enable`):**

1. Validate `risk_percentage`, `anti_collateral` via Python/JS math adapter.
2. Debit wallet (`anti_investment_collateral`).
3. Store risk mode in `investmentStore`.

`amount_at_risk = shipping_hold_2x × risk_percentage / 100`  
Expected anti-collateral = `amount_at_risk × risk_boundary_error` (default 0.15).

### Demo ideas

1. **Legal:** call `legal/documents/list` → enqueue → review; show Tome transition on `legal/document/review`.
2. **Investment:** evaluate eligibility → enable risky mode → show wallet debit and stored risk mode.
3. Run `npm run verify:pact` in each repo.

### Review questions

**Resaurce**

- **Contract drift:** LVM JSON lists only `legal/document/review`; handlers also define `list` + `enqueue`.
- **UI Tome vs manifest:** `module.yaml` allowed routes may omit list/enqueue.
- **Persistence gap** vs tax domain — intentional stub or follow-up?

**Saurce**

- Python math path defaults to sibling `inventory/backend/python-apis/wallet-ledger/wallet_math.py` — verify layout.
- In-memory stores — no durability.
- Hold-type edge cases and `risky_mode_enabled` payload override.

---

## Cross-topic dependency map

```mermaid
flowchart TB
  subgraph planetary [Planetary]
    PB[PlanetBody]
    PSS[PlanetMeshStreamingService]
  end

  subgraph continuuuum [Continuuuum HTTP]
    SL[serve_library.py]
    SP[/api/spatial/*]
  end

  subgraph pathing [Pathing]
    HPS[HierarchicalPathingSolver]
    PSR[PhysicalPathingSolverRegistry]
    CSS[CurvedSpacetimeSd2PathingSolver]
  end

  subgraph roads [Roads]
    RMB[RoadMeshBaker]
    RCM[RoadCorridorMarker]
  end

  subgraph mp [Multiplayer]
    SO[ServerOrchestrator]
    WNB[WeatherLodNetworkBridge]
  end

  subgraph cloudbake [Video to Cloud]
    V2S[video_to_script]
    CBV[cloud_bake_from_video]
    CPS[CloudPerspectiveBakeSolver]
    WPM[WeatherPhysicsManifold]
  end

  subgraph vocab [Lemma / Spatial]
    LM[lemma_merge]
    VR[vocabulary_render_masks]
  end

  SL --> PSS
  PB --> PSS
  RMB --> RCM
  RCM --> HPS
  PB --> CSS
  CSS --> PSR
  HPS --> PSR
  WNB --> SO
  RMB --> PB
  VR --> SP
  LM --> VR
  V2S --> CBV
  CBV --> CPS
  CPS --> WPM
  WNB --> WPM
```

**Talking point:** Planetary height feeds weather; roads rebake planetary composition; corridor markers feed Drive pathing; Space pathing activates with Planetary bootstrap (straight-line stub otherwise); optional **radiation-aware** pathing wraps hierarchical routes for spaceship scenes; vocabulary render masks populate the spatial treemap; weather eggs merge over the multiplayer tree stream; **video to cloud** is the offline authoring path into the same manifold live eggs update at runtime.

---

## Pre-presentation checklist

- [ ] Continuuuum running with populated `continuuuum.db` and a known episode UUID for spatial demo
- [ ] Lemma server on alternate port if doing cross-link demo
- [ ] Unity scene with PlanetBody + PlanetarySystemBootstrap for space pathing contrast
- [ ] Optional: spaceship scene with `RadiationPathingOptions` + manifold with non-zero `SampleRadiation`
- [ ] Optional: road bake scene with Drive-mode pathing
- [ ] Optional: multiplayer lobby or loopback for weather egg push
- [ ] Sample sky video + ffmpeg installed for CloudBake Python demo
- [ ] Unity scene with WeatherPhysicsManifold, Cloud, Wind for **Window → Weather → Cloud Perspective Bake**
- [ ] Optional: `cloud_bake_timeline.json` from Python CLI to compare with Unity video bake output
- [ ] resaurce + Saurce dev servers for Cave route demos
- [ ] Pact verify passing in resaurce and Saurce if presenting contracts

---

## Appendix: canonical architecture docs

| Topic | Doc |
|-------|-----|
| Planetary | `Drawer 2/Assets/Planetary/docs/PlanetaryArchitecture.md` |
| Networking | `Drawer 2/Assets/SystemDrawer/docs/NetworkingArchitecture.md` |
| Spatial volumes | `Drawer 2/Assets/HierarchicalPathFinding/docs/SpatialVolumeProvider.md` |
| Continuuuum merge plan | `continuuuum/CONTINUUUUM_SINGLE_SERVER.md` |
| Networking merge theory | `Drawer 2/.cursor/plans/networking_architecture.md` |
| Video storage / gradients | `Drawer 2/Scripts/video_storage_tool/README.md` |
| CloudBake JSON schema | `Drawer 2/Scripts/continuuuum_api/data/cloud_bake_schema.json` |
| resaurce runbook | `D:/Development/resaurce/README.md` |

---

## Appendix: reviewer “sharp questions” (leave time for these)

1. **Planetary:** What happens at the poles and longitude wrap for a new feature touching the shell grid?
2. **Roads:** Who owns `RoadCorridorMarker` long term—Roads or PathFinding?
3. **Space pathing:** Which Space solver wins in a typical planet scene, and is that deterministic? Is radiation-aware rerouting in scope for this scene (spaceship layer vs registry)?
4. **Spatial maps:** If `causality_leaf_id` doesn’t match bucket format, where does the highlight chain break?
5. **Weather merge:** Is timeout decay exponential (C#) or rational (Python society flywheel)—should they match?
6. **Multiplayer:** What is server-authoritative for weather eggs vs peer-transferable trees?
7. **Video to cloud:** Does Unity decode video frames per index, or only Python-supplied gradients—and when do the two timeline JSON formats merge?
8. **Legal / investment:** What is the persistence story before production?
