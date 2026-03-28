# Continuum Media Fixtures

Small deterministic fixtures used for USC + Continuum feature parity tests.

These are intentionally synthetic and short to keep test execution fast.

## Profiles and measures

Defined in `../media_fixture_specs.json`:

- `audio_heavy_short`
  - `audio_heavy_4s.mp4`
  - 4 seconds, 320x240, 24 FPS
  - max size: 3.0 MB
- `video_heavy_short`
  - `video_heavy_4s.mp4`
  - 4 seconds, 640x360, 30 FPS
  - max size: 6.0 MB
- `image_source`
  - `image_source.png`
  - 640x360
  - max size: 1.5 MB

## Generation

From the continuum repo root:

```bash
python tests/generate_media_fixtures.py
```

Requires `ffmpeg` on PATH.

## Why synthetic fixtures

- No licensing ambiguity
- Deterministic behavior for codec/reconstruct comparisons
- Tiny artifacts for fast CI and local iteration
