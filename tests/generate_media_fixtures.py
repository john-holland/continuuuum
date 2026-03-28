"""
Generate small deterministic media fixtures for USC/Continuum parity testing.

Requires ffmpeg on PATH.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPECS_PATH = HERE / "media_fixture_specs.json"
FIXTURE_DIR = HERE / "fixtures"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    specs = json.loads(SPECS_PATH.read_text(encoding="utf-8"))
    profiles = {p["id"]: p for p in specs["profiles"]}

    audio = profiles["audio_heavy_short"]
    audio_out = FIXTURE_DIR / audio["output_file"]
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=navy:s={audio['width']}x{audio['height']}:d={audio['duration_sec']}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:duration={audio['duration_sec']}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=660:duration={audio['duration_sec']}",
            "-filter_complex",
            "[1:a][2:a]amix=inputs=2:weights='1 0.5'[mix]",
            "-map",
            "0:v",
            "-map",
            "[mix]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(audio_out),
        ]
    )

    video = profiles["video_heavy_short"]
    video_out = FIXTURE_DIR / video["output_file"]
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={video['width']}x{video['height']}:rate={video['fps']}:duration={video['duration_sec']}",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=48000:cl=stereo:d={video['duration_sec']}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video_out),
        ]
    )

    image = profiles["image_source"]
    image_out = FIXTURE_DIR / image["output_file"]
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=steelblue:s={image['width']}x{image['height']}",
            "-frames:v",
            "1",
            str(image_out),
        ]
    )

    print(f"Generated fixtures in: {FIXTURE_DIR}")


if __name__ == "__main__":
    generate()
