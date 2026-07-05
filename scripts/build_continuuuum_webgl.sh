#!/usr/bin/env bash
set -euo pipefail

# Build WebGL Continuuuum Library UI from the Drawer 2 Unity project into library/continuuuum_editor_webgl/.
#
# Requires:
#   UNITY_EDITOR_PATH  — Unity Editor executable (same major.minor as Drawer 2 ProjectSettings/ProjectVersion.txt)
# Optional:
#   DRAWER2_PROJECT    — Unity project root containing Assets/ (default: sibling ../Drawer 2 next to this repo)
#   CONTINUUUUM_WEBGL_OUT — Output directory (default: this repo's library/continuuuum_editor_webgl)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DRAWER2="${DRAWER2_PROJECT:-$(cd "${REPO_ROOT}/.." && pwd)/Drawer 2}"
OUT="${CONTINUUUUM_WEBGL_OUT:-${REPO_ROOT}/library/continuuuum_editor_webgl}"

if [[ -z "${UNITY_EDITOR_PATH:-}" ]]; then
  echo "Set UNITY_EDITOR_PATH to your Unity Editor binary." >&2
  exit 1
fi

if [[ ! -d "${DRAWER2}/Assets" ]]; then
  echo "DRAWER2_PROJECT must point at a Unity project (missing Assets/). Got: ${DRAWER2}" >&2
  exit 1
fi

echo "Unity project: ${DRAWER2}"
echo "WebGL output:  ${OUT}"

"${UNITY_EDITOR_PATH}" -batchmode -nographics -quit \
  -projectPath "${DRAWER2}" \
  -executeMethod BuildContinuuuumWebGL.BuildFromCli \
  -continuuuumWebGlOut "${OUT}"
