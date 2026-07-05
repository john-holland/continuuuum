#Requires -Version 5.0
$ErrorActionPreference = "Stop"

# Build WebGL Continuuuum Library UI from Drawer 2 into library/continuuuum_editor_webgl/.
# Env:
#   UNITY_EDITOR_PATH   (required) e.g. "C:\Program Files\Unity\Hub\Editor\6000.3.2f1\Editor\Unity.exe"
# Optional:
#   DRAWER2_PROJECT     Unity project root containing Assets\
#   CONTINUUUUM_WEBGL_OUT Output folder

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$DefaultDrawer2 = Join-Path (Split-Path $RepoRoot -Parent) "Drawer 2"
$Drawer2 = if ($env:DRAWER2_PROJECT) { $env:DRAWER2_PROJECT } else { $DefaultDrawer2 }
$Out = if ($env:CONTINUUUUM_WEBGL_OUT) { $env:CONTINUUUUM_WEBGL_OUT } else { Join-Path $RepoRoot "library\continuuuum_editor_webgl" }

if (-not $env:UNITY_EDITOR_PATH) {
    Write-Error "Set UNITY_EDITOR_PATH to Unity.exe"
}
$Unity = $env:UNITY_EDITOR_PATH

if (-not (Test-Path (Join-Path $Drawer2 "Assets"))) {
    Write-Error "DRAWER2_PROJECT must point at a Unity project (missing Assets). Got: $Drawer2"
}

Write-Host "Unity project: $Drawer2"
Write-Host "WebGL output:  $Out"

& $Unity -batchmode -nographics -quit `
    -projectPath $Drawer2 `
    -executeMethod BuildContinuuuumWebGL.BuildFromCli `
    -continuuuumWebGlOut $Out
