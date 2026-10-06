# Create .venv-sdk: Python 3.14 with the ZED SDK Python API, RealSense, OpenCV, PyAV and trimesh, plus this package
# (editable). Runs every step except the hand models, and the capture apps.
# Needs: Python 3.14 (py launcher) and the ZED SDK 5.3 (https://www.stereolabs.com/developers/release).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\setup_sdk_env.ps1
param([string]$PythonVersion = '3.14')
$ErrorActionPreference = 'Stop'
function Run { $exe, $rest = $args; & $exe @rest; if ($LASTEXITCODE) { throw "failed ($LASTEXITCODE): $args" } }

$repo = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $repo '.venv-sdk'
$zed = if ($env:ZED_SDK_ROOT_DIR) { $env:ZED_SDK_ROOT_DIR } else { 'C:\Program Files (x86)\ZED SDK' }
if (-not (Test-Path (Join-Path $zed 'bin'))) { throw "ZED SDK not found at $zed. Install ZED SDK 5.3 first." }
if (-not (Test-Path $venv)) { Run py "-$PythonVersion" -m venv $venv }
$py = Join-Path $venv 'Scripts\python.exe'
Run $py -m pip install --upgrade pip
Run $py -m pip install -r (Join-Path $repo 'environment\requirements-sdk.txt')
Run $py -m pip install --no-deps -e $repo
Run $py -c "from mchoi.recorder.devices import load_zed; import pyrealsense2, cv2; print('ZED SDK', load_zed().Camera.get_sdk_version(), '| OpenCV', cv2.__version__)"
Write-Host "SDK environment ready: $py"
