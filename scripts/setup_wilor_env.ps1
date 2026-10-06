# Create .venv-wilor: Python 3.10 with CUDA PyTorch, WiLoR (pinned commit) and Ultralytics, plus this package (editable),
# then download the models. Used for hand detection/reconstruction, the MANO fit and the arm keypoints.
# Needs: Python 3.10 (py launcher), git, an NVIDIA GPU with a CUDA 12.8 capable driver.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\setup_wilor_env.ps1
param([string]$PythonVersion = '3.10')
$ErrorActionPreference = 'Stop'
function Run { $exe, $rest = $args; & $exe @rest; if ($LASTEXITCODE) { throw "failed ($LASTEXITCODE): $args" } }

$repo = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $repo '.venv-wilor'
$wilor = Join-Path $repo 'third_party\WiLoR'
$commit = 'fcb911312a38fa8badd30d9656a167485d61b8f9'
if (-not (Test-Path $venv)) { Run py "-$PythonVersion" -m venv $venv }
$py = Join-Path $venv 'Scripts\python.exe'
Run $py -m pip install --upgrade pip wheel 'setuptools<80'
Run $py -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
Run $py -m pip install numpy==1.23.5
Run $py -m pip install --no-build-isolation -r (Join-Path $repo 'environment\requirements-wilor.txt')
Run $py -m pip install --no-deps -e $repo
if (-not (Test-Path $wilor)) { Run git clone https://github.com/rolpotamias/WiLoR.git $wilor }
Run git -C $wilor fetch --quiet origin
Run git -C $wilor checkout --quiet $commit
Run $py -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
Run $py (Join-Path $repo 'scripts\download_models.py')
Write-Host "WiLoR environment ready: $py"
