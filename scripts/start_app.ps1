# Start the calibration or recording desktop app with the SDK environment (logs in <repo>\logs).
param([Parameter(Mandatory)][ValidateSet('calibration', 'recorder')][string]$App)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv-sdk\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Environment missing: run scripts\setup_sdk_env.ps1 (see README.md).' }
$logs = Join-Path $repo 'logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$env:PYTHONPATH = Join-Path $repo 'src'
Start-Process -FilePath $python -ArgumentList @('-m', "mchoi.$App") -WorkingDirectory $repo -WindowStyle Normal `
    -RedirectStandardOutput (Join-Path $logs "$App.stdout.log") -RedirectStandardError (Join-Path $logs "$App.stderr.log")
