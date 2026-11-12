$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$runFolder = 'outputs/dummy_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& '.\.venv\Scripts\python.exe' -m models_other.train --dummy --output $runFolder
if ($LASTEXITCODE -ne 0) { throw 'Training smoke test failed.' }
& '.\.venv\Scripts\python.exe' -m models_other.check_dummy $runFolder
if ($LASTEXITCODE -ne 0) { throw 'Export or Grad-CAM check failed.' }
