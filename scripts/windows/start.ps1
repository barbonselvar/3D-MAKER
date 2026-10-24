#Requires -Version 5.1
<#
.SYNOPSIS
    Starts 3D-MAKER on this computer: studio + API on http://127.0.0.1:8000 and the local GPU worker.

.PARAMETER Port
    Studio/API port (default 8000). The worker API uses Port + 1 and listens on 127.0.0.1 only.

.PARAMETER NoBrowser
    Do not open the studio in the default browser.

.PARAMETER NoWorker
    Start only the studio and API (no local GPU).

.PARAMETER SelfTest
    Re-run the GPU self-test now (for example after closing other GPU programs or updating the driver), then exit.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\windows\start.ps1
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$NoWorker,
    [switch]$SelfTest,
    [string]$Engines = 'sf3d'
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Py = Join-Path $Repo '.venv-gpu\Scripts\python.exe'
if (-not (Test-Path $Py)) {
    Write-Host 'The local environment is not installed yet. Run first:' -ForegroundColor Red
    Write-Host '  powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1'
    exit 1
}

Push-Location $Repo
try {
    if ($SelfTest) {
        & $Py -m worker.main --engines $Engines --selftest
        exit $LASTEXITCODE
    }
    $launchArgs = @('-m', 'app.local_stack', '--port', $Port, '--worker-port', ($Port + 1), '--engines', $Engines)
    if (-not $NoBrowser) { $launchArgs += '--open' }
    if ($NoWorker) { $launchArgs += '--no-worker' }
    Write-Host "Starting 3D-MAKER at http://127.0.0.1:$Port/  (press Ctrl+C to stop)" -ForegroundColor Green
    & $Py @launchArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
