#Requires -Version 5.1
<#
.SYNOPSIS
    Installs the 3D-MAKER local GPU worker natively on Windows 10/11 (no WSL, no Docker, no cloud).

.DESCRIPTION
    Checks the NVIDIA GPU and driver, Python 3.11, Git, Node.js, Visual Studio 2022 Build Tools and the
    CUDA Toolkit 12.4; creates .venv-gpu; installs PyTorch 2.6.0 (CUDA 12.4) and the pinned dependencies;
    downloads Stable Fast 3D at a fixed commit and compiles its two extensions; signs in to Hugging Face and
    downloads the gated SF3D weights at a recorded revision; pre-downloads the CPU background-removal model;
    then runs the GPU self-test that decides whether this machine shows "Ready".

    Nothing here starts a paid service. Re-running is safe: finished steps are skipped or repeated harmlessly.

.PARAMETER InstallPrerequisites
    Install missing system tools with winget (Python 3.11, Git, Node.js LTS, VS 2022 Build Tools).
    Without it the script only tells you the exact command. The CUDA Toolkit is never installed silently.

.PARAMETER HfToken
    A Hugging Face read token. If omitted and you are not signed in, the script asks for it (input hidden).

.PARAMETER SkipSelfTest
    Skip the GPU self-test at the end (the worker runs it on first start anyway).

.PARAMETER CheckOnly
    Only run the system checks and print what is missing; change nothing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
#>
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSReviewUnusedParameter', 'InstallPrerequisites',
    Justification = 'Read inside Install-WithWinget.')]
[CmdletBinding()]
param(
    [switch]$InstallPrerequisites,
    [string]$HfToken = $env:HF_TOKEN,
    [switch]$SkipSelfTest,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# --- pinned versions (keep in sync with worker/windows/*.txt and worker/engines/sf3d.py) -------------
$Sf3dCommit = 'ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2'
$Sf3dRepo = 'https://github.com/Stability-AI/stable-fast-3d.git'
$TorchIndex = 'https://download.pytorch.org/whl/cu124'
$TorchCuda = [version]'12.4'
$PythonWanted = '3.11'

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Venv = Join-Path $Repo '.venv-gpu'
$Py = Join-Path $Venv 'Scripts\python.exe'
$Sf3dDir = Join-Path $Repo 'third_party\stable-fast-3d'
$Constraints = Join-Path $Repo 'worker\windows\constraints-win-py311.txt'

function Write-Step([string]$Number, [string]$Text) { Write-Host "`n[$Number] $Text" -ForegroundColor Cyan }
function Write-Ok([string]$Text) { Write-Host "  OK    $Text" -ForegroundColor Green }
function Write-Warn([string]$Text) { Write-Host "  WARN  $Text" -ForegroundColor Yellow }
function Exit-Setup([string]$Text, [string]$Fix) {
    Write-Host "  FAIL  $Text" -ForegroundColor Red
    if ($Fix) { Write-Host "        Fix: $Fix" -ForegroundColor Red }
    throw "Setup stopped: $Text"
}

function Invoke-Native {
    # Windows PowerShell does not stop on a failing .exe; check the exit code explicitly.
    param([string]$Exe, [string[]]$Arguments, [string]$What)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { Exit-Setup "$What failed (exit code $LASTEXITCODE)." 'See the output above; docs\WINDOWS.md has a troubleshooting section.' }
}

function Invoke-Quiet([scriptblock]$Block) {
    # Windows PowerShell 5.1 turns redirected native stderr into terminating errors under 'Stop'.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Block } finally { $ErrorActionPreference = $previous }
}

function Invoke-Helper {
    # Runs worker/windows/setup_helpers.py and returns its JSON result.
    param([string[]]$Arguments)
    Push-Location $Repo
    try { $raw = & $Py -m worker.windows.setup_helpers @Arguments } finally { Pop-Location }
    $json = ($raw | Select-Object -Last 1) | ConvertFrom-Json
    return $json
}

function Get-NvidiaSmi {
    $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @("$env:SystemRoot\System32\nvidia-smi.exe", "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe")) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Get-VsInstall {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $null }
    $path = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $path) { return $null }
    $vcvars = Join-Path $path 'VC\Auxiliary\Build\vcvars64.bat'
    $toolsFile = Join-Path $path 'VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt'
    $tools = if (Test-Path $toolsFile) { (Get-Content $toolsFile -Raw).Trim() } else { $null }
    return [pscustomobject]@{ Path = $path; VcVars = $vcvars; Toolset = $tools }
}

function Get-CudaToolkit {
    $candidates = @($env:CUDA_PATH_V12_4, (Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA\v12.4'))
    foreach ($c in $candidates) {
        if ($c -and (Test-Path (Join-Path $c 'bin\nvcc.exe'))) {
            $text = (& (Join-Path $c 'bin\nvcc.exe') --version) -join "`n"
            if ($text -match 'release (\d+\.\d+)') { return [pscustomobject]@{ Path = $c; Version = [version]$Matches[1] } }
        }
    }
    return $null
}

function Import-VsDevEnv([string]$VcVars) {
    # Load the MSVC x64 build environment into this PowerShell session.
    $lines = cmd /c "`"$VcVars`" >nul 2>&1 && set"
    if ($LASTEXITCODE -ne 0) { Exit-Setup 'Could not load the Visual Studio build environment.' 'Repair "Desktop development with C++" in the Visual Studio Installer.' }
    foreach ($line in $lines) {
        if ($line -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
    }
}

function Install-WithWinget([string]$Id, [string]$Extra) {
    if (-not $InstallPrerequisites) { return $false }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { Write-Warn 'winget is not available; install the tool manually.'; return $false }
    Write-Host "  ...   winget install $Id" -ForegroundColor Gray
    $wingetArgs = @('install', '-e', '--id', $Id, '--accept-package-agreements', '--accept-source-agreements')
    if ($Extra) { $wingetArgs += @('--override', $Extra) }
    & winget @wingetArgs
    if ($LASTEXITCODE -eq 0) { $script:InstalledSomething = $true; return $true }
    return $false
}

Write-Host '3D-MAKER local GPU worker setup (Stable Fast 3D, native Windows)' -ForegroundColor White
Write-Host "Repository: $Repo"
$missing = @()
$InstalledSomething = $false

# --- 1. system checks -------------------------------------------------------------------------------
Write-Step '1/9' 'Checking Windows, GPU and driver'
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
    Exit-Setup 'This script needs 64-bit Windows 10 or 11.' 'On Linux use worker/README.md instead.'
}
Write-Ok ("Windows {0} (build {1})" -f [Environment]::OSVersion.Version.Major, [Environment]::OSVersion.Version.Build)
$smi = Get-NvidiaSmi
if (-not $smi) {
    Exit-Setup 'nvidia-smi was not found, so no NVIDIA driver is installed.' 'Install the latest Game Ready or Studio driver from https://www.nvidia.com/Download/index.aspx and reboot.'
}
$gpuLine = (& $smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | Select-Object -First 1)
$gpu = $gpuLine -split ',\s*'
$vramMiB = 0
[void][int]::TryParse($gpu[1], [ref]$vramMiB)  # '[N/A]' on some setups
$banner = (& $smi) -join "`n"
$driverCuda = if ($banner -match 'CUDA Version:\s*(\d+\.\d+)') { [version]$Matches[1] } else { $null }
Write-Ok ("GPU: {0}, {1} MiB VRAM, driver {2}, driver supports CUDA {3}" -f $gpu[0], $gpu[1], $gpu[2], $driverCuda)
if ($driverCuda -and $driverCuda -lt $TorchCuda) {
    Exit-Setup "The driver supports CUDA $driverCuda, but PyTorch here needs CUDA $TorchCuda." 'Update the NVIDIA driver (https://www.nvidia.com/Download/index.aspx) and run setup again.'
}
if ($vramMiB -gt 0 -and $vramMiB -lt 7000) {
    Write-Warn ("{0} MiB is close to Stable Fast 3D's published ~6 GB at default settings. The self-test will find what fits (smaller textures, low-VRAM mode). Close games, browsers with GPU acceleration and other GPU apps before generating." -f $gpu[1])
}

Write-Step '2/9' 'Checking Python 3.11, Git and Node.js'
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
$basePython = $null
if ($pyLauncher) {
    $ver = Invoke-Quiet { & py "-$PythonWanted" -c "import sys,struct;print('%d.%d %d' % (sys.version_info[0], sys.version_info[1], struct.calcsize('P')*8))" 2>$null }
    if ($LASTEXITCODE -eq 0 -and $ver -eq "$PythonWanted 64") { $basePython = @('py', "-$PythonWanted") }
}
if (-not $basePython) {
    if (Install-WithWinget 'Python.Python.3.11') { $basePython = @('py', "-$PythonWanted") }
    else { $missing += "Python $PythonWanted (64-bit): winget install -e --id Python.Python.3.11" }
} else { Write-Ok "Python $PythonWanted (64-bit) via the py launcher" }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    if (-not (Install-WithWinget 'Git.Git')) { $missing += 'Git: winget install -e --id Git.Git' }
} else { Write-Ok ((git --version) -join '') }
$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node -or ([int]((node --version) -replace '^v(\d+).*$', '$1')) -lt 18) {
    if (-not (Install-WithWinget 'OpenJS.NodeJS.LTS')) { $missing += 'Node.js 18+ (glTF validation): winget install -e --id OpenJS.NodeJS.LTS' }
} else { Write-Ok "Node.js $((node --version) -join '')" }

Write-Step '3/9' 'Checking Visual Studio 2022 Build Tools and the CUDA Toolkit (needed to compile SF3D)'
$vs = Get-VsInstall
if (-not $vs) {
    $vsOverride = '--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
    if (Install-WithWinget 'Microsoft.VisualStudio.2022.BuildTools' $vsOverride) { $vs = Get-VsInstall }
    if (-not $vs) { $missing += "Visual Studio 2022 Build Tools with C++: winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override `"$vsOverride`"" }
} else { Write-Ok "MSVC toolset $($vs.Toolset) at $($vs.Path)" }
$cuda = Get-CudaToolkit
if (-not $cuda) {
    $missing += 'CUDA Toolkit 12.4 (compiles the SF3D texture baker for the GPU): https://developer.nvidia.com/cuda-12-4-1-download-archive (choose Windows > x86_64 > exe (local); in Custom install you may untick "Driver components" if your driver is newer)'
} else { Write-Ok "CUDA Toolkit $($cuda.Version) at $($cuda.Path)" }

if ($InstalledSomething) {
    Write-Host "`nwinget installed new tools. Open a NEW PowerShell window (so PATH is refreshed) and run setup again." -ForegroundColor Yellow
    exit 0
}
if ($missing.Count -gt 0) {
    Write-Host "`nMissing prerequisites:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host "`nInstall them (or re-run with -InstallPrerequisites for the winget ones), open a NEW PowerShell window, then run setup again."
    exit 1
}
if ($CheckOnly) { Write-Host "`nAll prerequisites found. (-CheckOnly: nothing was changed.)" -ForegroundColor Green; exit 0 }

# --- 2. Python environment ---------------------------------------------------------------------------
Write-Step '4/9' "Creating the Python environment ($Venv)"
if (-not (Test-Path $Py)) { Invoke-Native $basePython[0] @($basePython[1], '-m', 'venv', $Venv) 'Creating the virtual environment' }
Invoke-Native $Py @('-m', 'pip', 'install', '--upgrade', 'pip') 'Upgrading pip'
Invoke-Native $Py @('-m', 'pip', 'install', '-c', $Constraints, 'setuptools', 'wheel') 'Installing setuptools 69.5.1 and wheel'
Write-Ok 'Virtual environment ready'

Write-Step '5/9' 'Installing PyTorch 2.6.0 (CUDA 12.4) and the pinned dependencies (several GB, takes a while)'
Invoke-Native $Py @('-m', 'pip', 'install', 'torch==2.6.0', 'torchvision==0.21.0', '--index-url', $TorchIndex) 'Installing PyTorch'
Invoke-Native $Py @('-m', 'pip', 'install', '-c', $Constraints, '-r', (Join-Path $Repo 'requirements.txt'), '-r', (Join-Path $Repo 'worker\requirements.txt'), '-r', (Join-Path $Repo 'worker\windows\requirements-sf3d.txt')) 'Installing dependencies'
$torch = Invoke-Helper @('torch')
if (-not $torch.ok) { Exit-Setup $torch.error 'Make sure the NVIDIA driver is current, then run setup again.' }
Write-Ok "PyTorch $($torch.version) sees $($torch.device) (compute capability $($torch.capability), bf16: $($torch.bf16))"
Invoke-Native 'npm' @('ci', '--prefix', (Join-Path $Repo 'worker'), '--no-audit', '--no-fund') 'Installing the Khronos glTF validator'
Write-Ok 'Khronos glTF-Validator installed'

# --- 3. Stable Fast 3D code + compiled extensions ----------------------------------------------------
Write-Step '6/9' "Fetching Stable Fast 3D at $($Sf3dCommit.Substring(0, 7)) and compiling its extensions"
if (-not (Test-Path (Join-Path $Sf3dDir '.git'))) {
    Invoke-Native 'git' @('clone', '--filter=blob:none', $Sf3dRepo, $Sf3dDir) 'Cloning Stable Fast 3D'
}
Invoke-Native 'git' @('-C', $Sf3dDir, 'fetch', '--quiet', 'origin', $Sf3dCommit) 'Fetching the pinned commit'
Invoke-Native 'git' @('-C', $Sf3dDir, 'checkout', '--quiet', '--force', $Sf3dCommit) 'Checking out the pinned commit'
$site = (& $Py -c "import sysconfig;print(sysconfig.get_paths()['purelib'])").Trim()
Set-Content -Path (Join-Path $site 'stable_fast_3d.pth') -Value $Sf3dDir -Encoding ASCII  # makes `import sf3d` work

Import-VsDevEnv $vs.VcVars
$env:CUDA_HOME = $cuda.Path
$env:CUDA_PATH = $cuda.Path
$env:DISTUTILS_USE_SDK = '1'
$env:USE_CUDA = '1'
$env:USE_NATIVE_ARCH = '0'            # upstream adds -march=native, which MSVC does not understand
$env:TORCH_CUDA_ARCH_LIST = $torch.capability
if ($vs.Toolset -and [version]$vs.Toolset -ge [version]'14.40') {
    # CUDA 12.4 predates MSVC 14.40 (VS 17.10+); nvcc refuses newer hosts unless told otherwise.
    $env:NVCC_APPEND_FLAGS = '-allow-unsupported-compiler'
    Write-Warn "MSVC $($vs.Toolset) is newer than CUDA 12.4 officially supports; compiling with -allow-unsupported-compiler. If the build fails, install the 'MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.39)' component."
}
foreach ($ext in @('texture_baker', 'uv_unwrapper')) {
    Invoke-Native $Py @('-m', 'pip', 'install', '--no-build-isolation', '--force-reinstall', '--no-deps', (Join-Path $Sf3dDir $ext)) "Compiling $ext"
}
Write-Ok 'texture_baker (CUDA) and uv_unwrapper compiled'

# --- 4. settings, Hugging Face, weights --------------------------------------------------------------
Write-Step '7/9' 'Signing in to Hugging Face and downloading the gated Stable Fast 3D weights'
$envResult = Invoke-Helper @('init-env')
Write-Ok "Settings file: $($envResult.env_file)"
$login = Invoke-Helper @('hf-login')
if (-not $login.ok) {
    if (-not $HfToken) {
        Write-Host '  A Hugging Face READ token is needed once (https://huggingface.co/settings/tokens).'
        $secure = Read-Host '  Paste token (hidden)' -AsSecureString
        $HfToken = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
    }
    $env:HF_TOKEN_INPUT = $HfToken
    try { $login = Invoke-Helper @('hf-login') } finally { Remove-Item env:HF_TOKEN_INPUT -ErrorAction SilentlyContinue }
    if (-not $login.ok) { Exit-Setup $login.error '' }
}
Write-Ok "Signed in to Hugging Face as $($login.user)"
$weights = Invoke-Helper @('weights')
if (-not $weights.ok) { Exit-Setup $weights.error '' }
Write-Ok "Weights stabilityai/stable-fast-3d at revision $($weights.revision) (recorded in the settings file)"

Write-Step '8/9' 'Downloading the CPU background-removal model (U^2-Net, ~170 MB)'
$u2 = Invoke-Helper @('u2net')
if (-not $u2.ok) { Exit-Setup 'Could not download u2net.onnx.' 'Check your internet connection and run setup again.' }
Write-Ok 'U^2-Net ready (runs on the CPU, leaves the GPU to the 3D model)'
$check = Invoke-Helper @('check')
if (-not $check.ok) { Exit-Setup $check.problem '' }
Write-Ok 'Stable Fast 3D installation check passed'

# --- 5. self-test -------------------------------------------------------------------------------------
Write-Step '9/9' 'GPU self-test: real generation on a test photo to find what fits in your VRAM'
if ($SkipSelfTest) {
    Write-Warn 'Skipped (-SkipSelfTest). The worker runs it on first start.'
} else {
    Push-Location $Repo
    try { & $Py -m worker.main --engines sf3d --selftest; $selftestCode = $LASTEXITCODE } finally { Pop-Location }
    if ($selftestCode -eq 0) { Write-Ok 'Self-test passed: Stable Fast 3D is Ready on this GPU (details above).' }
    else { Write-Warn 'Self-test did not pass (details above). The studio will show the status and reason; see docs\WINDOWS.md > Troubleshooting.' }
}

Write-Host "`nDone. Start 3D-MAKER with:" -ForegroundColor Green
Write-Host '  powershell -ExecutionPolicy Bypass -File scripts\windows\start.ps1'
