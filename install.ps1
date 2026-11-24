# HealthLens installer for Windows.
#
# Run it by double-clicking install.bat, or from PowerShell in this folder:
#     powershell -ExecutionPolicy Bypass -File install.ps1
#
# What it does, in order:
#   1. Looks for Python 3.9 or newer.
#   2. If none is found, offers to install the latest Python 3 from the
#      Microsoft-maintained winget package (Python.Python.3.13), after asking.
#   3. Creates a private Python environment in the .venv folder here.
#   4. Installs HealthLens and its libraries into it (from PyPI).
# It changes nothing outside this folder except installing Python if you agree.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Write-Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }

function Find-Python {
    # Prefer the 'py' launcher that the python.org and winget installers add.
    $candidates = @(
        @{ Exe = "py"; Extra = @("-3") },
        @{ Exe = "python"; Extra = @() },
        @{ Exe = "python3"; Extra = @() }
    )
    foreach ($candidate in $candidates) {
        $exe = $candidate.Exe
        $extra = $candidate.Extra
        $cmd = Get-Command $exe -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        # Skip the Microsoft Store placeholder that only opens the Store.
        if ($cmd.Source -like "*WindowsApps*") { continue }
        try {
            $version = & $exe @extra -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        } catch { continue }
        if (-not $version) { continue }
        $parts = $version.Trim().Split(".")
        if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 9) {
            return @{ Exe = $exe; Args = $extra; Version = $version.Trim() }
        }
    }
    return $null
}

Write-Step "Looking for Python 3.9 or newer"
$python = Find-Python

if (-not $python) {
    Write-Host "Python was not found."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "Please install Python from https://www.python.org/downloads/"
        Write-Host "On the installer's first screen, tick 'Add python.exe to PATH'. Then run this installer again."
        exit 1
    }
    $answer = Read-Host "Install Python 3.13 now with winget? (y/n)"
    if ($answer -notmatch "^[Yy]") {
        Write-Host "Nothing was installed. Install Python from https://www.python.org/downloads/ and run this again."
        exit 1
    }
    Write-Step "Installing Python 3.13 (Windows may ask for permission)"
    winget install --id Python.Python.3.13 --exact --source winget --accept-package-agreements --accept-source-agreements
    # Make the new install visible to this window without restarting it.
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
    $python = Find-Python
    if (-not $python) {
        Write-Host "Python was installed but this window cannot see it yet."
        Write-Host "Close this window, open a new one, and run the installer again."
        exit 1
    }
}
Write-Host "Using Python $($python.Version)"

Write-Step "Creating the HealthLens environment in .venv"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    $pyArgs = $python.Args
    & $python.Exe @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Host "Could not create the environment."; exit 1 }
}
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Step "Installing HealthLens and its libraries (a few minutes the first time)"
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -e . --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing the libraries failed. Check your internet connection and run this again."
    exit 1
}

Write-Step "Checking the installation"
& $venvPython -m healthlens --version

Write-Host ""
Write-Host "HealthLens is installed." -ForegroundColor Green
Write-Host "From this folder you can now run, for example:"
Write-Host "    .\healthlens demo"
Write-Host "    .\healthlens run C:\Users\$env:USERNAME\Downloads\export.zip"
Write-Host "Your data and reports will be kept in $env:USERPROFILE\HealthLens"
