# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
# build_exe.ps1 - one-click PyInstaller build for IOI Control Panel.
#
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File build_exe.ps1
#
# Output: ioi_control_panel.exe in this folder, next to the scripts it launches
#         (single file, ~40-50 MB with UPX). It looks for the scripts beside
#         itself, so run it from here -- use a shortcut to start it from
#         elsewhere. PyInstaller's intermediate files go in build\.
#
# Build interpreter. It needs PyInstaller, PySide6 and pyserial importable,
# and is chosen in this order:
#   1. -Python <path to python.exe>, if given
#   2. the project's .venv (.venv\Scripts\python.exe), if it exists
#   3. py -3.10
# setup.ps1 creates the .venv with everything the build needs (see INSTALL.md,
# "Building the standalone exe").
#
# The exe bundles only the GUI. The acquisition and analysis scripts still run
# as subprocesses through `py -3.10`, so the target machine needs the Python 3.10
# environment from INSTALL.md for anything that touches the camera.

param(
    [string]$Python = ""
)

Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$fallbackNote = ""
if ($Python) {
    $pyExe = $Python
    $pyArgs = @()
    $pyLabel = $Python
} elseif (Test-Path $venvPython) {
    $pyExe = $venvPython
    $pyArgs = @()
    $pyLabel = ".venv"
} else {
    $pyExe = "py"
    $pyArgs = @("-3.10")
    $pyLabel = "py -3.10"
    $fallbackNote = " (no .venv found)"
}

function Write-SetupHint {
    Write-Host ""
    Write-Host "Run setup first -- it creates .venv with everything the build needs:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File setup.ps1" -ForegroundColor Yellow
    Write-Host "then run build_exe.ps1 again. See INSTALL.md, 'Building the standalone exe'." -ForegroundColor Yellow
}

Write-Host "Building IOI Control Panel with $pyLabel$fallbackNote..." -ForegroundColor Cyan

if (-not (Get-Command $pyExe -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "ERROR: build interpreter not found: $pyExe" -ForegroundColor Red
    Write-SetupHint
    exit 1
}

# Check before building. Without PySide6, PyInstaller still reports success
# and produces an exe that closes on launch with no window and no message.
$probe = "import importlib.util as u; print(','.join(n for n in ('PyInstaller', 'PySide6', 'serial') if u.find_spec(n) is None))"
$probeOut = & $pyExe @pyArgs -c $probe
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: could not run the build interpreter ($pyLabel, exit code $LASTEXITCODE)." -ForegroundColor Red
    Write-SetupHint
    exit 1
}
$missing = "$(@($probeOut) | Select-Object -Last 1)".Trim()
if ($missing) {
    Write-Host ""
    Write-Host "ERROR: the build interpreter ($pyLabel) is missing: $($missing -replace ',', ', ')" -ForegroundColor Red
    Write-SetupHint
    exit 1
}

# The exe is written straight into this folder, replacing any earlier build.
# Windows locks a running exe, so check now rather than letting PyInstaller
# fail at the very end of the build.
$exePath = Join-Path $PSScriptRoot "ioi_control_panel.exe"
if (Test-Path $exePath) {
    try {
        $handle = [System.IO.File]::Open($exePath, 'Open', 'ReadWrite', 'None')
        $handle.Close()
    } catch {
        Write-Host ""
        Write-Host "ERROR: $exePath is in use or read-only." -ForegroundColor Red
        Write-Host "Close the IOI Control Panel, then run build_exe.ps1 again." -ForegroundColor Yellow
        exit 1
    }
}

# --distpath . puts the exe in this folder (the script already cd'd here)
# instead of PyInstaller's default dist\.
& $pyExe @pyArgs -m PyInstaller ioi_control_panel.spec --noconfirm --distpath .
$buildExit = $LASTEXITCODE

if ($buildExit -eq 0) {
    $size    = [math]::Round((Get-Item $exePath).Length / 1MB, 1)
    Write-Host ""
    Write-Host "Build succeeded!  ($size MB)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Executable: $exePath"
    Write-Host "Double-click it right here. To start it from the Desktop, make a shortcut;"
    Write-Host "don't move the exe -- it looks for the scripts beside itself."
    $staleExe = Join-Path $PSScriptRoot "dist\ioi_control_panel.exe"
    if (Test-Path $staleExe) {
        Write-Host ""
        Write-Host "Note: an older build is still in dist\. It can't find the scripts from" -ForegroundColor Yellow
        Write-Host "there, so delete it to avoid launching it by mistake." -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "The exe bundles the GUI only. Camera and analysis scripts still run"
    Write-Host "through py -3.10 -- see INSTALL.md for the script environment."
} else {
    Write-Host ""
    Write-Host "Build FAILED (exit code $buildExit)." -ForegroundColor Red
    Write-Host "Check the output above for errors."
    exit $buildExit
}
