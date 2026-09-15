# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
# setup.ps1 -- one-command setup for the IOI Control Panel.
#
# Usage, from this folder:
#   powershell -ExecutionPolicy Bypass -File setup.ps1 [-Yes] [-NoShortcuts]
#
# What it does, in order:
#   1. Finds Python 3.10, and offers to install it with winget if it's missing.
#   2. Installs PySpin from a Spinnaker wheel you downloaded, if it finds one.
#      It never downloads or redistributes the Spinnaker SDK itself.
#   3. Installs the acquisition and analysis packages (requirements_scripts.txt)
#      into Python 3.10.
#   4. Creates .venv and installs the GUI packages (requirements-gui.txt) into it.
#   5. Adds an "IOI Control Panel" shortcut to the Desktop and Start Menu.
#   6. Runs verify_install.py to report what this machine is ready for.
#
# Safe to run again: after pulling updates, or after installing the Spinnaker
# SDK on a machine that was first set up without it.
#
#   -Yes          answer setup's own [Y/n] prompts with yes
#   -NoShortcuts  skip step 5

param(
    [switch]$Yes,
    [switch]$NoShortcuts
)

Set-Location $PSScriptRoot

$WheelPattern = "spinnaker_python-*cp310*win_amd64.whl"

function Write-Section([string]$title) {
    Write-Host ""
    Write-Host $title -ForegroundColor Cyan
}

function Stop-Setup([string[]]$lines) {
    Write-Host ""
    Write-Host "ERROR: $($lines[0])" -ForegroundColor Red
    foreach ($line in $lines | Select-Object -Skip 1) {
        Write-Host "       $line" -ForegroundColor Red
    }
    Write-Host ""
    exit 1
}

function Confirm-Step([string]$question) {
    if ($Yes) {
        Write-Host "  $question [Y/n] y  (-Yes)" -ForegroundColor Cyan
        return $true
    }
    Write-Host "  $question [Y/n] " -ForegroundColor Cyan -NoNewline
    try {
        $answer = Read-Host
    } catch {
        # Not an interactive console (e.g. a remote or scheduled run).
        Write-Host ""
        Write-Host "  (no interactive console -- skipping; run setup again with -Yes to accept)" -ForegroundColor Yellow
        return $false
    }
    return ($answer -eq "" -or $answer -match "^[Yy]")
}

function Find-PyLauncher {
    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    # Per-user and all-users install locations of the python.org launcher.
    foreach ($candidate in @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Launcher\py.exe"),
        (Join-Path $env:WINDIR "py.exe")
    )) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Get-Python310Version([string]$launcher) {
    if (-not $launcher) { return $null }
    $out = & $launcher -3.10 -c "import sys; print(sys.version.split()[0])" 2>$null
    if ($LASTEXITCODE -eq 0 -and "$out" -match '^3\.10\.\d+') { return $Matches[0] }
    return $null
}

function Test-OnSavedPath([string]$dir) {
    # The shortcut-launched GUI gets PATH from the registry, not from this
    # console session, so check what is saved there.
    $saved = @(
        [Environment]::GetEnvironmentVariable("PATH", "Machine"),
        [Environment]::GetEnvironmentVariable("PATH", "User")
    ) -join ";"
    foreach ($entry in $saved -split ";") {
        if (-not $entry) { continue }
        $expanded = [Environment]::ExpandEnvironmentVariables($entry).TrimEnd("\")
        if ($expanded -ieq $dir.TrimEnd("\")) { return $true }
    }
    return $false
}

function New-IoiShortcut([string]$folder, [string]$root) {
    # pythonw.exe starts the GUI with no console window -- the same way the
    # packaged exe starts, which is what gui/script_runner.py's hidden-console
    # and graceful-stop handling were verified against.
    $shell = New-Object -ComObject WScript.Shell
    $path = Join-Path $folder "IOI Control Panel.lnk"
    $shortcut = $shell.CreateShortcut($path)
    $shortcut.TargetPath = Join-Path $root ".venv\Scripts\pythonw.exe"
    $shortcut.Arguments = '"' + (Join-Path $root "ioi_control_panel.py") + '"'
    $shortcut.WorkingDirectory = $root
    $shortcut.IconLocation = (Join-Path $root "ioi_icon.ico") + ",0"
    $shortcut.Description = "IOI Control Panel"
    $shortcut.Save()
    return $path
}

Write-Host ""
Write-Host "IOI Control Panel -- setup" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan

# --- 1. Python 3.10 ---------------------------------------------------------
Write-Section "[1/6] Python 3.10"
$launcher = Find-PyLauncher
$pyVersion = Get-Python310Version $launcher
if (-not $pyVersion) {
    Write-Host "  Python 3.10 was not found." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        if (Confirm-Step "Install Python 3.10 now with winget?") {
            winget install --id Python.Python.3.10 --exact --source winget
            # The installer updates PATH for new sessions only; refresh this one.
            $env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                        [Environment]::GetEnvironmentVariable("PATH", "User")
            $launcher = Find-PyLauncher
            $pyVersion = Get-Python310Version $launcher
        }
    } else {
        Write-Host "  winget isn't available on this machine, so setup can't install it for you." -ForegroundColor Yellow
    }
    if (-not $pyVersion) {
        Stop-Setup @(
            "Python 3.10 is required.",
            "Install it from https://www.python.org/downloads/release/python-31011/",
            "(check 'Add Python 3.10 to PATH' and 'Install launcher'), then run setup.ps1 again."
        )
    }
}
Write-Host "  Python $pyVersion  OK" -ForegroundColor Green
$launcherDir = Split-Path $launcher
if (-not (Test-OnSavedPath $launcherDir)) {
    Write-Host "  Warning: the 'py' launcher isn't on your saved PATH. The GUI starts every" -ForegroundColor Yellow
    Write-Host "  script with 'py -3.10', so add this folder to PATH, then sign out and in:" -ForegroundColor Yellow
    Write-Host "    $launcherDir" -ForegroundColor Yellow
}

# --- 2. PySpin --------------------------------------------------------------
Write-Section "[2/6] PySpin (Spinnaker SDK) -- only needed on machines that drive the camera"
$pyspin = & $launcher -3.10 -c "import PySpin; print('ok')" 2>$null
if ("$pyspin" -match "^ok") {
    Write-Host "  PySpin  OK" -ForegroundColor Green
} else {
    Write-Host "  PySpin is not installed yet." -ForegroundColor Yellow
    # Only ever installs a wheel you already downloaded from Teledyne yourself.
    $candidates = @()
    $candidates += Get-ChildItem -Path $PSScriptRoot, (Join-Path $env:USERPROFILE "Downloads") `
        -Filter $WheelPattern -ErrorAction SilentlyContinue
    $candidates += Get-ChildItem -Path (Join-Path $env:ProgramFiles "Teledyne\Spinnaker"), (Join-Path $env:ProgramFiles "FLIR Systems\Spinnaker") `
        -Filter $WheelPattern -Recurse -Depth 3 -ErrorAction SilentlyContinue
    $wheel = $candidates | Where-Object { $_ } | Sort-Object LastWriteTime -Descending | Select-Object -First 1

    if ($wheel) {
        Write-Host "  Found a Spinnaker wheel: $($wheel.FullName)" -ForegroundColor Cyan
        if (Confirm-Step "Install it now?") {
            & $launcher -3.10 -m pip install "$($wheel.FullName)"
            $pyspin = & $launcher -3.10 -c "import PySpin; print('ok')" 2>$null
            if ("$pyspin" -match "^ok") {
                Write-Host "  PySpin installed OK." -ForegroundColor Green
            } else {
                Write-Host "  The install ran, but PySpin still doesn't import -- check that the" -ForegroundColor Red
                Write-Host "  Spinnaker SDK itself is installed (INSTALL.md, section 1)." -ForegroundColor Red
            }
        }
    } else {
        Write-Host "  No downloaded Spinnaker wheel ($WheelPattern) in this folder," -ForegroundColor Yellow
        Write-Host "  Downloads, or the Spinnaker install folder. If it came as a .zip, extract" -ForegroundColor Yellow
        Write-Host "  the .whl first -- see INSTALL.md, section 1. Continuing without it." -ForegroundColor Yellow
    }
}

# --- 3. Script packages ----------------------------------------------------
Write-Section "[3/6] Acquisition and analysis packages (requirements_scripts.txt)"
& $launcher -3.10 -m pip install -r requirements_scripts.txt
if ($LASTEXITCODE -ne 0) {
    Stop-Setup @("pip could not install requirements_scripts.txt (exit code $LASTEXITCODE) -- see the output above.")
}

# --- 4. GUI environment ----------------------------------------------------
Write-Section "[4/6] GUI environment (.venv, requirements-gui.txt)"
$venvDir = Join-Path $PSScriptRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "  Creating .venv..."
    & $launcher -3.10 -m venv $venvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
        Stop-Setup @("could not create .venv (exit code $LASTEXITCODE) -- see the output above.")
    }
}
& $venvPython -m pip install -r requirements-gui.txt
if ($LASTEXITCODE -ne 0) {
    Stop-Setup @("pip could not install requirements-gui.txt into .venv (exit code $LASTEXITCODE) -- see the output above.")
}
& $venvPython -c "import PySide6, serial"
if ($LASTEXITCODE -ne 0) {
    Stop-Setup @("the GUI packages installed but don't import in .venv -- see the output above.")
}
Write-Host "  GUI environment  OK" -ForegroundColor Green

# --- 5. Shortcuts -----------------------------------------------------------
Write-Section "[5/6] Shortcuts"
if ($NoShortcuts) {
    Write-Host "  Skipped (-NoShortcuts)."
} else {
    foreach ($folder in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
        try {
            $made = New-IoiShortcut $folder $PSScriptRoot
            Write-Host "  $made" -ForegroundColor Green
        } catch {
            Write-Host "  Could not create a shortcut in $folder -- $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

# --- 6. Readiness report --------------------------------------------------
Write-Section "[6/6] What this machine is ready for (verify_install.py)"
& $launcher -3.10 verify_install.py
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Setup complete." -ForegroundColor Green
    if ($NoShortcuts) {
        Write-Host "Start the GUI with:  .venv\Scripts\pythonw ioi_control_panel.py" -ForegroundColor Green
    } else {
        Write-Host "Start the GUI from the IOI Control Panel shortcut on the Desktop or Start Menu." -ForegroundColor Green
    }
} else {
    Write-Host "Setup finished, but verify_install.py found a problem with a required package -- see above." -ForegroundColor Red
}
Write-Host ""
exit $verifyExit
