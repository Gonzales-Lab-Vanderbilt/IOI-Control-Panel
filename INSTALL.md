# IOI Control Panel — Installation Guide

Setup is one script. It installs everything that can be installed
automatically, then reports what this machine is ready for — offline
analysis, the full rig (camera + LEDs), and the visual stimulus server are
reported separately, since most machines only need one or two of them.

| This machine will... | Sections |
|---|---|
| Re-analyze captured sessions, or just run the GUI | [2. Run setup](#2-run-setup) |
| Drive the camera and LEDs | [1. Spinnaker SDK](#1-install-the-spinnaker-sdk-camera-machines-only), [2. Run setup](#2-run-setup), [3. Arduino firmware](#3-flash-the-arduino-firmware-rig-machines-only) |
| Show visual stimuli | also [4. Stimulus server](#4-start-the-stimulus-server-visual-stimuli-only) |

---

## 1. Install the Spinnaker SDK (camera machines only)

PySpin, the Python binding for FLIR/Teledyne cameras, comes from Teledyne and
is **not pip-installable** — this is the one step setup can't do for you.

1. Go to **https://www.teledynevisionsolutions.com/products/spinnaker-sdk/**
   (login required — create a free account).
2. Download and run the **Spinnaker SDK for Windows** installer (64-bit,
   matching your camera's tested version). Accept the defaults.
3. Get the Spinnaker Python wheel for Python 3.10,
   `spinnaker_python-<version>-cp310-cp310-win_amd64.whl`. Depending on the
   SDK version it is a separate download on the same page (extract it if it
   comes as a `.zip`) or sits inside the SDK's install folder. Put the `.whl`
   in this folder or your Downloads folder; setup looks there, and in the SDK
   install folder.

**Why isn't the wheel included here?** The Spinnaker SDK and its
`spinnaker_python` wheel are Teledyne's, under their own licence, and are not
redistributable — so this repository does not ship them.

If you install the SDK after running setup, just run setup again.

---

## 2. Run setup

Open PowerShell in this folder and run:

```
powershell -ExecutionPolicy Bypass -File setup.ps1
```

or right-click `setup.ps1` → **Run with PowerShell**. It:

1. Checks for Python 3.10 and, if it's missing, offers to install it with
   `winget`.
2. Installs PySpin from a downloaded Spinnaker wheel, if it finds one.
3. Installs the acquisition and analysis packages
   (`requirements_scripts.txt`) into Python 3.10.
4. Creates `.venv` and installs the GUI packages (`requirements-gui.txt`)
   into it, keeping them out of the acquisition scripts' environment.
5. Adds an **IOI Control Panel** shortcut to the Desktop and Start Menu.
6. Runs `verify_install.py` and prints what this machine is ready for.

Then **double-click the IOI Control Panel shortcut** to start the GUI.

Setup is safe to run again — do so after pulling updates, or after installing
the Spinnaker SDK on a machine you first set up without it. Options:

- `-Yes` answers setup's own prompts automatically. (On first use, `winget`
  may still ask you to accept its source agreement.)
- `-NoShortcuts` skips step 5. Start the GUI with
  `.venv\Scripts\pythonw ioi_control_panel.py` from this folder instead.

To re-check this machine any time, without installing anything:

```
py -3.10 verify_install.py
```

The GUI also runs this check a moment after it opens, and shows the result
as the **environment** indicator in its top bar; click it for the full report.

**Note on numpy:** PySpin is built against the numpy 1.x ABI and breaks with
numpy 2.x. `requirements_scripts.txt` pins `numpy<2.0` for exactly this
reason — don't remove that upper bound when updating dependencies.

### If setup can't install Python 3.10

Without `winget`, install it by hand: download the Windows 64-bit installer
from **https://www.python.org/downloads/release/python-31011/** (any 3.10.x
works), check **"Add Python 3.10 to PATH"** and **"Install launcher for all
users (recommended)"**, then run setup again.

---

## 3. Flash the Arduino firmware (rig machines only)

The Arduino controls the red and green LEDs and triggers the camera.

1. Install the **Arduino IDE** from https://www.arduino.cc/en/software.
2. In the IDE, open **Tools → Manage Libraries…** and install
   **Adafruit NeoPixel** — the sketch won't compile without it.
3. Open `intrinsic_arduino\intrinsic_arduino.ino`.
4. Connect the Arduino via USB, and select the board (Arduino Uno) and its COM
   port under **Tools**.
5. Click **Upload**.
6. Verify: in the GUI's Utilities tab, click **Red ON**. The LED should light
   up within a few seconds.

---

## 4. Start the stimulus server (visual stimuli only)

Setup already installed pygame. The server listens on UDP port 55000.

1. Start it from the GUI's **Utilities** tab (**Start Stim Server**), or from
   a terminal with `py -3.10 intrinsic_visual_stimulus.py`. Use
   `--list-monitors` first if you are unsure which screen is the stimulus
   monitor, then pass e.g. `--monitor 2`. On a single-machine rig, add
   `--bind-host 127.0.0.1` — the server otherwise binds `0.0.0.0` and accepts
   commands from any host on the network.
2. In the GUI's Run Session tab, enable **Visual stimulus**. The stim-server
   indicator (top bar) turns green when the server is reachable.
3. The stimulus server must be running **before** starting a session.

---

## Building the standalone exe

Optional — the setup shortcut already starts the GUI with a double-click, so
most labs don't need this. Build an exe if you want the GUI as a single
self-contained program. It bundles the GUI only: the acquisition and analysis
scripts still run through `py -3.10`, so the machine still needs setup for
anything that touches the camera.

1. Run setup first — the `.venv` it creates includes PyInstaller.
2. Build:
   ```
   powershell -ExecutionPolicy Bypass -File build_exe.ps1
   ```
   `build_exe.ps1` uses `.venv` automatically and checks that PyInstaller,
   PySide6, and pyserial are present before it starts, so a missing package
   stops the build with instructions instead of producing an exe that won't
   open. With no `.venv` it falls back to `py -3.10`; pass
   `-Python <path\to\python.exe>` to use a specific interpreter.
3. Double-click `ioi_control_panel.exe`, which the build writes into this
   folder, next to the scripts it launches (rebuilding replaces it; close the
   Control Panel first). The exe looks for the scripts beside itself, so
   don't move it — to start it from the Desktop, make a shortcut instead.

---

## Folder layout

```
setup.ps1                      ← one-command setup: Python 3.10, packages, .venv, shortcuts
ioi_control_panel.py           ← GUI entry point (what the setup shortcut runs)
gui\                           ← GUI modules (PySide6)
build_exe.ps1                  ← optional: builds a standalone .exe via PyInstaller
make_icon.py                   ← regenerates ioi_icon.ico from the lab logo (build-time only)
ioi_control_panel.spec         ← PyInstaller spec used by build_exe.ps1
requirements-gui.txt           ← pip packages for the GUI's .venv (PySide6, pyserial; PyInstaller for the exe build)
requirements_scripts.txt       ← pip packages for the scripts (Python 3.10)
verify_install.py              ← environment/hardware check -- setup runs it; run it any time
red.py / green.py              ← LED control (Arduino serial)
reset_blackfly_roi.py          ← reset camera ROI to full frame
capture_dark_reference.py      ← dark-noise reference stack
intrinsic_calibration.py       ← interactive ROI/exposure calibration
intrinsic_calibrated_imaging.py ← main acquisition entry point
intrinsic_imaging.py           ← acquisition engine (driven by the above)
intrinsic_analysis.py          ← offline reanalysis
statistical_analyses.py        ← stats pipeline (Analysis tab / GUI-driven)
session_poster_figures.py      ← auto-chained figures after a Statistics run
convert_raw_to_png.py          ← .raw → PNG converter
npy_to_tiff.py                 ← .npy → TIFF exporter
view_npy_image.py              ← quick-look viewer for .npy maps
send_stim.py                   ← manual UDP probe / stim trigger
intrinsic_visual_stimulus.py   ← visual stimulus server (needs pygame)
raw_image_opening.py           ← utility for raw image inspection
live_preview.py                ← live camera preview (Diagnostics tab / Run Session)
check_frame_timing.py          ← frame-timing diagnostic utility
render_crop_reference.py       ← renders the Analysis tab's crop-selector preview
intrinsic_arduino\             ← Arduino firmware
presets\                       ← saved parameter presets (JSON)
INSTALL.md                     ← this file
README.md                      ← project overview, safety model, analysis methods
LICENSE                        ← MIT
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Setup stops at "Python 3.10 is required" | Python 3.10 missing and `winget` unavailable or declined | Install it by hand — see [If setup can't install Python 3.10](#if-setup-cant-install-python-310) — then run setup again |
| Setup warns that the `py` launcher isn't on your saved PATH | Python installed without the launcher on PATH | Add the folder it prints to PATH and sign out and back in, or reinstall Python 3.10 with "Install launcher" checked |
| The shortcut says its target has changed or moved | The folder was moved, or `.venv` was deleted | Run setup again from the folder's current location; it recreates `.venv` if needed and rewrites the shortcut |
| `py -3.10` not recognized in a terminal | Python 3.10 or its launcher isn't installed | Run setup; it offers to install Python 3.10 |
| "PySpin not found" when running camera scripts | Spinnaker SDK or wheel not installed | Follow [section 1](#1-install-the-spinnaker-sdk-camera-machines-only), then run setup again |
| PySpin imports but camera scripts fail with an ABI/module error | numpy got upgraded to 2.x | `py -3.10 -m pip install "numpy<2.0"`, then re-run `verify_install.py` |
| PySpin installed, but `verify_install.py` reports 0 cameras | Camera not connected/powered, or USB issue | Check the USB cable and camera power; try a different USB port |
| COM port dropdown is empty | No serial drivers | Install Arduino drivers (via Arduino IDE) |
| Arduino IDE: `Adafruit_NeoPixel.h: No such file or directory` | NeoPixel library not installed | [Section 3](#3-flash-the-arduino-firmware-rig-machines-only), step 2 |
| LEDs don't turn off after a crash | Teardown didn't run | Use "Lights Off" button in GUI, or power-cycle the Arduino |
| Stim server shows DOWN | No stimulus server running | Start `intrinsic_visual_stimulus.py` first |
| Stimulus server won't start: "pygame is not installed" | pygame missing | Run setup again, or `py -3.10 -m pip install "pygame>=2.1"` |
| Stimulus opens on the wrong screen | Wrong monitor index | Run `py -3.10 intrinsic_visual_stimulus.py --list-monitors`, then pass `--monitor N` |
| `build_exe.ps1` stops with "missing: PyInstaller" (or PySide6) | `.venv` not set up | Run setup, then build again |
| `build_exe.ps1` says `ioi_control_panel.exe` is in use | The Control Panel is still open | Close it, then build again |
| A built exe shows a blank window or crashes immediately | Missing Visual C++ runtime | Install [VC++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| A built exe opens, but every step fails and the environment indicator shows issues | The exe was moved away from the scripts (or is an old build left in `dist\`) | Run the `ioi_control_panel.exe` next to the scripts; use a shortcut to start it from elsewhere — step 3 of [Building the standalone exe](#building-the-standalone-exe) |
