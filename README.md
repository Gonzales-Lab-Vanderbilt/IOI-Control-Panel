# Intrinsic Optical Imaging Control Panel

A desktop GUI and script toolchain for running an intrinsic optical imaging (IOI)
rig: guided red/green calibration, triggered acquisition, and offline analysis.

<!-- screenshot: docs/screenshot.png -->

## What it is

A thin [PySide6](https://doc.qt.io/qtforpython-6/) front end that lets lab members
operate the rig without touching a terminal. The GUI collects parameters from
forms and presets, assembles a command line, and runs the acquisition and
analysis scripts as **subprocesses**, streaming their stdout/stderr into a live
log pane.

## What it is not

The GUI is not the acquisition engine. It never imports PySpin, numpy, or any
acquisition code into its own interpreter — all camera and Arduino work happens
inside subprocesses launched through the `py -3.10` launcher. This keeps the GUI
environment completely decoupled from the Spinnaker SDK's pinned ABI, so the GUI
can be built and run on a machine that has no camera SDK installed at all.

## Safety model

Read this section before running the rig unattended.

- **Saturation / clip level is 65520, not 65535.** The IMX249 is a 12-bit sensor
  whose ADC output is left-shifted into Mono16: `4095 << 4 = 65520`. Any pixel at
  65520 is clipped. Treating 65535 as the sentinel silently under-reports
  saturation.

- **Stop sends `CTRL_BREAK_EVENT`** to a subprocess launched with
  `CREATE_NEW_PROCESS_GROUP`, with a hard `kill()` only as a last resort after a
  graceful stop times out (the UI warns when that happens).

- **Every stoppable script installs a `SIGBREAK` handler, and it is
  load-bearing.** On Windows, `CTRL_BREAK_EVENT` terminates a Python process
  outright with `STATUS_CONTROL_C_EXIT` (`0xC000013A`) unless the process
  installs one — no `KeyboardInterrupt`, no `finally`, and therefore no
  lights-off `teardown()`. Python handles `CTRL_C_EVENT` on its own but leaves
  `SIGBREAK` at `SIG_DFL`, so this has to be explicit. `intrinsic_imaging.py`,
  `intrinsic_calibrated_imaging.py`, `intrinsic_calibration.py`,
  `live_preview.py`, and `intrinsic_visual_stimulus.py` each call
  `install_break_handler()` as the first statement of `main()`. **Any new
  script the GUI can Stop must do the same**, or a Stop will leave the LEDs
  energized.

  **The exit code tells you whether teardown ran.** Only `0` and `130` mean the
  script reached its own `finally`, so those are the only two where the LEDs are
  known to be off. The GUI's status line translates these for you; the raw codes
  matter if you are running the scripts directly:

  | Exit | Meaning | LEDs |
  |---|---|---|
  | `0` | Ran to completion. | off, via teardown |
  | `130` | Stopped by you (`KeyboardInterrupt` / Stop / Ctrl+Break). | off, via teardown |
  | `1` | The script raised an error, or refused to start (missing PySpin, bad path). | **check the rig** |
  | `2` | Bad command line — argparse rejected it before anything ran. | never turned on |
  | `-1073741510` | `STATUS_CONTROL_C_EXIT` — killed before teardown could run. Means a stoppable script is missing its `SIGBREAK` handler. | **check the rig** |
  | other negative | Terminated by the OS (e.g. `0xC0000005`, an access violation). | **check the rig** |

  **The always-available Lights Off control remains the backstop**, for a hard
  `kill()` or a crash that outruns teardown. It is context-aware: if a session
  subprocess owns the COM port it triggers the graceful stop; if nothing is
  running it opens the port directly and sends `LIGHTS_OFF`.

## Repository layout

| Path | What it is |
|---|---|
| `ioi_control_panel.py`, `gui/` | The GUI (25 modules). PySide6 + pyserial only. |
| `intrinsic_calibrated_imaging.py` | Main entry point: interactive calibration, then triggered imaging. |
| `intrinsic_imaging.py` | Acquisition engine (`BlackflyCapture`, `TrialConfig`). |
| `intrinsic_calibration.py` | Calibration logic; opens interactive matplotlib windows. |
| `intrinsic_analysis.py` | Offline reanalysis of a saved session folder. |
| `intrinsic_visual_stimulus.py` | Drifting-grating visual stimulus server (UDP, port 55000). |
| `statistical_analyses.py` | ROI time-course, leave-one-out ROI, pixelwise t-map with permutation test. |
| `session_poster_figures.py` | Publication-style per-session figure panels. |
| `intrinsic_arduino/` | Arduino sync firmware. |
| `red.py`, `green.py`, `send_stim.py`, `reset_blackfly_roi.py`, … | Single-purpose utilities, each surfaced as a GUI button or small form. |
| `presets/` | Plain-JSON parameter sets that map onto argparse flags. |
| `build_exe.ps1`, `ioi_control_panel.spec` | PyInstaller packaging for the optional standalone exe. |
| `setup.ps1`, `verify_install.py` | One-command setup (Python 3.10, script and GUI packages, shortcuts), and a report of what this machine is ready for. |
| `make_icon.py` | Regenerates `ioi_icon.ico` from the lab logo. Build-time only; not needed to run anything. |

INSTALL.md carries the complete file-by-file listing.

## Requirements

- **Windows.** The acquisition host is Windows; the GUI uses Win32 process
  groups for its graceful stop.
- **Python 3.10 for the scripts**, reachable as `py -3.10`. This environment is
  separate from the GUI's.
- **FLIR / Teledyne Spinnaker SDK and a matching `spinnaker_python` (PySpin)
  wheel.** *These are not included in this repository and are not redistributable
  — download them from Teledyne.* Every camera-dependent script fails without
  PySpin.
- **numpy must stay below 2.0** in the script environment. PySpin is built
  against the numpy 1.x ABI; numpy 2.0 and later import fine on their own but
  break PySpin at runtime. The cap in `requirements_scripts.txt` is
  load-bearing.
- **Arduino IDE** to flash the sync firmware.
- **pygame** only if you run the visual stimulus server.

The GUI itself needs nothing but `PySide6` and `pyserial`
(`requirements-gui.txt`) and can run with no camera SDK present.

## Install

See [INSTALL.md](INSTALL.md).

## Quick start

**Setup** (any machine; camera machines install the Spinnaker SDK first — see
[INSTALL.md](INSTALL.md)):

```
powershell -ExecutionPolicy Bypass -File setup.ps1
```

This installs Python 3.10 if needed, the script and GUI packages, and an
**IOI Control Panel** shortcut on the Desktop and Start Menu, then reports what
this machine is ready for. Double-click the shortcut to start the GUI.

**Analysis only** (re-process a saved session folder):

```
py -3.10 intrinsic_analysis.py --folder .\captures\session_YYYYMMDD_HHMMSS --width 1920 --height 1200 --pixel-format Mono16
```

**Full rig** — flash the firmware, start the stimulus server if you are using
visual stimuli, then launch the GUI and use the Run Session tab.

The GUI runs `verify_install.py` for you a moment after it opens and reports
the result in the **environment** indicator in the top bar: green when the
core analysis packages are present, red otherwise. Click it for the full
report (Python version, package versions, camera count, serial ports) in the
Pre-Session Diagnostics tab. You can still run `py -3.10 verify_install.py`
directly in a terminal for the same output.

**Build a standalone executable** (optional — the setup shortcut already starts
the GUI with a double-click). The `.venv` that setup creates already has
PyInstaller (it is in `requirements-gui.txt`), so:

```
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```

This runs PyInstaller against `ioi_control_panel.spec` and writes
`ioi_control_panel.exe` into this folder, next to the scripts, ready to
double-click (details in [INSTALL.md](INSTALL.md#building-the-standalone-exe)).
No prebuilt binary ships in this repository. The
exe bundles the GUI only — the acquisition and analysis scripts still run as
subprocesses through `py -3.10`, so the target machine still needs the Python
3.10 environment from [INSTALL.md](INSTALL.md) for anything touching the camera.

## Analysis methods

`--analysis-method` selects how an activation map is computed:

| Value | What it computes |
|---|---|
| `raw_counts` *(default)* | Post-window mean minus baseline mean, in raw ADC counts, median-centered to remove global drift. |
| `fractional_reflectance` | Fractional reflectance change, with a denominator floor and low-denominator masking so near-zero baseline pixels cannot produce non-physiological values. |

`fractional` is accepted as shorthand for `fractional_reflectance`. These are
the only method names the toolchain recognises; a session folder written by an
older version whose `analysis_method` is not one of them will have every trial
skipped by the session pass, which then writes no session map and says so.

## Presets and configuration

Presets are plain JSON whose keys map directly onto argparse flags. They carry
the advanced analysis-tuning values so most users only ever see the common
controls (output folder, port, trials, ITI, stimulus, orientations).

Presets are **data only** — no live objects (serial connections, file handles,
camera handles) are ever serialized into a preset or any config file.

## Hardware

- FLIR/Teledyne Blackfly, IMX249-class sensor, Mono16.
- Arduino running `intrinsic_arduino/intrinsic_arduino.ino`, which drives the
  LEDs and emits the frame/stimulus marker protocol the acquisition scripts
  parse. The pin map is documented at the top of the `.ino`.
- Optional second monitor for the visual stimulus server.

## Known behavior

- **A session-analysis pass skips trials with missing arrays rather than
  failing.** `intrinsic_imaging.py` prints `Skipping trial ... missing analysis
  arrays` and moves on; if *every* trial is skipped it prints `Session analysis
  skipped: no trial maps available` and writes no session map at all. Nothing is
  silently averaged from an empty set, but a partial skip does quietly shrink the
  trial count behind the session mean — check that count in the log after a run.
- **The stimulus server binds `0.0.0.0` by default**, accepting UDP commands from
  any host on the network. That is intentional for a two-machine rig. On a
  single machine, pass `--bind-host 127.0.0.1`.
- **`session_poster_figures.py --reuse-extraction-cache` loads a pickle** from
  its own `--out-dir`. It only ever reads a cache the tool itself wrote; do not
  point it at a cache directory from an untrusted source.
- **Spatial calibration is rig-specific.** In the GUI, set it under Analysis →
  "Spatial calibration for your rig" (saved per computer in `rig_settings.json`).
  It starts unset, and figures get no scale bar until you enter a value; an
  unverified value asks for confirmation before each run. From the command line,
  `session_poster_figures.py --um-per-px` still defaults to a value measured on
  this lab's rig (3.682 um/px, a 7.07 mm field across 1920 px); pass your own
  value or `--no-scale-bar`. Re-measure for your own optics before quoting a
  physical distance.

## Attribution

Developed in the Gonzales Lab, Vanderbilt University. The "Gonzales Lab" name
and logo are retained in this repository as marks of the lab; the MIT grant
covers the code.

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

Research software, provided as-is. Not a medical device and not validated for
any clinical or diagnostic use.

## Contributing

Issues and pull requests are welcome. Note that the acquisition and analysis
scripts are validated against real experimental data — changes that alter
numerical output should say so explicitly and show a before/after comparison.
