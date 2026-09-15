#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
verify_install.py

Sanity-check this machine's Python environment against what the IOI
acquisition/analysis scripts need. Run standalone any time:

    py -3.10 verify_install.py

Also invoked by setup.ps1 right after installing pip packages, so a
bad install fails loudly during setup instead of mid-experiment.

Reports three independent tiers, since not every lab laptop needs all of
them:
    CORE    numpy / matplotlib / Pillow / tifffile / scipy / pyserial --
            needed for offline analysis (intrinsic_analysis.py,
            statistical_analyses.py, session_poster_figures.py) even on a
            laptop with no camera or Arduino attached.
    CAMERA  PySpin (Spinnaker SDK) -- needed only on the laptop actually
            driving the Blackfly camera (intrinsic_imaging.py,
            intrinsic_calibration.py, intrinsic_calibrated_imaging.py,
            capture_dark_reference.py, reset_blackfly_roi.py,
            live_preview.py). Not pip-installable -- see INSTALL.md, section 1.
    STIM    pygame -- needed only to run intrinsic_visual_stimulus.py
            (skip it if you are not running visual stimuli).

Exit code is 0 iff CORE is fully satisfied, regardless of CAMERA/STIM --
those are legitimately optional depending on what this particular laptop is
for, so their absence is reported but not treated as failure.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

_EXPECTED_PY = (3, 10)


def _header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _check_python() -> bool:
    _header("Python interpreter")
    v = sys.version_info
    print(f"  {sys.executable}")
    print(f"  {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) != _EXPECTED_PY:
        print(f"  NOTE: expected {_EXPECTED_PY[0]}.{_EXPECTED_PY[1]}.x -- the GUI always launches scripts")
        print("  via 'py -3.10' regardless of what ran this check, so re-run this as")
        print("  'py -3.10 verify_install.py' to check the interpreter the GUI will actually use.")
        return False
    print("  OK")
    return True


def _check_core() -> bool:
    _header("Core packages (required for analysis)")
    ok = True

    try:
        import numpy
        major = int(numpy.__version__.split(".")[0])
        if major >= 2:
            print(f"  numpy        {numpy.__version__}  FAIL -- PySpin needs numpy<2 (see requirements_scripts.txt)")
            ok = False
        else:
            print(f"  numpy        {numpy.__version__}  OK")
    except ImportError:
        print("  numpy        MISSING")
        ok = False

    try:
        import matplotlib
        print(f"  matplotlib   {matplotlib.__version__}  OK")
    except ImportError:
        print("  matplotlib   MISSING")
        ok = False

    try:
        import PIL
        print(f"  Pillow       {PIL.__version__}  OK")
    except ImportError:
        print("  Pillow       MISSING")
        ok = False

    try:
        import tifffile
        print(f"  tifffile     {tifffile.__version__}  OK")
    except ImportError:
        print("  tifffile     MISSING")
        ok = False

    try:
        import scipy
        print(f"  scipy        {scipy.__version__}  OK")
    except ImportError:
        print("  scipy        MISSING")
        ok = False

    try:
        import serial
        print(f"  pyserial     {serial.__version__}  OK")
    except ImportError:
        print("  pyserial     MISSING")
        ok = False

    if not ok:
        print()
        print("  Fix: py -3.10 -m pip install -r requirements_scripts.txt")

    return ok


def _check_camera() -> str:
    _header("Camera (Spinnaker SDK / PySpin) -- optional, only for the laptop driving the camera")
    try:
        import PySpin
    except ImportError:
        print("  PySpin       not installed")
        print("  Needed only to run intrinsic_imaging.py / intrinsic_calibration.py /")
        print("  intrinsic_calibrated_imaging.py / capture_dark_reference.py / reset_blackfly_roi.py /")
        print("  live_preview.py. Not pip-installable -- see INSTALL.md, section 1.")
        return "not installed (fine for a GUI-only or analysis-only laptop)"

    print("  PySpin       installed")
    try:
        system = PySpin.System.GetInstance()
        try:
            cam_list = system.GetCameras()
            count = cam_list.GetSize()
            cam_list.Clear()
            if count == 0:
                print("  cameras      0 detected -- check the USB connection and camera power")
                return "PySpin OK, but 0 cameras detected"
            plural = "" if count == 1 else "s"
            print(f"  cameras      {count} detected")
            return f"ready ({count} camera{plural} detected)"
        finally:
            system.ReleaseInstance()
    except Exception as exc:  # PySpin raises its own SpinnakerException hierarchy
        print(f"  cameras      could not enumerate -- {exc}")
        return "PySpin installed, but enumeration failed -- see above"


def _check_stim() -> str:
    _header("Visual stimulus server (pygame) -- optional, only for intrinsic_visual_stimulus.py")
    try:
        import pygame
        print(f"  pygame       {pygame.version.ver}  OK")
        return "ready"
    except ImportError:
        print("  pygame       not installed")
        print("  Needed only if you run the visual stimulus server.")
        print("  Fix: py -3.10 -m pip install \"pygame>=2.1\"")
        return "not installed (fine if you are not running visual stimuli)"


def _check_com_ports() -> None:
    _header("Serial ports (Arduino)")
    try:
        from serial.tools import list_ports
    except ImportError:
        print("  pyserial not installed -- can't enumerate ports (see Core packages above)")
        return
    ports = list(list_ports.comports())
    if not ports:
        print("  No serial ports found -- plug in the Arduino, or check drivers")
        print("  (Arduino IDE installs the necessary USB-serial drivers).")
    else:
        for p in ports:
            print(f"  {p.device}   {p.description}")


def main() -> int:
    print("IOI Control Panel -- environment check")
    print("=" * 39)

    py_ok = _check_python()
    core_ok = _check_core()
    camera_status = _check_camera()
    stim_status = _check_stim()
    _check_com_ports()

    _header("Summary")
    print(f"  Offline analysis (stats / figures):  {'READY' if core_ok else 'NOT READY -- see Core packages above'}")
    print(f"  Full rig (camera + LEDs):            {camera_status}")
    print(f"  Visual stimulus (Python server):     {stim_status}")

    if not py_ok:
        print()
        print("  NOTE: the Python interpreter check above did not find 3.10.x --")
        print("  re-run as 'py -3.10 verify_install.py' for an accurate check.")

    print()
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
