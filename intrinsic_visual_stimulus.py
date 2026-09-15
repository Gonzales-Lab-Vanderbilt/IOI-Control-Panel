#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
intrinsic_visual_stimulus.py

Visual stimulus server for intrinsic optical imaging. Opens a black fullscreen
stimulus window on the stimulus monitor and drives drifting sinusoidal gratings
in response to UDP commands from intrinsic_imaging.py /
intrinsic_calibrated_imaging.py (--visual-stim).

Requires neither MATLAB nor Psychtoolbox.

Intended workflow:
  1) Start this script first. It opens a black fullscreen window and listens.
  2) Start intrinsic_calibrated_imaging.py with --visual-stim enabled.
  3) The acquisition process sends UDP commands when it sees Arduino markers:
       STIM <orientation_deg> <duration_s> <trial_index>
       BLACK
       QUIT

This deliberately does NOT open or control the Arduino, and never touches the
camera. The acquisition process owns all hardware timing.

Usage:
    py -3.10 intrinsic_visual_stimulus.py
    py -3.10 intrinsic_visual_stimulus.py --monitor 2 --udp-port 55000
    py -3.10 intrinsic_visual_stimulus.py --list-monitors
    py -3.10 intrinsic_visual_stimulus.py --no-fullscreen      # windowed, for testing

Stop with the QUIT command, the Escape key, or Ctrl+C / Ctrl+Break. Every exit
path blanks the screen to black and saves the stimulus log.

Requires pygame (py -3.10 -m pip install "pygame>=2.1"). A slower tkinter
fallback exists behind --backend tk; see the warning it prints before you rely
on it for real data.

DURATION QUIRK (deliberate, for compatibility with validated sessions):
    showDriftingGrating() recomputes its own duration as
        nOrientations * gratingDurationS
    and ignores the duration carried by the STIM command. With the stock
    settings that is 4 * 1.25 = 5.0 s. In single-orientation mode it is
    1 * 1.25 = 1.25 s, again regardless of what was requested. The orientation
    in the STIM command is likewise ignored in multi-orientation mode.
    This behaviour is the default so timing matches previously validated
    sessions. Use --respect-requested-duration to opt out.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Silence pygame's "Hello from the pygame community" banner. The GUI streams
# this process's stdout into a log pane; keep it to real messages only.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame  # type: ignore
except ImportError:
    pygame = None  # checked in make_display(); --backend tk still works


# Number of phase steps the sinusoid is quantized into, both spatially and
# temporally. See build_phase_maps() for why it is quantized at all. At 1024
# steps the rendered frame is within 1/255 of the exact sinusoid - i.e. inside
# the display's own 8-bit rounding - while spatial quantization sits near
# 0.16 px and temporal quantization near 0.5 ms at 2 Hz. The step count does
# not affect per-frame cost (the lookup table stays a couple of kB) or memory
# (the per-pixel index stays uint16), so there is no reason to lower it.
PHASE_STEPS = 1024

# Commands that only report liveness. gui/stim_probe.py sends "PROBE" on a
# timer; recognizing it here keeps the health check from spamming the log with
# unknown-command warnings. It never starts or stops a stimulus.
PROBE_COMMANDS = ("PROBE", "PING")
PROBE_REPLY = b"IOI_STIM_READY"


def probe_reply(args: argparse.Namespace, orientations: list[float]) -> bytes:
    """PROBE_REPLY followed by a JSON description of which STIM fields this
    server will honor, so the GUI can show what will actually run. Anything
    that only checks the reply's prefix still sees PROBE_REPLY."""
    config = {
        "multi_orientation": bool(args.multi_orientation),
        "orientations_deg": [float(o) for o in orientations],
        "grating_duration_s": float(args.grating_duration_s),
        "respect_requested_duration": bool(args.respect_requested_duration),
        "randomize_orientations": bool(args.randomize_orientations),
    }
    return PROBE_REPLY + b" " + json.dumps(config, separators=(",", ":")).encode("utf-8")

_IDLE_POLL_S = 0.001  # idle tick between datagram polls


def log(message: str) -> None:
    """Print one timestamped line, flushed so the GUI log pane stays live."""
    stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{stamp}] {message}", flush=True)


def warn(message: str) -> None:
    log(f"WARNING: {message}")


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# --------------------------------------------------------------------------
# Monitor enumeration
# --------------------------------------------------------------------------

def windows_monitors() -> list[tuple[int, int, int, int]]:
    """
    Return [(left, top, width, height), ...] for each monitor, primary first.

    Best-effort via the Win32 API; returns [] if anything goes wrong. Used by
    the tkinter backend (which cannot enumerate monitors on its own) and by
    --list-monitors. The pygame backend uses SDL's own display indices instead.
    """
    if sys.platform != "win32":
        return []
    try:
        import ctypes
        from ctypes import wintypes

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        monitors: list[tuple[int, int, int, int]] = []

        callback_type = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.c_ulonglong,
            ctypes.c_ulonglong,
            ctypes.POINTER(RECT),
            ctypes.c_double,
        )

        def _callback(hmon, hdc, lprect, data):  # noqa: ANN001 - Win32 signature
            r = lprect.contents
            monitors.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
            return 1

        ctypes.windll.user32.EnumDisplayMonitors(
            wintypes.HDC(0), None, callback_type(_callback), 0
        )
        # Win32 enumeration order is not guaranteed, so put the primary
        # monitor (the one whose origin is (0, 0)) first, which is what a
        # 1-based --monitor index is expected to count from.
        monitors.sort(key=lambda m: (m[0] != 0 or m[1] != 0, m[0], m[1]))
        return monitors
    except Exception:
        return []


def describe_monitors() -> str:
    """Human-readable monitor list for --list-monitors and startup errors."""
    lines: list[str] = []

    sizes: list[tuple[int, int]] = []
    if pygame is not None:
        try:
            pygame.display.init()
            sizes = list(pygame.display.get_desktop_sizes())
            pygame.display.quit()
        except Exception:
            sizes = []

    rects = windows_monitors()
    count = max(len(sizes), len(rects))
    if count == 0:
        return "  (could not enumerate monitors)"

    for i in range(count):
        parts = [f"  --monitor {i + 1}"]
        if i < len(sizes):
            parts.append(f"{sizes[i][0]}x{sizes[i][1]}")
        if i < len(rects):
            left, top, w, h = rects[i]
            parts.append(f"at ({left}, {top}) [{w}x{h}]")
        if i == 0:
            parts.append("(primary)")
        lines.append("  ".join(parts))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Display backends
#
# Both backends share one convention: image arrays are shaped (width, height)
# or (width, height, 3) with index [x, y], x growing right and y growing DOWN.
# That is pygame's native surfarray layout, and it matches the top-left
# origin the geometry math below assumes, so the orientation angles map to
# gratings the same way in both backends.
# --------------------------------------------------------------------------

class Display:
    """Common interface: a black RGB canvas the grating is painted into."""

    width: int
    height: int

    def fill_black(self) -> None:
        raise NotImplementedError

    def draw_gray(self, gray: np.ndarray, x0: int, y0: int, blue_only: bool) -> None:
        """Paint a (w, h) uint8 luminance patch at (x0, y0)."""
        raise NotImplementedError

    def draw_box(self, rect: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
        raise NotImplementedError

    def present(self) -> None:
        raise NotImplementedError

    def user_quit_requested(self) -> bool:
        """True if the operator closed the window or pressed Escape."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class PygameDisplay(Display):
    """Fast path. Renders into an offscreen surface, then blits and flips."""

    name = "pygame"

    def __init__(self, monitor_index: int, fullscreen: bool) -> None:
        assert pygame is not None
        pygame.init()
        pygame.mouse.set_visible(False)

        sizes = list(pygame.display.get_desktop_sizes())
        if not sizes:
            raise SystemExit("pygame could not detect any monitors.")

        if monitor_index >= len(sizes):
            warn(
                f"Requested --monitor {monitor_index + 1} but only {len(sizes)} "
                f"monitor(s) detected. Using monitor 1."
            )
            monitor_index = 0

        size = sizes[monitor_index]
        if fullscreen:
            # Deliberately NOT pygame.FULLSCREEN: that flag is SDL's real
            # exclusive fullscreen, which does a display-mode switch and makes
            # Windows disable desktop composition (DWM) for this surface. On a
            # multi-monitor rig, clicking the OTHER monitor forces DWM to flip
            # composition back on and recomposite everything, which is seen as
            # every open window "refreshing" at once - a known Windows/SDL
            # interaction, not something specific to this stimulus. A borderless
            # (NOFRAME) window sized to exactly cover the monitor gets the same
            # fullscreen look without the exclusive-mode switch, so DWM
            # composition never turns off and nothing flickers.
            flags = pygame.NOFRAME
        else:
            # Windowed mode is for desk testing; use a smaller window so the
            # operator can still reach the taskbar.
            flags = 0
            size = (min(size[0], 1280), min(size[1], 720))

        self._screen = self._set_mode(size, flags, monitor_index)
        self.width, self.height = self._screen.get_size()
        self.monitor_index = monitor_index

        # Offscreen buffer in the display's own pixel format, so the per-frame
        # blit to the screen is a straight copy with no format conversion.
        self._surface = pygame.Surface((self.width, self.height)).convert()
        self._surface.fill((0, 0, 0))

    @staticmethod
    def _set_mode(size, flags, monitor_index):
        # vsync=1 keeps the drift free of tearing, but SDL refuses it on some
        # driver/monitor combinations. Fall back rather than fail to start.
        try:
            return pygame.display.set_mode(size, flags, display=monitor_index, vsync=1)
        except pygame.error:
            return pygame.display.set_mode(size, flags, display=monitor_index)

    def fill_black(self) -> None:
        self._surface.fill((0, 0, 0))

    def draw_gray(self, gray: np.ndarray, x0: int, y0: int, blue_only: bool) -> None:
        w, h = gray.shape
        # pixels3d() returns a live view into the surface memory and locks the
        # surface while that view exists, so writes land with no intermediate
        # copy. The surface must be unlocked again before present() can blit it,
        # which is what deleting the view in the finally block does.
        px = pygame.surfarray.pixels3d(self._surface)
        try:
            if blue_only:
                px[x0:x0 + w, y0:y0 + h, 2] = gray
            else:
                # Three separate plane writes, not px[..., :] = gray[:, :, None].
                # The broadcasting form takes ~8 ms at 1920x1080 versus ~2.9 ms
                # for these three, which is the difference between comfortable
                # and marginal against a 16.67 ms vsync budget.
                px[x0:x0 + w, y0:y0 + h, 0] = gray
                px[x0:x0 + w, y0:y0 + h, 1] = gray
                px[x0:x0 + w, y0:y0 + h, 2] = gray
        finally:
            del px

    def draw_box(self, rect, color) -> None:
        self._surface.fill(color, pygame.Rect(*rect))

    def present(self) -> None:
        self._screen.blit(self._surface, (0, 0))
        pygame.display.flip()

    def user_quit_requested(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_ESCAPE,
                pygame.K_q,
            ):
                return True
        return False

    def close(self) -> None:
        try:
            pygame.quit()
        except Exception:
            pass


class TkDisplay(Display):
    """
    Fallback path for machines without pygame.

    Every frame is converted through PIL and handed to Tk, which at 1920x1080
    costs tens of milliseconds. The resulting frame rate is low enough to make
    a 2 Hz drift visibly stepped, so this backend is opt-in only.
    """

    name = "tk"

    def __init__(self, monitor_index: int, fullscreen: bool) -> None:
        import tkinter as tk
        from PIL import Image, ImageTk

        self._tk = tk
        self._Image = Image
        self._ImageTk = ImageTk

        rects = windows_monitors()
        if monitor_index >= len(rects):
            if rects:
                warn(
                    f"Requested --monitor {monitor_index + 1} but only "
                    f"{len(rects)} monitor(s) detected. Using monitor 1."
                )
            monitor_index = 0
        left, top, w, h = rects[monitor_index] if rects else (0, 0, 1280, 720)

        if not fullscreen:
            w, h = min(w, 1280), min(h, 720)

        self._root = tk.Tk()
        self._root.title("Visual Stimulus Server")
        self._root.configure(bg="black")
        if fullscreen:
            # overrideredirect + explicit geometry places the window precisely
            # on the requested monitor. Tk's -fullscreen attribute only covers
            # whichever monitor the window already happens to be on.
            self._root.overrideredirect(True)
            self._root.attributes("-topmost", True)
        self._root.geometry(f"{w}x{h}+{left}+{top}")

        self.width, self.height = w, h
        self.monitor_index = monitor_index

        self._buf = np.zeros((w, h, 3), dtype=np.uint8)
        self._photo = ImageTk.PhotoImage(Image.new("RGB", (w, h), "black"))
        self._label = tk.Label(self._root, image=self._photo, borderwidth=0,
                               highlightthickness=0, bg="black")
        self._label.pack(fill="both", expand=True)

        self._quit_requested = False
        self._root.bind("<Escape>", self._on_quit_key)
        self._root.bind("<q>", self._on_quit_key)
        self._root.protocol("WM_DELETE_WINDOW", self._on_quit_key)
        self._root.focus_force()
        self._root.update()

    def _on_quit_key(self, event=None) -> None:  # noqa: ANN001 - Tk callback
        self._quit_requested = True

    def fill_black(self) -> None:
        self._buf[:] = 0

    def draw_gray(self, gray: np.ndarray, x0: int, y0: int, blue_only: bool) -> None:
        w, h = gray.shape
        if blue_only:
            self._buf[x0:x0 + w, y0:y0 + h, 2] = gray
        else:
            self._buf[x0:x0 + w, y0:y0 + h, :] = gray[:, :, None]

    def draw_box(self, rect, color) -> None:
        x, y, w, h = rect
        self._buf[x:x + w, y:y + h, :] = color

    def present(self) -> None:
        # PIL wants (row, col) = (height, width); our buffer is (width, height).
        image = self._Image.fromarray(np.transpose(self._buf, (1, 0, 2)))
        self._photo.paste(image)
        self._root.update()

    def user_quit_requested(self) -> bool:
        try:
            self._root.update()
        except Exception:
            return True
        return self._quit_requested

    def close(self) -> None:
        try:
            self._root.destroy()
        except Exception:
            pass


def make_display(backend: str, monitor_index: int, fullscreen: bool) -> Display:
    if backend == "tk":
        warn(
            "Using the tkinter backend. Frame rate is far below the pygame "
            "backend, so the drifting grating will look stepped. Acceptable "
            "for a wiring check; install pygame before collecting real data."
        )
        return TkDisplay(monitor_index, fullscreen)

    if pygame is None:
        raise SystemExit(
            "pygame is not installed in this Python environment.\n"
            "  Install it with:  py -3.10 -m pip install \"pygame>=2.1\"\n"
            "  Or run the slower fallback with:  --backend tk"
        )
    return PygameDisplay(monitor_index, fullscreen)


# --------------------------------------------------------------------------
# Stimulus geometry and precomputation
# --------------------------------------------------------------------------

@dataclass
class Geometry:
    """Screen and patch layout, plus the px/deg conversion."""

    screen_w: int
    screen_h: int
    patch_w: int
    patch_h: int
    x0: int
    y0: int
    px_per_deg: float
    spatial_freq_cpp: float


def compute_geometry(args: argparse.Namespace, screen_w: int, screen_h: int) -> Geometry:
    """Pixels-per-degree math and patch placement for the stimulus monitor."""
    deg_per_cm = 2.0 * math.atan2(0.5, args.view_dist_cm) * (180.0 / math.pi)
    cm_per_deg = 1.0 / deg_per_cm
    px_per_cm = args.monitor_res_px / args.monitor_width_cm
    px_per_deg = px_per_cm * cm_per_deg
    spatial_freq_cpp = args.spatial_freq_cpd / px_per_deg

    if args.patch_size_px is not None:
        patch_w, patch_h = args.patch_size_px
    elif args.fullscreen_stim:
        patch_w, patch_h = screen_w, screen_h
    else:
        square = int(min(screen_w, screen_h) * args.patch_size_frac)
        patch_w = patch_h = square

    patch_w = 2 * (patch_w // 2)  # keep even so the patch centres on a pixel
    patch_h = 2 * (patch_h // 2)

    x0 = (screen_w - patch_w) // 2
    y0 = (screen_h - patch_h) // 2

    if patch_w <= 0 or patch_h <= 0 or x0 < 0 or y0 < 0:
        raise SystemExit(
            f"Patch {patch_w}x{patch_h} does not fit on the stimulus monitor "
            f"({screen_w}x{screen_h}). Check --patch-size-px / --patch-size-frac."
        )

    return Geometry(
        screen_w=screen_w,
        screen_h=screen_h,
        patch_w=patch_w,
        patch_h=patch_h,
        x0=x0,
        y0=y0,
        px_per_deg=px_per_deg,
        spatial_freq_cpp=spatial_freq_cpp,
    )


def build_sine_lut(contrast: float, gain: float, offset: float) -> np.ndarray:
    """
    One temporal cycle of the grating waveform as 8-bit levels, tiled twice.

    Tiling lets a drifting phase be expressed as a plain slice offset:
    lut2[s : s + PHASE_STEPS] is the whole waveform rotated by s steps, with
    no modulo arithmetic needed per frame.
    """
    j = np.arange(PHASE_STEPS, dtype=np.float64)
    values = offset + 0.5 * contrast * np.sin(2.0 * np.pi * j / PHASE_STEPS)
    values = np.clip(gain * values, 0.0, 1.0)
    lut = np.rint(values * 255.0).astype(np.uint8)
    return np.concatenate([lut, lut])


def build_phase_maps(
    orientations_deg: list[float], geom: Geometry
) -> list[np.ndarray]:
    """
    Precompute, per orientation, each pixel's spatial phase as a LUT index.

    Evaluating sin() over a 1920x1080 grid every frame costs ~20 ms, which is
    too slow to keep a 2 Hz drift smooth. The grating is a pure sinusoid along
    the rotated axis, and drifting only shifts its phase, so the expensive part
    (the rotated coordinate and its sine) can be computed once per orientation
    and reduced to a per-pixel index. Each frame then becomes a single lookup.

    Memory is 2 bytes/pixel/orientation - about 17 MB for four orientations at
    1920x1080, versus the ~250 MB a precomputed frame stack would need.
    """
    xp = (np.arange(geom.patch_w, dtype=np.float64) - (geom.patch_w - 1) / 2.0)[:, None]
    yp = (np.arange(geom.patch_h, dtype=np.float64) - (geom.patch_h - 1) / 2.0)[None, :]

    maps: list[np.ndarray] = []
    for deg in orientations_deg:
        theta = math.radians(deg)
        x_rot = xp * math.cos(theta) + yp * math.sin(theta)
        cycles = geom.spatial_freq_cpp * x_rot
        idx = np.rint((cycles % 1.0) * PHASE_STEPS).astype(np.int32) % PHASE_STEPS
        maps.append(np.ascontiguousarray(idx, dtype=np.uint16))
    return maps


def photodiode_rect(args: argparse.Namespace, geom: Geometry) -> Optional[tuple[int, int, int, int]]:
    """(x, y, w, h) of the photodiode square, or None when it is disabled."""
    if not args.photodiode:
        return None

    size = args.pd_box_size
    margin = args.pd_margin
    corner = args.pd_corner.lower()

    if corner == "topleft":
        return (margin, margin, size, size)
    if corner == "topright":
        return (geom.screen_w - margin - size, margin, size, size)
    if corner == "bottomleft":
        return (margin, geom.screen_h - margin - size, size, size)
    if corner == "bottomright":
        return (geom.screen_w - margin - size, geom.screen_h - margin - size, size, size)
    raise SystemExit(f"Unknown --pd-corner: {args.pd_corner}")


def photodiode_color(value: float, use_white: bool) -> tuple[int, int, int]:
    """
    Photodiode square colour for a given 0-1 level.

    White mode drives all three channels; otherwise only blue is driven and
    red/green stay at zero.
    """
    level = int(round(max(0.0, min(1.0, value)) * 255.0))
    return (level, level, level) if use_white else (0, 0, level)


# --------------------------------------------------------------------------
# Stimulus server
# --------------------------------------------------------------------------

@dataclass
class StimServer:
    args: argparse.Namespace
    display: Display
    geom: Geometry
    sock: socket.socket
    orientations: list[float]
    lut2: np.ndarray
    pd_rect: Optional[tuple[int, int, int, int]]
    events: list[dict] = field(default_factory=list)

    # Single-orientation mode takes its orientation from each STIM command, so
    # a map may be needed for an angle that was never configured. Cache those,
    # but cap the cache: each entry is ~4 MB at 1920x1080, and a sender that
    # sweeps many distinct angles would otherwise grow memory without bound.
    _MAX_CACHED_MAPS = 16

    def __post_init__(self) -> None:
        self._gray = np.empty((self.geom.patch_w, self.geom.patch_h), dtype=np.uint8)
        self._rng = np.random.default_rng()
        self._probe_reply = probe_reply(self.args, self.orientations)
        self._map_cache: dict[float, np.ndarray] = {
            deg: phase_map
            for deg, phase_map in zip(
                self.orientations, build_phase_maps(self.orientations, self.geom)
            )
        }

    def _phase_map(self, deg: float) -> np.ndarray:
        """Phase-index map for one orientation, building and caching on demand."""
        key = round(float(deg), 4)
        cached = self._map_cache.get(key)
        if cached is None:
            if len(self._map_cache) >= self._MAX_CACHED_MAPS:
                self._map_cache.pop(next(iter(self._map_cache)))
            cached = build_phase_maps([key], self.geom)[0]
            self._map_cache[key] = cached
        return cached

    # -- UDP ---------------------------------------------------------------

    def _read_command(self) -> Optional[list[str]]:
        """
        Non-blocking read of one datagram, split into tokens.

        Reads a single datagram per call. Returns None when nothing is
        waiting.
        """
        try:
            data, addr = self.sock.recvfrom(4096)
        except BlockingIOError:
            return None
        except ConnectionResetError:
            # Windows surfaces ICMP port-unreachable from an earlier reply as
            # an error on this socket. Nothing is wrong with our listener.
            return None
        except OSError:
            return None

        message = data.decode("utf-8", errors="replace").strip()
        if not message:
            return None

        parts = message.split()
        if parts[0].upper() in PROBE_COMMANDS:
            try:
                self.sock.sendto(self._probe_reply, addr)
            except OSError:
                pass
            return None

        return parts

    # -- Drawing -----------------------------------------------------------

    def _show_black(self) -> None:
        self.display.fill_black()
        if self.pd_rect is not None:
            self.display.draw_box(
                self.pd_rect, photodiode_color(self.args.pd_off_val, self.args.pd_use_white)
            )
        self.display.present()

    def _run_grating(
        self, sequence_deg: list[float], requested_duration_s: float
    ) -> tuple[float, str, int]:
        """
        Drive one stimulus epoch. Returns (elapsed_s, stop_reason, frame_count).

        The epoch is split into equal segments, one per orientation in
        sequence_deg, each --grating-duration-s long.
        """
        n_orient = len(sequence_deg)
        segment_s = self.args.grating_duration_s

        if self.args.respect_requested_duration:
            total_s = requested_duration_s
        else:
            # Default behaviour: the requested duration is discarded and the
            # epoch always runs one full segment per orientation.
            total_s = n_orient * segment_s

        segment_maps = [self._phase_map(deg) for deg in sequence_deg]
        blue_only = not self.args.gray_stim
        pd_on_color = photodiode_color(self.args.pd_on_val, self.args.pd_use_white)

        stop_reason = "DURATION_COMPLETE"
        frames = 0
        start = time.perf_counter()

        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= total_s:
                break

            if self.display.user_quit_requested():
                stop_reason = "QUIT"
                break

            parts = self._read_command()
            if parts is not None:
                command = parts[0].upper()
                if command == "BLACK":
                    stop_reason = "BLACK"
                    break
                if command == "QUIT":
                    stop_reason = "QUIT"
                    break
                warn(f"Ignoring command received during active stimulus: {' '.join(parts)}")

            # Clamp to the last orientation if the epoch outlives its segments
            # (only reachable with --respect-requested-duration).
            segment_index = min(int(elapsed // segment_s), n_orient - 1)
            elapsed_in_segment = elapsed - segment_index * segment_s

            # Temporal phase as a LUT rotation. The waveform we want is
            # sin(2*pi*(f*x_rot - tf*t)); the spatial part is baked into the
            # phase map, so subtracting the temporal part is a slice offset.
            step = int(round((self.args.temporal_freq * elapsed_in_segment % 1.0) * PHASE_STEPS))
            shift = (PHASE_STEPS - step % PHASE_STEPS) % PHASE_STEPS
            window = self.lut2[shift:shift + PHASE_STEPS]

            # mode="clip" skips per-element bounds checking. Nothing is ever
            # actually clipped: build_phase_maps() guarantees every index is in
            # [0, PHASE_STEPS) and the window is exactly PHASE_STEPS long.
            np.take(window, segment_maps[segment_index], out=self._gray, mode="clip")

            self.display.fill_black()
            self.display.draw_gray(self._gray, self.geom.x0, self.geom.y0, blue_only)
            if self.pd_rect is not None:
                self.display.draw_box(self.pd_rect, pd_on_color)
            self.display.present()
            frames += 1

        return time.perf_counter() - start, stop_reason, frames

    # -- Command handling --------------------------------------------------

    def _handle_stim(self, parts: list[str]) -> bool:
        """Handle one STIM command. Returns False if the server should stop."""
        if len(parts) < 3:
            warn(f"Malformed STIM command: {' '.join(parts)}")
            return True

        try:
            orientation_deg = float(parts[1])
            duration_s = float(parts[2])
        except ValueError:
            warn(f"Invalid STIM command values: {' '.join(parts)}")
            return True

        trial_index: Optional[int] = None
        if len(parts) >= 4:
            try:
                trial_index = int(float(parts[3]))
            except ValueError:
                trial_index = None

        if math.isnan(orientation_deg) or math.isnan(duration_s) or duration_s <= 0:
            warn(f"Invalid STIM command values: {' '.join(parts)}")
            return True

        if self.args.multi_orientation:
            sequence = list(self.orientations)
            if self.args.randomize_orientations:
                sequence = [sequence[i] for i in self._rng.permutation(len(sequence))]
            log(
                f"STIM trial={trial_index} orientations={sequence} "
                f"duration={duration_s:.3f} s"
            )
            orientation_for_log: object = sequence
        else:
            # Single-orientation mode honours the orientation from the wire.
            sequence = [orientation_deg]
            log(
                f"STIM trial={trial_index} orientation={orientation_deg:.1f} deg "
                f"duration={duration_s:.3f} s"
            )
            orientation_for_log = orientation_deg

        row: dict = {
            "command": "STIM",
            "trialIndex": trial_index,
            "orientationDeg": orientation_for_log,
            "durationRequestedS": duration_s,
            "hostClockStart": _now_iso(),
        }

        elapsed_s, stop_reason, frames = self._run_grating(sequence, duration_s)

        self._show_black()

        row["stopReason"] = stop_reason
        row["hostClockEnd"] = _now_iso()
        row["elapsedS"] = elapsed_s
        row["frameCount"] = frames
        row["meanFrameRateHz"] = (frames / elapsed_s) if elapsed_s > 0 else 0.0
        self.events.append(row)

        log(
            f"  ended after {elapsed_s:.3f} s ({stop_reason}), "
            f"{frames} frames at {row['meanFrameRateHz']:.1f} Hz"
        )

        return stop_reason != "QUIT"

    def run(self) -> None:
        """Main command loop. Returns when QUIT is received or the window closes."""
        self._show_black()

        while True:
            if self.display.user_quit_requested():
                log("QUIT (window closed / Escape pressed)")
                self.events.append({"command": "QUIT", "hostClockStart": _now_iso(),
                                    "stopReason": "USER_CLOSED"})
                return

            parts = self._read_command()
            if parts is None:
                time.sleep(_IDLE_POLL_S)
                continue

            command = parts[0].upper()

            if command == "STIM":
                if not self._handle_stim(parts):
                    return

            elif command == "BLACK":
                self._show_black()
                log("BLACK")
                self.events.append({"command": "BLACK", "hostClockStart": _now_iso()})

            elif command == "QUIT":
                self._show_black()
                log("QUIT")
                self.events.append({"command": "QUIT", "hostClockStart": _now_iso()})
                return

            else:
                warn(f"Unknown visual stimulus command: {' '.join(parts)}")


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def save_log(
    args: argparse.Namespace,
    geom: Geometry,
    display: Display,
    orientations: list[float],
    events: list[dict],
    started_at: str,
) -> Optional[Path]:
    """
    Write the session log as JSON.

    JSON keeps the log readable with no extra tooling and matches how the
    rest of this project serializes plain data.
    """
    if not args.save_log:
        return None

    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"visual_stim_log_{datetime.now():%Y-%m-%d_%H-%M-%S}.json"

    payload = {
        "server": "intrinsic_visual_stimulus.py",
        "backend": getattr(display, "name", "unknown"),
        "startedAt": started_at,
        "endedAt": _now_iso(),
        "udpPort": args.udp_port,
        "display": {
            "monitor": args.monitor,
            "screenWidthPx": geom.screen_w,
            "screenHeightPx": geom.screen_h,
            "patchWidthPx": geom.patch_w,
            "patchHeightPx": geom.patch_h,
            "fullscreen": args.fullscreen,
            "fullscreenStim": args.fullscreen_stim,
            "patchSizePx": list(args.patch_size_px) if args.patch_size_px else None,
            "patchSizeFrac": args.patch_size_frac,
        },
        "geometry": {
            "monitorWidthCm": args.monitor_width_cm,
            "monitorResPx": args.monitor_res_px,
            "viewDistCm": args.view_dist_cm,
            "pxPerDeg": geom.px_per_deg,
        },
        "stimulus": {
            "spatialFreqCpd": args.spatial_freq_cpd,
            "spatialFreqCpp": geom.spatial_freq_cpp,
            "temporalFreqHz": args.temporal_freq,
            "contrast": args.contrast,
            "gratingGain": args.grating_gain,
            "gratingOffset": args.grating_offset,
            "gratingDurationS": args.grating_duration_s,
            "useBlueStim": not args.gray_stim,
            "useMultiOrientation": args.multi_orientation,
            "orientationsPerStimDeg": orientations,
            "randomizeOrientationOrder": args.randomize_orientations,
            "respectRequestedDuration": args.respect_requested_duration,
            "phaseSteps": PHASE_STEPS,
        },
        "photodiode": {
            "enabled": bool(args.photodiode),
            "boxSize": args.pd_box_size,
            "margin": args.pd_margin,
            "corner": args.pd_corner,
            "onVal": args.pd_on_val,
            "offVal": args.pd_off_val,
            "useWhite": args.pd_use_white,
        },
        "events": events,
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_orientations(text: str) -> list[float]:
    values = [float(item) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--orientations needs at least one value")
    return values


def parse_patch_size(text: str) -> tuple[int, int]:
    parts = [item for item in text.replace("x", ",").split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--patch-size-px expects WIDTH,HEIGHT")
    return int(parts[0]), int(parts[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visual stimulus server for intrinsic optical imaging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--list-monitors", action="store_true",
                        help="Print detected monitors and exit.")
    parser.add_argument("--backend", choices=["pygame", "tk"], default="pygame",
                        help="Rendering backend. 'tk' is a slow fallback; see --help notes.")

    # Server / display
    parser.add_argument("--udp-port", type=int, default=55000,
                        help="UDP port to listen on. Must match the acquisition --stim-port.")
    parser.add_argument("--bind-host", type=str, default="0.0.0.0",
                        help="Interface to bind. 0.0.0.0 accepts commands from another host.")
    parser.add_argument("--monitor", type=int, default=2,
                        help="Stimulus monitor, 1-based.")
    parser.add_argument("--no-fullscreen", dest="fullscreen", action="store_false",
                        help="Run in a window instead of fullscreen (for desk testing).")

    # Monitor geometry, for px/deg
    parser.add_argument("--monitor-width-cm", type=float, default=52.0,
                        help="Physical width of the stimulus monitor.")
    parser.add_argument("--monitor-res-px", type=float, default=1920.0,
                        help="Horizontal resolution used for the px/deg calculation.")
    parser.add_argument("--view-dist-cm", type=float, default=20.0,
                        help="Eye-to-screen distance.")

    # Stimulus
    parser.add_argument("--spatial-freq-cpd", type=float, default=0.08,
                        help="Spatial frequency in cycles/degree.")
    parser.add_argument("--temporal-freq", type=float, default=2.0,
                        help="Drift rate in cycles/second.")
    parser.add_argument("--contrast", type=float, default=1.0,
                        help="Michelson contrast, 0-1.")
    parser.add_argument("--grating-gain", type=float, default=1.0,
                        help="Overall display scaling applied after the sinusoid.")
    parser.add_argument("--grating-offset", type=float, default=0.5,
                        help="Mean level of the sinusoid.")
    parser.add_argument("--grating-duration-s", type=float, default=1.25,
                        help="Seconds spent on each orientation within one STIM epoch.")
    parser.add_argument("--gray", dest="gray_stim", action="store_true",
                        help="Gray bars instead of the default blue.")

    # Orientation sequencing
    parser.add_argument("--orientations", type=parse_orientations, default="135,180,225,270",
                        help="Comma-separated orientations swept within each STIM epoch.")
    parser.add_argument("--single-orientation", dest="multi_orientation",
                        action="store_false",
                        help="Use the orientation carried by each STIM command instead of "
                             "sweeping the --orientations list.")
    parser.add_argument("--randomize-orientations", action="store_true",
                        help="Shuffle the orientation order on every STIM epoch.")
    parser.add_argument("--respect-requested-duration", action="store_true",
                        help="Use the duration from the STIM command instead of "
                             "n_orientations * --grating-duration-s. Leave it off to match "
                             "previously validated sessions.")

    # Patch size
    parser.add_argument("--patch-size-px", type=parse_patch_size, default=None,
                        help="Explicit patch size as WIDTH,HEIGHT. Overrides the options below.")
    parser.add_argument("--no-fullscreen-stim", dest="fullscreen_stim", action="store_false",
                        help="Draw a centered square patch instead of filling the screen.")
    parser.add_argument("--patch-size-frac", type=float, default=1.0,
                        help="Patch size as a fraction of the smaller screen dimension. "
                             "Used only with --no-fullscreen-stim.")

    # Photodiode
    parser.add_argument("--photodiode", action="store_true",
                        help="Draw a photodiode square that is bright during stimulation.")
    parser.add_argument("--pd-box-size", type=int, default=60, help="Photodiode square size in px.")
    parser.add_argument("--pd-margin", type=int, default=20, help="Photodiode inset from the edge.")
    parser.add_argument("--pd-corner", type=str, default="bottomright",
                        choices=["topleft", "topright", "bottomleft", "bottomright"],
                        help="Which corner holds the photodiode square.")
    parser.add_argument("--pd-on-val", type=float, default=1.0, help="Photodiode level during stimulus.")
    parser.add_argument("--pd-off-val", type=float, default=0.0, help="Photodiode level when black.")
    parser.add_argument("--pd-blue", dest="pd_use_white", action="store_false",
                        help="Drive the photodiode square in blue only instead of white.")

    # Logging
    parser.add_argument("--no-save-log", dest="save_log", action="store_false",
                        help="Do not write a stimulus log on exit.")
    parser.add_argument("--log-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "visual_stim_logs"),
                        help="Where stimulus logs are written.")

    return parser


def install_break_handler() -> None:
    """
    Route Windows CTRL_BREAK_EVENT into the normal KeyboardInterrupt path.

    The GUI stops a subprocess with CTRL_BREAK_EVENT (see gui/script_runner.py).
    Python maps CTRL_C_EVENT to KeyboardInterrupt on its own, but SIGBREAK
    defaults to SIG_DFL, which lets the OS terminate the process outright with
    STATUS_CONTROL_C_EXIT (0xC000013A) - no except, no finally, no cleanup.
    Measured on this machine: without this handler the stimulus log is lost and
    the display is never blanked; with it, KeyboardInterrupt is raised normally
    and the process exits 0.
    """
    handler = getattr(signal, "SIGBREAK", None)
    if handler is not None:
        signal.signal(handler, signal.default_int_handler)


def main() -> int:
    install_break_handler()

    parser = build_parser()
    args = parser.parse_args()

    if args.list_monitors:
        print("Detected monitors:")
        print(describe_monitors())
        return 0

    # argparse applies the type converter to command-line strings but not to a
    # string default, so normalize here.
    if isinstance(args.orientations, str):
        args.orientations = parse_orientations(args.orientations)

    if args.grating_duration_s <= 0:
        parser.error("--grating-duration-s must be > 0")
    if args.monitor < 1:
        parser.error("--monitor is 1-based; use 1 for the primary monitor")

    started_at = _now_iso()

    try:
        display = make_display(args.backend, args.monitor - 1, args.fullscreen)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Could not open the stimulus window: {exc}", file=sys.stderr)
        print("Detected monitors:\n" + describe_monitors(), file=sys.stderr)
        return 1

    geom = compute_geometry(args, display.width, display.height)
    orientations = list(args.orientations)
    lut2 = build_sine_lut(args.contrast, args.grating_gain, args.grating_offset)
    pd_rect = photodiode_rect(args, geom)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    try:
        sock.bind((args.bind_host, args.udp_port))
    except OSError as exc:
        display.close()
        print(
            f"Could not listen on UDP port {args.udp_port}: {exc}\n"
            "Another stimulus server may "
            "already be running. Close it and try again.",
            file=sys.stderr,
        )
        return 1

    print()
    print("Visual stimulus server is ready.")
    print(f"Listening on UDP port {args.udp_port} ({args.bind_host}).")
    print(
        f"Stimulus monitor: {args.monitor} | Resolution: {geom.screen_w}x{geom.screen_h} px "
        f"| Patch: {geom.patch_w}x{geom.patch_h} centered | Backend: {getattr(display, 'name', '?')}"
    )
    print(
        f"{geom.px_per_deg:.2f} px/deg | {args.spatial_freq_cpd} cpd "
        f"({geom.spatial_freq_cpp:.6f} cycles/px) | {args.temporal_freq} Hz drift"
    )
    if args.multi_orientation:
        epoch_s = len(orientations) * args.grating_duration_s
        print(
            f"Multi-orientation: {orientations} x {args.grating_duration_s} s "
            f"= {epoch_s:.2f} s per STIM epoch"
            + ("" if not args.respect_requested_duration
               else "  (overridden by --respect-requested-duration)")
        )
    print("Commands: STIM <orientation_deg> <duration_s> [trial_index], BLACK, QUIT")
    print("Press Escape or close the window to stop.")
    print(flush=True)

    server = StimServer(
        args=args,
        display=display,
        geom=geom,
        sock=sock,
        orientations=orientations,
        lut2=lut2,
        pd_rect=pd_rect,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        # Ctrl+C, or the CTRL_BREAK_EVENT the GUI sends for a graceful stop.
        log("Interrupted; blanking the display.")
        server.events.append({"command": "QUIT", "hostClockStart": _now_iso(),
                              "stopReason": "KEYBOARD_INTERRUPT"})
    finally:
        try:
            server._show_black()
        except Exception:
            pass

        saved = None
        try:
            saved = save_log(args, geom, display, orientations, server.events, started_at)
        except Exception as exc:
            print(f"Warning: could not save the stimulus log: {exc}", file=sys.stderr)

        sock.close()
        display.close()

        if saved is not None:
            print(f"Stimulus log saved to:\n  {saved}", flush=True)
        print("Visual stimulus server closed.", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
