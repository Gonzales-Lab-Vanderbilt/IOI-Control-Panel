#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Blackfly external-trigger acquisition script with Arduino handshake,
count-closed phase transitions, graceful post-trial timeout,
support for a short green pre-baseline block, and optional software analysis ROI cropping.

Expected Arduino serial messages for the session-green protocol:
    ARDUINO_READY
    SESSION_GREEN_REFERENCE_START
    SESSION_GREEN_REFERENCE_END
    RED_STABILIZATION_START or RED_ON
    TRIAL_START
    RED_BASELINE_START
    GAP_START
    STIM_START
    POST_START
    STIM_END
    TRIAL_END or CYCLE_COMPLETE

Expected commands from Python to Arduino:
    SESSION_GREEN_REFERENCE
    SESSION_RED_ON
    START_TRIAL
    LIGHTS_OFF

With 2x2 software analysis binning, dark-frame subtraction, and -ΔR/R analysis while preserving full-resolution RAW acquisition:
py -3.10 intrinsic_imaging.py --output .\captures --port COM4 --trials 2 --binning 2 --save-format raw --analyze --dark-reference .\dark_refs\dark_reference_mean.npy --analysis-method raw_counts

Or, following calibration (intrinsic_calibration.py; ROI is analysis-only, not hardware capture):
py -3.10 intrinsic_imaging.py --output .\captures --port COM4 --trials 10 --analyze --open-trial-overlays --visual-stim --stim-duration-s 7.5 --analysis-method raw_counts
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import signal
import sys
import threading
import time
import os
import socket
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

try:
    import PySpin  # type: ignore
except ImportError as exc:
    raise SystemExit("PySpin is not installed or not visible to this Python environment.") from exc

try:
    import serial  # type: ignore
except ImportError as exc:
    raise SystemExit("pyserial is not installed. Install it with: py -3.10 -m pip install pyserial") from exc


# Analysis-method identifiers. "fractional" is accepted as a shorthand for
# "fractional_reflectance". Used as argparse `type=`, which runs before
# `choices=`, so --help advertises only the canonical names.
_ANALYSIS_METHOD_ALIASES = {
    "fractional": "fractional_reflectance",
}


def normalize_analysis_method(value: str) -> str:
    key = str(value).strip().lower()
    return _ANALYSIS_METHOD_ALIASES.get(key, key)


def _session_map_candidates(analysis_dir: Path, method: str) -> list[Path]:
    """Per-trial activation maps the session pass will average, best first.

    These basenames are a contract with the writer in analyze_trial() -- change
    them in one place only and every trial is skipped for "missing analysis
    arrays", which aborts the session pass (see the `if not trial_maps` guard
    in _run_session_analysis).
    """
    if method == "raw_counts":
        names = ["activation_map_raw_counts_smoothed.npy"]
    else:
        names = [
            "activation_map_fractional_reflectance_smoothed.npy",
            "activation_map_masked_fractional_reflectance.npy",
        ]
    return [analysis_dir / n for n in names]


@dataclass
class TrialConfig:
    num_trials: int = 1
    inter_trial_s: float = 0.0
    exposure_us: float = 23000.0
    green_exposure_us: Optional[float] = None
    gain_db: Optional[float] = None
    pixel_format: str = "Mono16"
    binning_horizontal: int = 1
    binning_vertical: int = 1
    trigger_line: str = "Line0"
    trigger_activation: str = "RisingEdge"
    save_format: str = "raw"
    overwrite: bool = False
    serial_port: str = ""
    serial_baud: int = 115200
    serial_timeout_s: float = 0.1
    marker_timeout_s: float = 30.0
    flush_input_on_start: bool = True
    save_gap_frames: bool = True
    green_frames: int = 30
    green_reference_trim_frames: int = 5
    baseline_frames: int = 40
    post_frames: int = 40
    trailing_timeout_s: float = 2.0
    post_idle_timeout_s: float = 2.0
    run_analysis: bool = False
    analysis_start_frame: int = 5
    analysis_end_frame: Optional[int] = 35
    analysis_smoothing_sigma: float = 5.0
    analysis_binning: int = 1
    analysis_mask_percentile: float = 20.0
    analysis_denominator_floor_percentile: float = 5.0
    analysis_denominator_floor_counts: float = 100.0
    analysis_median_filter_size: int = 3
    roi_config: Optional[Path] = None
    dark_reference_path: Optional[Path] = None
    save_analysis_arrays: bool = True
    analysis_method: str = "raw_counts"
    run_session_analysis: bool = True
    rescale_bit_depth: int = 16
    rescale_mode: str = "signed_symmetric"
    invert_display_signal: bool = False
    analyze: bool = False
    open_overlay: bool = False
    open_trial_overlays: bool = False
    open_session_overlay: bool = False
    visual_stim: bool = False
    # Per-trial stim/catch sequence, one bool per trial (True = fire stim,
    # False = identical-timing catch trial), length == num_trials. None (the
    # default) reproduces today's uniform behavior: every trial fires stim
    # whenever the configured modality (visual_stim / an electrical or LRA
    # trial_start_cmd) says to.
    trial_conditions: Optional[list[bool]] = None
    stim_host: str = "127.0.0.1"
    stim_port: int = 55000
    stim_orientations: str = "45"
    stim_duration_s: float = 7.0
    session_green_cmd: str = "SESSION_GREEN_REFERENCE"
    red_on_cmd: str = "SESSION_RED_ON"
    trial_start_cmd: str = "START_TRIAL"
    lights_off_cmd: str = "LIGHTS_OFF"
    red_stabilization_s: float = 600.0
    turn_lights_off_on_exit: bool = True
    precaptured_green_reference: bool = False
    red_settling_sample_interval_s: float = 0.0
    # Nominal camera trigger period. Must match the Arduino's triggerPeriodMs.
    # Used to convert camera-timestamp gaps into a count of missed triggers.
    trigger_period_ms: float = 100.0
    # Ticks per second in image.GetTimeStamp(). Spinnaker reports nanoseconds on
    # the BFLY-U3-23S6M-C, so 1e9. A wrong value here is caught by the startup
    # sanity check in _update_trigger_index(), which disables drop inference
    # rather than fabricating drop counts.
    camera_timestamp_hz: float = 1e9
    # Depth at which the disk-writer backlog is worth complaining about. The
    # queue is bounded well above this; crossing it means disk is the bottleneck.
    writer_queue_warn_depth: int = 50
    stream_buffer_count: int = 64


class CameraConfigError(RuntimeError):
    pass


class BlackflyCapture:
    # A GUI (or any external controller) drops a file at this path, inside the
    # session folder, to end the red-stabilization wait early. See
    # _wait_with_marker_polling(skip_signal_path=...).
    SKIP_RED_STABILIZATION_FILENAME = "skip_red_stabilization.flag"

    VALID_MARKERS = {
        "ARDUINO_READY",
        "TRIAL_START",
        "GREEN_BASELINE_START",
        "RED_BASELINE_START",
        "GAP_START",
        "POST_START",
        "POST_TRAILING_START",
        "TRIAL_END",
        "STIM_START",
        "STIM_END",
        "CYCLE_COMPLETE",
        "SESSION_GREEN_REFERENCE_START",
        "SESSION_GREEN_REFERENCE_END",
        "RED_STABILIZATION_START",
        "RED_STABILIZATION_END",
        "RED_ON",
        "LIGHTS_OFF",
    }

    # recognized informational markers: into marker_log.csv, but never treated as phase drivers and never written into last_marker / markers_seen.
    INFO_MARKERS = {
        "SESSION_GREEN_FIRST_FRAME_TRIGGER",
        "SESSION_GREEN_LAST_FRAME_TRIGGER",
        "BASELINE_FIRST_FRAME_TRIGGER",
        "BASELINE_LAST_FRAME_TRIGGER",
        "POST_FIRST_FRAME_TRIGGER",
        "POST_LAST_FRAME_TRIGGER",
        "BUSY_TRIAL_RUNNING",
        "RED_ON_AUTO",
        "CAL_TRIGGER",
        "CAL_GREEN_ON",
        "CAL_RED_ON",
        "UNKNOWN_COMMAND",
    }

    def __init__(
        self,
        output_dir: Path,
        cfg: TrialConfig,
        external_serial: Optional["serial.Serial"] = None,
        session_dir_override: Optional[Path] = None,
    ) -> None:
        self.output_dir = output_dir
        self.cfg = cfg
        self.system = None
        self.cam_list = None
        self.cam = None
        self.ser = None
        self._external_serial = external_serial
        self._owns_serial = external_serial is None
        if session_dir_override is not None:
            # Reuse a session folder created by an earlier stage (e.g. a
            # standalone --green-reference-only run) instead of starting a
            # new timestamped one, so a staged workflow can span several
            # separate process invocations while writing into one session.
            self.session_dir = Path(session_dir_override)
            self.session_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.session_dir = self._make_session_dir(output_dir, cfg.overwrite)
        self.log_csv_path = self.session_dir / "session_log.csv"
        self.marker_log_csv_path = self.session_dir / "marker_log.csv"
        self._csv_fp = None
        self._csv_writer = None
        self._marker_csv_fp = None
        self._marker_csv_writer = None
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self.active_roi: Optional[dict] = None
        self.stim_sock: Optional[socket.socket] = None
        self._lights_off_sent = False
        # Instance state (not a run() local) so a standalone green-reference-
        # only capture and a later precaptured-green-reference run both keep
        # correct, continuous frame numbering in session_log.csv.
        self._global_frame_count = 0

        # Non-blocking serial line assembly. _read_serial_marker() must never
        # stall the acquisition loop waiting on a port timeout, so it drains
        # whatever bytes are already buffered and keeps the partial tail here.
        self._serial_rx_buf = bytearray()

        # Latest arduino_millis / trigger index seen for each marker name, reset
        # per trial. Phase labels are derived from the trigger indices in here
        # instead of from whatever phase the consumer happened to be in when it
        # dequeued a frame.
        self._marker_arduino_millis: dict[str, int] = {}
        self._marker_trigger_index: dict[str, int] = {}

        # Background disk writer. Frames are copied out of the PySpin buffer and
        # handed off so the acquisition loop never blocks on file I/O.
        self._writer_queue: Optional["queue.Queue"] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_error: Optional[BaseException] = None
        self._writer_queue_peak = 0
        self._writer_warned = False
        self._writer_enqueued = 0
        self._writer_completed = 0

        # Camera-timestamp scale validation (see _update_trigger_index).
        self._ts_scale_samples: list[float] = []
        self._ts_scale_checked = False
        self._ts_scale_ok = True

    @staticmethod
    def _make_session_dir(base: Path, overwrite: bool) -> Path:
        timestamp = time.strftime("session_%Y%m%d_%H%M%S")
        session_dir = base / timestamp
        if session_dir.exists() and not overwrite:
            raise FileExistsError(f"Output directory already exists: {session_dir}")
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def setup(self) -> None:
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        if self.cam_list.GetSize() < 1:
            self.teardown()
            raise RuntimeError("No Blackfly/Spinnaker camera detected.")

        self.cam = self.cam_list.GetByIndex(0)
        self.cam.Init()
        self._configure_camera()
        self.frame_width = self._get_int_value("Width")
        self.frame_height = self._get_int_value("Height")
        self._print_camera_settings()
        self._open_serial()
        self._open_csv_log()
        self._open_marker_log()
        self._open_visual_stim_socket()

    def teardown(self) -> None:
        # Get queued frames onto disk before anything else is torn down.
        self._stop_frame_writer()

        try:
            if self.cam is not None:
                try:
                    self.cam.EndAcquisition()
                except Exception:
                    pass
                try:
                    self.cam.DeInit()
                except Exception:
                    pass
        finally:
            self.cam = None

        if self.cam_list is not None:
            self.cam_list.Clear()
            self.cam_list = None

        if self.system is not None:
            self.system.ReleaseInstance()
            self.system = None

        if self.ser is not None:
            self._send_lights_off_if_needed()
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

        if self.stim_sock is not None:
            try:
                self._send_visual_stim_command("BLACK")
            except Exception:
                pass
            try:
                self.stim_sock.close()
            except Exception:
                pass
            self.stim_sock = None

        if self._csv_fp is not None:
            try:
                self._csv_fp.flush()
            except Exception:
                pass
            self._csv_fp.close()
            self._csv_fp = None

        if self._marker_csv_fp is not None:
            self._marker_csv_fp.close()
            self._marker_csv_fp = None

    def _open_visual_stim_socket(self) -> None:
        """Open a UDP socket used to command the visual stimulus server."""
        if not self.cfg.visual_stim:
            return

        self.stim_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.stim_sock.settimeout(0.2)
        print(
            "Visual stimulus UDP enabled: "
            f"{self.cfg.stim_host}:{self.cfg.stim_port}; "
            f"orientations={self._stim_orientation_list()}; "
            f"duration={self.cfg.stim_duration_s:.3f} s"
        )
        # Force the stimulus display to black before acquisition begins.
        self._send_visual_stim_command("BLACK")

    def _stim_orientation_list(self) -> list[float]:
        """Parse comma-separated stimulus orientations from the command line."""
        values: list[float] = []
        for item in str(self.cfg.stim_orientations).split(","):
            item = item.strip()
            if not item:
                continue
            values.append(float(item))
        if not values:
            values = [45.0]
        return values

    def _stim_orientation_for_trial(self, trial_index: int) -> float:
        """Return the orientation assigned to this trial, cycling if needed."""
        values = self._stim_orientation_list()
        return values[(int(trial_index) - 1) % len(values)]

    def _send_visual_stim_command(
        self,
        command: str,
        orientation_deg: Optional[float] = None,
        duration_s: Optional[float] = None,
        trial_index: Optional[int] = None,
    ) -> None:
        """Send one UDP command to the visual stimulus server.

        Commands understood by the stimulus server:
            BLACK
            STIM <orientation_deg> <duration_s> <trial_index>
            QUIT
        """
        if not self.cfg.visual_stim or self.stim_sock is None:
            return

        command = command.upper().strip()
        if command == "STIM":
            if orientation_deg is None:
                orientation_deg = 45.0
            if duration_s is None:
                duration_s = self.cfg.stim_duration_s
            if trial_index is None:
                message = f"STIM {float(orientation_deg):.6g} {float(duration_s):.6g}"
            else:
                message = f"STIM {float(orientation_deg):.6g} {float(duration_s):.6g} {int(trial_index)}"
        else:
            message = command

        self.stim_sock.sendto(
            message.encode("utf-8"),
            (self.cfg.stim_host, int(self.cfg.stim_port)),
        )
        print(f"[visual_stim] sent: {message}")

    def _send_arduino_command(self, command: str) -> None:
        """Send one line command to the Arduino, if the serial port is open."""
        if self.ser is None:
            return
        message = command.strip()
        if not message:
            return
        self.ser.write((message + "\n").encode("utf-8"))
        self.ser.flush()
        print(f"[arduino_command] sent: {message}")

    def _send_lights_off_if_needed(self) -> None:
        """Ask the Arduino to turn all illumination off once, usually during cleanup."""
        if not self.cfg.turn_lights_off_on_exit or self._lights_off_sent:
            return
        if self.ser is None:
            return
        try:
            if self.cfg.visual_stim:
                self._send_visual_stim_command("BLACK")
            self._send_arduino_command(self.cfg.lights_off_cmd)
            self._lights_off_sent = True
        except Exception as exc:
            print(f"Warning: could not send lights-off cleanup command: {exc}")

    def _sample_red_settling(self, timeout_ms: int = 500) -> Optional[float]:
        """Fire one CAL_TRIGGER pulse, grab the resulting frame, and return its
        mean brightness within the active analysis ROI (full frame if none is
        set). CAL_TRIGGER is safe to send here: the Arduino only honors it
        while no timed trial is running, which is always true during the
        red-stabilization wait. Returns None on any capture hiccup — a missed
        sample should never interrupt the wait itself."""
        try:
            self._send_arduino_command("CAL_TRIGGER")
            image = self.cam.GetNextImage(timeout_ms)
        except PySpin.SpinnakerException:
            return None
        try:
            if image.IsIncomplete():
                return None
            arr = image.GetNDArray().copy()
        finally:
            try:
                image.Release()
            except Exception:
                pass
        cropped = self._crop_stack_to_analysis_roi(arr)
        self._save_settling_sample_image(cropped)
        return float(np.mean(cropped))

    def _save_settling_sample_image(self, cropped: "np.ndarray") -> None:
        """Quick-look percentile-scaled PNG of the most recent red-settling
        sample frame (the same frame _sample_red_settling just measured),
        overwritten in place each sample so a GUI polling this path always
        sees the latest one. Written to a temp file and renamed into place
        (atomic on the same filesystem) so a concurrent reader never sees a
        half-written PNG. Best-effort: a failure here must never interrupt
        the red-stabilization wait."""
        try:
            from PIL import Image
        except ImportError as exc:
            print(f"Settling sample image skipped: missing dependency: {exc}")
            return
        try:
            lo = float(np.percentile(cropped, 1.0))
            hi = float(np.percentile(cropped, 99.0))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.min(cropped)), float(np.max(cropped))
            if hi <= lo:
                hi = lo + 1.0
            scaled = np.clip((cropped.astype(np.float64) - lo) / (hi - lo), 0, 1)
            img = Image.fromarray((scaled * 255).astype(np.uint8))
            final_path = self.session_dir / "settling_sample.png"
            tmp_path = final_path.with_suffix(".png.tmp")
            # format= explicit: PIL infers format from the path suffix by
            # default, and the .tmp extension isn't a format it recognizes.
            img.save(tmp_path, format="PNG")
            tmp_path.replace(final_path)
        except Exception as exc:
            print(f"Warning: could not save settling sample image: {exc}")

    def _wait_with_marker_polling(
        self,
        duration_s: float,
        trial_index: int = 0,
        label: str = "Red trials start in",
        skip_signal_path: Optional[Path] = None,
        settling_sample_interval_s: float = 0.0,
    ) -> None:
        """Wait while logging serial markers, draining frames, and showing a countdown.

        If skip_signal_path is given and that file appears on disk mid-wait
        (a GUI writes it in response to a user "skip" action), the wait ends
        early and the file is consumed. Only red stabilization opts into this;
        other callers pass no path and behave exactly as before.

        If settling_sample_interval_s > 0, periodically (that often) triggers
        one camera frame and prints its mean brightness within the analysis
        ROI, plus its percent change relative to the first sample of this
        wait, so a caller can plot LED settling over time. Only red
        stabilization opts into this; other callers pass 0 (default) and
        behave exactly as before.
        """
        total_s = max(0.0, float(duration_s))
        deadline = time.time() + total_s
        start_t = time.time()
        markers_seen: list[str] = []
        last_remaining: Optional[int] = None
        last_sample_t = start_t - float(settling_sample_interval_s)
        first_sample_mean: Optional[float] = None

        if total_s <= 0:
            print(f"{label}: 00:00 remaining.")
            return

        while True:
            now = time.time()
            remaining_s = max(0, int(deadline - now + 0.999))

            if remaining_s != last_remaining:
                minutes, seconds = divmod(remaining_s, 60)
                print(
                    f"\r{label}: {minutes:02d}:{seconds:02d} remaining ",
                    end="",
                    flush=True,
                )
                last_remaining = remaining_s

            if now >= deadline:
                break

            if skip_signal_path is not None and skip_signal_path.exists():
                try:
                    skip_signal_path.unlink()
                except OSError:
                    pass
                print(f"\r{label}: skipped by user request.        ")
                return

            if settling_sample_interval_s > 0 and (now - last_sample_t) >= settling_sample_interval_s:
                last_sample_t = now
                mean_val = self._sample_red_settling()
                if mean_val is not None:
                    if first_sample_mean is None:
                        first_sample_mean = mean_val
                    rel_pct = (
                        0.0 if not first_sample_mean
                        else (mean_val - first_sample_mean) / first_sample_mean * 100.0
                    )
                    print(
                        f"Red settling sample: t={now - start_t:.1f}s "
                        f"mean={mean_val:.2f} rel={rel_pct:+.2f}%"
                    )

            self._poll_all_markers(markers_seen, trial_index=trial_index)

            # No frames are expected during red stabilization, but if the Arduino
            # accidentally emits camera triggers, drain them so the acquisition
            # buffer does not contaminate the first baseline frames.
            image = None
            try:
                image = self.cam.GetNextImage(20)
            except PySpin.SpinnakerException:
                image = None
            finally:
                if image is not None:
                    try:
                        image.Release()
                    except Exception:
                        pass

            time.sleep(0.01)

        print(f"\r{label}: 00:00 remaining.        ")

    def _wait_minimum_inter_trial_interval(
            self,
            trial_index: int,
            interval_start_wall: float,
    ) -> None:
        """Wait only long enough to enforce a minimum post-trial interval.
        
        The interval starts when the previous trial finishes acquisition/drain.
        Trial analysis and session analysis count toward the interval.
        
        Example with --iti 45:
            processing took 12 s -> wait 33 s
            processing took 45 s -> wait 0 s
            processing took 70 s -> wait 0 s
        """
        min_interval_s = max(0.0, float(self.cfg.inter_trial_s))
        if min_interval_s <= 0:
            return
        
        elapsed_s = max(0.0, time.time() - float(interval_start_wall))
        remaining_s = max(0.0, min_interval_s - elapsed_s)

        print(
            f"Inter-trial interval target: {min_interval_s:.1f} s, "
            f"processing elapsed: {elapsed_s:.1f} s, "
            f"waiting: {remaining_s:.1f} s."
        )

        if remaining_s > 0:
            self._wait_with_marker_polling(
                remaining_s,
                trial_index=trial_index,
                label="Next trial starts in",
            )
        else:
            print("Processing time exceeded the minimum inter-trial interval; starting next trial immediately.")

    def _open_serial(self) -> None:
        if self._external_serial is not None:
            self.ser = self._external_serial
            self._owns_serial = False
            print("Reusing the calibration's existing Arduino serial connection (no reopen, no reset risk).")
            if self.cfg.flush_input_on_start:
                self.ser.reset_input_buffer()
            return

        # timeout=0 (non-blocking): _read_serial_marker() reads only bytes that
        # are already buffered and reassembles lines itself, and it runs once per
        # frame in the acquisition loop. A blocking timeout here would put a wait
        # of up to serial_timeout_s directly into the frame budget.
        self.ser = serial.Serial(
            port=self.cfg.serial_port,
            baudrate=self.cfg.serial_baud,
            timeout=0,
        )
        self._owns_serial = True
        time.sleep(2.0)
        if self.cfg.flush_input_on_start:
            self.ser.reset_input_buffer()
        print("Waiting for ARDUINO_READY...")
        try:
            self._wait_for_marker({"ARDUINO_READY"}, 5.0, trial_index=0)
            print("Arduino is ready.")
        except TimeoutError:
            print("No ARDUINO_READY within 5 seconds; expected with the auto-reset cap installed; proceeding.")

    def _open_csv_log(self) -> None:
        # Append (and skip the header) when reusing a session_dir from an
        # earlier stage (e.g. a standalone --green-reference-only run), so
        # that stage's rows aren't clobbered by this process's own log.
        file_exists = self.log_csv_path.exists()
        if file_exists:
            # A session folder written by an older build has an 8-column header.
            # Appending 10-column rows to it would produce a file no reader can
            # parse consistently, so say so loudly rather than corrupting it.
            try:
                with open(self.log_csv_path, "r", newline="", encoding="utf-8") as fp:
                    existing_header = next(csv.reader(fp), [])
                if existing_header and "trigger_index" not in existing_header:
                    print(
                        f"Warning: {self.log_csv_path} was written by an older version "
                        f"without the trigger_index/dropped_before columns. New rows will "
                        f"have 10 fields, existing rows have {len(existing_header)}."
                    )
            except OSError:
                pass

        self._csv_fp = open(self.log_csv_path, "a" if file_exists else "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_fp)
        if not file_exists:
            self._csv_writer.writerow([
                "trial_index", "phase", "frame_index_in_phase", "global_frame_count",
                "host_timestamp_iso", "camera_timestamp", "filename", "last_serial_marker",
                "trigger_index", "dropped_before"
            ])
            self._csv_fp.flush()

    def _open_marker_log(self) -> None:
        file_exists = self.marker_log_csv_path.exists()
        self._marker_csv_fp = open(self.marker_log_csv_path, "a" if file_exists else "w", newline="", encoding="utf-8")
        self._marker_csv_writer = csv.writer(self._marker_csv_fp)
        if not file_exists:
            self._marker_csv_writer.writerow([
                "trial_index",
                "host_timestamp_iso",
                "marker",
                "arduino_millis",
                "raw_text",
            ])
            self._marker_csv_fp.flush()

    def _get_node(self, name: str):
        node = self.cam.GetNodeMap().GetNode(name)
        if node is None or not PySpin.IsAvailable(node):
            raise CameraConfigError(f"Node not available: {name}")
        return node

    def _set_enum(self, name: str, entry_name: str) -> None:
        node = PySpin.CEnumerationPtr(self._get_node(name))
        if not PySpin.IsWritable(node):
            raise CameraConfigError(f"Enum node not writable: {name}")
        entry = node.GetEntryByName(entry_name)
        if entry is None or not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
            raise CameraConfigError(f"Enum entry {entry_name} not available for {name}")
        node.SetIntValue(entry.GetValue())

    def _set_float(self, name: str, value: float) -> None:
        node = PySpin.CFloatPtr(self._get_node(name))
        if not PySpin.IsWritable(node):
            raise CameraConfigError(f"Float node not writable: {name}")
        node.SetValue(max(node.GetMin(), min(node.GetMax(), value)))

    def _get_int_limits(self, name: str) -> tuple[int, int, int]:
        node = PySpin.CIntegerPtr(self._get_node(name))
        if not PySpin.IsReadable(node):
            raise CameraConfigError(f"Integer node not readable: {name}")
        inc = int(node.GetInc()) if hasattr(node, "GetInc") else 1
        return int(node.GetMin()), int(node.GetMax()), max(1, inc)

    def _set_int(self, name: str, value: int) -> int:
        node = PySpin.CIntegerPtr(self._get_node(name))
        if not PySpin.IsWritable(node):
            raise CameraConfigError(f"Integer node not writable: {name}")
        min_value = int(node.GetMin())
        max_value = int(node.GetMax())
        inc = int(node.GetInc()) if hasattr(node, "GetInc") else 1
        inc = max(1, inc)
        clipped = max(min_value, min(max_value, int(value)))
        aligned = min_value + ((clipped - min_value) // inc) * inc
        aligned = max(min_value, min(max_value, aligned))
        node.SetValue(aligned)
        return int(node.GetValue())

    def _get_stream_node(self, name: str):
        """Fetch a node from the transport-layer stream nodemap.

        Buffer-pool settings live on the TLStream nodemap, not the device
        nodemap that _get_node()/_set_enum()/_set_int() use, so they need their
        own accessors.
        """
        node = self.cam.GetTLStreamNodeMap().GetNode(name)
        if node is None or not PySpin.IsAvailable(node):
            raise CameraConfigError(f"Stream node not available: {name}")
        return node

    def _set_stream_enum(self, name: str, entry_name: str) -> None:
        node = PySpin.CEnumerationPtr(self._get_stream_node(name))
        if not PySpin.IsWritable(node):
            raise CameraConfigError(f"Stream enum node not writable: {name}")
        entry = node.GetEntryByName(entry_name)
        if entry is None or not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
            raise CameraConfigError(f"Stream enum entry {entry_name} not available for {name}")
        node.SetIntValue(entry.GetValue())

    def _set_stream_int(self, name: str, value: int) -> int:
        node = PySpin.CIntegerPtr(self._get_stream_node(name))
        if not PySpin.IsWritable(node):
            raise CameraConfigError(f"Stream integer node not writable: {name}")
        min_value = int(node.GetMin())
        max_value = int(node.GetMax())
        inc = int(node.GetInc()) if hasattr(node, "GetInc") else 1
        inc = max(1, inc)
        clipped = max(min_value, min(max_value, int(value)))
        aligned = min_value + ((clipped - min_value) // inc) * inc
        aligned = max(min_value, min(max_value, aligned))
        node.SetValue(aligned)
        return int(node.GetValue())

    def _configure_stream_buffers(self) -> None:
        """Enlarge the acquisition buffer pool and make overflow explicit.

        The Spinnaker default is ~11 buffers. At 10 Hz that is barely a second
        of slack: any consumer hiccup fills the pool and the camera starts
        overwriting frames that were never delivered, which shows up downstream
        as silently mistimed data rather than as an error.

        OldestFirst keeps delivery in capture order and lets a genuine overflow
        surface as a camera-timestamp gap (which _update_trigger_index turns
        into a dropped_before count) instead of silently reordering frames.

        Raises CameraConfigError if the nodes are missing or read-only -- a
        silently unconfigured pool is the exact failure this fixes.
        """
        self._set_stream_enum("StreamBufferCountMode", "Manual")
        applied = self._set_stream_int("StreamBufferCountManual", int(self.cfg.stream_buffer_count))
        self._set_stream_enum("StreamBufferHandlingMode", "OldestFirst")

        print(
            f"Stream buffers: mode=Manual count={applied} "
            f"(requested {int(self.cfg.stream_buffer_count)}) handling=OldestFirst"
        )
        if applied < int(self.cfg.stream_buffer_count):
            print(
                f"Warning: camera capped the buffer pool at {applied} buffers; "
                f"requested {int(self.cfg.stream_buffer_count)}."
            )

    def _try_set_enum(self, name: str, entry_name: str) -> bool:
        """Best-effort enum setter for optional camera features."""
        try:
            node = PySpin.CEnumerationPtr(self._get_node(name))
            if not PySpin.IsWritable(node):
                return False
            entry = node.GetEntryByName(entry_name)
            if entry is None or not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
                return False
            node.SetIntValue(entry.GetValue())
            return True
        except Exception:
            return False

    def _configure_binning(self) -> None:
        """Configure camera-side pixel binning when supported by the Blackfly.

        Use --binning 2 for 2x2 binning. The camera's Width/Height limits usually
        change after binning is applied, so this must run before full-frame ROI setup.
        """
        h_req = max(1, int(self.cfg.binning_horizontal))
        v_req = max(1, int(self.cfg.binning_vertical))

        # Some GenICam cameras expose BinningSelector. It is optional, so do not
        # fail if absent; the direct horizontal/vertical nodes are the important part.
        self._try_set_enum("BinningSelector", "All")

        applied_h = 1
        applied_v = 1

        for node_name, requested in (
            ("BinningHorizontal", h_req),
            ("BinningVertical", v_req),
        ):
            try:
                applied = self._set_int(node_name, requested)
            except CameraConfigError as exc:
                if requested == 1:
                    print(f"Binning node unavailable; continuing without {node_name}.")
                    applied = 1
                else:
                    raise CameraConfigError(
                        f"Requested {node_name}={requested}, but this camera/setting "
                        f"does not expose writable hardware binning. Try --binning 1 "
                        f"or use software binning during analysis. Original error: {exc}"
                    ) from exc

            if applied != requested:
                print(
                    f"Warning: requested {node_name}={requested}, "
                    f"camera applied {applied}."
                )

            if node_name == "BinningHorizontal":
                applied_h = applied
            else:
                applied_v = applied

        print(f"Camera binning: horizontal={applied_h}, vertical={applied_v}")

    def _load_roi_config(self) -> Optional[dict]:
        if self.cfg.roi_config is None:
            return None

        roi_path = Path(self.cfg.roi_config)
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI config not found: {roi_path}")

        with open(roi_path, "r", encoding="utf-8") as fp:
            roi = json.load(fp)

        if not bool(roi.get("enabled", True)):
            print(f"ROI config {roi_path} is disabled; using full camera frame.")
            return None

        for key in ("x", "y", "width", "height"):
            if key not in roi:
                raise RuntimeError(f"ROI config is missing required key: {key}")

        return {
            "x": int(roi["x"]),
            "y": int(roi["y"]),
            "width": int(roi["width"]),
            "height": int(roi["height"]),
            "source_path": str(roi_path),
        }

    def _apply_camera_roi(self) -> None:
        """Force full-frame capture and store any calibration ROI for analysis only.

        Older versions used the selected calibration ROI as a hardware camera ROI,
        which reduced the pixels streamed/saved by the Blackfly. This version
        always resets the camera to the maximum available full-frame image. If an
        ROI JSON is provided, that ROI is clipped to the full-frame coordinates
        and used later only to crop stacks during analysis/plotting.
        """
        requested = self._load_roi_config()

        try:
            # Always return the camera to full-frame capture. Offset must be zero
            # before expanding Width/Height back to their maxima.
            self._set_int("OffsetX", 0)
            self._set_int("OffsetY", 0)
            _, width_max, _ = self._get_int_limits("Width")
            _, height_max, _ = self._get_int_limits("Height")
            full_width = self._set_int("Width", width_max)
            full_height = self._set_int("Height", height_max)

            if requested is None:
                self.active_roi = None
                print(
                    "Full-frame capture enabled with no analysis ROI: "
                    f"width={full_width}, height={full_height}"
                )
                return

            x_req = max(0, min(int(requested["x"]), full_width - 1))
            y_req = max(0, min(int(requested["y"]), full_height - 1))
            w_req = max(1, min(int(requested["width"]), full_width - x_req))
            h_req = max(1, min(int(requested["height"]), full_height - y_req))

            self.active_roi = {
                "mode": "analysis_only_full_frame_capture",
                "requested": requested,
                "analysis_crop": {
                    "x": x_req,
                    "y": y_req,
                    "width": w_req,
                    "height": h_req,
                    "full_width": full_width,
                    "full_height": full_height,
                },
            }

            print(
                "Full-frame capture enabled; analysis ROI will be cropped in software: "
                f"x={x_req}, y={y_req}, width={w_req}, height={h_req} "
                f"within full frame width={full_width}, height={full_height} "
                f"(requested x={requested['x']}, y={requested['y']}, "
                f"width={requested['width']}, height={requested['height']})"
            )
        except CameraConfigError as exc:
            raise CameraConfigError(
                f"Could not set full-frame capture / analysis ROI from "
                f"{requested.get('source_path', self.cfg.roi_config) if requested else self.cfg.roi_config}: {exc}"
            ) from exc

    def _analysis_roi_bounds(self, image_shape: tuple[int, int]) -> Optional[tuple[int, int, int, int]]:
        """Return clipped analysis ROI bounds as y0, y1, x0, x1 for an image shape."""
        if self.active_roi is None:
            return None

        crop = self.active_roi.get("analysis_crop") or self.active_roi.get("actual")
        if not crop:
            return None

        height, width = int(image_shape[0]), int(image_shape[1])
        x0 = max(0, min(int(crop.get("x", 0)), width - 1))
        y0 = max(0, min(int(crop.get("y", 0)), height - 1))
        x1 = max(x0 + 1, min(x0 + int(crop.get("width", width)), width))
        y1 = max(y0 + 1, min(y0 + int(crop.get("height", height)), height))
        return y0, y1, x0, x1

    def _crop_stack_to_analysis_roi(self, stack: "np.ndarray") -> "np.ndarray":
        """Crop a frame stack to the selected analysis ROI without affecting saved raw data."""
        bounds = self._analysis_roi_bounds(stack.shape[-2:])
        if bounds is None:
            return stack
        y0, y1, x0, x1 = bounds
        if stack.ndim == 2:
            return stack[y0:y1, x0:x1]
        return stack[..., y0:y1, x0:x1]

    @staticmethod
    def _software_bin_spatial(arr: "np.ndarray", factor: int) -> "np.ndarray":
        """Average non-overlapping factor x factor pixel blocks for analysis only.

        This preserves the original acquired/saved TIFF or RAW frames. If the image
        dimensions are not exactly divisible by factor, the extra bottom/right pixels
        are cropped before binning. Works on either a single 2D image or a stack with
        frame dimensions in the final two axes.
        """
        import numpy as np

        factor = max(1, int(factor))
        if factor <= 1:
            return arr
        if arr.ndim < 2:
            return arr

        height = int(arr.shape[-2])
        width = int(arr.shape[-1])
        binned_height = height // factor
        binned_width = width // factor
        if binned_height < 1 or binned_width < 1:
            raise RuntimeError(
                f"Analysis binning factor {factor} is too large for image shape {arr.shape}."
            )

        cropped = arr[..., :binned_height * factor, :binned_width * factor]
        if arr.ndim == 2:
            return cropped.reshape(binned_height, factor, binned_width, factor).mean(axis=(1, 3))

        leading = cropped.shape[:-2]
        reshaped = cropped.reshape(*leading, binned_height, factor, binned_width, factor)
        return reshaped.mean(axis=(-3, -1))

    def _configure_camera(self) -> None:
        self._configure_stream_buffers()
        self._set_enum("TriggerMode", "Off")
        self._set_enum("AcquisitionMode", "Continuous")
        self._set_enum("PixelFormat", self.cfg.pixel_format)
        self._configure_binning()
        self._apply_camera_roi()
        self._set_enum("ExposureAuto", "Off")
        self._set_enum("ExposureMode", "Timed")
        self._set_float("ExposureTime", self.cfg.exposure_us)
        try:
            self._set_enum("GainAuto", "Off")
            if self.cfg.gain_db is not None:
                self._set_float("Gain", self.cfg.gain_db)
        except CameraConfigError:
            pass
        self._set_enum("TriggerSelector", "FrameStart")
        self._set_enum("TriggerSource", self.cfg.trigger_line)
        self._set_enum("TriggerActivation", self.cfg.trigger_activation)
        self._set_enum("TriggerMode", "On")

    def start_acquisition(self) -> None:
        self.cam.BeginAcquisition()

    # ------------------------------------------------------------------
    # Background frame writer
    #
    # out_path.write_bytes() opens, writes, and closes a handle per frame, on
    # the acquisition thread, between two GetNextImage() calls. At 10 Hz that
    # cost lands directly in the frame budget and pushes the consumer behind
    # the camera. The acquisition loop now copies the buffer, hands it to this
    # thread, and releases the PySpin image immediately.
    # ------------------------------------------------------------------
    def _ensure_frame_writer(self) -> None:
        if self._writer_thread is not None and self._writer_thread.is_alive():
            return
        # Bounded, but far above any sane backlog: put() must not become the
        # new blocking call in the hot loop. Hitting this bound at all means
        # disk cannot keep up, and blocking is then the correct backpressure.
        self._writer_queue = queue.Queue(maxsize=1024)
        self._writer_thread = threading.Thread(
            target=self._frame_writer_loop,
            name="frame-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _frame_writer_loop(self) -> None:
        while True:
            item = self._writer_queue.get()
            if item is None:
                return
            try:
                out_path, payload = item
                out_path.write_bytes(payload)
            except Exception as exc:
                # Never let one bad frame kill the thread: a dead writer would
                # strand every later frame and hang the end-of-trial drain.
                self._writer_error = exc
                print(f"Frame writer error: {exc}")
            finally:
                self._writer_completed += 1

    def _writer_backlog(self) -> int:
        # Only the acquisition thread increments _writer_enqueued and only the
        # writer thread increments _writer_completed, so this needs no lock.
        return self._writer_enqueued - self._writer_completed

    def _drain_frame_writer(self, timeout_s: float = 60.0) -> None:
        """Block until every queued frame is on disk. Called at trial end so no
        metadata is written describing frames that are not there yet."""
        if self._writer_queue is None:
            return

        deadline = time.time() + timeout_s
        while self._writer_backlog() > 0:
            if self._writer_thread is None or not self._writer_thread.is_alive():
                print(
                    f"Warning: the frame writer thread is not running; "
                    f"{self._writer_backlog()} frame(s) may be unwritten."
                )
                break
            if time.time() > deadline:
                print(
                    f"Warning: frame writer still has {self._writer_backlog()} frame(s) "
                    f"queued after {timeout_s:.0f} s."
                )
                break
            time.sleep(0.005)

        if self._writer_error is not None:
            err = self._writer_error
            self._writer_error = None
            print(f"Warning: at least one frame failed to write: {err}")

    def _stop_frame_writer(self) -> None:
        if self._writer_queue is None or self._writer_thread is None:
            return
        try:
            self._drain_frame_writer()
            self._writer_queue.put(None)
            self._writer_thread.join(timeout=10.0)
        except Exception:
            pass
        self._writer_queue = None
        self._writer_thread = None

    def _save_image(self, image, out_path: Path) -> None:
        """Queue a frame for the writer thread (raw) or save it inline.

        Only the raw path is asynchronous. png/tiff go through Spinnaker's own
        image.Save(), which needs the PySpin image object alive, and deep-copying
        that per frame would reintroduce the cost this exists to avoid. raw is
        the acquisition default and the only format used for timed trials; the
        encoded formats stay on the original synchronous path.
        """
        fmt = self.cfg.save_format.lower()
        if fmt != "raw":
            image.Save(str(out_path))
            return

        data = image.GetData()
        # .tobytes() copies the buffer as-is regardless of dtype, matching what
        # write_bytes(image.GetData()) used to put on disk byte for byte.
        payload = data.tobytes() if hasattr(data, "tobytes") else bytes(data)

        self._ensure_frame_writer()
        depth = self._writer_backlog()
        if depth > self._writer_queue_peak:
            self._writer_queue_peak = depth
        if depth > self.cfg.writer_queue_warn_depth and not self._writer_warned:
            self._writer_warned = True
            print(
                f"Warning: frame writer backlog reached {depth} frames "
                f"(threshold {self.cfg.writer_queue_warn_depth}). Disk is not "
                f"keeping up with acquisition."
            )
        self._writer_queue.put((out_path, payload))
        self._writer_enqueued += 1

    def _make_trial_dirs(self, trial_index: int) -> Tuple[Path, Path, Path, Path, Path, Path]:
        trial_dir = self.session_dir / f"trial_{trial_index:03d}"
        green_dir = trial_dir / "green"
        baseline_dir = trial_dir / "baseline"
        gap_dir = trial_dir / "gap"
        post_dir = trial_dir / "post"
        meta_dir = trial_dir / "meta"
        analysis_dir = trial_dir / "analysis"
        for d in (green_dir, baseline_dir, gap_dir, post_dir, meta_dir, analysis_dir):
            d.mkdir(parents=True, exist_ok=True)
        return green_dir, baseline_dir, gap_dir, post_dir, meta_dir, analysis_dir

    def _write_trial_metadata(
        self,
        trial_index: int,
        green_dir: Path,
        baseline_dir: Path,
        gap_dir: Path,
        post_dir: Path,
        meta_dir: Path,
        trial_start_wall: float,
        trial_end_wall: float,
        green_count: int,
        baseline_count: int,
        gap_count: int,
        post_count: int,
        markers_seen: list[str],
        visual_stim_metadata: Optional[dict] = None,
        trial_condition: str = "stim",
        dropped_frame_count: int = 0,
        dropped_frames: Optional[list[dict]] = None,
        phase_boundary_trigger_indices: Optional[dict] = None,
        trigger_labeling: bool = False,
    ) -> None:
        trial_config = asdict(self.cfg)
        if trial_config.get("roi_config") is not None:
            trial_config["roi_config"] = str(trial_config["roi_config"])

        metadata = {
            "trial_index": trial_index,
            "trial_config": trial_config,
            "image_width_px": self.frame_width,
            "image_height_px": self.frame_height,
            "binning_horizontal": int(self.cfg.binning_horizontal),
            "binning_vertical": int(self.cfg.binning_vertical),
            "pixel_format": self.cfg.pixel_format,
            "active_roi": self.active_roi,
            "trial_start_iso": datetime.fromtimestamp(trial_start_wall).isoformat(),
            "trial_end_iso": datetime.fromtimestamp(trial_end_wall).isoformat(),
            "green_dir": str(green_dir),
            "baseline_dir": str(baseline_dir),
            "gap_dir": str(gap_dir),
            "post_dir": str(post_dir),
            "green_frame_count": green_count,
            "baseline_frame_count": baseline_count,
            "gap_frame_count": gap_count,
            "post_frame_count": post_count,
            "markers_seen": markers_seen,
            "visual_stim_metadata": visual_stim_metadata,
            # "stim" or "catch" -- ground truth for which condition this trial
            # actually ran, independent of any modality-level flag. Read by
            # statistical_analyses.py's --condition filter for interleaved
            # sessions; "stim" (today's implicit default) for every trial when
            # trial_conditions is unused.
            "trial_condition": trial_condition,
            # Number of camera triggers whose frames never reached the host,
            # inferred from gaps in the camera timestamp. Must be 0 for a trial
            # to be trusted for timing-sensitive analysis.
            "dropped_frame_count": int(dropped_frame_count),
            "dropped_frames": dropped_frames or [],
            # Inclusive first trigger index of each phase, as reported by the
            # Arduino. Lets analysis re-derive any frame's phase independently.
            "phase_boundary_trigger_indices": phase_boundary_trigger_indices or {},
            # False means the firmware did not report trigger indices and phase
            # labels came from the legacy dequeue-time state machine, which is
            # sensitive to acquisition backlog.
            "trigger_index_labeling": bool(trigger_labeling),
        }
        with open(meta_dir / "trial_metadata.json", "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, default=str)

    def _write_trial_conditions_manifest(self, max_trial_index: int) -> None:
        """Write session_dir/trial_conditions.json: every completed trial's
        stim/catch condition in one place, for interleaved sessions. No-op
        when interleaving is unused (self.cfg.trial_conditions is None) --
        the per-trial trial_metadata.json is the only condition record for a
        plain session, matching every other interleaving-only artifact this
        session.

        Re-derived from each trial's own trial_metadata.json on every call
        (not accumulated in memory), matching this file's existing
        re-scan-from-disk pattern for session-level state, and rewritten
        after every trial so an interrupted session still leaves an accurate
        manifest for whatever trials completed so far.
        """
        if self.cfg.trial_conditions is None:
            return

        trials = []
        for trial_index in range(1, max_trial_index + 1):
            meta_path = self.session_dir / f"trial_{trial_index:03d}" / "meta" / "trial_metadata.json"
            if not meta_path.exists():
                continue
            try:
                trial_cond = json.loads(meta_path.read_text(encoding="utf-8")).get("trial_condition")
            except (OSError, ValueError):
                trial_cond = None
            if trial_cond is None:
                continue
            trials.append({"trial_index": trial_index, "condition": trial_cond})

        manifest = {
            "session": self.session_dir.name,
            "num_trials_total": int(self.cfg.num_trials),
            "num_trials_completed": len(trials),
            "num_stim": sum(1 for t in trials if t["condition"] == "stim"),
            "num_catch": sum(1 for t in trials if t["condition"] == "catch"),
            "trials": trials,
        }
        try:
            with open(self.session_dir / "trial_conditions.json", "w", encoding="utf-8") as fp:
                json.dump(manifest, fp, indent=2)
        except OSError as exc:
            print(f"Warning: could not write trial_conditions.json: {exc}")

    def _read_serial_marker(self) -> Optional[tuple[str, Optional[int], str]]:
        """Return one complete marker line, or None if none is buffered yet.

        Never blocks. The old implementation called ser.readline(), which waits
        out the full port timeout (0.1 s by default) whenever no newline has
        arrived -- and this runs before every GetNextImage(), so an idle port
        cost roughly one entire frame period per frame. Bytes are drained from
        the OS buffer instead and the partial tail is kept in _serial_rx_buf.

        Accepted formats:
            MARKER,arduino_millis                 (older firmware)
            MARKER,arduino_millis,trigger_index   (current firmware)
        """
        if self.ser is None:
            return None

        try:
            pending = self.ser.in_waiting
        except Exception:
            pending = 0

        if pending:
            try:
                self._serial_rx_buf.extend(self.ser.read(pending))
            except Exception:
                pass

        newline_at = self._serial_rx_buf.find(b"\n")
        if newline_at < 0:
            return None

        line = bytes(self._serial_rx_buf[:newline_at])
        del self._serial_rx_buf[:newline_at + 1]

        raw_text = line.decode("utf-8", errors="ignore").strip()
        if not raw_text:
            return None

        parts = raw_text.split(",")
        marker = parts[0].strip()

        arduino_millis = None
        if len(parts) >= 2:
            try:
                arduino_millis = int(parts[1].strip())
            except ValueError:
                arduino_millis = None

        trigger_index = None
        if len(parts) >= 3:
            try:
                trigger_index = int(parts[2].strip())
            except ValueError:
                trigger_index = None

        if trigger_index is not None:
            self._marker_trigger_index[marker] = trigger_index
        if arduino_millis is not None:
            self._marker_arduino_millis[marker] = arduino_millis

        return marker, arduino_millis, raw_text

    def _reset_marker_tracking(self) -> None:
        """Clear per-trial marker bookkeeping so one trial's phase boundaries
        can never label the next trial's frames."""
        self._marker_arduino_millis.clear()
        self._marker_trigger_index.clear()

    def _boundary_trigger_index(self, *marker_names: str) -> Optional[int]:
        """First available trigger index among equivalent boundary markers."""
        for name in marker_names:
            value = self._marker_trigger_index.get(name)
            if value is not None:
                return value
        return None

    # Phase segments in trigger order. Each entry is (label, marker aliases);
    # the marker's trigger index is the inclusive first trigger of that phase.
    PHASE_SEGMENTS = (
        ("baseline_collect", ("RED_BASELINE_START", "BASELINE_FIRST_FRAME_TRIGGER")),
        ("gap", ("GAP_START",)),
        ("post_collect", ("POST_START", "POST_FIRST_FRAME_TRIGGER")),
        ("post_trailing", ("POST_TRAILING_START",)),
    )

    def _has_phase_boundaries(self) -> bool:
        """True when the firmware is reporting trigger indices for this trial.

        Two-field firmware reports none, in which case the caller falls back to
        the legacy live-phase labeling rather than discarding every frame.
        """
        return any(
            self._boundary_trigger_index(*aliases) is not None
            for _, aliases in self.PHASE_SEGMENTS
        )

    def _phase_from_trigger_index(self, trigger_index: int) -> tuple[str, Optional[int]]:
        """Map a frame's trigger index onto its phase and 1-based index within it.

        This is the fix for queue-depth-dependent phase labels: the answer
        depends only on which trigger produced the frame, so it is identical
        whether the frame was dequeued immediately or eleven frames later.
        """
        phase_name = "idle"
        start_ti: Optional[int] = None
        for name, aliases in self.PHASE_SEGMENTS:
            boundary = self._boundary_trigger_index(*aliases)
            if boundary is not None and trigger_index >= boundary:
                phase_name = name
                start_ti = boundary
        if start_ti is None:
            return "idle", None
        return phase_name, trigger_index - start_ti + 1

    def _update_trigger_index(
        self,
        previous_index: int,
        cam_ts: int,
        prev_cam_ts: Optional[int],
    ) -> tuple[int, int]:
        """Advance the trigger index for a newly received frame.

        Returns (trigger_index, dropped_before). Normally the index advances by
        one; when the camera timestamp jumped by more than 1.5 nominal periods
        the extra periods are triggers whose frames never reached us, so the
        index skips over them and the shortfall is reported.
        """
        nominal = (float(self.cfg.trigger_period_ms) / 1000.0) * float(self.cfg.camera_timestamp_hz)

        if prev_cam_ts is None or nominal <= 0:
            return previous_index + 1, 0

        delta = float(cam_ts - prev_cam_ts)

        # One-shot sanity check on the assumed timestamp scale. If the observed
        # cadence does not look like the nominal period, the tick rate is wrong
        # and every inferred drop would be fiction -- so stop inferring.
        #
        # No drop inference runs until this completes: with a wrong tick rate a
        # single frame could otherwise advance the index by hundreds of triggers
        # before the check had the samples to notice. These are the first frames
        # after TRIAL_START, where the queue is empty and drops are not credible.
        if not self._ts_scale_checked:
            if delta > 0:
                self._ts_scale_samples.append(delta)
            if len(self._ts_scale_samples) >= 8:
                self._ts_scale_checked = True
                ordered = sorted(self._ts_scale_samples)
                median = ordered[len(ordered) // 2]
                ratio = median / nominal if nominal else 0.0
                if not (0.75 <= ratio <= 1.25):
                    self._ts_scale_ok = False
                    print(
                        "Warning: camera timestamps do not match the expected scale "
                        f"(median inter-frame delta {median:.0f} ticks vs an expected "
                        f"{nominal:.0f} for {self.cfg.trigger_period_ms:.1f} ms at "
                        f"{self.cfg.camera_timestamp_hz:.3g} Hz; ratio {ratio:.3f}). "
                        "Dropped-frame detection is disabled for this session. Fix "
                        "--camera-timestamp-hz / --trigger-period-ms to re-enable it."
                    )
            return previous_index + 1, 0

        if not self._ts_scale_ok:
            return previous_index + 1, 0

        if delta > 1.5 * nominal:
            step = max(1, int(round(delta / nominal)))
            return previous_index + step, step - 1

        return previous_index + 1, 0


    def _log_marker(self, trial_index: int, marker: str, arduino_millis: Optional[int], raw_text: str) -> None:
        if self._marker_csv_writer is None:
            return

        self._marker_csv_writer.writerow([
            trial_index,
            datetime.now().isoformat(timespec="milliseconds"),
            marker,
            arduino_millis if arduino_millis is not None else "",
            raw_text,
        ])
        self._marker_csv_fp.flush()

    def _wait_for_marker(self, expected: set[str], timeout_s: float, trial_index: int = 0) -> str:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            result = self._read_serial_marker()
            if result is None:
                # _read_serial_marker() no longer blocks on the port timeout, so
                # yield briefly instead of spinning a core flat out. 1 ms is far
                # finer than the 100 ms trigger period this ever waits on.
                time.sleep(0.001)
                continue

            marker, arduino_millis, raw_text = result
            self._log_marker(trial_index, marker, arduino_millis, raw_text)

            print(
                f"[marker] trial={trial_index} host={datetime.now().isoformat(timespec='milliseconds')} "
                f"arduino_ms={arduino_millis} marker={marker}"
            )

            if marker in expected:
                return marker
            if marker in self.VALID_MARKERS:
                print(f"Ignoring out-of-order marker while waiting: {raw_text}")
            elif marker in self.INFO_MARKERS:
                pass # recognized; stay quiet
            else:
                print(f"Ignoring unknown serial text: {raw_text}")

        raise TimeoutError(f"Timed out waiting for marker(s): {sorted(expected)}")

    def _poll_all_markers(self, markers_seen: list[str], trial_index: int) -> list[str]:
        seen = []
        while True:
            result = self._read_serial_marker()
            if result is None:
                break

            marker, arduino_millis, raw_text = result
            self._log_marker(trial_index, marker, arduino_millis, raw_text)

            print(
                f"[marker] trial={trial_index} host={datetime.now().isoformat(timespec='milliseconds')} "
                f"arduino_ms={arduino_millis} marker={marker}"
            )

            if marker in self.VALID_MARKERS:
                markers_seen.append(marker)
                seen.append(marker)
            elif marker in self.INFO_MARKERS:
                pass # recognized; already logged to marker CSV, stay quiet
            else:
                print(f"Ignoring unknown serial text: {raw_text}")

        return seen

    def _load_dark_reference_for_shape(self, image_shape: tuple[int, int]) -> Optional["np.ndarray"]:
        """Load a master dark reference image for analysis-time dark subtraction.

        The dark reference should be acquired with the same camera, exposure, gain,
        pixel format, frame size, and lens-cap/LED-off conditions. It is subtracted
        from green, baseline, and post stacks before ROI cropping, software binning,
        averaging, and intrinsic-signal calculations.
        """
        if self.cfg.dark_reference_path is None:
            return None

        import numpy as np

        dark_path = Path(str(self.cfg.dark_reference_path))
        if not dark_path.exists():
            print(
                f"WARNING: dark reference not found at {dark_path}. Proceeding WITHOUT dark-frame subtraction."
            )
            return None

        if dark_path.suffix.lower() == ".npy":
            dark = np.load(dark_path).astype(np.float64)
        else:
            from PIL import Image
            dark = np.array(Image.open(dark_path)).astype(np.float64)

        if dark.ndim != 2:
            raise RuntimeError(f"Dark reference must be a 2D image; got shape {dark.shape} from {dark_path}")

        expected_shape = tuple(int(v) for v in image_shape)
        if tuple(dark.shape) != expected_shape:
            raise RuntimeError(
                f"Dark reference shape {dark.shape} does not match loaded frame shape {expected_shape}. "
                "Use a dark reference acquired with the same Width/Height, pixel format, exposure, gain, "
                "and camera ROI settings as the current acquisition."
            )

        return dark

    def _load_image_file_for_analysis(self, path: Path) -> "np.ndarray":
        """Load one saved frame as float32 for analysis.

        TIFF/PNG files are read through PIL. RAW files are headerless, so this
        uses the current camera width/height and pixel format saved during setup.
        """
        import numpy as np

        ext = path.suffix.lower()
        if ext == ".raw":
            if self.frame_width is None or self.frame_height is None:
                raise RuntimeError("Cannot read RAW files because frame dimensions are unknown.")

            dtype_by_format = {
                "Mono8": np.uint8,
                "Mono12": np.uint16,
                "Mono16": np.uint16,
            }
            dtype = dtype_by_format.get(self.cfg.pixel_format)
            if dtype is None:
                raise RuntimeError(f"Unsupported RAW pixel format for analysis: {self.cfg.pixel_format}")

            data = np.fromfile(path, dtype=dtype)
            expected = int(self.frame_width) * int(self.frame_height)
            if data.size != expected:
                raise RuntimeError(
                    f"RAW frame has {data.size} pixels, expected {expected}. "
                    f"Check width/height/pixel format. File: {path}"
                )
            return data.reshape((int(self.frame_height), int(self.frame_width))).astype(np.float32)

        from PIL import Image
        return np.array(Image.open(path)).astype(np.float32)

    def _load_stack_for_analysis(self, folder: Path) -> "np.ndarray":
        import numpy as np

        patterns = ["*.tiff", "*.tif", "*.png", "*.raw"]
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(folder.glob(pattern)))
        files = sorted(files)

        if not files:
            raise RuntimeError(f"No image files found for analysis in: {folder}")

        return np.stack([self._load_image_file_for_analysis(f) for f in files], axis=0)

    @staticmethod
    def _display_scale_for_analysis(img: "np.ndarray") -> "np.ndarray":
        """Return a 0-1 display version without changing quantitative arrays."""
        import numpy as np

        finite = np.isfinite(img)
        if not np.any(finite):
            return np.zeros_like(img, dtype=np.float32)

        lo, hi = np.nanpercentile(img, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(img))
            hi = float(np.nanmax(img))
        if hi <= lo:
            return np.zeros_like(img, dtype=np.float32)

        return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)

    # IMX249: 12-bit ADC left-shifted into a 16-bit Mono16 container.
    # Clipping happens at 65520, NOT 65535. Comparing against the dtype max never fires.
    SENSOR_SATURATION_COUNT = 65520.0

    @staticmethod
    def _saturated_fraction(stack: "np.ndarray", saturation_count: float = SENSOR_SATURATION_COUNT) -> float:
        import numpy as np
        arr = np.asarray(stack)
        return float(np.mean(arr >= float(saturation_count))) if arr.size else 0.0

    @staticmethod
    def _stack_qc_stats(
        stack: "np.ndarray",
        phase: str,
        bit_depth: int = 16,
        saturation_count: float = SENSOR_SATURATION_COUNT,
    ) -> dict:
        """Return basic image-quality statistics for a frame stack.

        saturation_count is the true sensor clip level in raw ADC counts (65520
        for the IMX249 in Mono16), not the container dtype max.
        """
        import numpy as np

        finite = np.isfinite(stack)
        dtype_max = float((2 ** bit_depth) - 1)
        if np.issubdtype(stack.dtype, np.integer):
            try:
                dtype_max = float(np.iinfo(stack.dtype).max)
            except ValueError:
                pass

        sat_threshold = float(saturation_count)
        values = stack[finite]

        if values.size == 0:
            return {
                "phase": phase,
                "frames": int(stack.shape[0]) if stack.ndim >= 3 else 0,
                "min": None, "p01": None, "p50": None, "p95": None, "p99": None,
                "max": None, "mean": None,
                "dtype_max": dtype_max,
                "saturation_threshold": sat_threshold,
                "saturated_fraction": None,
            }

        return {
            "phase": phase,
            "frames": int(stack.shape[0]) if stack.ndim >= 3 else 0,
            "min": float(np.nanmin(values)),
            "p01": float(np.nanpercentile(values, 1)),
            "p50": float(np.nanpercentile(values, 50)),
            "p95": float(np.nanpercentile(values, 95)),
            "p99": float(np.nanpercentile(values, 99)),
            "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
            "dtype_max": dtype_max,
            "saturation_threshold": sat_threshold,
            "saturated_fraction": float(np.mean(stack >= sat_threshold)),
        }

    @staticmethod
    def _rescale_signed_symmetric_to_bit_depth(img: "np.ndarray", bit_depth: int = 16) -> "np.ndarray":
        """Scale a signed map to unsigned integer range using zero as mid-gray.

        Mapping:
            -max_abs -> 0
             0       -> midpoint
            +max_abs -> 2^bit_depth - 1
        """
        import numpy as np

        finite = np.isfinite(img)
        max_value = int((2 ** int(bit_depth)) - 1)
        dtype = np.uint16 if int(bit_depth) > 8 else np.uint8

        if not np.any(finite):
            return np.zeros_like(img, dtype=dtype)

        max_abs = float(np.nanmax(np.abs(img[finite])))
        if not np.isfinite(max_abs) or max_abs <= 0:
            return np.full_like(img, int(round(max_value / 2.0)), dtype=dtype)

        scaled = img / max_abs
        scaled = np.clip(scaled, -1.0, 1.0)
        unsigned = (scaled + 1.0) * (max_value / 2.0)

        return np.clip(unsigned, 0, max_value).astype(dtype)

    @staticmethod
    def _rescale_minmax_to_bit_depth(img: "np.ndarray", bit_depth: int = 16) -> "np.ndarray":
        """Scale a map to unsigned integer range using min-max normalization.

        Mapping:
            finite minimum -> 0
            finite maximum -> 2^bit_depth - 1

        Note: zero does not necessarily map to mid-gray with this mode.
        """
        import numpy as np

        finite = np.isfinite(img)
        max_value = int((2 ** int(bit_depth)) - 1)
        dtype = np.uint16 if int(bit_depth) > 8 else np.uint8

        if not np.any(finite):
            return np.zeros_like(img, dtype=dtype)

        lo = float(np.nanmin(img[finite]))
        hi = float(np.nanmax(img[finite]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(img, dtype=dtype)

        scaled = (img - lo) / (hi - lo)
        unsigned = scaled * max_value

        return np.clip(unsigned, 0, max_value).astype(dtype)

    @classmethod
    def _rescale_to_bit_depth(cls, img: "np.ndarray", bit_depth: int = 16, mode: str = "signed_symmetric") -> "np.ndarray":
        """Scale a floating-point analysis map to an unsigned integer image.

        Args:
            img: Floating-point map to export.
            bit_depth: Output bit depth, usually 16.
            mode:
                "signed_symmetric" keeps zero at mid-gray.
                "minmax" maps the finite minimum to 0 and finite maximum to max.
        """
        mode = str(mode).lower().strip().replace("-", "_")
        if mode in {"signed", "symmetric", "signed_symmetric", "symmetric_signed"}:
            return cls._rescale_signed_symmetric_to_bit_depth(img, bit_depth)
        if mode in {"minmax", "min_max", "min-max"}:
            return cls._rescale_minmax_to_bit_depth(img, bit_depth)
        raise ValueError(f"Unsupported rescale mode: {mode!r}. Use 'signed_symmetric' or 'minmax'.")

    def _run_trial_analysis(
        self,
        trial_index: int,
        green_dir: Path,
        baseline_dir: Path,
        post_dir: Path,
        analysis_dir: Path,
    ) -> None:
        """Create intrinsic optical signal maps for one completed trial.

        The quantitative arrays stay in floating point. Rescaled 16-bit images and PNGs
        are saved only for visualization/export.
        """
        try:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            print(f"Analysis skipped for trial {trial_index}: missing dependency: {exc}")
            return

        try:
            from scipy.ndimage import gaussian_filter, median_filter  # type: ignore
        except ImportError:
            gaussian_filter = None
            median_filter = None

        try:
            green_stack = self._load_stack_for_analysis(green_dir).astype(np.float64)
            baseline_stack = self._load_stack_for_analysis(baseline_dir).astype(np.float64)
            post_stack = self._load_stack_for_analysis(post_dir).astype(np.float64)

            raw_saturation = {
                "green": self._saturated_fraction(green_stack),
                "baseline": self._saturated_fraction(baseline_stack),
                "post_all": self._saturated_fraction(post_stack),
            }

            full_frame_shape = tuple(int(v) for v in baseline_stack.shape[-2:])

            dark_reference = self._load_dark_reference_for_shape(full_frame_shape)
            dark_reference_applied = dark_reference is not None
            if dark_reference_applied:
                green_stack = green_stack - dark_reference
                baseline_stack = baseline_stack - dark_reference
                post_stack = post_stack - dark_reference
                print(
                    f"Applied dark-frame subtraction using: {str(self.cfg.dark_reference_path)}"
                )

            green_stack = self._crop_stack_to_analysis_roi(green_stack)
            baseline_stack = self._crop_stack_to_analysis_roi(baseline_stack)
            post_stack = self._crop_stack_to_analysis_roi(post_stack)
            analysis_frame_shape_pre_binning = tuple(int(v) for v in baseline_stack.shape[-2:])

            analysis_binning = max(1, int(self.cfg.analysis_binning))
            if analysis_binning > 1:
                green_stack = self._software_bin_spatial(green_stack, analysis_binning)
                baseline_stack = self._software_bin_spatial(baseline_stack, analysis_binning)
                post_stack = self._software_bin_spatial(post_stack, analysis_binning)
                print(
                    f"Applied {analysis_binning}x{analysis_binning} software binning for analysis only; "
                    "saved camera frames are unchanged."
                )

            analysis_frame_shape = tuple(int(v) for v in baseline_stack.shape[-2:])

            trim = max(0, int(self.cfg.green_reference_trim_frames))

            if trim > 0 and green_stack.shape[0] > 2 * trim:
                green_stack_for_reference = green_stack[trim:-trim]
            else:
                green_stack_for_reference = green_stack

            green_reference = np.nanmean(green_stack_for_reference, axis=0)
            baseline_mean = np.nanmean(baseline_stack, axis=0)

            start = max(0, int(self.cfg.analysis_start_frame))
            end = self.cfg.analysis_end_frame
            if end is None or end <= 0 or end > post_stack.shape[0]:
                end = post_stack.shape[0]
            end = int(end)
            if start >= end:
                start = 0
                end = post_stack.shape[0]

            post_window = post_stack[start:end]
            post_mean = np.nanmean(post_window, axis=0)

            bit_depth = int(self.cfg.rescale_bit_depth)
            qc_rows = [
                self._stack_qc_stats(green_stack, "green", bit_depth=bit_depth),
                self._stack_qc_stats(baseline_stack, "baseline", bit_depth=bit_depth),
                self._stack_qc_stats(post_stack, "post_all", bit_depth=bit_depth),
                self._stack_qc_stats(post_window, "post_analysis_window", bit_depth=bit_depth),
            ]

            qc_rows[0]["saturated_fraction"] = raw_saturation["green"]
            qc_rows[1]["saturated_fraction"] = raw_saturation["baseline"]
            qc_rows[2]["saturated_fraction"] = raw_saturation["post_all"]
            qc_rows[3]["saturated_fraction"] = raw_saturation["post_all"]  # window ⊆ post

            # The camera saves full-frame images. If a calibration ROI was provided,
            # analysis is cropped to that ROI here, after loading the saved frames.
            analysis_mask = np.isfinite(baseline_mean)

            analysis_method = self.cfg.analysis_method.lower().strip()
            eps = max(float(np.nanmedian(baseline_mean)) * 1e-6, 1e-6)

            denominator_floor = None
            denominator_floor_percentile = None
            denominator_floor_counts = None
            denominator_mask_fraction_kept = None
            median_filter_size = 0
            median_filter_applied = False

            if analysis_method == "raw_counts":
                # Raw pixel-count processing:
                # 1) average baseline and stimulation/post frames as floating point
                # 2) subtract baseline mean from stimulation/post mean
                # 3) subtract the median of the difference image to remove global drift
                # 4) Gaussian filter the centered difference image
                raw_diff = post_mean - baseline_mean
                median_reference = float(np.nanmedian(raw_diff[analysis_mask])) if np.any(analysis_mask) else float(np.nanmedian(raw_diff))
                centered_map = raw_diff - median_reference
                quantitative_label = "stim_minus_baseline_median_centered"
                colorbar_label = "post - baseline, median-centered"
            elif analysis_method == "fractional_reflectance":
                # Fractional reflectance option, reflectance decreases appear negative.
                #
                # IMPORTANT: after dark-frame subtraction, very dim or capped pixels can have
                # baseline values near zero. Dividing by those pixels produces enormous,
                # non-physiological ΔR/R values. Use a denominator floor/mask so only pixels
                # with sufficient baseline signal contribute to the fractional reflectance map.
                finite_baseline = np.isfinite(baseline_mean)
                positive_baseline = baseline_mean[finite_baseline & (baseline_mean > 0)]

                denominator_floor_percentile = float(self.cfg.analysis_denominator_floor_percentile)
                denominator_floor_counts = float(self.cfg.analysis_denominator_floor_counts)

                if positive_baseline.size > 0:
                    percentile_floor = float(np.nanpercentile(positive_baseline, denominator_floor_percentile))
                else:
                    percentile_floor = float("nan")

                if not np.isfinite(percentile_floor):
                    denominator_floor = denominator_floor_counts
                else:
                    denominator_floor = max(percentile_floor, denominator_floor_counts)

                safe_baseline = baseline_mean.copy()
                denominator_mask = finite_baseline & (safe_baseline >= denominator_floor)
                denominator_mask_fraction_kept = float(np.mean(denominator_mask)) if denominator_mask.size else None
                safe_baseline[~denominator_mask] = np.nan

                raw_diff = (post_mean - baseline_mean) / safe_baseline

                # Remove isolated hot/dead-pixel spikes from the fractional map before
                # median centering and Gaussian smoothing. This is not display clipping;
                # it is spatial outlier suppression on the quantitative ΔR/R image.
                median_filter_size = int(self.cfg.analysis_median_filter_size)
                median_filter_applied = False
                if median_filter is not None and median_filter_size > 1:
                    if median_filter_size % 2 == 0:
                        median_filter_size += 1

                    finite_raw = np.isfinite(raw_diff)
                    if np.any(finite_raw):
                        raw_fill_value = float(np.nanmedian(raw_diff[finite_raw]))
                        raw_for_filter = np.where(finite_raw, raw_diff, raw_fill_value)
                        raw_filtered = median_filter(raw_for_filter, size=median_filter_size)
                        raw_diff = np.where(finite_raw, raw_filtered, np.nan)
                        median_filter_applied = True
                else:
                    median_filter_size = 0 if median_filter_size <= 1 else median_filter_size

                drr_analysis_mask = analysis_mask & np.isfinite(raw_diff)
                median_reference = float(np.nanmedian(raw_diff[drr_analysis_mask])) if np.any(drr_analysis_mask) else float(np.nanmedian(raw_diff))
                centered_map = raw_diff - median_reference
                quantitative_label = "delta_r_over_r_median_centered"
                colorbar_label = "ΔR/R, median-centered"
            else:
                raise RuntimeError(f"Unsupported analysis_method: {self.cfg.analysis_method}")

            if gaussian_filter is not None and self.cfg.analysis_smoothing_sigma > 0:
                smoothed_map = gaussian_filter(
                    centered_map,
                    sigma=float(self.cfg.analysis_smoothing_sigma),
                )
            else:
                smoothed_map = centered_map

            masked_map = smoothed_map

            # For red-light reflectance data, a true activation often appears as a decrease
            # in reflected light. The quantitative raw-counts map is signed; display_signal can
            # optionally invert it so decreases plot as positive/hot.
            if self.cfg.invert_display_signal and analysis_method == "raw_counts":
                display_signal = -masked_map
                display_label = "-(post - baseline), median-centered"
            else:
                display_signal = masked_map
                display_label = colorbar_label

            analysis_dir.mkdir(parents=True, exist_ok=True)

            # Save QC table.
            qc_path = analysis_dir / "frame_qc_stats.csv"
            with open(qc_path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(qc_rows[0].keys()))
                writer.writeheader()
                writer.writerows(qc_rows)

            for row in qc_rows:
                sat = row.get("saturated_fraction")
                if sat is not None and sat > 0:
                    print(
                        f"Warning: trial {trial_index} {row['phase']} has "
                        f"{100.0 * float(sat):.5f}% saturated pixels."
                    )

            if self.cfg.save_analysis_arrays:
                np.save(analysis_dir / "green_reference.npy", green_reference)
                np.save(analysis_dir / "baseline_reference.npy", baseline_mean)
                np.save(analysis_dir / "post_mean_analysis_window.npy", post_mean)
                np.save(analysis_dir / f"activation_map_raw_{quantitative_label}.npy", raw_diff)
                np.save(analysis_dir / f"activation_map_centered_{quantitative_label}.npy", centered_map)
                np.save(analysis_dir / f"activation_map_smoothed_{quantitative_label}.npy", smoothed_map)
                np.save(analysis_dir / f"activation_map_masked_{quantitative_label}.npy", masked_map)
                np.save(analysis_dir / "activation_map_display_signal.npy", display_signal)

                # Method-named copies of the same arrays. These basenames are a
                # contract with the session-analysis reader below (see
                # _session_map_candidates) -- change them in both places or the
                # session pass silently finds no trials.
                if analysis_method == "raw_counts":
                    np.save(analysis_dir / "activation_map_raw_counts.npy", centered_map)
                    np.save(analysis_dir / "activation_map_raw_counts_smoothed.npy", smoothed_map)
                    np.save(analysis_dir / "activation_map_raw_counts_masked.npy", masked_map)
                else:
                    np.save(analysis_dir / "activation_map_fractional_reflectance.npy", centered_map)
                    np.save(analysis_dir / "activation_map_fractional_reflectance_smoothed.npy", smoothed_map)
                    np.save(analysis_dir / "activation_map_masked_fractional_reflectance.npy", masked_map)

            # Save a true 16-bit rescaled visualization image without corrupting the float data.
            try:
                from PIL import Image
                rescaled = self._rescale_to_bit_depth(smoothed_map, bit_depth=bit_depth, mode=self.cfg.rescale_mode)
                Image.fromarray(rescaled).save(analysis_dir / f"activation_map_rescaled_{bit_depth}bit.tiff")
            except Exception as exc:
                print(f"Warning: could not save rescaled TIFF for trial {trial_index}: {exc}")

            # Save anatomical/reference figures.
            plt.figure(figsize=(8, 8))
            plt.imshow(self._display_scale_for_analysis(green_reference), cmap="gray")
            plt.title("Green anatomical reference")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(analysis_dir / "green_reference.png", dpi=200)
            plt.close()

            plt.figure(figsize=(8, 8))
            plt.imshow(self._display_scale_for_analysis(baseline_mean), cmap="gray")
            plt.title("Red baseline reference")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(analysis_dir / "baseline_reference.png", dpi=200)
            plt.close()

            finite_display = np.isfinite(display_signal)
            abs_lim = float(np.nanpercentile(np.abs(display_signal[finite_display]), 99)) if np.any(finite_display) else 1.0
            if not np.isfinite(abs_lim) or abs_lim <= 0:
                abs_lim = 1.0

            plt.figure(figsize=(8, 8))
            # Trial activation map
            im = plt.imshow(display_signal, cmap="gray", vmin=-abs_lim, vmax=abs_lim)
            plt.colorbar(im, fraction=0.046, pad=0.04, label=display_label)
            plt.title(f"Activation map, post frames {start + 1}-{end}")
            plt.axis("off")
            plt.tight_layout()
            activation_map_path = analysis_dir / "activation_map.png"
            plt.savefig(activation_map_path, dpi=200)
            plt.close()

            plt.figure(figsize=(8, 8))
            # Trial overlay
            plt.imshow(self._display_scale_for_analysis(green_reference), cmap="gray")
            im = plt.imshow(display_signal, cmap="gray", alpha=0.55, vmin=-abs_lim, vmax=abs_lim)

            plt.colorbar(im, fraction=0.046, pad=0.04, label=display_label)
            plt.title("Intrinsic signal overlay on green reference")
            plt.axis("off")
            plt.tight_layout()

            overlay_path = analysis_dir / "activation_overlay.png"
            plt.savefig(overlay_path, dpi=200)
            plt.close()

            # Per-trial auto-opening is handled by cumulative session analysis
            # after each trial when --open-trial-overlays is enabled. In this
            # cumulative-overlay version, we do not auto-open the individual
            # trial overlay here.

            finite_masked = np.isfinite(masked_map)
            finite_display = np.isfinite(display_signal)
            summary = {
                "trial_index": trial_index,
                "analysis_method": analysis_method,
                "quantitative_label": quantitative_label,
                "analysis_window_post_frame_start_1_indexed": start + 1,
                "analysis_window_post_frame_end_1_indexed_inclusive": end,
                "smoothing_sigma_px": self.cfg.analysis_smoothing_sigma,
                "mask_percentile": None,
                "masking": "disabled; optional calibration ROI is used as a software analysis crop only",
                "active_roi": self.active_roi,
                "full_frame_shape_yx": list(full_frame_shape),
                "analysis_frame_shape_yx_before_binning": list(analysis_frame_shape_pre_binning),
                "analysis_frame_shape_yx": list(analysis_frame_shape),
                "analysis_binning": int(analysis_binning),
                "dark_reference_path": str(self.cfg.dark_reference_path) if self.cfg.dark_reference_path is not None else None,
                "dark_reference_applied": bool(dark_reference_applied),
                "denominator_floor": float(denominator_floor) if denominator_floor is not None and np.isfinite(denominator_floor) else None,
                "denominator_floor_percentile": denominator_floor_percentile,
                "denominator_floor_counts": denominator_floor_counts,
                "denominator_mask_fraction_kept": denominator_mask_fraction_kept,
                "median_filter_size": int(median_filter_size),
                "median_filter_applied": bool(median_filter_applied),
                "peak_quantitative_signal": float(np.nanmax(masked_map)) if np.any(finite_masked) else None,
                "min_quantitative_signal": float(np.nanmin(masked_map)) if np.any(finite_masked) else None,
                "peak_display_signal": float(np.nanmax(display_signal)) if np.any(finite_display) else None,
                "trough_display_signal": float(np.nanmin(display_signal)) if np.any(finite_display) else None,
                "green_frames_used": int(green_stack.shape[0]),
                "baseline_frames_used": int(baseline_stack.shape[0]),
                "post_frames_available": int(post_stack.shape[0]),
                "post_frames_used_for_analysis": int(post_window.shape[0]),
                "median_subtracted_value": median_reference,
                "rescale_bit_depth": bit_depth,
                "rescale_mode": self.cfg.rescale_mode,
                "invert_display_signal": bool(self.cfg.invert_display_signal),
                "notes": (
                    "This method saves signed floating-point maps; TIFF/PNG outputs are display/export products. "
                    "Camera acquisition/saved frames are full-frame; selected ROI, if any, is applied only as a software crop during analysis. "
                    "Software analysis binning, if enabled, averages non-overlapping pixel blocks after loading/cropping and never changes saved TIFF/RAW frames. Dark-frame subtraction, if enabled, is applied before ROI cropping/binning and before ΔR/R analysis. For ΔR/R, low-denominator pixels are masked before division and an optional spatial median filter suppresses isolated hot/dead-pixel spikes. "
                    "For red-light imaging, activation-related reflectance decreases may appear as negative values in the signed map."
                ),
            }
            with open(analysis_dir / "analysis_summary.json", "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2, default=str)

            print(
                f"Analysis complete for trial {trial_index}: method={analysis_method}, "
                f"peak_display={summary['peak_display_signal']}, "
                f"trough_display={summary['trough_display_signal']}"
            )
        except Exception as exc:
            print(f"Analysis failed for trial {trial_index}: {exc}")

    def _save_trial_vs_running_average_panel(self, trial_index: int) -> None:
        """Save one side-by-side PNG for post-trial review: three panels
        normally, four for an interleaved (stim/catch) session.

        Non-interleaved panels:
            1) This trial's red-subtraction / activation display map.
            2) Running-average red-subtraction / activation display map through this trial.
            3) Green anatomical reference image.

        Interleaved panels (self.cfg.trial_conditions is not None):
            1) This trial's red-subtraction / activation display map.
            2) Running-average red-subtraction map, STIM trials only.
            3) Running-average red-subtraction map, CATCH trials only.
            4) Green anatomical reference image.
        A condition with zero trials completed so far shows a placeholder
        instead of a map.

        Each red-subtraction panel has its own independent color scale and
        colorbar (99th-percentile |signal| within that panel), so a weak
        single-trial map isn't washed out by a strong running average or vice
        versa. The green reference is shown separately in grayscale and is
        not drawn underneath the red-subtraction panels.
        """
        try:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            print(f"Post-trial summary image skipped for trial {trial_index}: missing dependency: {exc}")
            return

        interleaved = self.cfg.trial_conditions is not None

        try:
            trial_analysis_dir = self.session_dir / f"trial_{trial_index:03d}" / "analysis"
            session_analysis_dir = self.session_dir / "session_analysis"

            trial_display_path = trial_analysis_dir / "activation_map_display_signal.npy"
            trial_green_path = trial_analysis_dir / "green_reference.npy"

            required_paths = [trial_display_path, trial_green_path]
            if not interleaved:
                required_paths.append(session_analysis_dir / "session_mean_display_signal.npy")

            missing = [str(path) for path in required_paths if not path.exists()]
            if missing:
                print(
                    f"Post-trial summary image skipped for trial {trial_index}: "
                    f"missing files: {missing}"
                )
                return

            trial_display = np.load(trial_display_path).astype(np.float64)
            trial_green = np.load(trial_green_path).astype(np.float64)

            if interleaved:
                stim_display_path = session_analysis_dir / "session_mean_display_signal_stim.npy"
                catch_display_path = session_analysis_dir / "session_mean_display_signal_catch.npy"
                stim_display = np.load(stim_display_path).astype(np.float64) if stim_display_path.exists() else None
                catch_display = np.load(catch_display_path).astype(np.float64) if catch_display_path.exists() else None

                stim_included_path = session_analysis_dir / "included_trials_stim.npy"
                catch_included_path = session_analysis_dir / "included_trials_catch.npy"
                n_stim = int(np.load(stim_included_path).size) if stim_included_path.exists() else 0
                n_catch = int(np.load(catch_included_path).size) if catch_included_path.exists() else 0

            else:
                session_display = np.load(session_analysis_dir / "session_mean_display_signal.npy").astype(np.float64)

                included_trials_path = session_analysis_dir / "included_trials.npy"
                if included_trials_path.exists():
                    included_trials = np.load(included_trials_path).astype(int).tolist()
                    n_included = len(included_trials)
                else:
                    included_trials = list(range(1, int(trial_index) + 1))
                    n_included = int(trial_index)

            def _panel_vmax(arr: np.ndarray) -> float:
                """Independent 99th-percentile |signal| scale for one panel —
                each red-subtraction panel gets its own vmin/vmax/colorbar
                rather than sharing one scale across the whole figure."""
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return 1.0
                panel_vmax = float(np.nanpercentile(np.abs(finite), 99))
                if not np.isfinite(panel_vmax) or panel_vmax <= 0:
                    return 1.0
                return panel_vmax

            trial_vmax = _panel_vmax(trial_display)

            if interleaved:
                fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

                im_trial = axes[0].imshow(trial_display, cmap="gray", vmin=-trial_vmax, vmax=trial_vmax)
                condition_label = "stim" if self.cfg.trial_conditions[trial_index - 1] else "catch"
                axes[0].set_title(f"Trial {trial_index:03d} red subtraction\n({condition_label})")
                axes[0].axis("off")
                fig.colorbar(im_trial, ax=axes[0], fraction=0.046, pad=0.04, label="display signal")

                stim_vmax: float | None = None
                if stim_display is not None:
                    stim_vmax = _panel_vmax(stim_display)
                    im_stim = axes[1].imshow(stim_display, cmap="gray", vmin=-stim_vmax, vmax=stim_vmax)
                    axes[1].set_title(f"Running average (STIM)\nn={n_stim}")
                    fig.colorbar(im_stim, ax=axes[1], fraction=0.046, pad=0.04, label="display signal")
                else:
                    axes[1].text(0.5, 0.5, "no stim trials yet", ha="center", va="center", transform=axes[1].transAxes)
                axes[1].axis("off")

                catch_vmax: float | None = None
                if catch_display is not None:
                    catch_vmax = _panel_vmax(catch_display)
                    im_catch = axes[2].imshow(catch_display, cmap="gray", vmin=-catch_vmax, vmax=catch_vmax)
                    axes[2].set_title(f"Running average (CATCH)\nn={n_catch}")
                    fig.colorbar(im_catch, ax=axes[2], fraction=0.046, pad=0.04, label="display signal")
                else:
                    axes[2].text(0.5, 0.5, "no catch trials yet", ha="center", va="center", transform=axes[2].transAxes)
                axes[2].axis("off")

                axes[3].imshow(self._display_scale_for_analysis(trial_green), cmap="gray")
                axes[3].set_title("Green reference")
                axes[3].axis("off")

                panels_list = [
                    "trial red subtraction",
                    "running average red subtraction (stim)",
                    "running average red subtraction (catch)",
                    "green reference",
                ]
            else:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

                im_trial = axes[0].imshow(trial_display, cmap="gray", vmin=-trial_vmax, vmax=trial_vmax)
                axes[0].set_title(f"Trial {trial_index:03d} red subtraction")
                axes[0].axis("off")
                fig.colorbar(im_trial, ax=axes[0], fraction=0.046, pad=0.04, label="display signal")

                running_vmax = _panel_vmax(session_display)
                im_running = axes[1].imshow(session_display, cmap="gray", vmin=-running_vmax, vmax=running_vmax)
                axes[1].set_title(f"Running average red subtraction\nthrough trial {trial_index:03d} (n={n_included})")
                axes[1].axis("off")
                fig.colorbar(im_running, ax=axes[1], fraction=0.046, pad=0.04, label="display signal")

                axes[2].imshow(self._display_scale_for_analysis(trial_green), cmap="gray")
                axes[2].set_title("Green reference")
                axes[2].axis("off")

                panels_list = [
                    "trial red subtraction",
                    "running average red subtraction",
                    "green reference",
                ]

            fig.suptitle("Post-trial intrinsic imaging summary", fontsize=14)

            trial_panel_path = trial_analysis_dir / "trial_running_average_green_reference_panel.png"
            session_panel_path = session_analysis_dir / f"trial_{trial_index:03d}_running_average_green_reference_panel.png"
            fig.savefig(trial_panel_path, dpi=200)
            fig.savefig(session_panel_path, dpi=200)
            plt.close(fig)

            if interleaved:
                display_vmax = {"trial": float(trial_vmax)}
                if stim_vmax is not None:
                    display_vmax["running_average_stim"] = float(stim_vmax)
                if catch_vmax is not None:
                    display_vmax["running_average_catch"] = float(catch_vmax)
            else:
                display_vmax = {"trial": float(trial_vmax), "running_average": float(running_vmax)}

            panel_summary = {
                "trial_index": int(trial_index),
                "interleaved": bool(interleaved),
                "trial_panel_path": str(trial_panel_path),
                "session_panel_path": str(session_panel_path),
                "display_vmax_p99_abs": display_vmax,
                "panels": panels_list,
                "notes": "Each red-subtraction panel has its own independent color scale and colorbar; the green reference is shown separately in grayscale.",
            }
            if interleaved:
                panel_summary["num_stim_trials_included"] = n_stim
                panel_summary["num_catch_trials_included"] = n_catch
            else:
                panel_summary["included_trials"] = included_trials
                panel_summary["num_trials_included"] = int(n_included)
            with open(trial_analysis_dir / "trial_running_average_green_reference_panel_summary.json", "w", encoding="utf-8") as fp:
                json.dump(panel_summary, fp, indent=2, default=str)

            print(f"Saved post-trial summary image: {trial_panel_path}")

            if self.cfg.open_trial_overlays:
                try:
                    os.startfile(trial_panel_path)
                except Exception as exc:
                    print(f"Warning: could not open post-trial summary image: {exc}")

        except Exception as exc:
            print(f"Post-trial summary image failed for trial {trial_index}: {exc}")

    def _run_session_analysis(
        self, max_trial_index: Optional[int] = None, open_overlay: bool = True,
        condition: Optional[str] = None,
    ) -> None:
        """Average processed trial-level maps across completed trial maps.

        If max_trial_index is provided, only trials up to that index are included.
        This lets the script regenerate a cumulative/session mean overlay after
        each trial instead of waiting until the end of acquisition.

        condition: None (default) reproduces today's behavior exactly, including
            every output filename. "stim" or "catch" additionally restricts the
            included trials to that trial_metadata.json's trial_condition, and
            appends "_{condition}" to every output filename so a "stim" call and
            a "catch" call against the same session never clobber each other.
        """
        try:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            print(f"Session analysis skipped: missing dependency: {exc}")
            return

        session_analysis_dir = self.session_dir / "session_analysis"
        session_analysis_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{condition}" if condition else ""
        label = f" ({condition})" if condition else ""

        if max_trial_index is None:
            max_trial_index = self.cfg.num_trials
        max_trial_index = max(1, min(int(max_trial_index), int(self.cfg.num_trials)))

        trial_maps: list[np.ndarray] = []
        display_maps: list[np.ndarray] = []
        green_refs: list[np.ndarray] = []
        included_trials: list[int] = []
        summaries: list[dict] = []

        for trial_index in range(1, max_trial_index + 1):
            analysis_dir = self.session_dir / f"trial_{trial_index:03d}" / "analysis"
            summary_path = analysis_dir / "analysis_summary.json"
            if not summary_path.exists():
                continue

            if condition is not None:
                meta_path = self.session_dir / f"trial_{trial_index:03d}" / "meta" / "trial_metadata.json"
                try:
                    trial_cond = json.loads(meta_path.read_text(encoding="utf-8")).get("trial_condition")
                except (OSError, ValueError):
                    trial_cond = None
                if trial_cond != condition:
                    continue

            try:
                with open(summary_path, "r", encoding="utf-8") as fp:
                    summary = json.load(fp)

                method = normalize_analysis_method(
                    summary.get("analysis_method", self.cfg.analysis_method)
                )
                candidates = _session_map_candidates(analysis_dir, method)
                map_path = next((c for c in candidates if c.exists()), candidates[0])

                display_path = analysis_dir / "activation_map_display_signal.npy"
                green_path = analysis_dir / "green_reference.npy"

                if not map_path.exists() or not display_path.exists() or not green_path.exists():
                    print(f"Skipping trial {trial_index} in session analysis: missing analysis arrays.")
                    continue

                trial_maps.append(np.load(map_path).astype(np.float64))
                display_maps.append(np.load(display_path).astype(np.float64))
                green_refs.append(np.load(green_path).astype(np.float64))
                included_trials.append(trial_index)
                summaries.append(summary)
            except Exception as exc:
                print(f"Skipping trial {trial_index} in session analysis: {exc}")

        if not trial_maps:
            print(f"Session analysis{label} skipped: no trial maps available.")
            return

        mean_map = np.nanmean(np.stack(trial_maps, axis=0), axis=0)
        mean_display = np.nanmean(np.stack(display_maps, axis=0), axis=0)
        mean_green = np.nanmean(np.stack(green_refs, axis=0), axis=0)

        # Re-center after averaging to remove any remaining global offset.
        finite_mean = np.isfinite(mean_map)
        if np.any(finite_mean):
            mean_map = mean_map - float(np.nanmedian(mean_map[finite_mean]))
        finite_display = np.isfinite(mean_display)
        if np.any(finite_display):
            mean_display = mean_display - float(np.nanmedian(mean_display[finite_display]))

        np.save(session_analysis_dir / f"session_mean_activation_map{suffix}.npy", mean_map)
        np.save(session_analysis_dir / f"session_mean_display_signal{suffix}.npy", mean_display)
        np.save(session_analysis_dir / f"session_mean_green_reference{suffix}.npy", mean_green)
        np.save(session_analysis_dir / f"included_trials{suffix}.npy", np.array(included_trials, dtype=int))

        try:
            from PIL import Image
            rescaled = self._rescale_to_bit_depth(mean_map, bit_depth=int(self.cfg.rescale_bit_depth), mode=self.cfg.rescale_mode)
            Image.fromarray(rescaled).save(session_analysis_dir / f"session_mean_activation_rescaled_{self.cfg.rescale_bit_depth}bit{suffix}.tiff")
        except Exception as exc:
            print(f"Warning: could not save session rescaled TIFF{label}: {exc}")

        finite_display = np.isfinite(mean_display)
        abs_lim = float(np.nanpercentile(np.abs(mean_display[finite_display]), 99)) if np.any(finite_display) else 1.0
        if not np.isfinite(abs_lim) or abs_lim <= 0:
            abs_lim = 1.0

        plt.figure(figsize=(8, 8))
        # session mean activation map
        im = plt.imshow(mean_display, cmap="gray", vmin=-abs_lim, vmax=abs_lim)
        plt.colorbar(im, fraction=0.046, pad=0.04, label="session mean display signal")
        plt.title(f"Session mean intrinsic signal{label}, n={len(included_trials)} trials")
        plt.axis("off")
        plt.tight_layout()
        activation_map_path = session_analysis_dir / f"session_mean_activation_map{suffix}.png"
        plt.savefig(activation_map_path, dpi=200)
        cumulative_map_path = session_analysis_dir / f"session_mean_activation_map{suffix}_after_trial_{max_trial_index:03d}.png"
        plt.savefig(cumulative_map_path, dpi=200)
        plt.close()

        plt.figure(figsize=(8, 8))
        # session mean overlay
        plt.imshow(self._display_scale_for_analysis(mean_green), cmap="gray")
        im = plt.imshow(mean_display, cmap="gray", alpha=0.55, vmin=-abs_lim, vmax=abs_lim)
        plt.colorbar(im, fraction=0.046, pad=0.04, label="session mean display signal")
        plt.title(f"Session mean overlay on green reference{label}")
        plt.axis("off")
        plt.tight_layout()
        overlay_path = session_analysis_dir / f"session_mean_activation_overlay{suffix}.png"
        plt.savefig(overlay_path, dpi=200)

        # Also save a trial-indexed snapshot so the cumulative overlay after
        # each trial is preserved instead of overwritten by the next trial.
        cumulative_overlay_path = session_analysis_dir / f"session_mean_activation_overlay{suffix}_after_trial_{max_trial_index:03d}.png"
        plt.savefig(cumulative_overlay_path, dpi=200)
        plt.close()

        # Trial summary CSV: useful for finding bad/outlier trials.
        trial_summary_path = session_analysis_dir / f"trial_analysis_summary{suffix}.csv"
        fieldnames = [
            "trial_index",
            "analysis_method",
            "peak_display_signal",
            "peak_quantitative_signal",
            "post_frames_available",
            "post_frames_used_for_analysis",
            "median_subtracted_value",
        ]
        with open(trial_summary_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for summary in summaries:
                writer.writerow({name: summary.get(name, "") for name in fieldnames})

        session_summary = {
            "condition": condition,
            "included_trials": included_trials,
            "num_trials_included": len(included_trials),
            "max_trial_index_requested": max_trial_index,
            "analysis_method": self.cfg.analysis_method,
            "rescale_mode": self.cfg.rescale_mode,
            "peak_session_display_signal": float(np.nanmax(mean_display)) if np.any(finite_display) else None,
            "session_analysis_dir": str(session_analysis_dir),
            "latest_overlay_path": str(overlay_path),
            "cumulative_overlay_snapshot_path": str(cumulative_overlay_path),
            "notes": "Average of available trial-level processed intrinsic maps through max_trial_index_requested. Use the final cumulative map for primary localization, not individual noisy trials. Centroid detection is disabled in this version.",
        }
        with open(session_analysis_dir / f"session_analysis_summary{suffix}.json", "w", encoding="utf-8") as fp:
            json.dump(session_summary, fp, indent=2, default=str)

        if open_overlay and self.cfg.open_session_overlay:
            try:
                os.startfile(overlay_path)
            except Exception as exc:
                print(f"Warning: could not open session overlay{label}: {exc}")

        print(
            f"Cumulative session analysis{label} complete through trial {max_trial_index}: "
            f"n={len(included_trials)} trials. Saved overlay to {overlay_path}"
        )

    def _capture_session_green_reference(self, session_green_dir: Path) -> int:
        """Capture the one-time session green reference stack. Returns frame count.

        Extracted so it can also run standalone (via --green-reference-only,
        its own camera/serial session) as the first stage of a workflow that
        pauses for a live-feed refocus before red calibration. See main().
        """
        green_exposure_us = (
            float(self.cfg.green_exposure_us)
            if self.cfg.green_exposure_us is not None
            else float(self.cfg.exposure_us)
        )
        if self.cfg.green_exposure_us is not None:
            self._set_float("ExposureTime", green_exposure_us)
            print(f"Set camera exposure for green reference: {green_exposure_us:.3f} us")
        else:
            print(
                "No separate green exposure provided; using red/trial exposure "
                f"for green reference: {green_exposure_us:.3f} us"
            )

        print(f"Requesting session green reference stack: {self.cfg.green_frames} frames...")
        self._send_arduino_command(self.cfg.session_green_cmd)

        marker = self._wait_for_marker(
            {"SESSION_GREEN_REFERENCE_START", "GREEN_BASELINE_START"},
            self.cfg.marker_timeout_s,
            trial_index=0,
        )
        session_markers_seen = [marker]
        green_count = 0
        green_end_seen = False

        while True:
            for marker in self._poll_all_markers(session_markers_seen, trial_index=0):
                if marker == "SESSION_GREEN_REFERENCE_END":
                    green_end_seen = True

            image = None
            try:
                image = self.cam.GetNextImage(100)
            except PySpin.SpinnakerException:
                image = None

            if image is None:
                if green_count >= self.cfg.green_frames and green_end_seen:
                    break
                continue

            try:
                if image.IsIncomplete():
                    continue

                host_ts = datetime.now().isoformat(timespec="milliseconds")
                cam_ts = int(image.GetTimeStamp())
                if green_count < self.cfg.green_frames:
                    green_count += 1
                    filename = f"green_{green_count:05d}.{self.cfg.save_format.lower()}"
                    self._save_image(image, session_green_dir / filename)
                else:
                    filename = ""
                if self._csv_writer is not None:
                    self._csv_writer.writerow([
                        0,
                        "session_green_collect" if filename else "session_green_extra",
                        green_count if filename else "",
                        self._global_frame_count,
                        host_ts,
                        cam_ts,
                        filename,
                        session_markers_seen[-1] if session_markers_seen else "",
                        "",
                        "",
                    ])
                self._global_frame_count += 1
            finally:
                image.Release()

            if green_count >= self.cfg.green_frames and green_end_seen:
                break

        # Flushed once here rather than per frame: a per-frame flush is a
        # synchronous disk round trip inside a 10 Hz acquisition loop.
        self._drain_frame_writer()
        if self._csv_fp is not None:
            self._csv_fp.flush()

        print(f"Session green reference complete: {green_count} frames saved to {session_green_dir}")

        if self.cfg.green_exposure_us is not None:
            self._set_float("ExposureTime", float(self.cfg.exposure_us))
            print(f"Restored camera exposure for red trials: {float(self.cfg.exposure_us):.3f} us")

        return green_count

    def _count_precaptured_green_reference(self, session_green_dir: Path) -> int:
        """Count frames already saved by an earlier --green-reference-only run."""
        ext = self.cfg.save_format.lower()
        existing = sorted(session_green_dir.glob(f"green_*.{ext}"))
        count = len(existing)
        if count == 0:
            raise RuntimeError(
                f"--precaptured-green-reference set but no green_*.{ext} frames found "
                f"in {session_green_dir}. Run --green-reference-only first."
            )
        if count != self.cfg.green_frames:
            print(
                f"Warning: expected {self.cfg.green_frames} pre-captured green reference "
                f"frames in {session_green_dir}, found {count}. Continuing with what's there."
            )
        return count

    def run(self) -> None:
        """Run session-green intrinsic imaging protocol.

        Session structure:
            1) Acquire one green anatomical/reference stack (or reuse one
               already captured by an earlier --green-reference-only run).
            2) Turn red illumination on and wait for stabilization.
            3) Run repeated trials using red illumination only:
               baseline -> gap/stim onset -> post/stimulation.
            4) Keep red illumination on until all trials finish or cleanup runs.
        """
        self.start_acquisition()

        session_green_dir = self.session_dir / "session_green_reference"
        session_green_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving session to: {self.session_dir}")
        print(f"Listening on serial port {self.cfg.serial_port} at {self.cfg.serial_baud} baud")
        print("Session-green protocol enabled:")
        print(f"  1) Acquire {self.cfg.green_frames} green reference frames once.")
        print(f"  2) Turn red LED on and wait {self.cfg.red_stabilization_s:.1f} s.")
        print(f"  3) Run {self.cfg.num_trials} red-illumination trials.")
        print("Press Ctrl+C to stop early.")

        try:
            # -------------------------------------------------------------
            # 1) Single session-level green reference stack
            # -------------------------------------------------------------
            if self.cfg.precaptured_green_reference:
                green_count = self._count_precaptured_green_reference(session_green_dir)
                print(
                    f"Using pre-captured session green reference: {green_count} frames "
                    f"already in {session_green_dir} (skipping capture)."
                )
            else:
                green_count = self._capture_session_green_reference(session_green_dir)

            # -------------------------------------------------------------
            # 2) Turn red illumination on once and wait for stabilization
            # -------------------------------------------------------------
            print("Turning red illumination on for session hold...")
            self._send_arduino_command(self.cfg.red_on_cmd)

            try:
                self._wait_for_marker(
                    {"RED_STABILIZATION_START", "RED_ON"},
                    min(5.0, self.cfg.marker_timeout_s),
                    trial_index=0,
                )
            except TimeoutError:
                print(
                    "Warning: did not receive RED_STABILIZATION_START/RED_ON marker. "
                    "Continuing with stabilization wait anyway."
                )

            print(f"Waiting {self.cfg.red_stabilization_s:.1f} s for red illumination stabilization...")
            self._wait_with_marker_polling(
                self.cfg.red_stabilization_s,
                trial_index=0,
                label="Red trials start in",
                skip_signal_path=self.session_dir / self.SKIP_RED_STABILIZATION_FILENAME,
                settling_sample_interval_s=self.cfg.red_settling_sample_interval_s,
            )
            print("Red stabilization wait complete.")

            # -------------------------------------------------------------
            # 3) Red-only trials: baseline -> gap/stim -> post
            # -------------------------------------------------------------
            consecutive_empty_trials = 0
            for trial_index in range(1, self.cfg.num_trials + 1):
                green_dir, baseline_dir, gap_dir, post_dir, meta_dir, analysis_dir = self._make_trial_dirs(trial_index)
                # The trial-level green directory is intentionally unused in this
                # protocol. Analysis uses the session-level green reference.
                baseline_count = 0
                gap_count = 0
                post_count = 0
                markers_seen: list[str] = []
                last_marker = ""
                trial_start_wall = 0.0
                trial_end_wall = 0.0

                # Trigger-index bookkeeping. trigger_index counts camera trigger
                # pulses within this trial, matching the Arduino's own counter,
                # so a frame's phase depends on the trigger that produced it
                # rather than on how far behind the consumer had fallen.
                self._reset_marker_tracking()
                trigger_index = 0
                prev_cam_ts: Optional[int] = None
                trial_dropped_total = 0
                dropped_frames: list[dict] = []

                red_baseline_requested = False
                gap_requested = False
                post_requested = False
                trailing_frames_requested = False
                trial_end_requested = False
                phase = "idle"
                last_post_frame_time: Optional[float] = None
                trial_end_request_time: Optional[float] = None
                stim_running = False
                # Per-trial condition: True unless an explicit --trial-conditions
                # sequence says otherwise, which reproduces today's always-on
                # behavior when the feature is unused.
                trial_is_stim = (
                    self.cfg.trial_conditions[trial_index - 1]
                    if self.cfg.trial_conditions is not None else True
                )
                trial_stim_active = self.cfg.visual_stim and trial_is_stim
                trial_stim_orientation = self._stim_orientation_for_trial(trial_index) if trial_stim_active else None
                trial_visual_stim_metadata = {
                    "enabled": bool(trial_stim_active),
                    "host": self.cfg.stim_host if trial_stim_active else None,
                    "port": int(self.cfg.stim_port) if trial_stim_active else None,
                    "orientation_deg": trial_stim_orientation,
                    # Ceiling only: this is the duration handed to the stim server,
                    # which is how long the grating would run if nothing stopped it.
                    # The host cuts it short by sending BLACK on STIM_END, so the
                    # stimulus that actually happened is measured_duration_s.
                    "duration_s": float(self.cfg.stim_duration_s) if trial_stim_active else None,
                    "duration_s_is_ceiling": True,
                    "measured_duration_s": None,
                    "measured_duration_source": None,
                    "triggered_by_marker": "STIM_START" if trial_stim_active else None,
                    "off_markers": ["STIM_END", "TRIAL_END", "CYCLE_COMPLETE"] if trial_stim_active else [],
                    "session_green_reference_dir": str(session_green_dir),
                    "red_led_session_hold": True,
                    "red_stabilization_s": float(self.cfg.red_stabilization_s),
                }

                if self.cfg.trial_conditions is not None:
                    trial_start_cmd = f"{self.cfg.trial_start_cmd},{1 if trial_is_stim else 0}"
                else:
                    trial_start_cmd = self.cfg.trial_start_cmd
                print(f"Sending {trial_start_cmd} for trial {trial_index}/{self.cfg.num_trials}...")
                self._send_arduino_command(trial_start_cmd)

                marker = self._wait_for_marker(
                    {"TRIAL_START", "RED_BASELINE_START"},
                    self.cfg.marker_timeout_s,
                    trial_index=trial_index,
                )
                markers_seen.append(marker)
                last_marker = marker
                trial_start_wall = time.time()

                if marker == "RED_BASELINE_START":
                    red_baseline_requested = True
                    phase = "baseline_collect"
                else:
                    phase = "idle"

                while True:
                    # Grab the frame BEFORE reading markers. The Arduino emits a
                    # phase-boundary marker at the same instant it fires that
                    # phase's first trigger, so those bytes only reach the host
                    # while GetNextImage() is already blocking on that frame.
                    # Polling first reads the port as it stood a frame ago, which
                    # puts every boundary one trigger late even with an empty
                    # queue -- on top of the backlog-sized error.
                    image = None
                    try:
                        image = self.cam.GetNextImage(100)
                    except PySpin.SpinnakerException:
                        image = None

                    for marker in self._poll_all_markers(markers_seen, trial_index):
                        last_marker = marker
                        if marker == "RED_BASELINE_START":
                            red_baseline_requested = True
                        elif marker == "GAP_START":
                            gap_requested = True
                        elif marker == "STIM_START":
                            if trial_stim_active and not stim_running:
                                self._send_visual_stim_command(
                                    "STIM",
                                    orientation_deg=trial_stim_orientation,
                                    duration_s=self.cfg.stim_duration_s,
                                    trial_index=trial_index,
                                )
                                stim_running = True
                        elif marker == "POST_START":
                            post_requested = True
                        elif marker == "POST_TRAILING_START":
                            trailing_frames_requested = True
                        elif marker == "STIM_END":
                            if self.cfg.visual_stim and stim_running:
                                self._send_visual_stim_command("BLACK")
                                stim_running = False
                        elif marker in {"TRIAL_END", "CYCLE_COMPLETE"}:
                            if self.cfg.visual_stim and stim_running:
                                self._send_visual_stim_command("BLACK")
                                stim_running = False
                            trial_end_requested = True
                            if trial_end_request_time is None:
                                trial_end_request_time = time.time()

                    # Each frame's phase label comes from its own trigger index
                    # (see below), so once the firmware reports trigger indices
                    # this state machine only has to follow the markers. The
                    # count-closed baseline condition is kept for legacy firmware
                    # but must not apply here: a single dropped baseline frame
                    # would hold baseline_count one short forever, the phase would
                    # never reach post_collect, and the trial would never exit.
                    use_trigger_labels = self._has_phase_boundaries()

                    if phase == "idle" and red_baseline_requested:
                        phase = "baseline_collect"
                    if (
                        phase == "baseline_collect"
                        and gap_requested
                        and (use_trigger_labels or baseline_count >= self.cfg.baseline_frames)
                    ):
                        phase = "gap"
                    if phase == "gap" and post_requested:
                        phase = "post_collect"

                    # Decided rather than broken out of directly: the frame grabbed
                    # above still has to be released before leaving the loop.
                    should_end = False
                    if phase == "post_collect" and trial_end_requested:
                        if post_count >= self.cfg.post_frames:
                            should_end = True
                        elif last_post_frame_time is not None and (time.time() - last_post_frame_time) > self.cfg.post_idle_timeout_s:
                            print(
                                f"Warning: post frame target not reached before idle timeout "
                                f"(got {post_count}/{self.cfg.post_frames})."
                            )
                            should_end = True
                        elif trial_end_request_time is not None and last_post_frame_time is None and (time.time() - trial_end_request_time) > self.cfg.post_idle_timeout_s:
                            print(
                                f"Warning: no post frames arrived after trial end request "
                                f"(got {post_count}/{self.cfg.post_frames})."
                            )
                            should_end = True

                    if should_end:
                        if image is not None:
                            image.Release()
                        break

                    if image is None:
                        continue

                    try:
                        if image.IsIncomplete():
                            continue

                        host_ts = datetime.now().isoformat(timespec="milliseconds")
                        cam_ts = int(image.GetTimeStamp())

                        trigger_index, dropped_before = self._update_trigger_index(
                            trigger_index, cam_ts, prev_cam_ts
                        )
                        prev_cam_ts = cam_ts

                        out_dir = None
                        frame_idx = None
                        filename = ""
                        ext = self.cfg.save_format.lower()

                        # Phase comes from the frame's own trigger index against
                        # the Arduino's marker boundaries. The live `phase`
                        # variable still drives loop control, but it flips on the
                        # next *dequeued* frame, so using it as a label made the
                        # error scale with acquisition backlog.
                        if use_trigger_labels:
                            frame_phase, idx_in_phase = self._phase_from_trigger_index(trigger_index)
                        else:
                            # Firmware without trigger indices: fall back to the
                            # legacy behaviour rather than dropping every frame.
                            frame_phase, idx_in_phase = phase, None

                        phase_for_log = frame_phase

                        if frame_phase == "baseline_collect":
                            idx = idx_in_phase if idx_in_phase is not None else baseline_count + 1
                            if idx <= self.cfg.baseline_frames:
                                baseline_count += 1
                                out_dir = baseline_dir
                                frame_idx = idx
                                filename = f"baseline_{frame_idx:05d}.{ext}"
                            else:
                                phase_for_log = "baseline_full"
                        elif frame_phase == "gap":
                            gap_count += 1
                            idx = idx_in_phase if idx_in_phase is not None else gap_count
                            if self.cfg.save_gap_frames:
                                out_dir = gap_dir
                                frame_idx = idx
                                filename = f"gap_{frame_idx:05d}.{ext}"
                        elif frame_phase == "post_collect":
                            idx = idx_in_phase if idx_in_phase is not None else post_count + 1
                            if idx <= self.cfg.post_frames:
                                post_count += 1
                                out_dir = post_dir
                                frame_idx = idx
                                filename = f"post_{frame_idx:05d}.{ext}"
                                last_post_frame_time = time.time()
                            elif use_trigger_labels:
                                # Overflow inside the post block: the Arduino sends
                                # postFrames triggers but only the first
                                # --post-frames are analysis data. Camera still
                                # triggers continuously through here (acquisition
                                # stays gap-free); nothing past the target is
                                # written. Kept distinct from post_trailing so the
                                # POST_TRAILING_START boundary stays checkable.
                                phase_for_log = "post_full"
                            elif trailing_frames_requested:
                                phase_for_log = "post_trailing"
                            else:
                                phase_for_log = "post_full"
                        elif frame_phase == "post_trailing":
                            phase_for_log = "post_trailing"
                        else:
                            phase_for_log = "idle"

                        if dropped_before:
                            trial_dropped_total += dropped_before
                            dropped_frames.append({
                                "trigger_index": trigger_index,
                                "dropped_before": int(dropped_before),
                                "phase": phase_for_log,
                                "frame_index_in_phase": frame_idx,
                                "host_timestamp_iso": host_ts,
                            })
                            print(
                                f"Warning: trial {trial_index} missed {dropped_before} trigger(s) "
                                f"before trigger_index={trigger_index} (phase {phase_for_log})."
                            )

                        if out_dir is not None and frame_idx is not None:
                            self._save_image(image, out_dir / filename)

                        if self._csv_writer is not None:
                            self._csv_writer.writerow([
                                trial_index,
                                phase_for_log,
                                frame_idx if frame_idx is not None else "",
                                self._global_frame_count,
                                host_ts,
                                cam_ts,
                                filename,
                                last_marker,
                                trigger_index,
                                dropped_before,
                            ])
                        self._global_frame_count += 1
                    finally:
                        image.Release()

                # Drains any frames still queued in the camera's buffer after TRIAL_END so
                # they don't bleed into the next trial's baseline. Post-collection is already
                # done at this point, so nothing here is saved (see the "stop saving after the
                # 40th post frame" policy above) -- this is a pure buffer flush.
                drain_deadline = time.time() + self.cfg.trailing_timeout_s
                while time.time() < drain_deadline:
                    self._poll_all_markers(markers_seen, trial_index)
                    image = None
                    try:
                        image = self.cam.GetNextImage(20)
                    except PySpin.SpinnakerException:
                        image = None
                    if image is None:
                        continue
                    image.Release()

                if self.cfg.visual_stim and stim_running:
                    self._send_visual_stim_command("BLACK")
                    stim_running = False

                # Frame writes and the CSV are flushed once per trial instead of
                # once per frame. Everything must be on disk before the metadata
                # that describes it is written.
                self._drain_frame_writer()
                if self._csv_fp is not None:
                    self._csv_fp.flush()

                phase_boundaries = {
                    name: self._boundary_trigger_index(*aliases)
                    for name, aliases in self.PHASE_SEGMENTS
                }
                phase_boundaries["stim_start"] = self._boundary_trigger_index("STIM_START")
                phase_boundaries["stim_end"] = self._boundary_trigger_index("STIM_END")
                phase_boundaries["trial_end"] = self._boundary_trigger_index("TRIAL_END")

                # Record the stimulus that actually ran. stim_duration_s is only
                # the ceiling handed to the stim server; the host cuts the grating at
                # STIM_END, so the two differ by design.
                stim_start_ms = self._marker_arduino_millis.get("STIM_START")
                stim_end_ms = self._marker_arduino_millis.get("STIM_END")
                if stim_start_ms is not None and stim_end_ms is not None:
                    trial_visual_stim_metadata["measured_duration_s"] = (
                        (stim_end_ms - stim_start_ms) / 1000.0
                    )
                    trial_visual_stim_metadata["measured_duration_source"] = (
                        "arduino_millis(STIM_END - STIM_START)"
                    )

                trial_end_wall = time.time()
                self._write_trial_metadata(
                    trial_index, session_green_dir, baseline_dir, gap_dir, post_dir, meta_dir,
                    trial_start_wall, trial_end_wall,
                    green_count, baseline_count, gap_count, post_count, markers_seen,
                    visual_stim_metadata=trial_visual_stim_metadata,
                    trial_condition="stim" if trial_is_stim else "catch",
                    dropped_frame_count=trial_dropped_total,
                    dropped_frames=dropped_frames,
                    phase_boundary_trigger_indices=phase_boundaries,
                    trigger_labeling=self._has_phase_boundaries(),
                )
                self._write_trial_conditions_manifest(trial_index)
                print(
                    f"Trial {trial_index} complete: session_green={green_count}, "
                    f"baseline={baseline_count}, gap={gap_count}, post={post_count}"
                )
                if trial_dropped_total:
                    locations = ", ".join(
                        f"{d['phase']}@trigger {d['trigger_index']} (-{d['dropped_before']})"
                        for d in dropped_frames
                    )
                    print(
                        f"Trial {trial_index} DROPPED {trial_dropped_total} frame(s): {locations}"
                    )
                else:
                    print(f"Trial {trial_index} dropped frames: 0")
                if self._writer_queue_peak:
                    print(f"Trial {trial_index} peak frame-writer backlog: {self._writer_queue_peak} frames")
                self._writer_queue_peak = 0

                # A trial that captured zero baseline AND zero post frames --
                # not a partial drop, a total absence -- while the Arduino
                # markers above show it ran its full timed cycle means the
                # camera has stopped delivering frames to the host entirely.
                # That is a USB3/driver-level stream stall this script has no
                # way to see or recover from, not a bug in this loop. Two in
                # a row is a hard signal every remaining trial will be empty
                # too (observed on real hardware: it did not self-recover
                # across 11 further trials), so abort here rather than
                # silently burning through the rest of the session producing
                # no data. Breaking (not raising) lets the normal end-of-run
                # analysis and the finally-block teardown (lights off) below
                # run exactly as they would on a clean finish.
                if baseline_count == 0 and post_count == 0:
                    consecutive_empty_trials += 1
                else:
                    consecutive_empty_trials = 0

                if consecutive_empty_trials >= 2:
                    print(
                        f"ERROR: {consecutive_empty_trials} consecutive trials (ending at "
                        f"trial {trial_index}) captured zero frames even though the Arduino "
                        f"ran its full trigger cycle each time. The camera has stopped "
                        f"delivering frames to the host -- likely a USB3/driver-level stream "
                        f"stall. Aborting the remaining {self.cfg.num_trials - trial_index} "
                        f"planned trial(s) to avoid further data loss. Check the camera's USB "
                        f"connection (cable/port/hub) and start a new session."
                    )
                    break

                if self.cfg.run_analysis:
                    self._run_trial_analysis(trial_index, session_green_dir, baseline_dir, post_dir, analysis_dir)

                    if self.cfg.run_session_analysis:
                        if self.cfg.trial_conditions is not None:
                            # Interleaved session: pooling stim+catch into one
                            # running average would defeat the point of
                            # interleaving, so compute both separately instead
                            # of the single unfiltered call. Only the condition
                            # matching *this* trial auto-opens (if configured),
                            # so a lab member doesn't get two overlay windows
                            # popping up every trial.
                            matching = "stim" if trial_is_stim else "catch"
                            other = "catch" if trial_is_stim else "stim"
                            self._run_session_analysis(
                                max_trial_index=trial_index,
                                open_overlay=self.cfg.open_trial_overlays,
                                condition=matching,
                            )
                            self._run_session_analysis(
                                max_trial_index=trial_index,
                                open_overlay=False,
                                condition=other,
                            )
                        else:
                            self._run_session_analysis(
                                max_trial_index=trial_index,
                                open_overlay=self.cfg.open_trial_overlays,
                            )
                        self._save_trial_vs_running_average_panel(trial_index)

                if trial_index < self.cfg.num_trials:
                    self._wait_minimum_inter_trial_interval(
                        trial_index=trial_index,
                        interval_start_wall=trial_end_wall,
                    )

            if (
                self.cfg.run_analysis
                and self.cfg.run_session_analysis
                and self.cfg.open_session_overlay
                and not self.cfg.open_trial_overlays
            ):
                if self.cfg.trial_conditions is not None:
                    # One-time final call, not per-trial -- open both overlays
                    # for a final side-by-side review, no popup-spam concern.
                    self._run_session_analysis(max_trial_index=self.cfg.num_trials, open_overlay=True, condition="stim")
                    self._run_session_analysis(max_trial_index=self.cfg.num_trials, open_overlay=True, condition="catch")
                else:
                    self._run_session_analysis(max_trial_index=self.cfg.num_trials, open_overlay=True)

        finally:
            # Leave red illumination on only during the experiment. On normal
            # completion, Ctrl+C, or an error, command the Arduino to shut down.
            if self.cfg.visual_stim:
                try:
                    self._send_visual_stim_command("BLACK")
                except Exception:
                    pass
            self._send_lights_off_if_needed()


    def _get_enum_symbolic(self, name: str) -> str:
        node = PySpin.CEnumerationPtr(self._get_node(name))
        entry = node.GetCurrentEntry()
        if entry is None or not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
            return "UNKNOWN"
        return entry.GetSymbolic()

    def _get_float_value(self, name: str) -> float:
        node = PySpin.CFloatPtr(self._get_node(name))
        if not PySpin.IsReadable(node):
            return float("nan")
        return float(node.GetValue())

    def _get_int_value(self, name: str) -> int:
        node = PySpin.CIntegerPtr(self._get_node(name))
        if not PySpin.IsReadable(node):
            return -1
        return int(node.GetValue())

    def _print_camera_settings(self) -> None:
        print("Camera settings:")
        print(f"  PixelFormat: {self._get_enum_symbolic('PixelFormat')}")
        print(f"  Width:       {self._get_int_value('Width')}")
        print(f"  Height:      {self._get_int_value('Height')}")
        try:
            print(f"  Binning H:   {self._get_int_value('BinningHorizontal')}")
            print(f"  Binning V:   {self._get_int_value('BinningVertical')}")
        except CameraConfigError:
            print("  Binning:     unavailable")
        try:
            print(f"  OffsetX:     {self._get_int_value('OffsetX')}")
            print(f"  OffsetY:     {self._get_int_value('OffsetY')}")
        except CameraConfigError:
            pass
        print(f"  Exposure:    {self._get_float_value('ExposureTime'):.2f} us")
        if self.cfg.green_exposure_us is not None:
            print(f"  Green exp.:  {float(self.cfg.green_exposure_us):.2f} us")

        try:
            print(f"  Gain:        {self._get_float_value('Gain'):.2f} dB")
        except CameraConfigError:
            print("  Gain:        unavailable")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blackfly triggered capture with Arduino serial sync")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--iti", type=float, default=20.0, help="Minimum inter-trial interval in seconds, measured from the end of one trial's acquisition period to the START_TRIAL command for the next trial. Analysis/processing time counts toward this interval.")
    parser.add_argument("--exposure-us", type=float, default=23000.0, help="Red/trial exposure time in microseconds.")
    parser.add_argument("--green-exposure-us", type=float, default=None, help="Optional separate exposure time in microseconds for the session green reference stack. If omitted, uses --exposure-us.")
    parser.add_argument("--gain-db", type=float, default=None)
    parser.add_argument("--pixel-format", type=str, default="Mono16", choices=["Mono8", "Mono12", "Mono16"])
    parser.add_argument("--binning", type=int, default=2, help="Software analysis binning factor. --binning 2 averages 2x2 pixel blocks during analysis only; saved TIFF/RAW frames remain full resolution.")
    parser.add_argument("--analysis-binning", type=int, default=None, help="Explicit software analysis binning factor. Overrides --binning when provided.")
    parser.add_argument("--camera-binning", type=int, default=1, help="Advanced: request symmetric camera hardware binning. Leave at 1 unless your camera exposes writable hardware binning.")
    parser.add_argument("--camera-binning-horizontal", type=int, default=None, help="Advanced: override horizontal camera hardware binning. Defaults to --camera-binning.")
    parser.add_argument("--camera-binning-vertical", type=int, default=None, help="Advanced: override vertical camera hardware binning. Defaults to --camera-binning.")
    parser.add_argument("--save-format", type=str, default="raw", choices=["png", "tiff", "raw"])
    parser.add_argument("--marker-timeout", type=float, default=30.0)
    parser.add_argument("--green-frames", type=int, default=30)
    parser.add_argument("--green-reference-trim-frames", type = int, default = 5, help="Number of initial green reference frames to discard before averaging, to allow LED and camera to stabilize. Only used if --green-frames is sufficiently large.")
    parser.add_argument("--baseline-frames", type=int, default=40)
    parser.add_argument("--post-frames", type=int, default=40)
    parser.add_argument("--trailing-timeout-s", type=float, default=2.0)
    parser.add_argument("--post-idle-timeout-s", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-gap-frames", action="store_true", help="Deprecated no-op: gap/debounce frames are saved by default now. Kept only for backward compatibility.")
    parser.add_argument("--no-save-gap-frames", action="store_true", help="Don't save frames captured during the inter-phase gap/debounce windows (baseline->post transition). The camera still triggers continuously through those windows; this only controls whether the frames are written to disk.")
    parser.add_argument("--analyze", action="store_true", help="Run intrinsic optical signal analysis after each trial.")
    parser.add_argument("--analysis-start-frame", type=int, default=5, help="0-indexed first post frame for activation averaging.")
    parser.add_argument("--analysis-end-frame", type=int, default=35, help="0-indexed exclusive end post frame for activation averaging. Use <=0 for all remaining post frames.")
    parser.add_argument("--analysis-smoothing-sigma", type=float, default=5.0, help="Gaussian smoothing sigma in pixels. Use 0 to disable.")
    parser.add_argument("--roi-config", type=Path, default=None, help="Path to selected_roi.json from the calibration script. If provided, the camera still captures full frame; analysis is cropped to this ROI in software.")
    parser.add_argument("--dark-reference", type=Path, default=None, help="Path to master dark reference .npy or .tiff. Subtracted during analysis before ROI cropping, software binning, and ΔR/R calculation.")
    parser.add_argument("--analysis-mask-percentile", type=float, default=20.0, help="Ignored; kept only for backward compatibility. ROI cropping is software-only.")
    parser.add_argument("--analysis-denominator-floor-percentile", type=float, default=5.0, help="For ΔR/R analysis, mask pixels whose dark-corrected baseline is below this percentile of positive baseline pixels.")
    parser.add_argument("--analysis-denominator-floor-counts", type=float, default=100.0, help="For ΔR/R analysis, also require dark-corrected baseline to be at least this many camera counts before division.")
    parser.add_argument("--analysis-median-filter-size", type=int, default=3, help="Odd spatial median-filter size applied to the ΔR/R map before Gaussian smoothing. Use 1 to disable.")
    parser.add_argument("--analysis-method", type=normalize_analysis_method, default="raw_counts", choices=["raw_counts", "fractional_reflectance"], help="Intrinsic signal processing method. 'raw_counts' uses post mean - baseline mean. 'fractional_reflectance' uses fractional reflectance with a denominator floor, median centering, smoothing, and display rescaling. Method names used by earlier versions are still accepted.")
    parser.add_argument("--no-session-analysis", action="store_true", help="Skip session-level averaging across trial maps.")
    parser.add_argument("--rescale-bit-depth", type=int, default=16, help="Bit depth used when exporting rescaled display TIFFs.")
    parser.add_argument("--rescale-mode", type=str, default="signed_symmetric", choices=["signed_symmetric", "minmax"], help="TIFF scaling mode. signed_symmetric maps zero to mid-gray; minmax maps finite min to 0 and max to 65535.")
    parser.add_argument("--invert-display-signal", action="store_true", help="Inverts the signal output for display; generally, do not use.")
    parser.add_argument("--no-save-analysis-arrays", action="store_true", help="Save PNG summaries only; skip .npy arrays.")
    parser.add_argument("--open-overlay", action="store_true", help="Open only the final session_mean_activation_overlay.png after all trials finish.")
    parser.add_argument("--open-trial-overlays", action="store_true", help="Auto-open the cumulative session_mean_activation_overlay.png after each completed trial, using all analyzed trials so far.")
    parser.add_argument("--visual-stim", action="store_true", help="Send UDP commands to the visual stimulus server at STIM_START/STIM_END markers.")
    parser.add_argument("--stim-host", type=str, default="127.0.0.1", help="IP/host running the visual stimulus server.")
    parser.add_argument("--stim-port", type=int, default=55000, help="UDP port used by the visual stimulus server.")
    parser.add_argument("--stim-orientations", type=str, default="45", help="Comma-separated grating orientations in degrees. Values cycle across trials, e.g. '45' or '45,135'.")
    parser.add_argument("--stim-duration-s", type=float, default=7.0, help="Ceiling, in seconds, for each drifting-grating command sent to the stimulus server. This is NOT the delivered stimulus duration: the host sends BLACK on the Arduino's STIM_END marker, which arrives at gapMs + stimActiveFrames*triggerPeriodMs (5.0 s with stock firmware). Keep this above that so the server never ends the grating first. The delivered duration is measured per trial and recorded as visual_stim_metadata.measured_duration_s.")
    parser.add_argument("--stream-buffer-count", type=int, default=64, help="Number of transport-layer stream buffers. The Spinnaker default (~11) is under two seconds of slack at 10 Hz, so any consumer hiccup overwrites undelivered frames.")
    parser.add_argument("--trigger-period-ms", type=float, default=100.0, help="Nominal camera trigger period in milliseconds. Must match the Arduino's triggerPeriodMs. Used to convert camera-timestamp gaps into a count of missed triggers.")
    parser.add_argument("--camera-timestamp-hz", type=float, default=1e9, help="Ticks per second in the camera's image timestamp (1e9 = nanoseconds, correct for the BFLY-U3-23S6M-C). A wrong value is detected at run time and disables dropped-frame detection rather than inventing drops.")
    parser.add_argument("--session-green-cmd", type=str, default="SESSION_GREEN_REFERENCE", help="Arduino command that acquires the one-time session green reference stack.")
    parser.add_argument("--red-on-cmd", type=str, default="SESSION_RED_ON", help="Arduino command that turns red illumination on for the whole session.")
    parser.add_argument("--trial-start-cmd", type=str, default="START_TRIAL", help="Arduino command that starts one red-only baseline/gap/post trial.")
    parser.add_argument("--lights-off-cmd", type=str, default="LIGHTS_OFF", help="Arduino cleanup command sent at end/error/Ctrl+C.")
    parser.add_argument("--red-stabilization-s", type=float, default=600.0, help="Seconds to wait after turning red illumination on before starting trials.")
    parser.add_argument("--red-settling-sample-interval-s", type=float, default=0.0, help="If > 0, every this-many seconds during red stabilization, trigger one frame and print its mean ROI brightness (and %% change from the first sample) for live LED-settling monitoring. 0 disables sampling.")
    parser.add_argument("--leave-lights-on-exit", action="store_true", help="Do not send the lights-off cleanup command when Python exits.")
    parser.add_argument(
        "--session-dir", type=Path, default=None,
        help="Reuse an existing session folder (e.g. one created by an earlier "
             "--green-reference-only run) instead of creating a new timestamped "
             "one. Lets a staged workflow span several separate process runs.",
    )
    parser.add_argument(
        "--green-reference-only", action="store_true",
        help="Capture just the session green reference stack, then exit "
             "(skip red stabilization and trials). Used to insert a live-feed "
             "refocus pause between green reference capture and red calibration.",
    )
    parser.add_argument(
        "--precaptured-green-reference", action="store_true",
        help="Skip capturing the session green reference in run(); reuse frames "
             "already saved in session_green_reference/ (from an earlier "
             "--green-reference-only run in the same --session-dir).",
    )
    return parser.parse_args(argv)

def install_break_handler() -> None:
    """
    Route Windows CTRL_BREAK_EVENT into the normal KeyboardInterrupt path.

    The GUI stops a subprocess with CTRL_BREAK_EVENT (see gui/script_runner.py).
    Python maps CTRL_C_EVENT to KeyboardInterrupt on its own, but SIGBREAK
    defaults to SIG_DFL, which lets the OS terminate the process outright with
    STATUS_CONTROL_C_EXIT (0xC000013A) - no except, no finally, and therefore no
    lights-off teardown. With this installed, KeyboardInterrupt is raised
    normally, the finally blocks run, and the LEDs are commanded off on the way
    out. Same handler intrinsic_visual_stimulus.py installs, for the same reason.
    """
    handler = getattr(signal, "SIGBREAK", None)
    if handler is not None:
        signal.signal(handler, signal.default_int_handler)


def main(argv: list[str]) -> int:
    install_break_handler()
    args = parse_args(argv)
    cfg = TrialConfig(
        num_trials=args.trials,
        inter_trial_s=args.iti,
        exposure_us=args.exposure_us,
        green_exposure_us=args.green_exposure_us,
        gain_db=args.gain_db,
        pixel_format=args.pixel_format,
        binning_horizontal=args.camera_binning_horizontal if args.camera_binning_horizontal is not None else args.camera_binning,
        binning_vertical=args.camera_binning_vertical if args.camera_binning_vertical is not None else args.camera_binning,
        analysis_binning=args.analysis_binning if args.analysis_binning is not None else args.binning,
        save_format=args.save_format,
        overwrite=args.overwrite,
        serial_port=args.port,
        serial_baud=args.baud,
        marker_timeout_s=args.marker_timeout,
        save_gap_frames=not args.no_save_gap_frames,
        green_frames=args.green_frames,
        green_reference_trim_frames=args.green_reference_trim_frames,
        baseline_frames=args.baseline_frames,
        post_frames=args.post_frames,
        trailing_timeout_s=args.trailing_timeout_s,
        post_idle_timeout_s=args.post_idle_timeout_s,
        run_analysis=args.analyze,
        analysis_start_frame=args.analysis_start_frame,
        analysis_end_frame=args.analysis_end_frame if args.analysis_end_frame > 0 else None,
        analysis_smoothing_sigma=args.analysis_smoothing_sigma,
        analysis_mask_percentile=args.analysis_mask_percentile,
        analysis_denominator_floor_percentile=args.analysis_denominator_floor_percentile,
        analysis_denominator_floor_counts=args.analysis_denominator_floor_counts,
        analysis_median_filter_size=args.analysis_median_filter_size,
        roi_config=args.roi_config,
        dark_reference_path=args.dark_reference,
        analysis_method=args.analysis_method,
        run_session_analysis=not args.no_session_analysis,
        rescale_bit_depth=args.rescale_bit_depth,
        rescale_mode=args.rescale_mode,
        invert_display_signal=args.invert_display_signal,
        save_analysis_arrays=not args.no_save_analysis_arrays,
        analyze=args.analyze,
        open_overlay=args.open_overlay,
        open_trial_overlays=args.open_trial_overlays,
        open_session_overlay=args.open_overlay,
        visual_stim=args.visual_stim,
        stim_host=args.stim_host,
        stim_port=args.stim_port,
        stim_orientations=args.stim_orientations,
        stim_duration_s=args.stim_duration_s,
        session_green_cmd=args.session_green_cmd,
        red_on_cmd=args.red_on_cmd,
        trial_start_cmd=args.trial_start_cmd,
        lights_off_cmd=args.lights_off_cmd,
        red_stabilization_s=args.red_stabilization_s,
        red_settling_sample_interval_s=args.red_settling_sample_interval_s,
        turn_lights_off_on_exit=not args.leave_lights_on_exit,
        precaptured_green_reference=args.precaptured_green_reference,
        stream_buffer_count=args.stream_buffer_count,
        trigger_period_ms=args.trigger_period_ms,
        camera_timestamp_hz=args.camera_timestamp_hz,
    )

    capture = BlackflyCapture(args.output, cfg, session_dir_override=args.session_dir)

    if args.green_reference_only:
        try:
            capture.setup()
            capture.start_acquisition()
            session_green_dir = capture.session_dir / "session_green_reference"
            session_green_dir.mkdir(parents=True, exist_ok=True)
            capture._capture_session_green_reference(session_green_dir)
            print(f"Session folder: {capture.session_dir}")
            print("Done.")
            return 0
        except KeyboardInterrupt:
            print("Interrupted by user.")
            return 130
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        finally:
            capture.teardown()

    try:
        capture.setup()
        capture.run()
        print("Done.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        capture.teardown()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))