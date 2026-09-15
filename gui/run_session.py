# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
RunSessionWidget — Run Session tab.

Guided Vertical Stepper (replaces the old single Start/Stop session). Six
steps, one active at a time; done steps collapse to a check + result, later
steps are locked. Two of the six are camera-feed checkpoints — no subprocess,
advanced with a "Continue" button — interleaved with the four launchable
stages:
    1. Focus on surface vasculature   — live_preview.py (feed checkpoint)
    2. Calibrate Green Exposure       — intrinsic_calibration.py --stage green
    3. Capture Green Reference        — intrinsic_imaging.py --green-reference-only
    4. Refocus below surface vasculature — live_preview.py (feed checkpoint)
    5. Calibrate Red Exposure + ROI   — intrinsic_calibration.py --stage red
    6. Start Trials                   — intrinsic_calibrated_imaging.py --final-stage

Each stage is a separate subprocess launch on the shared session ScriptRunner,
threading results forward via parsed stdout (exposure values, ROI path,
session folder) — the same pattern already used for "Daily output folder:".
Splitting the old one-shot flow this way is what allows a live-feed pause
between stages; see gui/diagnostics.py and CLAUDE.md for why a pause can't
live inside a single subprocess (no stdin channel back into a running script).

_STAGES stays the four launchable entries; _STEPPER_ITEMS is the six-item view
list. _refresh_stepper_visuals() is a pure view over the existing state
(_stage_status[key] "done" prop, _current_stage, _stage_failed, _feed_done).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui import presets as _presets
from gui.lights_controller import CAL_GREEN_ON_CMD, CAL_RED_ON_CMD, send_command_async
from gui.panel_image_view import PanelImageBrowser, PanelImageView
from gui.paths import project_root
from gui.port_indicator import PortIndicator
from gui.runner_bridge import RunnerBridge
from gui.safety_bar import SafetyBar
from gui.script_runner import ScriptRunner
from gui.settling_plot import SettlingPlotWidget
from gui.stim_probe import StimServerConfig, StimState, probe_config_async, probe_once
from gui.theme import (
    CHROME_ACCENT_HEX,
    DANGER_HEX,
    GREEN_HEX,
    HINT_STYLE,
    STYLE_START,
    STYLE_STOP,
    WARNING_STYLE,
    text_rgba,
)
from gui.trial_sequence import (
    compress_to_pattern,
    conditions_to_argv_value,
    format_sequence,
    generate_trial_conditions,
    parse_manual_pattern,
)

_PROJECT_ROOT = project_root()
_CALIBRATION_SCRIPT = str(_PROJECT_ROOT / "intrinsic_calibration.py")
_IMAGING_SCRIPT = str(_PROJECT_ROOT / "intrinsic_imaging.py")
_CALIBRATED_IMAGING_SCRIPT = str(_PROJECT_ROOT / "intrinsic_calibrated_imaging.py")
_LIVE_PREVIEW_SCRIPT = str(_PROJECT_ROOT / "live_preview.py")
# Peek never passes --shared-crop, so --roi-within-region would be a no-op
# here regardless; the peek run is bit-identical to the Analysis tab default.
_PEEK_ROI_SCRIPT = str(_PROJECT_ROOT / "statistical_analyses.py")
_PEEK_OUT_SUBFOLDER = "peek"
_GRACEFUL_STOP_TIMEOUT_MS = 15_000
_LIVE_FEED_EXPOSURE_US = 15_000.0
_LIVE_FEED_FPS = 15.0

# Must match BlackflyCapture.SKIP_RED_STABILIZATION_FILENAME in
# intrinsic_imaging.py. Not imported (the GUI never imports acquisition code)
# — this filename is a small file-based protocol between the two processes:
# the GUI touches this file inside the session folder while the red-warm-up
# wait is running, and the subprocess's wait loop polls for it and ends the
# wait early when it appears.
_SKIP_RED_STABILIZATION_FILENAME = "skip_red_stabilization.flag"

# Cadence approved for the live LED-settling monitor (see
# intrinsic_imaging.py's --red-settling-sample-interval-s).
_SETTLING_SAMPLE_INTERVAL_S = 10.0

# intrinsic_arduino.ino: gapMs + stimActiveFrames * triggerPeriodMs =
# 1000 + 40 * 100 ms. The host sends BLACK at STIM_END, so no visual stimulus
# outlasts this regardless of what the stim server was asked for.
_FIRMWARE_STIM_WINDOW_S = 5.0

# ── Trials-stage duration model ──────────────────────────────────────────────
# Plain literals derived by *reading* the acquisition scripts and the Arduino
# firmware — never importing them (the GUI stays free of PySpin / numpy<2).
# Every value here is an estimate; the UI always renders it with a leading "~".
#
# _HW_TRIAL_S — one triggered trial's fixed hardware time. intrinsic_imaging.py
#   TrialConfig.trigger_period_ms = 100.0 ("Must match the Arduino's
#   triggerPeriodMs"); the GUI never overrides it, so the cadence is always
#   1000/100 = 10 triggers/s. intrinsic_arduino.ino runRedOnlyTrial() /
#   runLraTrial() fire a compile-time-fixed 45 baseline + 4 guard + 10 stim
#   lead-in + 60 post + 4 trailing = 123 triggers -> 12.3 s. The GUI's
#   --baseline-frames / --post-frames (default 40/40) are Python save-and-exit
#   targets *below* those firmware counts and the trial loop only exits on the
#   firmware TRIAL_END marker, so lowering them does not shorten a trial.
_HW_TRIAL_S = 12.3
# intrinsic_imaging.py TrialConfig.trailing_timeout_s; the post-trial drain
# loop has no early break, so it is always paid in full.
_TRAILING_DRAIN_S = 2.0
# Writer-queue drain + CSV flush + trial metadata + conditions manifest +
# next-trial marker handshake (intrinsic_imaging.py run() per-trial tail).
# Disk-dependent (~0.1-1 s); 0.7 is a mid estimate.
_PER_TRIAL_IO_S = 0.7
# One raw_counts trial analysis + cumulative session pass + trial-vs-running-
# average panel (~5-15 s). Hidden inside the ITI for every trial except the
# last, because the ITI is measured from a wall clock captured *before*
# analysis and analysis time counts toward the interval
# (intrinsic_imaging.py _wait_minimum_inter_trial_interval).
_PER_TRIAL_ANALYSIS_S = 10.0
# One-time flat pad: camera setup() + serial open (sleep 2 s + <=5 s
# ARDUINO_READY) + CSV/socket open + <=5 s RED_ON marker wait + ~3 s teardown.
# Small next to a 600 s warm-up.
_SESSION_SETUP_S = 12.0
# Slack on top of the exact red-warm-up value for the RED_ON marker wait that
# precedes the stabilization countdown.
_WARMUP_MARKER_SLACK_S = 5.0
# intrinsic_imaging.py post_idle_timeout_s: paid per trial only when a
# Frames-tab override pushes --baseline-frames above 45 or --post-frames
# above 60 (defaults 40/40 stay below the firmware counts).
_POST_IDLE_TIMEOUT_S = 2.0
_FIRMWARE_BASELINE_FRAMES = 45
_FIRMWARE_POST_FRAMES = 60

# ── Log line patterns ──────────────────────────────────────────────────────────
_RE_DAILY               = re.compile(r"Daily output folder:\s*(.+)")
_RE_WARMUP_START        = re.compile(r"Waiting ([\d.]+) s for red illumination stabilization")
_RE_WARMUP_DONE         = re.compile(r"Red stabilization wait complete")
_RE_COUNTDOWN           = re.compile(r"Red trials start in:\s+(\d+):(\d+) remaining")
_RE_ITI_COUNTDOWN       = re.compile(r"Next trial starts in:\s+(\d+):(\d+) remaining")
_RE_SETTLING_SAMPLE     = re.compile(r"Red settling sample: t=([\d.]+)s mean=([\d.eE+\-]+) rel=([+\-][\d.]+)%")
_RE_TRIAL               = re.compile(r"Sending \S+ for trial (\d+)/(\d+)")
_RE_DONE                = re.compile(r"^Done\.$")
# Health-summary patterns — parsed in parallel during the trials stage.
_RE_TRIAL_COMPLETE      = re.compile(r"^Trial (\d+) complete:")
_RE_ANALYSIS_PEAK       = re.compile(
    r"Analysis complete for trial (\d+):.*peak_display=([\d.eE+\-]+)"
    r"(?:,\s*trough_display=([\d.eE+\-]+))?"
)
_RE_WARNING             = re.compile(r"^Warning:", re.IGNORECASE)
# Staged-workflow result patterns.
_RE_GREEN_CAL_DONE      = re.compile(r"green exposure_us = ([\d.]+)")
_RE_RED_CAL_DONE        = re.compile(r"red exposure_us = ([\d.]+)")
_RE_ROI_CONFIG          = re.compile(r"roi_config\s*= (.+)")
_RE_GREEN_REF_SESSION   = re.compile(r"Session folder:\s*(.+)")
# statistical_analyses.py's own final print("Done ->", out_dir).
_RE_PEEK_DONE           = re.compile(r"^Done -> (.+)$")
# intrinsic_imaging.py's _save_trial_vs_running_average_panel() final print.
# The trial index is pulled out of the path itself (trial_XXX/analysis/...)
# rather than adding a second print line on the script side.
_RE_PANEL_SAVED         = re.compile(r"^Saved post-trial summary image:\s*(.+[\\/]trial_(\d+)[\\/]analysis[\\/].+)$")

_STAGES = [
    ("green_cal", "Calibrate Green Exposure"),
    ("green_ref", "Capture Green Reference"),
    ("red_cal",   "Calibrate Red Exposure + ROI"),
    ("trials",    "Start Trials"),
]
_STAGE_TITLE = dict(_STAGES)

# ── Guided Vertical Stepper ─────────────────────────────────────────────────
# The panel renders these items top-to-bottom; the operator works exactly one
# at a time. Two are camera-feed checkpoints ("feed") — no subprocess, no
# result: the operator opens the live feed, checks focus, closes it, and
# clicks Continue. The rest are the launchable _STAGES entries ("stage").
# Step numbering is structural (no "N." baked into any label).
_FEED_SURFACE = "feed_surface"
_FEED_BELOW = "feed_below"
_STEPPER_ITEMS: list[tuple[str, str, str]] = [
    ("feed",  _FEED_SURFACE, "Focus on surface vasculature"),
    ("stage", "green_cal",   "Calibrate Green Exposure"),
    ("stage", "green_ref",   "Capture Green Reference"),
    ("feed",  _FEED_BELOW,   "Refocus below surface vasculature"),
    ("stage", "red_cal",     "Calibrate Red Exposure + ROI"),
    ("stage", "trials",      "Start Trials"),
]
_STEP_NUM = {item_id: i for i, (_k, item_id, _l) in enumerate(_STEPPER_ITEMS, start=1)}
_STEP_TITLE = {item_id: label for _k, item_id, label in _STEPPER_ITEMS}

# Plain "check" — already used in the health panel; swap to ASCII "v" if a
# deployment font ever lacks it.
_CHECK_GLYPH = "✓"

_STEP_DESC = {
    _FEED_SURFACE: "Turn on green light and open the live preview, position the "
                   "animal, and focus on the surface vasculature. Close the "
                   "preview and press Continue when the vessels look sharp.",
    "green_cal":   "Interactive — a matplotlib window opens for green exposure "
                   "selection. (Set a Green exposure override in Advanced "
                   "settings to skip this and run the next step immediately.)",
    "green_ref":   "Runs unattended — captures this session's green-reference "
                   "frames and creates the session folder. Watch the Stage "
                   "output log on the right.",
    _FEED_BELOW:   "Open the live preview again and refocus slightly below the "
                   "surface vasculature. Close the preview and press Continue "
                   "when ready.",
    "red_cal":     "Interactive — turns red illumination on, then opens windows "
                   "to pick the red exposure and draw the ROI / crop.",
    "trials":      "Red-LED warm-up, then the trial sequence — the long run. "
                   "The warm-up countdown and per-trial progress appear in the "
                   "Progress area above this panel.",
}
_STEP_FEED_PROMPT = {
    _FEED_SURFACE: "Open the preview, check focus, then close it:",
    _FEED_BELOW:   "Open the preview, refocus, then close it:",
}

# Arduino trial-start commands (see intrinsic_arduino.ino handleCommand()).
# START_TRIAL runs runRedOnlyTrial() (electrical-marker/no-stim trials, and
# visual trials — the UDP visual-stim calls ride on the same TTL-marker
# timing). STIM_LRA runs runLraTrial(), which mirrors that timing exactly but
# drives the AD9833 DDS + PAM8302A amp instead of the stimPin TTL.
_DEFAULT_TRIAL_START_CMD = "START_TRIAL"
_LRA_TRIAL_START_CMD = "STIM_LRA"


def _dated_output_folder(base: str) -> str:
    """Same convention as intrinsic_calibrated_imaging.py's dated_output_folder():
    captures/ -> captures_MMDDYYYY/. Deterministic per calendar day, so every
    stage's separate process lands in the same dated folder."""
    base_path = Path(base)
    today = datetime.now().strftime("%m%d%Y")
    if base_path.name.endswith(f"_{today}"):
        return str(base_path)
    return str(base_path.with_name(f"{base_path.name}_{today}"))


def estimate_trials_stage_seconds(
    *,
    trials: int,
    iti_s: float,
    warmup_s: float,
    analyze: bool,
    extra_per_trial_s: float = 0.0,
) -> float:
    """Pre-run wall-clock estimate (seconds) for the trials stage only: fixed
    session setup + red warm-up + N triggered trials + (N-1) ITIs. Pure — no
    Qt, no import of the acquisition scripts. Always an estimate; callers
    render it with a leading "~".

    Per-trial analysis is hidden inside the ITI for every trial except the
    last when the ITI can absorb it (>= _PER_TRIAL_ANALYSIS_S), so it is
    added once; when the ITI is shorter than one analysis pass it is paid
    every trial instead.
    """
    n = max(0, int(trials))
    if n < 1:
        return 0.0
    iti_s = max(0.0, float(iti_s))
    warmup_s = max(0.0, float(warmup_s))
    per_trial = (
        _HW_TRIAL_S + _TRAILING_DRAIN_S + _PER_TRIAL_IO_S
        + max(0.0, float(extra_per_trial_s))
    )
    if not analyze:
        analysis_term = 0.0
    elif iti_s >= _PER_TRIAL_ANALYSIS_S:
        analysis_term = _PER_TRIAL_ANALYSIS_S
    else:
        analysis_term = _PER_TRIAL_ANALYSIS_S * n
    return (
        _SESSION_SETUP_S
        + warmup_s + _WARMUP_MARKER_SLACK_S
        + n * per_trial
        + (n - 1) * iti_s
        + analysis_term
    )


def format_duration_hm(seconds: float) -> str:
    """Short human duration: '< 1 min' / '~34 min' / '~1 h 12 min' /
    '> 1 day (~30 h)'."""
    seconds = max(0.0, float(seconds))
    if seconds >= 86400:
        return f"> 1 day (~{seconds / 3600:.0f} h)"
    total_min = int(round(seconds / 60.0))
    if total_min < 1:
        return "< 1 min"
    if total_min < 60:
        return f"~{total_min} min"
    h, m = divmod(total_min, 60)
    return f"~{h} h {m:02d} min"


def format_finish_clock(dt: datetime, *, with_weekday: bool = False) -> str:
    """Wall-clock finish time, e.g. '3:12 PM' (or 'Wed 3:12 PM' past 24 h).
    Uses .lstrip('0') rather than the non-portable %-I / %#I."""
    core = f"{dt:%I:%M %p}".lstrip("0")
    return f"{dt:%a} {core}" if with_weekday else core


def format_time_left(seconds: float) -> str:
    """Live 'time remaining' string: '~2h 05m left' (>1 h), '~7m 30s left'
    (>90 s), '~45s left' (<=90 s), 'finishing…' (<5 s or after Done)."""
    seconds = max(0.0, float(seconds))
    if seconds < 5:
        return "finishing…"
    if seconds <= 90:
        return f"~{int(round(seconds))}s left"
    total_s = int(round(seconds))
    if total_s >= 3600:
        h, rem = divmod(total_s, 3600)
        return f"~{h}h {rem // 60:02d}m left"
    m, s = divmod(total_s, 60)
    return f"~{m}m {s:02d}s left"


_DOT_STIM: dict[StimState, str] = {
    StimState.UP:      "<span style='color:#27ae60;'>●</span> visual stim <b>up</b>",
    StimState.DOWN:    f"<span style='color:{DANGER_HEX};'>●</span> visual stim <b>down</b>",
    StimState.UNKNOWN: f"<span style='color:{text_rgba(0.55)};'>●</span> visual stim <i>?</i>",
}


class _RunSessionStimBridge(QObject):
    """Marshals background probe result (StimState, StimServerConfig | None) to the main thread."""
    probed = Signal(object, object)


class _RunSessionLightsBridge(QObject):
    """Marshals lights_controller.send_command_async callbacks to the main thread."""
    message = Signal(str)
    finished = Signal(bool, str)


# ── RunSessionWidget ──────────────────────────────────────────────────────────

class RunSessionWidget(QWidget):
    """
    Form + log pane for a staged calibrated-imaging session.

    The caller supplies the shared session ScriptRunner and its RunnerBridge so
    that the SafetyBar's context-aware Lights Off works while any stage runs.
    """

    # Session folder to open in the Analysis tab (MainWindow switches tabs).
    analyzeSessionRequested = Signal(str)

    def __init__(
        self,
        session_runner: ScriptRunner,
        session_bridge: RunnerBridge,
        safety_bar: SafetyBar,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = session_runner
        self._bridge = session_bridge
        self._safety_bar = safety_bar

        self._daily_folder: str | None = None
        self._warmup_total_s: float = 0.0
        self._num_trials: int = 1
        # Health summary — reset at trials start, populated at trials end.
        self._trials_completed: int = 0
        # (trial_index, peak_display, trough_display) — trough is None for
        # sessions run before intrinsic_imaging.py emitted trough_display=.
        self._peak_signals: list[tuple[int, float, float | None]] = []
        self._health_warnings: list[str] = []

        # Staged-workflow state, threaded forward between separate subprocess
        # launches via parsed stdout (see the module docstring).
        self._current_stage: str = ""
        self._stage_results: dict[str, Any] = {
            "green_exposure_us": None,
            "session_dir": None,
            "red_exposure_us": None,
            "roi_json_path": None,
        }
        # Guided Vertical Stepper view-state. _stage_touched: stage keys that
        # have had a real status set by _mark_stage_status — the pure view fn
        # never stomps a resolved "Done — 8000 µs" / "Failed (exit N)" for
        # these back to "Not started". _stage_failed: stage keys whose last
        # attempt failed (drives the red card border + "Retry …"). _feed_done:
        # the two camera-feed checkpoints the operator has clicked Continue
        # on. All three cleared by _reset_stage_state.
        self._stage_touched: set[str] = set()
        self._stage_failed: set[str] = set()
        self._feed_done: set[str] = set()
        self._summary_seen: bool = False

        self._stop_timer = QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.setInterval(_GRACEFUL_STOP_TIMEOUT_MS)
        self._stop_timer.timeout.connect(self._on_stop_timeout)

        # Live "time remaining" readout for the trials stage. Ticks once a
        # second and interpolates between the events parsed in _on_trials_line;
        # only meaningful while self._current_stage == "trials".
        self._eta_timer = QTimer(self)
        self._eta_timer.setInterval(1000)
        self._eta_timer.timeout.connect(self._tick_eta)
        self._eta_phase: str = "idle"   # idle | warmup | trials | trial-running | iti
        self._eta_warmup_left: float = 0.0
        self._eta_iti_left: float = 0.0
        self._eta_iti_s: float = 0.0    # the ITI of the *current run* (may differ
                                        # from the form on a follow-up session)
        self._eta_total_trials: int = 0
        self._eta_trial_k: int = 0
        self._eta_secs_in_phase: int = 0
        self._eta_measured_trial_cost: float | None = None
        self._eta_trial1_monotonic: float | None = None

        self._bridge.line_received.connect(self._on_line)
        self._bridge.run_finished.connect(self._on_done)

        # Live camera feed — separate runner from the staged session above,
        # since it's a long-running, user-stopped process independent of
        # whichever stage is currently running (mirrors gui/diagnostics.py).
        self._live_feed_open_buttons: list[QPushButton] = []
        self._live_feed_stop_buttons: list[QPushButton] = []
        self._live_runner = ScriptRunner()
        self._live_bridge = RunnerBridge(self._live_runner)
        self._live_bridge.line_received.connect(self._on_live_line)
        self._live_bridge.run_finished.connect(self._on_live_done)
        self._live_stop_timer = QTimer(self)
        self._live_stop_timer.setSingleShot(True)
        self._live_stop_timer.setInterval(_GRACEFUL_STOP_TIMEOUT_MS)
        self._live_stop_timer.timeout.connect(self._on_live_stop_timeout)

        # ROI time-course "peek" — its own runner, since it's a read-only,
        # camera/serial-free CPU job (no PySpin, no Arduino) that should be
        # runnable at any time, including while a trial subprocess is still
        # writing into the very same session folder.
        self._peek_runner = ScriptRunner()
        self._peek_bridge = RunnerBridge(self._peek_runner)
        self._peek_bridge.line_received.connect(self._on_peek_line)
        self._peek_bridge.run_finished.connect(self._on_peek_done)
        self._peek_output_dir: str | None = None

        # Light automation for the live-feed shortcut buttons and the Step 3
        # (red calibration) launch — a direct one-shot serial command, only
        # ever sent while no stage subprocess owns the COM port.
        self._lights_bridge = _RunSessionLightsBridge(self)
        self._lights_bridge.message.connect(self._on_lights_message)
        self._lights_bridge.finished.connect(self._on_lights_done)
        self._lights_busy: bool = False
        self._lights_pending_success: Callable[[], None] | None = None
        self._lights_pending_failure: Callable[[], None] | None = None

        self._stim_bridge = _RunSessionStimBridge(self)
        self._stim_bridge.probed.connect(self._on_stim_probed)
        self._stim_server_state: StimState = StimState.UNKNOWN
        self._stim_server_config: StimServerConfig | None = None
        self._last_active_step: str | None = None

        self._stim_probe_timer = QTimer(self)
        self._stim_probe_timer.setInterval(5_000)
        self._stim_probe_timer.timeout.connect(self._fire_stim_probe)

        self._build_ui()
        self._refresh_presets()
        self._reset_stage_state()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setSpacing(10)
        outer.setContentsMargins(14, 10, 14, 10)

        # Left column: forms and controls, in a scroll area so that toggling
        # "Advanced settings" (or any other content growth) scrolls instead
        # of forcing the whole window to grow to fit — without this, opening
        # a section with no widget left to absorb the extra height pushes the
        # window's minimum size past its current size, which Qt satisfies by
        # growing the window, potentially off the bottom of the screen.
        left = QWidget()
        left_outer = QVBoxLayout(left)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Stashed so _launch_stage can scroll the pinned footer "Stop" button
        # into view when the long trials run starts.
        self._left_scroll = scroll
        scroll_w = QWidget()
        layout = QVBoxLayout(scroll_w)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(8)
        scroll.setWidget(scroll_w)
        left_outer.addWidget(scroll)

        layout.addLayout(self._build_preset_bar())
        layout.addWidget(self._build_common_form())
        layout.addWidget(self._build_trial_sequence_panel())
        layout.addWidget(self._build_advanced_section())
        layout.addWidget(self._build_progress_area())
        layout.addWidget(self._build_health_panel())
        layout.addLayout(self._build_post_run_row())
        layout.addWidget(self._build_stage_panel())
        layout.addWidget(self._build_peek_panel())
        # Trailing stretch: without it, Qt spreads any leftover vertical
        # space evenly between the widgets above (awkward gaps) instead of
        # collecting it here at the bottom.
        layout.addStretch()

        # Right column: the two command-line output boxes, stacked.
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(self._build_log_panel(), stretch=1)
        right_layout.addWidget(self._build_panel_image_group(), stretch=2)

        outer.addWidget(left, stretch=1)
        outer.addWidget(right, stretch=1)

        # Wired up last: the handler touches self._trial_start_cmd, which
        # lives in the advanced panel built above. Connecting before that
        # panel exists would crash the first time the combo's index settles.
        self._stim_modality_combo.currentIndexChanged.connect(self._on_stim_modality_changed)

        # Session-duration estimate: recompute whenever an input that feeds it
        # changes. The slots are self-guarding, so wiring order vs. other
        # panels doesn't matter.
        for _w in (
            self._trials_spin, self._iti_spin, self._warmup_spin,
            self._baseline_frames, self._post_frames,
        ):
            _w.valueChanged.connect(self._refresh_trials_estimate)
        self._analyze_check.toggled.connect(self._refresh_trials_estimate)
        self._interleave_check.toggled.connect(self._refresh_trials_estimate)
        self._refresh_trials_estimate()

    # ── Preset bar ────────────────────────────────────────────────────────────

    def _build_preset_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("Preset:"))

        self._preset_combo = QComboBox()
        self._preset_combo.setMinimumWidth(240)
        self._preset_combo.setToolTip("Select a saved preset to load")
        row.addWidget(self._preset_combo, stretch=1)

        btn_load = QPushButton("Load")
        btn_load.setToolTip("Apply selected preset to the form below")
        btn_load.clicked.connect(self._load_preset)
        row.addWidget(btn_load)

        btn_save = QPushButton("Save as…")
        btn_save.setToolTip("Save current settings as a new preset")
        btn_save.clicked.connect(self._save_preset)
        row.addWidget(btn_save)

        # A unicode glyph ("↺") isn't guaranteed to be in every font's
        # glyph set (it rendered blank under the app's new font) — a real
        # QIcon always renders regardless of font/glyph availability.
        btn_refresh = QPushButton()
        btn_refresh.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        # Override the app-wide flush-left QPushButton rule (gui/theme.py) —
        # it also shifts an icon-only button's icon left instead of
        # centering it, since Qt aligns icon+text as one block.
        btn_refresh.setStyleSheet("QPushButton{text-align:center;}")
        btn_refresh.setFixedWidth(28)
        btn_refresh.setToolTip("Refresh preset list from disk")
        btn_refresh.clicked.connect(self._refresh_presets)
        row.addWidget(btn_refresh)

        return row

    # ── Common form ───────────────────────────────────────────────────────────

    def _build_common_form(self) -> QGroupBox:
        box = QGroupBox("Session parameters")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(7)

        # Output folder
        out_row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Select or type a folder path…")
        out_row.addWidget(self._output_edit, stretch=1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_output)
        out_row.addWidget(btn_browse)
        form.addRow("Output folder:", out_row)

        # COM port (shared — set at the top of the window)
        form.addRow("Arduino port:", PortIndicator(self._safety_bar))

        # Trials
        self._trials_spin = QSpinBox()
        self._trials_spin.setRange(1, 9999)
        self._trials_spin.setValue(1)
        self._trials_spin.setSuffix("  trial(s)")
        form.addRow("Trials:", self._trials_spin)

        # ITI
        self._iti_spin = QDoubleSpinBox()
        self._iti_spin.setRange(0.0, 3600.0)
        self._iti_spin.setValue(45.0)
        self._iti_spin.setSuffix(" s")
        self._iti_spin.setDecimals(1)
        self._iti_spin.setToolTip("Inter-trial interval (ITI): minimum wait between trials.")
        form.addRow("Inter-trial interval:", self._iti_spin)

        # Red warm-up
        self._warmup_spin = QDoubleSpinBox()
        self._warmup_spin.setRange(0.0, 7200.0)
        self._warmup_spin.setValue(600.0)
        self._warmup_spin.setSuffix(" s")
        self._warmup_spin.setDecimals(0)
        form.addRow("Red warm-up:", self._warmup_spin)

        self._settling_monitor_check = QCheckBox(
            f"Live-monitor LED settling (sample every {_SETTLING_SAMPLE_INTERVAL_S:.0f} s)"
        )
        self._settling_monitor_check.setChecked(True)
        self._settling_monitor_check.setToolTip(
            "During red warm-up, periodically triggers one camera frame and "
            "plots its mean brightness (within the calibrated ROI) relative "
            "to the first sample, so you can watch the LED settle."
        )
        self._settling_monitor_check.toggled.connect(self._on_settling_monitor_toggled)
        form.addRow("", self._settling_monitor_check)

        # Analyze
        self._analyze_check = QCheckBox("Make response maps after each trial")
        self._analyze_check.setChecked(True)
        self._analyze_check.setToolTip(
            "Passes --analyze to the acquisition script: after every trial it "
            "saves that trial's response arrays and updates the session average."
        )
        analyze_hint = QLabel(
            "A ΔR/R map per trial plus the Running session image on the right. "
            "Peek (interim statistics) and the Analysis tab (final statistics and "
            "figures) both need these maps."
        )
        analyze_hint.setWordWrap(True)
        analyze_hint.setStyleSheet(HINT_STYLE)
        # Checkbox and hint share one field widget: a word-wrapped label in its
        # own QFormLayout row was given far more height than its text needs.
        analyze_widget = QWidget()
        analyze_col = QVBoxLayout(analyze_widget)
        analyze_col.setContentsMargins(0, 0, 0, 0)
        analyze_col.setSpacing(2)
        analyze_col.addWidget(self._analyze_check)
        analyze_col.addWidget(analyze_hint)
        form.addRow("Acquisition maps:", analyze_widget)

        # Stimulus modality: external stimulus / TTL marker, visual
        # (visual-stimulus server over UDP), or vibrotactile (LRA, driven entirely by the Arduino).
        stim_widget = QWidget()
        stim_hl = QHBoxLayout(stim_widget)
        stim_hl.setContentsMargins(0, 0, 0, 0)
        self._stim_modality_combo = QComboBox()
        self._stim_modality_combo.addItem("External stimulus / TTL marker", "none")
        self._stim_modality_combo.addItem("Visual grating (stim server)", "visual")
        self._stim_modality_combo.addItem("Vibrotactile (LRA)", "lra")
        self._stim_modality_combo.setToolTip(
            "External stimulus / TTL marker: the Arduino outputs its stimulus TTL "
            "marker each trial; drive any external stimulator from it (or use it "
            "for no-stimulus sessions). This app drives no stimulus itself.\n"
            "Visual grating: the Arduino marker also commands the visual stimulus "
            "server (start it from the Utilities tab).\n"
            "Vibrotactile (LRA): the Arduino drives the actuator directly."
        )
        stim_hl.addWidget(self._stim_modality_combo)
        self._stim_indicator = QLabel()
        self._stim_indicator.setTextFormat(Qt.TextFormat.RichText)
        self._stim_indicator.setToolTip(
            "UDP probe to the visual-stimulus server.\n"
            "Green = reachable, Red = nothing listening.\n"
            "Host/port configurable in Advanced → Lights / Markers."
        )
        self._stim_indicator.hide()
        stim_hl.addWidget(self._stim_indicator)
        stim_hl.addStretch()
        form.addRow("Stimulus:", stim_widget)

        self._orientations_label = QLabel("Orientations:")
        self._orientations_edit = QLineEdit("45")
        self._orientations_edit.setToolTip(
            "Comma-separated grating orientations in degrees, e.g. 0,45,90,135; "
            "cycled across trials.\n\n"
            "Only used if the stim server was started with --single-orientation. "
            "By default it sweeps its own --orientations list instead, and this "
            "field is locked."
        )
        form.addRow(self._orientations_label, self._orientations_edit)

        self._duration_label = QLabel("Stim duration:")
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(0.1, 60.0)
        self._duration_spin.setValue(5.0)
        self._duration_spin.setSuffix(" s")
        self._duration_spin.setDecimals(1)
        self._duration_spin.setToolTip(
            "Only used if the stim server was started with "
            "--respect-requested-duration (a separate flag from "
            "--single-orientation). Otherwise each stimulus lasts "
            "(number of orientations) × --grating-duration-s, and this field is "
            "locked.\n\n"
            f"Either way, the stock Arduino firmware ends every stimulus "
            f"{_FIRMWARE_STIM_WINDOW_S:g} s after it starts."
        )
        form.addRow(self._duration_label, self._duration_spin)

        self._stim_effective_label = QLabel()
        self._stim_effective_label.setWordWrap(True)
        self._stim_effective_label.setTextFormat(Qt.TextFormat.RichText)
        form.addRow("Will run:", self._stim_effective_label)

        self._lra_note_label = QLabel(
            "Vibrotactile stim is driven entirely by the Arduino (AD9833 + amp) "
            "once trials start — no stim server needed. Timing (~5 s) and "
            "amplitude/frequency are fixed in firmware; use LRA Bench Controls "
            "in the Utilities tab to test/tune the actuator beforehand."
        )
        self._lra_note_label.setWordWrap(True)
        self._lra_note_label.setStyleSheet(HINT_STYLE)
        form.addRow("", self._lra_note_label)
        self._common_form = form

        self._set_visual_rows_visible(False)
        form.setRowVisible(self._lra_note_label, False)
        self._update_stim_effective()
        return box

    # ── Trial sequence (stim / catch interleaving) ────────────────────────────

    def _build_trial_sequence_panel(self) -> QGroupBox:
        # Hidden unless a stim modality is selected, and collapsed to just the
        # checkbox until interleaving is on (see _on_sequence_mode_changed), so
        # unused settings don't push the staged workflow below the fold.
        box = QGroupBox("Trial sequence")
        self._trial_seq_box = box
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(7)

        hint_text = (
            "Randomly interleave stim and no-stim (\"catch\") trials within this "
            "one session instead of running separate stim/no-stim sessions, so "
            "session-level confounds (LED warm-up, time-of-day drift) are shared "
            "between conditions. Requires firmware with per-trial stim gating "
            "(START_TRIAL,0/1 / STIM_LRA,0/1)."
        )
        self._interleave_check = QCheckBox("Interleave stim / catch trials")
        self._interleave_check.setEnabled(False)  # enabled once a stim modality is chosen
        self._interleave_check.setToolTip(hint_text)
        self._interleave_check.toggled.connect(self._on_interleave_toggled)
        form.addRow("", self._interleave_check)

        self._trial_seq_hint = QLabel(hint_text)
        self._trial_seq_hint.setWordWrap(True)
        self._trial_seq_hint.setStyleSheet(HINT_STYLE)
        form.addRow("", self._trial_seq_hint)

        self._sequence_mode_combo = QComboBox()
        self._sequence_mode_combo.addItem("Random (fraction + seed)", "random")
        self._sequence_mode_combo.addItem("Manual pattern", "manual")
        self._sequence_mode_combo.setEnabled(False)
        self._sequence_mode_combo.setToolTip(
            "Random: shuffled sequence from the fraction/seed/max-consecutive "
            "controls below.\n"
            "Manual pattern: an explicit sequence you type in, e.g. for a "
            "block design a random shuffle can't reliably produce."
        )
        self._sequence_mode_combo.currentIndexChanged.connect(self._on_sequence_mode_changed)
        form.addRow("Sequence mode:", self._sequence_mode_combo)

        self._stim_fraction_spin = QDoubleSpinBox()
        self._stim_fraction_spin.setRange(0.0, 1.0)
        self._stim_fraction_spin.setSingleStep(0.05)
        self._stim_fraction_spin.setValue(0.5)
        self._stim_fraction_spin.setDecimals(2)
        self._stim_fraction_spin.setEnabled(False)
        self._stim_fraction_spin.setToolTip(
            "Fraction of trials that get real stim; the rest are catch trials."
        )
        self._stim_fraction_spin.valueChanged.connect(self._update_sequence_preview)
        form.addRow("Stim fraction:", self._stim_fraction_spin)

        self._trial_seed_edit = QLineEdit()
        self._trial_seed_edit.setEnabled(False)
        self._trial_seed_edit.setPlaceholderText("blank = auto-generated")
        self._trial_seed_edit.setToolTip(
            "Pin a specific integer seed for a reproducible sequence, or leave "
            "blank — an auto-generated seed fills in here as soon as a sequence "
            "is generated, so what you see previewed is exactly what runs."
        )
        self._trial_seed_edit.textChanged.connect(self._update_sequence_preview)
        form.addRow("Seed:", self._trial_seed_edit)

        self._max_consecutive_spin = QSpinBox()
        self._max_consecutive_spin.setRange(1, 50)
        self._max_consecutive_spin.setValue(3)
        self._max_consecutive_spin.setEnabled(False)
        self._max_consecutive_spin.setSuffix("  max in a row")
        self._max_consecutive_spin.setToolTip(
            "Reshuffle to avoid runs of the same condition longer than this, so "
            "a lucky/unlucky streak doesn't dominate the session."
        )
        self._max_consecutive_spin.valueChanged.connect(self._update_sequence_preview)
        form.addRow("Max consecutive:", self._max_consecutive_spin)

        self._manual_pattern_edit = QLineEdit()
        self._manual_pattern_edit.setEnabled(False)
        self._manual_pattern_edit.setVisible(False)
        self._manual_pattern_edit.setPlaceholderText("e.g. 25S,25C   or   S,S,N,S,N,...")
        self._manual_pattern_edit.setToolTip(
            "Explicit per-trial sequence instead of a random one. Comma- and/or "
            "space-separated tokens, each S/stim (or 1) or C/N/catch (or 0), "
            "optionally prefixed with a repeat count — '25S,25C' means 25 stim "
            "trials followed by 25 catch trials. Must expand to exactly as many "
            "trials as 'Trials' above.\n\n"
            "Note: unlike Random mode, a block pattern like this shares none of "
            "the session-level confound balancing (LED warm-up drift, time-of-"
            "day) between conditions — everything early in the session lands in "
            "one condition, everything late in the other."
        )
        self._manual_pattern_edit.textChanged.connect(self._update_sequence_preview)
        self._manual_pattern_label = QLabel("Manual pattern:")
        self._manual_pattern_label.setVisible(False)
        form.addRow(self._manual_pattern_label, self._manual_pattern_edit)

        self._sequence_preview_label = QLabel("—")
        self._sequence_preview_label.setWordWrap(True)
        self._sequence_preview_label.setStyleSheet(f"color:{text_rgba(0.8)};")
        form.addRow("Sequence:", self._sequence_preview_label)

        self._trials_spin.valueChanged.connect(self._update_sequence_preview)
        self._trial_seq_form = form

        self._on_sequence_mode_changed()
        box.setVisible(False)
        return box

    def _on_sequence_mode_changed(self, _index: int = 0) -> None:
        manual = self._sequence_mode_combo.currentData() == "manual"
        interleaving_on = self._interleave_check.isChecked()
        form = self._trial_seq_form

        for w in (self._trial_seq_hint, self._sequence_mode_combo, self._sequence_preview_label):
            form.setRowVisible(w, interleaving_on)

        for w in (self._stim_fraction_spin, self._trial_seed_edit, self._max_consecutive_spin):
            form.setRowVisible(w, interleaving_on and not manual)
            w.setEnabled(interleaving_on and not manual)

        form.setRowVisible(self._manual_pattern_edit, interleaving_on and manual)
        self._manual_pattern_edit.setEnabled(interleaving_on and manual)

        if manual and interleaving_on and not self._manual_pattern_edit.text().strip():
            # Prefill with a compact encoding of whatever Random mode
            # currently shows, so switching to Manual starts from a
            # concrete, valid sequence instead of a blank field.
            try:
                conditions, _seed = generate_trial_conditions(
                    self._trials_spin.value(), self._stim_fraction_spin.value(),
                    seed=None, max_consecutive=self._max_consecutive_spin.value(),
                )
                self._manual_pattern_edit.blockSignals(True)
                self._manual_pattern_edit.setText(compress_to_pattern(conditions))
                self._manual_pattern_edit.blockSignals(False)
            except Exception:
                pass

        self._update_sequence_preview()

    def _on_interleave_toggled(self, checked: bool) -> None:
        self._sequence_mode_combo.setEnabled(checked)
        # Delegate the random-vs-manual field enable/visibility split to
        # _on_sequence_mode_changed rather than duplicating it here.
        self._on_sequence_mode_changed()

    def _current_trial_conditions_and_seed(self, n_trials: int | None = None) -> tuple[list[bool], int | None]:
        """Generate (or re-read, if already pinned) the trial-condition
        sequence for a launch. Random mode: if the seed field is blank, the
        auto-generated seed is written back into it so repeated calls —
        Show command, the actual launch, and the log line written at launch
        — all agree on the same sequence instead of each drawing a fresh
        random one. Manual mode: parses the pattern field; seed is always
        None (there's no randomness to record).

        Raises ValueError (message meant for the user directly) if manual
        mode's pattern is empty, unparseable, or doesn't expand to exactly
        n_trials entries — callers that launch anything must check
        _trial_sequence_error_message() first so this is never reached with
        bad input mid-launch.
        """
        if n_trials is None:
            n_trials = self._trials_spin.value()

        if self._sequence_mode_combo.currentData() == "manual":
            conditions = parse_manual_pattern(self._manual_pattern_edit.text(), n_trials)
            return conditions, None

        seed_text = self._trial_seed_edit.text().strip()
        seed = int(seed_text) if seed_text.lstrip("-").isdigit() else None
        conditions, seed_used = generate_trial_conditions(
            n_trials,
            self._stim_fraction_spin.value(),
            seed=seed,
            max_consecutive=self._max_consecutive_spin.value(),
        )
        if seed is None:
            self._trial_seed_edit.blockSignals(True)
            self._trial_seed_edit.setText(str(seed_used))
            self._trial_seed_edit.blockSignals(False)
        return conditions, seed_used

    def _trial_sequence_error_message(self, n_trials: int | None = None) -> str | None:
        """None if the current trial-sequence configuration (random or
        manual) is valid for n_trials; otherwise a user-facing message.
        Call before launching anything that will build --trial-conditions,
        so an invalid manual pattern is caught here with a friendly dialog
        instead of raising ValueError deep inside argv building."""
        if not self._interleave_check.isChecked():
            return None
        try:
            self._current_trial_conditions_and_seed(n_trials)
        except ValueError as exc:
            return str(exc)
        return None

    def _update_sequence_preview(self, *_args) -> None:
        if not self._interleave_check.isChecked():
            self._sequence_preview_label.setText("—")
            self._sequence_preview_label.setStyleSheet(f"color:{text_rgba(0.8)};")
            return
        try:
            conditions, seed = self._current_trial_conditions_and_seed()
        except ValueError as exc:
            self._sequence_preview_label.setStyleSheet(f"color:{DANGER_HEX};")
            self._sequence_preview_label.setText(str(exc))
            return
        self._sequence_preview_label.setStyleSheet(f"color:{text_rgba(0.8)};")
        n_stim = sum(conditions)
        seed_part = f", seed {seed}" if seed is not None else ""
        self._sequence_preview_label.setText(
            f"{format_sequence(conditions)}\n"
            f"({n_stim} stim / {len(conditions) - n_stim} catch{seed_part})"
        )

    def _log_trial_sequence(self, n_trials: int | None = None) -> None:
        """Write the realized stim/catch sequence to the run log. Called
        right after argv is built for a "trials" launch, so the sequence
        read back here (seed already pinned by that argv build, for random
        mode) matches exactly what was sent to the subprocess."""
        conditions, seed = self._current_trial_conditions_and_seed(n_trials)
        n_stim = sum(conditions)
        seed_part = f", seed {seed}" if seed is not None else ", manual pattern"
        self._log.appendPlainText(
            f"Trial sequence ({n_stim} stim / {len(conditions) - n_stim} catch"
            f"{seed_part}): {format_sequence(conditions)}"
        )

    # ── Advanced section (collapsible) ────────────────────────────────────────

    def _build_advanced_section(self) -> QWidget:
        container = QWidget()
        vlay = QVBoxLayout(container)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(4)

        self._adv_toggle = QPushButton("▶  Advanced settings")
        self._adv_toggle.setCheckable(True)
        self._adv_toggle.setChecked(False)
        self._adv_toggle.setStyleSheet(
            "QPushButton{padding:3px 8px;}"
            "QPushButton:checked{font-weight:bold;}"
        )
        self._adv_toggle.toggled.connect(self._on_adv_toggled)
        vlay.addWidget(self._adv_toggle)

        self._adv_panel = self._build_advanced_panel()
        self._adv_panel.hide()
        vlay.addWidget(self._adv_panel)

        return container

    def _on_adv_toggled(self, checked: bool) -> None:
        self._adv_toggle.setText(
            "▼  Advanced settings" if checked else "▶  Advanced settings"
        )
        self._adv_panel.setVisible(checked)

    def _build_advanced_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 4, 0, 4)

        tabs = QTabWidget()
        tabs.addTab(self._build_cal_tab(), "Calibration")
        tabs.addTab(self._build_frames_tab(), "Frames")
        tabs.addTab(self._build_binning_tab(), "Binning / Format")
        tabs.addTab(self._build_analysis_tab(), "Analysis")
        tabs.addTab(self._build_lights_tab(), "Lights / Markers")
        layout.addWidget(tabs)

        return panel

    def _make_form(self) -> tuple[QWidget, QFormLayout]:
        w = QWidget()
        f = QFormLayout(w)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f.setSpacing(6)
        f.setContentsMargins(8, 8, 8, 8)
        return w, f

    def _build_cal_tab(self) -> QWidget:
        w, f = self._make_form()

        self._cal_min_us = QDoubleSpinBox()
        self._cal_min_us.setRange(100.0, 5000.0)
        self._cal_min_us.setValue(1000.0)
        self._cal_min_us.setSuffix(" µs")
        self._cal_min_us.setDecimals(0)
        f.addRow("Min exposure (cal):", self._cal_min_us)

        self._cal_max_us = QDoubleSpinBox()
        self._cal_max_us.setRange(100.0, 12000.0)
        self._cal_max_us.setValue(12000.0)
        self._cal_max_us.setSuffix(" µs")
        self._cal_max_us.setDecimals(0)
        f.addRow("Max exposure (cal):", self._cal_max_us)

        self._cal_red_max_us = QDoubleSpinBox()
        self._cal_red_max_us.setRange(100.0, 10000.0)
        self._cal_red_max_us.setValue(10000.0)
        self._cal_red_max_us.setSuffix(" µs")
        self._cal_red_max_us.setDecimals(0)
        f.addRow("Red max exposure (cal):", self._cal_red_max_us)

        self._cal_steps = QSpinBox()
        self._cal_steps.setRange(2, 500)
        self._cal_steps.setValue(50)
        f.addRow("Cal steps:", self._cal_steps)

        self._cal_fps = QDoubleSpinBox()
        self._cal_fps.setRange(1.0, 100.0)
        self._cal_fps.setValue(10.0)
        self._cal_fps.setSuffix(" fps")
        self._cal_fps.setDecimals(1)
        f.addRow("Cal frame rate:", self._cal_fps)

        self._green_exposure_us = QDoubleSpinBox()
        self._green_exposure_us.setRange(0.0, 5000.0)
        self._green_exposure_us.setValue(0.0)
        self._green_exposure_us.setSuffix(" µs")
        self._green_exposure_us.setDecimals(0)
        self._green_exposure_us.setSpecialValueText("auto (run green cal)")
        self._green_exposure_us.setToolTip(
            "Set > 0 to skip step 2 (Calibrate Green Exposure) and use this "
            "value directly. Leave at 0 to run green calibration normally."
        )
        f.addRow("Green exposure override:", self._green_exposure_us)

        self._skip_roi_selection = QCheckBox("Skip ROI selection (use full frame)")
        f.addRow("", self._skip_roi_selection)

        self._cal_external_trigger = QCheckBox("Use Arduino trigger for cal frames")
        self._cal_external_trigger.setChecked(True)
        f.addRow("", self._cal_external_trigger)

        return w

    def _build_frames_tab(self) -> QWidget:
        w, f = self._make_form()

        self._green_frames = QSpinBox()
        self._green_frames.setRange(1, 500)
        self._green_frames.setValue(30)
        self._green_frames.setSuffix("  frames")
        f.addRow("Green reference frames:", self._green_frames)

        self._green_ref_trim = QSpinBox()
        self._green_ref_trim.setRange(0, 100)
        self._green_ref_trim.setValue(5)
        self._green_ref_trim.setSuffix("  frames")
        f.addRow("Green ref trim:", self._green_ref_trim)

        self._baseline_frames = QSpinBox()
        self._baseline_frames.setRange(1, 500)
        self._baseline_frames.setValue(40)
        self._baseline_frames.setSuffix("  frames")
        f.addRow("Baseline frames:", self._baseline_frames)

        self._post_frames = QSpinBox()
        self._post_frames.setRange(1, 500)
        self._post_frames.setValue(40)
        self._post_frames.setSuffix("  frames")
        f.addRow("Post-stim frames:", self._post_frames)

        self._save_gap_frames = QCheckBox("Save inter-phase gap frames")
        self._save_gap_frames.setChecked(True)
        self._save_gap_frames.setToolTip(
            "The camera now triggers continuously through the baseline->post "
            "transition (previously a dead window). Checked, those frames are "
            "written to a per-trial gap/ folder alongside baseline/post; "
            "unchecked, they're still triggered but not saved."
        )
        f.addRow("", self._save_gap_frames)

        return w

    def _build_binning_tab(self) -> QWidget:
        w, f = self._make_form()

        self._binning = QSpinBox()
        self._binning.setRange(1, 8)
        self._binning.setValue(2)
        f.addRow("Analysis binning:", self._binning)

        self._analysis_binning = QSpinBox()
        self._analysis_binning.setRange(0, 8)
        self._analysis_binning.setValue(0)
        self._analysis_binning.setSpecialValueText("same as binning")
        f.addRow("Analysis binning override:", self._analysis_binning)

        self._camera_binning = QSpinBox()
        self._camera_binning.setRange(1, 8)
        self._camera_binning.setValue(1)
        f.addRow("Camera (hardware) binning:", self._camera_binning)

        self._save_format = QComboBox()
        self._save_format.addItems(["raw", "tiff", "png"])
        f.addRow("Save format:", self._save_format)

        return w

    def _build_analysis_tab(self) -> QWidget:
        w, f = self._make_form()

        self._analysis_start_frame = QSpinBox()
        self._analysis_start_frame.setRange(0, 500)
        self._analysis_start_frame.setValue(5)
        self._analysis_start_frame.setSuffix("  frames")
        f.addRow("Analysis start frame:", self._analysis_start_frame)

        self._analysis_end_frame = QSpinBox()
        self._analysis_end_frame.setRange(1, 500)
        self._analysis_end_frame.setValue(35)
        self._analysis_end_frame.setSuffix("  frames")
        f.addRow("Analysis end frame:", self._analysis_end_frame)

        self._analysis_smoothing = QDoubleSpinBox()
        self._analysis_smoothing.setRange(0.0, 50.0)
        self._analysis_smoothing.setValue(5.0)
        self._analysis_smoothing.setDecimals(1)
        self._analysis_smoothing.setSuffix("  px σ")
        f.addRow("Smoothing sigma:", self._analysis_smoothing)

        self._analysis_method = QComboBox()
        self._analysis_method.addItem("Raw pixel counts", "raw_counts")
        self._analysis_method.addItem("Fractional reflectance", "fractional_reflectance")
        f.addRow("Analysis method:", self._analysis_method)

        self._analysis_mask_pct = QDoubleSpinBox()
        self._analysis_mask_pct.setRange(0.0, 100.0)
        self._analysis_mask_pct.setValue(20.0)
        self._analysis_mask_pct.setSuffix(" %")
        self._analysis_mask_pct.setDecimals(1)
        f.addRow("Mask percentile:", self._analysis_mask_pct)

        self._denom_floor_pct = QDoubleSpinBox()
        self._denom_floor_pct.setRange(0.0, 100.0)
        self._denom_floor_pct.setValue(5.0)
        self._denom_floor_pct.setSuffix(" %")
        self._denom_floor_pct.setDecimals(1)
        f.addRow("Denom floor percentile:", self._denom_floor_pct)

        self._denom_floor_counts = QDoubleSpinBox()
        self._denom_floor_counts.setRange(0.0, 100000.0)
        self._denom_floor_counts.setValue(100.0)
        self._denom_floor_counts.setDecimals(0)
        f.addRow("Denom floor counts:", self._denom_floor_counts)

        self._median_filter = QSpinBox()
        self._median_filter.setRange(0, 21)
        self._median_filter.setValue(3)
        self._median_filter.setSingleStep(2)
        self._median_filter.setSuffix("  px (odd)")
        f.addRow("Median filter size:", self._median_filter)

        self._rescale_bit_depth = QSpinBox()
        self._rescale_bit_depth.setRange(8, 32)
        self._rescale_bit_depth.setValue(16)
        self._rescale_bit_depth.setSuffix("  bit")
        f.addRow("Rescale bit depth:", self._rescale_bit_depth)

        self._rescale_mode = QComboBox()
        self._rescale_mode.addItems(["signed_symmetric", "minmax"])
        f.addRow("Rescale mode:", self._rescale_mode)

        self._invert_signal = QCheckBox("Invert display signal")
        f.addRow("", self._invert_signal)

        return w

    def _build_lights_tab(self) -> QWidget:
        w, f = self._make_form()

        self._session_green_cmd = QLineEdit("SESSION_GREEN_REFERENCE")
        f.addRow("Green ref command:", self._session_green_cmd)

        self._red_on_cmd = QLineEdit("SESSION_RED_ON")
        f.addRow("Red on command:", self._red_on_cmd)

        self._trial_start_cmd = QLineEdit("START_TRIAL")
        f.addRow("Trial start command:", self._trial_start_cmd)

        self._lights_off_cmd = QLineEdit("LIGHTS_OFF")
        f.addRow("Lights off command:", self._lights_off_cmd)

        self._leave_lights_on = QCheckBox("Leave lights on when session exits")
        self._leave_lights_on.setChecked(True)
        self._leave_lights_on.setToolTip(
            "Checked (default): skip the teardown lights-off, so red stays warm for "
            "a follow-up session's shortened warm-up. Use the safety bar's Lights "
            "Off (or uncheck this) when you're actually done with the animal."
        )
        f.addRow("", self._leave_lights_on)

        hdr = QLabel("Visual stim server network")
        hdr.setStyleSheet(f"font-weight:bold; margin-top:8px; color:{text_rgba(0.70)};")
        hdr.setToolTip("Only used when Stimulus is set to Visual (UDP) above.")
        f.addRow(hdr)

        self._stim_host_edit = QLineEdit("127.0.0.1")
        self._stim_host_edit.setToolTip(
            "IP or hostname of the visual-stimulus server.\n"
            "Almost always 127.0.0.1 (same machine)."
        )
        f.addRow("Visual stim host:", self._stim_host_edit)

        self._stim_port_spin = QSpinBox()
        self._stim_port_spin.setRange(1, 65535)
        self._stim_port_spin.setValue(55000)
        self._stim_port_spin.setToolTip("UDP port the visual-stimulus server listens on (default 55000).")
        f.addRow("Visual stim port:", self._stim_port_spin)

        return w

    # ── Progress area ─────────────────────────────────────────────────────────

    def _build_progress_area(self) -> QWidget:
        w = QWidget()
        self._progress_area = w
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        phase_row = QHBoxLayout()
        self._phase_label = QLabel("Idle.")
        phase_row.addWidget(self._phase_label, stretch=1)

        # Live "time remaining" readout. Kept separate from
        # self._progress.setFormat() (which already carries the per-phase
        # "Warm-up: MM:SS remaining" / "Trial k / N"). Driven at 1 Hz by
        # self._eta_timer, re-synced on parsed log lines in _on_trials_line.
        # A fixed min-width keeps the 1 Hz updates from reflowing the row.
        self._eta_label = QLabel("")
        self._eta_label.setStyleSheet(f"color:{text_rgba(0.7)};")
        self._eta_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._eta_label.setMinimumWidth(110)
        phase_row.addWidget(self._eta_label)

        self._skip_stabilization_btn = QPushButton("⏭  Skip stabilization")
        self._skip_stabilization_btn.setToolTip(
            "End the red-LED warm-up wait immediately and start trials now.\n"
            "Only available while the warm-up countdown is running."
        )
        self._skip_stabilization_btn.setEnabled(False)
        self._skip_stabilization_btn.clicked.connect(self._on_skip_stabilization_clicked)
        phase_row.addWidget(self._skip_stabilization_btn)
        v.addLayout(phase_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("")
        v.addWidget(self._progress)

        settling_row = QHBoxLayout()
        self._settling_plot = SettlingPlotWidget()
        self._settling_plot.hide()
        settling_row.addWidget(self._settling_plot, stretch=3)

        self._settling_image = PanelImageView(
            placeholder_text="Settling sample image will appear here once sampling starts…"
        )
        self._settling_image.hide()
        self._settling_image.setMaximumWidth(220)
        settling_row.addWidget(self._settling_image, stretch=1, alignment=Qt.AlignmentFlag.AlignRight)
        v.addLayout(settling_row)

        return w

    # ── Log ───────────────────────────────────────────────────────────────────

    def _build_log(self) -> QPlainTextEdit:
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Stage output will appear here…")
        return self._log

    def _build_log_panel(self) -> QGroupBox:
        box = QGroupBox("Stage output")
        vl = QVBoxLayout(box)
        vl.addWidget(self._build_log())
        return box

    # ── Health summary panel ──────────────────────────────────────────────────

    def _build_health_panel(self) -> QGroupBox:
        self._health_box = QGroupBox("Post-run summary")
        vl = QVBoxLayout(self._health_box)
        vl.setSpacing(4)

        self._health_summary_lbl = QLabel()
        self._health_summary_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._health_summary_lbl.setWordWrap(True)
        vl.addWidget(self._health_summary_lbl)

        self._health_detail = QPlainTextEdit()
        self._health_detail.setReadOnly(True)
        self._health_detail.setMaximumHeight(90)
        self._health_detail.hide()
        vl.addWidget(self._health_detail)

        self._health_box.hide()
        return self._health_box

    def _populate_health_panel(self) -> None:
        requested = self._num_trials
        completed = self._trials_completed
        warnings  = self._health_warnings
        peaks     = self._peak_signals

        if completed >= requested:
            trial_html = (
                f"<b>Trials:</b> <span style='color:#27ae60'>"
                f"{completed} / {requested} complete ✓</span>"
            )
        elif completed > 0:
            trial_html = (
                f"<b>Trials:</b> <span style='color:#e67e22'>"
                f"{completed} / {requested} complete</span>"
            )
        else:
            trial_html = (
                f"<b>Trials:</b> <span style='color:{DANGER_HEX}'>"
                f"0 / {requested} complete</span>"
            )

        sat_warnings = [w for w in warnings if "saturated" in w.lower()]
        if sat_warnings:
            sat_html = (
                f"<br><b>Saturation (65520):</b> "
                f"<span style='color:{DANGER_HEX}'>{len(sat_warnings)} warning(s) — see details</span>"
            )
        else:
            sat_html = (
                "<br><b>Saturation (65520):</b> "
                "<span style='color:#27ae60'>none detected</span>"
            )

        other_warnings = [w for w in warnings if "saturated" not in w.lower()]
        warn_count = len(other_warnings)
        if warn_count:
            warn_html = (
                f"<br><b>Other warnings:</b> "
                f"<span style='color:#e67e22'>{warn_count}</span>"
            )
        else:
            warn_html = ""

        self._health_summary_lbl.setText(trial_html + sat_html + warn_html)

        detail_lines: list[str] = []
        if warnings:
            detail_lines.append("=== Warnings ===")
            detail_lines.extend(warnings)
        if peaks:
            detail_lines.append("=== Display signal peak / trough (per-trial analysis, rounded) ===")
            for idx, peak, trough in sorted(peaks):
                trough_str = str(round(trough)) if trough is not None else "n/a"
                detail_lines.append(
                    f"  Trial {idx:03d}: peak {round(peak):>7}   trough {trough_str:>7}"
                )

        if detail_lines:
            self._health_detail.setPlainText("\n".join(detail_lines))
            self._health_detail.show()
        else:
            self._health_detail.hide()

        self._health_box.show()

    # ── Post-run row ──────────────────────────────────────────────────────────

    def _build_post_run_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._post_run_label = QLabel()
        self._post_run_label.hide()
        row.addWidget(self._post_run_label)

        self._open_folder_btn = QPushButton("Open folder")
        self._open_folder_btn.hide()
        self._open_folder_btn.clicked.connect(self._open_output_folder)
        row.addWidget(self._open_folder_btn)

        row.addStretch()
        return row

    # ── Staged session panel ──────────────────────────────────────────────────

    def _build_live_feed_row(self, hint_text: str) -> QWidget:
        """A compact 'open the live feed right here' shortcut row, so the user
        doesn't have to switch to Pre-Session Diagnostics and back between
        stages. Uses fixed sensible defaults; use the Diagnostics tab instead
        if you need to tune exposure/fps for the live feed itself."""
        row_widget = QWidget()
        col = QVBoxLayout(row_widget)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        lbl = QLabel(hint_text)
        lbl.setStyleSheet(HINT_STYLE)
        lbl.setWordWrap(True)
        col.addWidget(lbl)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        open_btn = QPushButton("📷 Turn on green light && preview")
        open_btn.setToolTip(
            "Sends green-on to the Arduino, then opens the live camera window. "
            "(The Diagnostics tab's preview leaves the LEDs unchanged.)"
        )
        open_btn.clicked.connect(self._on_live_feed_start)
        row.addWidget(open_btn)
        self._live_feed_open_buttons.append(open_btn)

        stop_btn = QPushButton("Close preview")
        stop_btn.setToolTip("Closes the live camera window. Green light stays on.")
        stop_btn.setEnabled(False)
        stop_btn.clicked.connect(self._on_live_feed_stop)
        row.addWidget(stop_btn)
        self._live_feed_stop_buttons.append(stop_btn)
        row.addStretch()
        col.addLayout(row)

        return row_widget

    def _build_stage_panel(self) -> QGroupBox:
        """Guided Vertical Stepper. All six items are always visible; exactly
        one is 'active' (bright call-to-action), earlier ones collapse to a
        one-line check + result, later ones are dimmed and locked. The two
        camera-feed items are discrete steps: open the feed, check focus,
        close it, Continue. A pure view over the existing state machine —
        _refresh_stepper_visuals() re-derives every card's look; nothing here
        changes how a stage subprocess is launched, stopped, or parsed."""
        box = QGroupBox("Staged session")
        self._stage_box = box
        vl = QVBoxLayout(box)
        vl.setSpacing(6)

        # KEPT names/contracts: _stage_status[key] carries the "done" bool prop
        # read by _next_stage_key(); _stage_buttons[key] keeps the _on_stage_run
        # slot. Only stage items get entries in these two.
        self._stage_status: dict[str, QLabel] = {}
        self._stage_buttons: dict[str, QPushButton] = {}
        # Per-stepper-item widget registries (feed items included).
        self._step_frames: dict[str, QFrame] = {}
        self._step_badges: dict[str, QLabel] = {}
        self._step_titles: dict[str, QLabel] = {}
        self._step_descs: dict[str, QLabel] = {}
        self._step_gates: dict[str, QLabel] = {}
        self._step_rules: dict[str, QFrame] = {}
        self._step_feeds: dict[str, QWidget] = {}
        self._feed_continue_btns: dict[str, QPushButton] = {}

        # ── Pinned header: the persistent "Next:" line (stays out of the
        #    collapsing card area, as does the footer below). ─────────────────
        self._stage_next_label = QLabel()
        self._stage_next_label.setWordWrap(True)
        self._stage_next_label.setStyleSheet("font-weight:bold;")
        self._stage_next_label.setToolTip(
            "\n".join(f"{_STEP_NUM[i]}. {t}" for (_k, i, t) in _STEPPER_ITEMS)
        )
        vl.addWidget(self._stage_next_label)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color:{text_rgba(0.22)};")
        vl.addWidget(divider)

        # Created here so it can be nested inside the "Start Trials" card.
        self._trials_estimate_label = QLabel()
        self._trials_estimate_label.setWordWrap(True)
        self._trials_estimate_label.setMinimumHeight(18)
        self._trials_estimate_label.setStyleSheet(f"color:{text_rgba(0.7)};")

        for kind, item_id, title in _STEPPER_ITEMS:
            num = _STEP_NUM[item_id]

            frame = QFrame()
            frame.setObjectName("stageStep")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(8, 6, 8, 6)
            fl.setSpacing(4)

            head = QHBoxLayout()
            head.setSpacing(8)
            badge = QLabel(str(num))
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setFixedSize(20, 20)
            head.addWidget(badge)
            self._step_badges[item_id] = badge

            title_lbl = QLabel(title)
            title_lbl.setStyleSheet("font-weight:bold;")
            head.addWidget(title_lbl)
            self._step_titles[item_id] = title_lbl
            head.addStretch(1)

            if kind == "stage":
                status = QLabel("Not started")
                status.setStyleSheet(f"color:{text_rgba(0.55)};")
                status.setWordWrap(True)
                status.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                )
                head.addWidget(status)
                self._stage_status[item_id] = status
            fl.addLayout(head)

            desc = QLabel(_STEP_DESC[item_id])
            desc.setWordWrap(True)
            desc.setStyleSheet(f"color:{text_rgba(0.70)};")
            fl.addWidget(desc)
            self._step_descs[item_id] = desc

            if kind == "stage":
                gate = QLabel("")
                gate.setWordWrap(True)
                gate.setStyleSheet(HINT_STYLE)
                gate.hide()
                fl.addWidget(gate)
                self._step_gates[item_id] = gate

            if kind == "feed":
                feed = self._build_live_feed_row(_STEP_FEED_PROMPT[item_id])
                fl.addWidget(feed)
                self._step_feeds[item_id] = feed

            if item_id == "trials":
                fl.addWidget(self._trials_estimate_label)

            rule = QFrame()
            rule.setFrameShape(QFrame.Shape.HLine)
            rule.setStyleSheet(f"color:{text_rgba(0.18)};")
            fl.addWidget(rule)
            self._step_rules[item_id] = rule

            act = QHBoxLayout()
            act.addStretch(1)
            if kind == "stage":
                btn = QPushButton(f"Run step {num}")
                btn.clicked.connect(
                    lambda _=False, k=item_id: self._on_stage_run(k)
                )
                self._stage_buttons[item_id] = btn
                act.addWidget(btn)
            else:
                cbtn = QPushButton("Continue  ▶")
                cbtn.clicked.connect(
                    lambda _=False, fid=item_id: self._on_feed_continue(fid)
                )
                self._feed_continue_btns[item_id] = cbtn
                act.addWidget(cbtn)
            fl.addLayout(act)

            vl.addWidget(frame)
            self._step_frames[item_id] = frame

        # ── "Session so far" — collapsible; keeps resolved calibration values
        #    visible after their step cards have collapsed. ───────────────────
        self._summary_toggle = QPushButton("Session so far  (0/4 captured)")
        self._summary_toggle.setCheckable(True)
        self._summary_toggle.setChecked(False)
        self._summary_toggle.setStyleSheet(
            "QPushButton{border:none;background:transparent;"
            "text-align:left;padding:2px 0;}"
        )
        self._summary_toggle.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight)
        )
        self._summary_toggle.toggled.connect(self._on_summary_toggled)
        vl.addWidget(self._summary_toggle)

        self._summary_body = QWidget()
        sform = QFormLayout(self._summary_body)
        sform.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        sform.setContentsMargins(12, 2, 4, 6)
        sform.setSpacing(4)
        _sel = Qt.TextInteractionFlag.TextSelectableByMouse
        self._sum_green_val = QLabel("—")
        self._sum_green_val.setTextInteractionFlags(_sel)
        self._sum_session_val = QLabel("—")
        self._sum_session_val.setWordWrap(True)
        self._sum_session_val.setTextInteractionFlags(_sel)
        self._sum_red_val = QLabel("—")
        self._sum_red_val.setTextInteractionFlags(_sel)
        self._sum_roi_val = QLabel("—")
        self._sum_roi_val.setWordWrap(True)
        self._sum_roi_val.setTextInteractionFlags(_sel)
        sform.addRow("Green exposure:", self._sum_green_val)
        sform.addRow("Session folder:", self._sum_session_val)
        sform.addRow("Red exposure:", self._sum_red_val)
        sform.addRow("ROI config:", self._sum_roi_val)
        self._summary_body.setVisible(False)
        vl.addWidget(self._summary_body)

        divider2 = QFrame()
        divider2.setFrameShape(QFrame.Shape.HLine)
        divider2.setStyleSheet(f"color:{text_rgba(0.22)};")
        vl.addWidget(divider2)

        # ── Pinned footer: same four controls, same slots, kept out of the
        #    collapsing area so the red Stop never scrolls away mid-run. ──────
        # Two rows: as one row this footer was the widest thing in the column
        # and forced a horizontal scrollbar at 100% zoom.
        tools_row = QHBoxLayout()

        self._show_cmd_btn = QPushButton("Show next command…")
        self._show_cmd_btn.clicked.connect(self._show_command)
        tools_row.addWidget(self._show_cmd_btn)

        self._reset_btn = QPushButton("Reset (new animal)")
        self._reset_btn.setToolTip("Clear staged progress and start over from step 1.")
        self._reset_btn.clicked.connect(self._reset_stage_state)
        tools_row.addWidget(self._reset_btn)
        tools_row.addStretch()
        vl.addLayout(tools_row)

        btn_row = QHBoxLayout()

        self._followup_btn = QPushButton("Follow-up session…")
        self._followup_btn.setToolTip(
            "Run another batch of trials for the same animal, reusing the "
            "green reference, exposures, and ROI/crop from the calibration "
            "steps above (skips calibration entirely), with its own trial "
            "count/inter-trial interval/red warm-up — typically a shortened "
            "warm-up since red is already warm from the session just run."
        )
        self._followup_btn.setEnabled(False)
        self._followup_btn.clicked.connect(self._on_run_followup)
        btn_row.addWidget(self._followup_btn)

        btn_row.addStretch()

        self._stop_btn = QPushButton("■  Stop session")
        self._stop_btn.setToolTip(
            "Gracefully stop the running session step (same as Stop Session at "
            "the top of the window). Does not close the live preview."
        )
        self._stop_btn.setStyleSheet(STYLE_STOP)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)

        vl.addLayout(btn_row)
        return box

    # ── ROI time-course "peek" panel ──────────────────────────────────────────

    def _build_peek_panel(self) -> QGroupBox:
        box = QGroupBox("Interim statistics (Peek) — this session so far")
        vl = QVBoxLayout(box)

        hint = QLabel(
            "Three kinds of analysis: <b>acquisition maps</b> (per-trial, made "
            "during the run), <b>interim statistics</b> (this Peek: a quick ROI "
            "time course and t-map of the trials finished so far, needs 3+, safe "
            "mid-run, written to a 'peek' subfolder), and <b>final statistics "
            "and figures</b> (the Analysis tab, for the finished session)."
        )
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.TextFormat.RichText)
        hint.setStyleSheet(HINT_STYLE)
        vl.addWidget(hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("Permutations:"))
        self._peek_nperm = QSpinBox()
        self._peek_nperm.setRange(10, 10_000)
        self._peek_nperm.setValue(200)
        self._peek_nperm.setToolTip(
            "Sign-flip permutations for the cluster test. Lower = faster peek; "
            "the script's own default (used for a final formal analysis) is 1000."
        )
        row.addWidget(self._peek_nperm)

        self._peek_neg_only = QCheckBox("Negative-only clusters")
        self._peek_neg_only.setChecked(True)
        self._peek_neg_only.setToolTip(
            "Activation at 630 nm is a reflectance decrease. A two-sided test can let a "
            "positive-going artifact (e.g. edge vignetting) outrank the real response as "
            "the 'largest cluster'. Leave checked unless you specifically need two-sided."
        )
        row.addWidget(self._peek_neg_only)

        self._peek_condition_combo = QComboBox()
        self._peek_condition_combo.addItem("All trials", "all")
        self._peek_condition_combo.addItem("Stim only", "stim")
        self._peek_condition_combo.addItem("Catch only", "catch")
        self._peek_condition_combo.setToolTip(
            "For an interleaved session, restrict the peek to one condition "
            "instead of mixing stim and catch trials together."
        )
        row.addWidget(self._peek_condition_combo)
        row.addStretch()

        vl.addLayout(row)

        action_row = QHBoxLayout()
        self._peek_btn = QPushButton("🔍  Run interim statistics")
        self._peek_btn.setEnabled(False)
        self._peek_btn.clicked.connect(self._on_run_peek)
        action_row.addWidget(self._peek_btn)

        self._peek_open_folder_btn = QPushButton("Open folder")
        self._peek_open_folder_btn.hide()
        self._peek_open_folder_btn.clicked.connect(self._open_peek_output_folder)
        action_row.addWidget(self._peek_open_folder_btn)
        action_row.addStretch()

        self._analyze_session_btn = QPushButton("Analyze this session  →")
        self._analyze_session_btn.setToolTip(
            "Open the Analysis tab with this session's folder filled in, for "
            "final statistics and figures."
        )
        self._analyze_session_btn.setEnabled(False)
        self._analyze_session_btn.clicked.connect(self._on_analyze_session_clicked)
        action_row.addWidget(self._analyze_session_btn)
        vl.addLayout(action_row)

        self._peek_status_label = QLabel("Idle.")
        self._peek_status_label.setWordWrap(True)
        vl.addWidget(self._peek_status_label)

        return box

    def _on_analyze_session_clicked(self) -> None:
        session_dir = self._stage_results.get("session_dir")
        if session_dir:
            self.analyzeSessionRequested.emit(str(session_dir))

    def _build_panel_image_group(self) -> QGroupBox:
        box = QGroupBox("Running session image")
        vl = QVBoxLayout(box)
        self._panel_image = PanelImageBrowser()
        vl.addWidget(self._panel_image)
        return box

    # ── Stim field visibility ─────────────────────────────────────────────────

    def _set_visual_rows_visible(self, visible: bool) -> None:
        for w in (self._orientations_edit, self._duration_spin, self._stim_effective_label):
            self._common_form.setRowVisible(w, visible)

    def _on_stim_modality_changed(self, _index: int = 0) -> None:
        modality = self._stim_modality_combo.currentData()
        is_visual = modality == "visual"
        is_lra = modality == "lra"

        self._set_visual_rows_visible(is_visual)
        self._common_form.setRowVisible(self._lra_note_label, is_lra)
        self._trial_seq_box.setVisible(is_visual or is_lra)
        self._safety_bar.set_visual_stim_required(is_visual)

        if is_visual:
            self._stim_indicator.show()
            self._fire_stim_probe()
            self._stim_probe_timer.start()
        else:
            self._stim_indicator.hide()
            self._stim_probe_timer.stop()

        # Drive the Arduino trial-start command from the modality choice.
        # Advanced users can still hand-edit it afterward in Advanced ->
        # Lights / Markers; that edit sticks until modality changes again.
        self._trial_start_cmd.setText(
            _LRA_TRIAL_START_CMD if is_lra else _DEFAULT_TRIAL_START_CMD
        )

        # Interleaving needs an actual stim modality to interleave against.
        has_stim = is_visual or is_lra
        self._interleave_check.setEnabled(has_stim)
        if not has_stim and self._interleave_check.isChecked():
            self._interleave_check.setChecked(False)
        self._update_sequence_preview()

    def _on_settling_monitor_toggled(self, checked: bool) -> None:
        if not checked:
            self._settling_plot.hide()
            self._settling_plot.clear_points()
            self._settling_image.hide()
            self._settling_image.clear_image()

    def _fire_stim_probe(self) -> None:
        host = self._stim_host_edit.text().strip() or "127.0.0.1"
        port = self._stim_port_spin.value()
        probe_config_async(self._stim_bridge.probed.emit, host=host, port=port)

    def _on_stim_probed(self, state: StimState, config: StimServerConfig | None) -> None:
        self._stim_indicator.setText(_DOT_STIM.get(state, _DOT_STIM[StimState.UNKNOWN]))
        if state == self._stim_server_state and config == self._stim_server_config:
            return
        self._stim_server_state = state
        self._stim_server_config = config
        self._update_stim_effective()

    def _update_stim_effective(self) -> None:
        """Show what the stim server will actually display, from the settings
        it reported, and lock whichever of the two fields it will ignore.
        Both stay locked until a server has reported."""
        cfg = self._stim_server_config
        self._orientations_edit.setEnabled(cfg is not None and cfg.honors_orientation)
        self._duration_spin.setEnabled(cfg is not None and cfg.honors_duration)

        cutoff = (
            f"The Arduino firmware ends every stimulus {_FIRMWARE_STIM_WINDOW_S:g} s "
            "after it starts, so anything longer is cut there."
        )
        if cfg is None:
            if self._stim_server_state is StimState.UP:
                text = (
                    "⚠ The stim server is running but didn't report its settings "
                    "(an older intrinsic_visual_stimulus.py?), so this can't be "
                    "confirmed. With stock settings it sweeps 135°, 180°, 225°, "
                    "270° for 1.25 s each and ignores both fields above."
                )
            else:
                text = (
                    "⚠ Stim server not reachable. Start it (Utilities tab) to see "
                    "what it will show; Orientations and Stim duration stay locked "
                    "until it reports which of them it uses."
                )
            self._stim_effective_label.setStyleSheet(WARNING_STYLE)
            self._stim_effective_label.setText(text)
            return

        orients = ", ".join(f"{o:g}°" for o in cfg.orientations_deg)
        seg = cfg.grating_duration_s
        if cfg.multi_orientation:
            order = " in random order" if cfg.randomize_orientations else ""
            what = f"A sweep of {orients}{order}"
        else:
            what = "The orientation(s) above, one per trial in turn,"

        if cfg.honors_duration:
            how_long = "for the stim duration above"
            if cfg.multi_orientation:
                how_long += f" ({seg:g} s per orientation)"
        else:
            n = len(cfg.orientations_deg) if cfg.multi_orientation else 1
            how_long = f"for <b>{n * seg:g} s</b>"
            if cfg.multi_orientation:
                how_long += f" ({seg:g} s per orientation)"

        ignored = []
        if not cfg.honors_orientation:
            ignored.append("Orientations (server started without --single-orientation)")
        if not cfg.honors_duration:
            ignored.append("Stim duration (server started without --respect-requested-duration)")
        ignored_text = (
            "<br>Locked because the server ignores them: " + "; ".join(ignored) + "."
            if ignored else ""
        )
        self._stim_effective_label.setStyleSheet(HINT_STYLE)
        self._stim_effective_label.setText(
            f"<b>{what}</b> {how_long} per stimulus.{ignored_text}<br>{cutoff}"
        )

    # ── Browse ────────────────────────────────────────────────────────────────

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._output_edit.setText(path)

    # ── Preset helpers ────────────────────────────────────────────────────────

    def _refresh_presets(self) -> None:
        prev_path = self._preset_combo.currentData()
        self._preset_combo.clear()
        self._preset_combo.addItem("— select a preset —", None)
        for p in _presets.list_presets():
            try:
                data = _presets.load_preset(p)
            except Exception:
                data = {}
            self._preset_combo.addItem(_presets.display_name(p, data), str(p))
        if prev_path:
            idx = self._preset_combo.findData(prev_path)
            if idx >= 0:
                self._preset_combo.setCurrentIndex(idx)

    def _load_preset(self) -> None:
        path_str = self._preset_combo.currentData()
        if not path_str:
            QMessageBox.information(self, "No preset selected", "Select a preset from the dropdown first.")
            return
        try:
            data = _presets.load_preset(Path(path_str))
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", f"Could not read preset:\n{exc}")
            return
        self.from_dict(data)

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Save preset", "Preset name:",
            text=self._preset_combo.currentText()
            if self._preset_combo.currentData() else ""
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        safe = re.sub(r'[^\w\- ]', '_', name).replace(' ', '_').lower()
        path = _presets.PRESETS_DIR / f"{safe}.json"
        data = self.to_dict()
        data["_name"] = name
        try:
            _presets.save_preset(path, data)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", f"Could not save preset:\n{exc}")
            return
        self._refresh_presets()
        idx = self._preset_combo.findData(str(path))
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)

    # ── Form serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return all form values keyed by argparse attribute name."""
        return {
            # common
            "output": self._output_edit.text().strip(),
            "port": self._safety_bar.selected_port() or "",
            "trials": self._trials_spin.value(),
            "iti": self._iti_spin.value(),
            "red_stabilization_s": self._warmup_spin.value(),
            "red_settling_monitor": self._settling_monitor_check.isChecked(),
            "analyze": self._analyze_check.isChecked(),
            "stim_modality": self._stim_modality_combo.currentData(),
            "visual_stim": self._stim_modality_combo.currentData() == "visual",
            "stim_orientations": self._orientations_edit.text().strip(),
            "stim_duration_s": self._duration_spin.value(),
            "stim_host": self._stim_host_edit.text().strip() or "127.0.0.1",
            "stim_port": self._stim_port_spin.value(),
            # trial sequence (stim/catch interleaving)
            "interleave_stim": self._interleave_check.isChecked(),
            "sequence_mode": self._sequence_mode_combo.currentData(),
            "stim_fraction": self._stim_fraction_spin.value(),
            "trial_seed": self._trial_seed_edit.text().strip(),
            "max_consecutive": self._max_consecutive_spin.value(),
            "manual_pattern": self._manual_pattern_edit.text().strip(),
            # calibration
            "cal_min_us": self._cal_min_us.value(),
            "cal_max_us": self._cal_max_us.value(),
            "cal_red_max_us": self._cal_red_max_us.value(),
            "cal_steps": self._cal_steps.value(),
            "cal_fps": self._cal_fps.value(),
            "green_exposure_us": self._green_exposure_us.value() or None,
            "skip_roi_selection": self._skip_roi_selection.isChecked(),
            "cal_external_trigger": self._cal_external_trigger.isChecked(),
            # frames
            "green_frames": self._green_frames.value(),
            "green_reference_trim_frames": self._green_ref_trim.value(),
            "baseline_frames": self._baseline_frames.value(),
            "post_frames": self._post_frames.value(),
            "save_gap_frames": self._save_gap_frames.isChecked(),
            # binning
            "binning": self._binning.value(),
            "analysis_binning": self._analysis_binning.value() or None,
            "camera_binning": self._camera_binning.value(),
            "save_format": self._save_format.currentText(),
            # analysis
            "analysis_start_frame": self._analysis_start_frame.value(),
            "analysis_end_frame": self._analysis_end_frame.value(),
            "analysis_smoothing_sigma": self._analysis_smoothing.value(),
            "analysis_method": self._analysis_method.currentData(),
            "analysis_mask_percentile": self._analysis_mask_pct.value(),
            "analysis_denominator_floor_percentile": self._denom_floor_pct.value(),
            "analysis_denominator_floor_counts": self._denom_floor_counts.value(),
            "analysis_median_filter_size": self._median_filter.value(),
            "rescale_bit_depth": self._rescale_bit_depth.value(),
            "rescale_mode": self._rescale_mode.currentText(),
            "invert_display_signal": self._invert_signal.isChecked(),
            # lights
            "session_green_cmd": self._session_green_cmd.text().strip(),
            "red_on_cmd": self._red_on_cmd.text().strip(),
            "trial_start_cmd": self._trial_start_cmd.text().strip(),
            "lights_off_cmd": self._lights_off_cmd.text().strip(),
            "leave_lights_on_exit": self._leave_lights_on.isChecked(),
        }

    def from_dict(self, d: dict[str, Any]) -> None:
        """Populate form widgets from a preset dict. Unknown keys are ignored."""

        def _str(key: str, default: str = "") -> str:
            return str(d.get(key, default))

        def _int(key: str, default: int = 0) -> int:
            try:
                return int(d[key])
            except (KeyError, TypeError, ValueError):
                return default

        def _float(key: str, default: float = 0.0) -> float:
            try:
                return float(d[key])
            except (KeyError, TypeError, ValueError):
                return default

        def _bool(key: str, default: bool = False) -> bool:
            v = d.get(key, default)
            return bool(v)

        if "output" in d and d["output"]:
            self._output_edit.setText(_str("output"))
        if "port" in d and d["port"]:
            # Best-effort: select the preset's port in the shared top-bar
            # control if it's currently plugged in. Presets don't own the
            # port — the top bar is the single source of truth.
            self._safety_bar.try_select_port(_str("port"))
        if "trials" in d:
            self._trials_spin.setValue(_int("trials", 1))
        if "iti" in d:
            self._iti_spin.setValue(_float("iti", 45.0))
        if "red_stabilization_s" in d:
            self._warmup_spin.setValue(_float("red_stabilization_s", 600.0))
        if "red_settling_monitor" in d:
            self._settling_monitor_check.setChecked(_bool("red_settling_monitor"))
        if "analyze" in d:
            self._analyze_check.setChecked(_bool("analyze"))
        # Apply stim network settings before switching modality so the probe
        # (fired by the modality-changed signal) uses the correct host/port.
        if "stim_host" in d:
            self._stim_host_edit.setText(_str("stim_host", "127.0.0.1"))
        if "stim_port" in d:
            self._stim_port_spin.setValue(_int("stim_port", 55000))
        if "stim_modality" in d:
            idx = self._stim_modality_combo.findData(_str("stim_modality", "none"))
            if idx >= 0:
                self._stim_modality_combo.setCurrentIndex(idx)
        elif "visual_stim" in d or "trial_start_cmd" in d:
            # Back-compat for presets saved before the modality selector
            # existed: infer from the old visual_stim flag / trial command.
            if _bool("visual_stim"):
                modality = "visual"
            elif _str("trial_start_cmd", _DEFAULT_TRIAL_START_CMD) == _LRA_TRIAL_START_CMD:
                modality = "lra"
            else:
                modality = "none"
            idx = self._stim_modality_combo.findData(modality)
            if idx >= 0:
                self._stim_modality_combo.setCurrentIndex(idx)
        if "stim_orientations" in d:
            self._orientations_edit.setText(_str("stim_orientations", "45"))
        if "stim_duration_s" in d:
            self._duration_spin.setValue(_float("stim_duration_s", 5.0))
        # Trial sequence — restore fraction/seed/max-consecutive before the
        # checkbox, so when the checkbox's toggled signal regenerates the
        # preview it uses the preset's values rather than the form defaults.
        # manual_pattern restored before sequence_mode, so if the preset is
        # manual mode, the currentIndexChanged handler that fires when
        # sequence_mode is applied below sees the real pattern already in
        # place and doesn't overwrite it with a fresh random-mode prefill.
        if "manual_pattern" in d:
            self._manual_pattern_edit.setText(_str("manual_pattern", ""))
        if "stim_fraction" in d:
            self._stim_fraction_spin.setValue(_float("stim_fraction", 0.5))
        if "trial_seed" in d:
            self._trial_seed_edit.setText(_str("trial_seed", ""))
        if "max_consecutive" in d:
            self._max_consecutive_spin.setValue(_int("max_consecutive", 3))
        if "sequence_mode" in d:
            idx = self._sequence_mode_combo.findData(_str("sequence_mode", "random"))
            if idx >= 0:
                self._sequence_mode_combo.setCurrentIndex(idx)
        if "interleave_stim" in d and self._interleave_check.isEnabled():
            self._interleave_check.setChecked(_bool("interleave_stim"))
        # calibration
        if "cal_min_us" in d:
            self._cal_min_us.setValue(_float("cal_min_us", 1000.0))
        if "cal_max_us" in d:
            self._cal_max_us.setValue(_float("cal_max_us", 12000.0))
        if "cal_red_max_us" in d:
            self._cal_red_max_us.setValue(_float("cal_red_max_us", 10000.0))
        if "cal_steps" in d:
            self._cal_steps.setValue(_int("cal_steps", 50))
        if "cal_fps" in d:
            self._cal_fps.setValue(_float("cal_fps", 10.0))
        if "green_exposure_us" in d:
            self._green_exposure_us.setValue(_float("green_exposure_us") or 0.0)
        if "skip_roi_selection" in d:
            self._skip_roi_selection.setChecked(_bool("skip_roi_selection"))
        if "cal_external_trigger" in d:
            self._cal_external_trigger.setChecked(_bool("cal_external_trigger"))
        # frames
        if "green_frames" in d:
            self._green_frames.setValue(_int("green_frames", 30))
        if "green_reference_trim_frames" in d:
            self._green_ref_trim.setValue(_int("green_reference_trim_frames", 5))
        if "baseline_frames" in d:
            self._baseline_frames.setValue(_int("baseline_frames", 40))
        if "post_frames" in d:
            self._post_frames.setValue(_int("post_frames", 40))
        if "save_gap_frames" in d:
            self._save_gap_frames.setChecked(_bool("save_gap_frames", True))
        # binning
        if "binning" in d:
            self._binning.setValue(_int("binning", 2))
        if "analysis_binning" in d:
            self._analysis_binning.setValue(_int("analysis_binning") or 0)
        if "camera_binning" in d:
            self._camera_binning.setValue(_int("camera_binning", 1))
        if "save_format" in d:
            idx = self._save_format.findText(_str("save_format", "raw"))
            if idx >= 0:
                self._save_format.setCurrentIndex(idx)
        # analysis
        if "analysis_start_frame" in d:
            self._analysis_start_frame.setValue(_int("analysis_start_frame", 5))
        if "analysis_end_frame" in d:
            self._analysis_end_frame.setValue(_int("analysis_end_frame", 35))
        if "analysis_smoothing_sigma" in d:
            self._analysis_smoothing.setValue(_float("analysis_smoothing_sigma", 5.0))
        if "analysis_method" in d:
            idx = self._analysis_method.findData(_str("analysis_method", "raw_counts"))
            if idx >= 0:
                self._analysis_method.setCurrentIndex(idx)
        if "analysis_mask_percentile" in d:
            self._analysis_mask_pct.setValue(_float("analysis_mask_percentile", 20.0))
        if "analysis_denominator_floor_percentile" in d:
            self._denom_floor_pct.setValue(_float("analysis_denominator_floor_percentile", 5.0))
        if "analysis_denominator_floor_counts" in d:
            self._denom_floor_counts.setValue(_float("analysis_denominator_floor_counts", 100.0))
        if "analysis_median_filter_size" in d:
            self._median_filter.setValue(_int("analysis_median_filter_size", 3))
        if "rescale_bit_depth" in d:
            self._rescale_bit_depth.setValue(_int("rescale_bit_depth", 16))
        if "rescale_mode" in d:
            idx = self._rescale_mode.findText(_str("rescale_mode", "signed_symmetric"))
            if idx >= 0:
                self._rescale_mode.setCurrentIndex(idx)
        if "invert_display_signal" in d:
            self._invert_signal.setChecked(_bool("invert_display_signal"))
        # lights
        if "session_green_cmd" in d:
            self._session_green_cmd.setText(_str("session_green_cmd", "SESSION_GREEN_REFERENCE"))
        if "red_on_cmd" in d:
            self._red_on_cmd.setText(_str("red_on_cmd", "SESSION_RED_ON"))
        if "trial_start_cmd" in d:
            self._trial_start_cmd.setText(_str("trial_start_cmd", "START_TRIAL"))
        if "lights_off_cmd" in d:
            self._lights_off_cmd.setText(_str("lights_off_cmd", "LIGHTS_OFF"))
        if "leave_lights_on_exit" in d:
            self._leave_lights_on.setChecked(_bool("leave_lights_on_exit"))

        # A loaded preset can change trials / iti / warm-up / analyze; keep the
        # static session-duration estimate in step with the new form values.
        self._refresh_trials_estimate()

    # ── Argv builders ─────────────────────────────────────────────────────────

    def _dated_output(self) -> str:
        return _dated_output_folder(self._output_edit.text().strip())

    def _build_green_cal_argv(self) -> list[str]:
        d = self.to_dict()
        args = [
            "--output", self._dated_output(),
            "--port", d["port"],
            "--label", "green",
            "--stage", "green",
            "--min-us", str(d["cal_min_us"]),
            "--max-us", str(d["cal_max_us"]),
            "--steps", str(d["cal_steps"]),
            "--cal-fps", str(d["cal_fps"]),
        ]
        if d["cal_external_trigger"]:
            args.append("--external-trigger-calibration")
        return args

    def _build_green_ref_argv(self) -> list[str]:
        d = self.to_dict()
        args = [
            "--output", self._dated_output(),
            "--port", d["port"],
            "--green-reference-only",
            "--green-frames", str(d["green_frames"]),
            "--exposure-us", str(self._stage_results["green_exposure_us"]),
            "--green-exposure-us", str(self._stage_results["green_exposure_us"]),
            "--session-green-cmd", d["session_green_cmd"],
        ]
        return args

    def _build_red_cal_argv(self) -> list[str]:
        d = self.to_dict()
        args = [
            "--output", self._dated_output(),
            "--port", d["port"],
            "--label", "red",
            "--stage", "red",
            "--min-us", str(d["cal_min_us"]),
            "--max-us", str(d["cal_max_us"]),
            "--cal-red-max-us", str(d["cal_red_max_us"]),
            "--steps", str(d["cal_steps"]),
            "--cal-fps", str(d["cal_fps"]),
        ]
        if d["skip_roi_selection"]:
            args.append("--skip-roi-selection")
        if d["cal_external_trigger"]:
            args.append("--external-trigger-calibration")
        return args

    def _build_final_stage_argv(
        self,
        *,
        trials: int | None = None,
        iti: float | None = None,
        red_stabilization_s: float | None = None,
        session_dir: Path | str | None = None,
    ) -> list[str]:
        """Build argv for the trials stage. trials/iti/red_stabilization_s
        override the form's values when given — used by the follow-up-session
        flow to run a shortened-warm-up repeat batch without touching the
        main form (and thus without disturbing its defaults for a fresh
        session started later). session_dir overrides which session folder
        trial_NNN folders are written into — the follow-up flow points this
        at a fresh folder (see _prepare_followup_session_dir) so its trials
        never collide with the session they're based on."""
        d = self.to_dict()
        args: list[str] = []

        args += ["--output", self._dated_output()]
        args += ["--port", d["port"]]
        args += ["--final-stage"]
        args += ["--red-exposure-us", str(self._stage_results["red_exposure_us"])]
        args += ["--green-exposure-us", str(self._stage_results["green_exposure_us"])]
        args += ["--roi-config", str(self._stage_results["roi_json_path"])]
        args += ["--session-dir", str(session_dir if session_dir is not None else self._stage_results["session_dir"])]

        args += ["--trials", str(trials if trials is not None else d["trials"])]
        args += ["--iti", str(iti if iti is not None else d["iti"])]
        args += [
            "--red-stabilization-s",
            str(int(red_stabilization_s if red_stabilization_s is not None else d["red_stabilization_s"])),
        ]
        if d["red_settling_monitor"]:
            args += ["--red-settling-sample-interval-s", str(_SETTLING_SAMPLE_INTERVAL_S)]
        if d["analyze"]:
            args.append("--analyze")
        if d["visual_stim"]:
            args.append("--visual-stim")
            if d["stim_orientations"]:
                args += ["--stim-orientations", d["stim_orientations"]]
            args += ["--stim-duration-s", str(d["stim_duration_s"])]
            args += ["--stim-host", d["stim_host"]]
            args += ["--stim-port", str(d["stim_port"])]

        if d["interleave_stim"]:
            n_trials_for_seq = trials if trials is not None else d["trials"]
            conditions, _seed = self._current_trial_conditions_and_seed(n_trials_for_seq)
            args += ["--trial-conditions", conditions_to_argv_value(conditions)]

        # Frames
        args += ["--green-frames", str(d["green_frames"])]
        args += ["--green-reference-trim-frames", str(d["green_reference_trim_frames"])]
        args += ["--baseline-frames", str(d["baseline_frames"])]
        args += ["--post-frames", str(d["post_frames"])]
        if not d["save_gap_frames"]:
            args.append("--no-save-gap-frames")

        # Binning / format
        args += ["--binning", str(d["binning"])]
        if d["analysis_binning"]:
            args += ["--analysis-binning", str(d["analysis_binning"])]
        args += ["--camera-binning", str(d["camera_binning"])]
        args += ["--save-format", d["save_format"]]

        # Analysis
        args += ["--analysis-start-frame", str(d["analysis_start_frame"])]
        args += ["--analysis-end-frame", str(d["analysis_end_frame"])]
        args += ["--analysis-smoothing-sigma", str(d["analysis_smoothing_sigma"])]
        args += ["--analysis-method", d["analysis_method"]]
        args += ["--analysis-mask-percentile", str(d["analysis_mask_percentile"])]
        args += [
            "--analysis-denominator-floor-percentile",
            str(d["analysis_denominator_floor_percentile"]),
        ]
        args += [
            "--analysis-denominator-floor-counts",
            str(d["analysis_denominator_floor_counts"]),
        ]
        args += ["--analysis-median-filter-size", str(d["analysis_median_filter_size"])]
        args += ["--rescale-bit-depth", str(d["rescale_bit_depth"])]
        args += ["--rescale-mode", d["rescale_mode"]]
        if d["invert_display_signal"]:
            args.append("--invert-display-signal")

        # Lights / markers
        args += ["--session-green-cmd", d["session_green_cmd"]]
        args += ["--red-on-cmd", d["red_on_cmd"]]
        args += ["--trial-start-cmd", d["trial_start_cmd"]]
        args += ["--lights-off-cmd", d["lights_off_cmd"]]
        if d["leave_lights_on_exit"]:
            args.append("--leave-lights-on-exit")

        return args

    def _script_and_argv_for_stage(self, key: str) -> tuple[str, list[str]]:
        if key == "green_cal":
            return _CALIBRATION_SCRIPT, self._build_green_cal_argv()
        if key == "green_ref":
            return _IMAGING_SCRIPT, self._build_green_ref_argv()
        if key == "red_cal":
            return _CALIBRATION_SCRIPT, self._build_red_cal_argv()
        return _CALIBRATED_IMAGING_SCRIPT, self._build_final_stage_argv()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_common(self) -> str | None:
        out = self._output_edit.text().strip()
        if not out:
            return "Output folder is required."
        try:
            Path(out).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"Cannot create output folder: {exc}"

        if not self._safety_bar.selected_port():
            return "Select an Arduino COM port at the top of the window."

        return None

    # ── Show command ──────────────────────────────────────────────────────────

    def _next_stage_key(self) -> str | None:
        for key, _label in _STAGES:
            if not self._stage_status[key].property("done"):
                return key
        return None

    def _show_command(self) -> None:
        key = self._next_stage_key()
        if key is None:
            QMessageBox.information(self, "All stages complete", "Every stage has already run. Use Reset to start a new session.")
            return
        err = self._validate_common()
        if not err and key == "trials":
            err = self._trial_sequence_error_message(self._trials_spin.value())
        script, stage_args = self._script_and_argv_for_stage(key) if not err else ("", [])
        argv = self._runner.launcher + [script] + (stage_args if not err else [])
        text = " ".join(argv) if not err else f"(cannot build command yet: {err})"
        if not err and key == "trials":
            try:
                _est_line = self._trials_estimate_text()
            except Exception:
                _est_line = ""
            if _est_line:
                text += "\n\n# Estimated: " + _est_line
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Command for: {dict(_STAGES)[key]}")
        dlg.resize(720, 140)
        lay = QVBoxLayout(dlg)
        edit = QPlainTextEdit(text)
        edit.setReadOnly(True)
        edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        lay.addWidget(edit)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        dlg.exec()

    # ── Stage sequencing ──────────────────────────────────────────────────────

    def _reset_stage_state(self) -> None:
        if self._runner.is_running:
            QMessageBox.warning(self, "Stage running", "Stop the current stage before resetting.")
            return
        self._current_stage = ""
        self._stage_results = {
            "green_exposure_us": None,
            "session_dir": None,
            "red_exposure_us": None,
            "roi_json_path": None,
        }
        for key, _label in _STAGES:
            self._stage_status[key].setProperty("done", False)
            self._stage_status[key].setText("Not started")
            self._stage_status[key].setStyleSheet(f"color:{text_rgba(0.55)};")
        self._stage_touched.clear()
        self._stage_failed.clear()
        self._feed_done.clear()
        self._summary_seen = False
        self._summary_toggle.setChecked(False)
        self._summary_body.setVisible(False)
        self._stage_buttons["green_cal"].setEnabled(True)
        self._stage_buttons["green_ref"].setEnabled(False)
        self._stage_buttons["red_cal"].setEnabled(False)
        self._stage_buttons["trials"].setEnabled(False)
        self._followup_btn.setEnabled(False)
        self._log.clear()
        self._set_progress_idle()
        self._progress.setFormat("")
        self._health_box.hide()
        self._post_run_label.hide()
        self._open_folder_btn.hide()
        self._skip_stabilization_btn.setEnabled(False)
        self._settling_plot.hide()
        self._settling_plot.clear_points()
        self._settling_image.hide()
        self._settling_image.clear_image()
        self._peek_btn.setEnabled(False)
        self._analyze_session_btn.setEnabled(False)
        self._panel_image.clear_all()
        # Live ETA readout: stop and clear; recompute the static estimate for
        # the (now reset) form.
        self._eta_timer.stop()
        self._eta_label.setText("")
        self._eta_phase = "idle"
        self._refresh_trials_estimate()
        self._refresh_stepper_visuals()

    def _mark_stage_status(self, key: str, text: str, *, done: bool, failed: bool = False) -> None:
        self._stage_touched.add(key)
        if failed:
            self._stage_failed.add(key)
        else:
            self._stage_failed.discard(key)
        lbl = self._stage_status[key]
        lbl.setProperty("done", done)
        lbl.setText(text)
        if failed:
            lbl.setStyleSheet(f"color:{DANGER_HEX};")
        elif done:
            lbl.setStyleSheet(f"color:{GREEN_HEX};")
        else:
            lbl.setStyleSheet(f"color:{text_rgba(0.55)};")
        self._refresh_stepper_visuals()

    def _set_progress_idle(self) -> None:
        """Nothing is running: hide the progress row so it doesn't take space
        above the stepper (see _begin_stage_launch for where it reappears)."""
        self._phase_label.setText("Idle.")
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress_area.hide()

    def _begin_stage_launch(self, key: str) -> None:
        """Shared bookkeeping right before a stage subprocess is launched."""
        self._current_stage = key
        self._progress_area.show()
        self._log.clear()
        # Launching a stage implicitly satisfies the feed checkpoint(s) before
        # it — a green calibration means the animal is positioned, a red one
        # means it's refocused below the surface.
        if key == "green_cal":
            self._feed_done.add(_FEED_SURFACE)
        elif key == "red_cal":
            self._feed_done.update((_FEED_SURFACE, _FEED_BELOW))
        elif key == "trials":
            self._feed_done.update((_FEED_SURFACE, _FEED_BELOW))
        self._mark_stage_status(key, "running…", done=False)
        for k, _label in _STAGES:
            self._stage_buttons[k].setEnabled(False)
        self._followup_btn.setEnabled(False)
        self._refresh_stepper_visuals()

    def _reset_trials_run_state(self, num_trials: int, iti_s: float) -> None:
        """Reset the progress/health-panel state for a trials-stage run
        (used for both the normal step-6 launch and a follow-up session).
        iti_s is the ITI of *this run* — the follow-up dialog's value can
        differ from the main form's, and the live ETA must use the run's."""
        self._daily_folder = None
        self._num_trials = num_trials
        self._trials_completed = 0
        self._peak_signals = []
        self._health_warnings = []
        self._health_box.hide()
        self._phase_label.setText("Phase: red stabilization")
        self._progress.setRange(0, 0)
        self._progress.setFormat("")
        self._post_run_label.hide()
        self._open_folder_btn.hide()
        self._settling_plot.clear_points()
        self._settling_plot.setVisible(self._settling_monitor_check.isChecked())
        # Unlike the plot, the settling image only appears once the warm-up
        # wait actually starts sampling (_RE_WARMUP_START) -- reset it to
        # hidden here rather than pre-showing an empty placeholder.
        self._settling_image.hide()
        self._settling_image.clear_image()
        self._panel_image.clear_all()

        # Live "time remaining": (re)anchor for a fresh trials-stage run.
        # Reached from both the normal launch and the follow-up flow, so a
        # follow-up's shortened / zero warm-up is handled here too. The first
        # parsed warm-up line (_RE_WARMUP_START / _RE_COUNTDOWN) or
        # _RE_WARMUP_DONE corrects _eta_phase / _eta_warmup_left.
        self._eta_phase = "warmup"
        self._eta_warmup_left = 0.0
        self._eta_iti_left = 0.0
        self._eta_iti_s = max(0.0, float(iti_s))
        self._eta_total_trials = num_trials
        self._eta_trial_k = 0
        self._eta_secs_in_phase = 0
        self._eta_measured_trial_cost = None
        self._eta_trial1_monotonic = None
        self._eta_label.setText("")
        self._eta_label.show()
        self._eta_timer.start()

    def _on_stage_run(self, key: str) -> None:
        if self._runner.is_running:
            QMessageBox.warning(self, "Already running", "A stage is already running. Stop it first.")
            return

        if self._live_runner.is_running:
            QMessageBox.warning(
                self, "Live feed open",
                "Close the live camera feed before starting a step — they "
                "can't share the camera.",
            )
            return

        err = self._validate_common()
        if err:
            QMessageBox.warning(self, "Cannot start", err)
            return

        if key == "green_cal":
            override = self._green_exposure_us.value()
            if override > 0:
                self._stage_results["green_exposure_us"] = override
                self._mark_stage_status("green_cal", f"Using manual override: {override:.0f} µs", done=True)
                self._stage_buttons["green_ref"].setEnabled(True)
                # Steps 1 and 2 are one combined action from the user's
                # perspective (see _on_done's green_cal branch for the
                # subprocess-calibration path) -- chain straight into
                # capturing the green reference instead of waiting for a
                # second click.
                self._on_stage_run("green_ref")
                return

        if key == "green_ref" and self._stage_results["green_exposure_us"] is None:
            QMessageBox.warning(self, "Cannot start", "Complete step 2 (Calibrate Green Exposure) first.")
            return

        if key == "red_cal" and self._stage_results["session_dir"] is None:
            QMessageBox.warning(self, "Cannot start", "Complete step 3 (Capture Green Reference) first.")
            return

        if key == "trials":
            if self._stage_results["red_exposure_us"] is None or self._stage_results["roi_json_path"] is None:
                QMessageBox.warning(self, "Cannot start", "Complete step 5 (Calibrate Red Exposure + ROI) first.")
                return
            stim_err = self._visual_stim_unreachable_message()
            if stim_err:
                QMessageBox.warning(self, "Cannot start trials", stim_err)
                return
            seq_err = self._trial_sequence_error_message(self._trials_spin.value())
            if seq_err:
                QMessageBox.warning(self, "Invalid trial sequence", seq_err)
                return

        script, argv = self._script_and_argv_for_stage(key)
        self._begin_stage_launch(key)

        if key == "trials":
            self._reset_trials_run_state(
                self._trials_spin.value(), self._iti_spin.value()
            )
            if self._interleave_check.isChecked():
                self._log_trial_sequence()
        else:
            self._phase_label.setText(f"Running: {dict(_STAGES)[key]}")
            self._progress.setRange(0, 0)
            self._progress.setFormat("")

        if key == "red_cal":
            # Red calibration is where illumination switches green -> red; turn
            # red on up front so it's lit (and stays lit from here on) before
            # the calibration subprocess itself takes over the port.
            self._mark_stage_status(key, "turning on red illumination…", done=False)
            self._send_light_command(
                CAL_RED_ON_CMD, "red illumination",
                on_success=lambda: self._launch_stage(key, script, argv),
                on_failure=lambda: self._cancel_stage_launch(key),
            )
        else:
            self._launch_stage(key, script, argv)

    def _launch_stage(self, key: str, script: str, argv: list[str]) -> None:
        self._mark_stage_status(key, "running…", done=False)
        self._stop_btn.setEnabled(True)
        if key == "trials":
            # Guarantee the pinned red Stop button is on screen before the
            # long run — the panel lives in the left QScrollArea.
            self._left_scroll.ensureWidgetVisible(self._stage_box)
        try:
            self._runner.start(script, argv)
        except (RuntimeError, OSError, FileNotFoundError) as exc:
            self._mark_stage_status(key, f"Launch failed: {exc}", done=False, failed=True)
            self._current_stage = ""
            self._eta_timer.stop()
            self._eta_label.setText("")
            self._reenable_stage_buttons()
            self._stop_btn.setEnabled(False)
            return
        self._safety_bar.set_session_active(True)
        # Repaint now that the runner is actually running, so the active card
        # flips from "active" to "running…" (the earlier repaints in
        # _begin_stage_launch / _mark_stage_status ran before start()).
        self._refresh_stepper_visuals()

    def _cancel_stage_launch(self, key: str) -> None:
        self._mark_stage_status(key, "Cancelled — red illumination command failed", done=False, failed=True)
        self._current_stage = ""
        self._set_progress_idle()
        self._reenable_stage_buttons()

    def _reenable_stage_buttons(self) -> None:
        self._stage_buttons["green_cal"].setEnabled(True)
        self._stage_buttons["green_ref"].setEnabled(self._stage_results["green_exposure_us"] is not None)
        self._stage_buttons["red_cal"].setEnabled(self._stage_results["session_dir"] is not None)
        calibration_ready = (
            self._stage_results["red_exposure_us"] is not None
            and self._stage_results["roi_json_path"] is not None
        )
        self._stage_buttons["trials"].setEnabled(calibration_ready)
        self._followup_btn.setEnabled(calibration_ready)
        self._peek_btn.setEnabled(
            self._stage_results["session_dir"] is not None and not self._peek_runner.is_running
        )
        self._analyze_session_btn.setEnabled(self._stage_results["session_dir"] is not None)
        self._refresh_stepper_visuals()

    # ── Guided Vertical Stepper — pure view over the existing state ────────────

    def _on_feed_continue(self, feed_id: str) -> None:
        """A camera-feed checkpoint step's primary action: mark it passed and
        let the stepper advance. Refuses while the feed is still open (the
        next step needs the camera)."""
        if self._runner.is_running or self._live_runner.is_running:
            QMessageBox.information(
                self, "Busy",
                "Stop the live camera feed (and any running step) first, then "
                "press Continue.",
            )
            return
        self._feed_done.add(feed_id)
        self._refresh_stepper_visuals()

    def _refresh_stepper_visuals(self) -> None:
        """Re-derive every step card's appearance from the existing state: the
        per-stage ``done`` property (still the sole input to _next_stage_key /
        "Show next command"), self._current_stage, self._stage_failed,
        self._feed_done and self._runner.is_running. Pure view — never writes
        the ``done`` property, and only writes status text for stage steps the
        user has not seen a real status for (self._stage_touched), so
        "Done — 8000 µs" / "Failed (exit N)" from _mark_stage_status stay
        intact."""
        # Guard on the LAST widget this method touches (created after the
        # _STEPPER_ITEMS loop) — so a hypothetical mid-construction call can
        # never pass the guard and then AttributeError on _summary_toggle.
        if not hasattr(self, "_summary_body"):
            return
        running = self._runner.is_running

        # A feed checkpoint is implicitly satisfied once any stage after it in
        # the stepper has been attempted (touched) — you can't have reached
        # that stage without passing the focus check. Keeps the stepper
        # consistent across reset / preset load / re-run / a mid-sequence
        # failure without special-casing every launch path.
        for idx, (kind, item_id, _t) in enumerate(_STEPPER_ITEMS):
            if kind != "feed":
                continue
            if any(
                k2 == "stage" and id2 in self._stage_touched
                for (k2, id2, _l) in _STEPPER_ITEMS[idx + 1:]
            ):
                self._feed_done.add(item_id)

        def _is_done(kind: str, item_id: str) -> bool:
            if kind == "feed":
                return item_id in self._feed_done
            return bool(self._stage_status[item_id].property("done"))

        active_id: str | None = None
        if running and self._current_stage:
            active_id = self._current_stage
        else:
            for kind, item_id, _t in _STEPPER_ITEMS:
                if not _is_done(kind, item_id):
                    active_id = item_id
                    break

        prev_title: str | None = None
        for kind, item_id, title in _STEPPER_ITEMS:
            num = _STEP_NUM[item_id]
            done = _is_done(kind, item_id)
            failed = kind == "stage" and item_id in self._stage_failed
            if done:
                state = "done"
            elif running and kind == "stage" and item_id == self._current_stage:
                state = "running"
            elif item_id == active_id:
                state = "active"
            else:
                state = "locked"
            expanded = state in ("running", "active")

            # Number badge (glyph + colour; fixed size => no relayout).
            if state == "done":
                btxt, bcol = _CHECK_GLYPH, GREEN_HEX
            elif state == "running":
                btxt, bcol = str(num), CHROME_ACCENT_HEX
            elif state == "active":
                btxt = str(num)
                bcol = DANGER_HEX if failed else CHROME_ACCENT_HEX
            else:
                btxt, bcol = str(num), text_rgba(0.45)
            badge = self._step_badges[item_id]
            badge.setText(btxt)
            badge.setStyleSheet(
                f"QLabel{{color:{bcol};border:1px solid {bcol};"
                f"border-radius:10px;font-weight:bold;}}"
            )

            self._step_titles[item_id].setStyleSheet(
                "font-weight:bold;"
                + (f"color:{text_rgba(0.45)};" if state == "locked" else "")
            )
            border_col = {
                "done": text_rgba(0.15),
                "running": CHROME_ACCENT_HEX,
                "active": DANGER_HEX if failed else CHROME_ACCENT_HEX,
                "locked": text_rgba(0.10),
            }[state]
            self._step_frames[item_id].setStyleSheet(
                f"QFrame#stageStep{{border:1px solid {border_col};"
                f"border-radius:4px;}}"
            )

            self._step_descs[item_id].setVisible(expanded)
            self._step_rules[item_id].setVisible(expanded)

            if kind == "feed":
                self._step_feeds[item_id].setVisible(state == "active")
                cbtn = self._feed_continue_btns[item_id]
                cbtn.setVisible(state == "active")
                if state == "active":
                    cbtn.setText("Continue  ▶")
                    cbtn.setStyleSheet(STYLE_START)
            else:
                gate = self._step_gates[item_id]
                if state == "locked":
                    gate.setText(
                        f"Complete step {num - 1} ({prev_title}) first."
                        if prev_title else "Complete the earlier steps first."
                    )
                    gate.show()
                else:
                    gate.hide()

                if item_id not in self._stage_touched:
                    slbl = self._stage_status[item_id]
                    slbl.setText("Not started")
                    slbl.setStyleSheet(f"color:{text_rgba(0.55)};")

                btn = self._stage_buttons[item_id]
                if state == "locked":
                    btn.setVisible(False)
                elif state == "done":
                    btn.setVisible(True)
                    btn.setText(f"Redo {num}…")
                    btn.setStyleSheet("")
                elif state == "running":
                    btn.setVisible(True)
                    btn.setText("running…")
                    btn.setStyleSheet(STYLE_START if item_id == "trials" else "")
                else:  # active
                    btn.setVisible(True)
                    if failed:
                        btn.setText(f"▶  Retry {title}")
                    elif item_id == "trials":
                        btn.setText("▶  Start Trials")
                    else:
                        btn.setText(f"Run step {num}")
                    btn.setStyleSheet(STYLE_START)

            prev_title = title

        self._update_stage_next_label(active_id, running)
        self._refresh_session_summary()

        # Keep the current step and its primary button on screen as the
        # workflow advances. Not on the very first refresh: at launch the
        # parameters above need to stay visible.
        if active_id != self._last_active_step:
            if self._last_active_step is not None and active_id is not None:
                frame = self._step_frames[active_id]
                QTimer.singleShot(0, lambda f=frame: self._left_scroll.ensureWidgetVisible(f, 0, 24))
            self._last_active_step = active_id

    def _update_stage_next_label(
        self, active_id: str | None, running: bool
    ) -> None:
        """The persistent 'Next:' header line — deliberately redundant with the
        highlighted card, for reassurance."""
        if active_id is None:
            self._stage_next_label.setText("All steps complete.")
            self._stage_next_label.setStyleSheet(
                f"font-weight:bold;color:{GREEN_HEX};"
            )
            return
        num = _STEP_NUM[active_id]
        title = _STEP_TITLE[active_id]
        if running and active_id == self._current_stage:
            self._stage_next_label.setText(f"Running now:  {num}. {title}")
            self._stage_next_label.setStyleSheet(
                f"font-weight:bold;color:{CHROME_ACCENT_HEX};"
            )
        elif active_id in self._stage_failed:
            self._stage_next_label.setText(
                f"Retry {num}. {title} — previous attempt failed, see log"
            )
            self._stage_next_label.setStyleSheet(
                f"font-weight:bold;color:{DANGER_HEX};"
            )
        else:
            self._stage_next_label.setText(f"Next:  ▶  {num}. {title}")
            self._stage_next_label.setStyleSheet("font-weight:bold;")

    def _refresh_session_summary(self) -> None:
        """Mirror the cached _stage_results into the 'Session so far' form so
        resolved calibration values stay visible after their step cards have
        collapsed. Auto-expands once, on the first captured result."""
        r = self._stage_results

        def _us(v: Any) -> str:
            return f"{v:.0f} µs" if v is not None else "—"

        self._sum_green_val.setText(_us(r["green_exposure_us"]))
        self._sum_session_val.setText(
            str(r["session_dir"]) if r["session_dir"] else "—"
        )
        self._sum_red_val.setText(_us(r["red_exposure_us"]))
        self._sum_roi_val.setText(
            str(r["roi_json_path"]) if r["roi_json_path"] else "—"
        )
        n = sum(1 for v in r.values() if v is not None)
        self._summary_toggle.setText(f"Session so far  ({n}/{len(r)} captured)")
        if n and not self._summary_seen:
            self._summary_seen = True
            self._summary_toggle.setChecked(True)   # -> _on_summary_toggled

    def _on_summary_toggled(self, checked: bool) -> None:
        self._summary_body.setVisible(checked)
        self._summary_toggle.setIcon(self.style().standardIcon(
            QStyle.StandardPixmap.SP_ArrowDown if checked
            else QStyle.StandardPixmap.SP_ArrowRight
        ))

    # ── Trials-stage duration estimate ───────────────────────────────────────

    def _estimate_trials_stage_seconds(
        self, *, trials: int, iti: float, warmup_s: float, analyze: bool
    ) -> float:
        """Instance wrapper around estimate_trials_stage_seconds() that folds
        in the small per-trial post-idle-timeout penalty when a Frames-tab
        override pushes --baseline-frames / --post-frames above the firmware
        trigger counts. Shared by the static form estimate and the follow-up
        dialog."""
        extra = 0.0
        try:
            if (
                self._post_frames.value() > _FIRMWARE_POST_FRAMES
                or self._baseline_frames.value() > _FIRMWARE_BASELINE_FRAMES
            ):
                extra = _POST_IDLE_TIMEOUT_S
        except Exception:
            extra = 0.0
        return estimate_trials_stage_seconds(
            trials=trials, iti_s=iti, warmup_s=warmup_s, analyze=analyze,
            extra_per_trial_s=extra,
        )

    def _trials_estimate_text(self) -> str:
        """One-line human estimate for the trials stage built from the main
        form, or "" if it can't be computed. Shared by the static card label
        and the Show-command dialog."""
        n = self._trials_spin.value()
        if n < 1:
            return ""
        iti_s = self._iti_spin.value()
        warmup_s = self._warmup_spin.value()
        analyze = self._analyze_check.isChecked()
        est = self._estimate_trials_stage_seconds(
            trials=n, iti=iti_s, warmup_s=warmup_s, analyze=analyze,
        )
        finish = datetime.now() + timedelta(seconds=est)
        over_day = est >= 86400
        clock = format_finish_clock(finish, with_weekday=over_day)

        extra = _POST_IDLE_TIMEOUT_S if (
            self._post_frames.value() > _FIRMWARE_POST_FRAMES
            or self._baseline_frames.value() > _FIRMWARE_BASELINE_FRAMES
        ) else 0.0
        cycle_s = int(round(
            _HW_TRIAL_S + _TRAILING_DRAIN_S + _PER_TRIAL_IO_S + extra
            + max(0.0, iti_s)
        ))
        if warmup_s >= 60:
            warm_part = f"{int(round(warmup_s / 60))} min warm-up"
        elif warmup_s > 0:
            warm_part = f"{int(round(warmup_s))} s warm-up"
        else:
            warm_part = "no warm-up"

        text = (
            f"{format_duration_hm(est)} total  —  finishes ~ {clock}"
            f"   ({warm_part} + {n} trial{'s' if n != 1 else ''} × ~{cycle_s} s)"
        )
        # Green reference not captured yet -> its ~20 s is still to pay.
        if self._stage_results.get("session_dir") is None:
            text = "+ ~20 s green reference  —  " + text
        return text

    def _refresh_trials_estimate(self) -> None:
        """Recompute the static pre-run estimate label. Never raises into the
        UI — on any failure the label is simply blanked."""
        lbl = getattr(self, "_trials_estimate_label", None)
        if lbl is None:
            return
        try:
            lbl.setText(self._trials_estimate_text())
        except Exception:
            try:
                lbl.setText("")
            except Exception:
                pass

    def _tick_eta(self) -> None:
        """1 Hz interpolation of the live 'time remaining' readout between the
        anchored log-line events parsed in _on_trials_line. Clamped so it never
        goes negative; wrapped so a bad state never surfaces a traceback."""
        try:
            if self._current_stage != "trials" or self._eta_phase == "idle":
                self._eta_label.setText("")
                return

            self._eta_secs_in_phase += 1
            iti_s = max(0.0, self._eta_iti_s)
            analyze = self._analyze_check.isChecked()

            per_trial_cost = (
                self._eta_measured_trial_cost
                if self._eta_measured_trial_cost is not None
                else (_HW_TRIAL_S + _TRAILING_DRAIN_S + _PER_TRIAL_IO_S)
            )

            total_trials = max(self._eta_total_trials, self._eta_trial_k)
            k = self._eta_trial_k

            if self._eta_phase == "warmup":
                self._eta_warmup_left = max(0.0, self._eta_warmup_left - 1.0)
                phase_left = self._eta_warmup_left
                trials_ahead = max(0, total_trials - k)
                itis_ahead = max(0, trials_ahead - 1)
            elif self._eta_phase == "iti":
                self._eta_iti_left = max(0.0, self._eta_iti_left - 1.0)
                phase_left = self._eta_iti_left
                trials_ahead = max(0, total_trials - k)
                itis_ahead = max(0, trials_ahead - 1)
            elif self._eta_phase == "trial-running":
                phase_left = max(0.0, per_trial_cost - self._eta_secs_in_phase)
                trials_ahead = max(0, total_trials - k)
                itis_ahead = trials_ahead
            else:  # "trials": warm-up done, first trial not announced yet
                phase_left = 0.0
                trials_ahead = total_trials
                itis_ahead = max(0, total_trials - 1)

            remaining = (
                phase_left
                + trials_ahead * per_trial_cost
                + itis_ahead * iti_s
            )
            if analyze and k < total_trials:
                remaining += _PER_TRIAL_ANALYSIS_S

            self._eta_label.setText(format_time_left(max(0.0, remaining)))
        except Exception:
            try:
                self._eta_label.setText("")
            except Exception:
                pass

    def _visual_stim_unreachable_message(self) -> str | None:
        """None if trials can start; otherwise the reason they can't, for a
        warning dialog. Shared by the normal trials launch and follow-up."""
        if self._stim_modality_combo.currentData() != "visual":
            return None
        host = self._stim_host_edit.text().strip() or "127.0.0.1"
        port = self._stim_port_spin.value()
        if probe_once(host=host, port=port) is StimState.DOWN:
            return (
                f"Visual stim is enabled but the stim server is not "
                f"responding on {host}:{port}.\n\n"
                "Start intrinsic_visual_stimulus.py (Utilities tab) before launching."
            )
        return None

    # ── Follow-up session (reuse cached calibration) ──────────────────────────

    def _on_run_followup(self) -> None:
        if self._runner.is_running:
            QMessageBox.warning(self, "Already running", "A stage is already running. Stop it first.")
            return

        results = self._stage_results
        if (
            results["red_exposure_us"] is None
            or results["roi_json_path"] is None
            or results["session_dir"] is None
        ):
            QMessageBox.warning(
                self, "Cannot start",
                "Run the calibration steps at least once first — a follow-up "
                "session reuses their cached green reference, exposure, and "
                "ROI/crop.",
            )
            return

        err = self._validate_common()
        if err:
            QMessageBox.warning(self, "Cannot start", err)
            return

        stim_err = self._visual_stim_unreachable_message()
        if stim_err:
            QMessageBox.warning(self, "Cannot start trials", stim_err)
            return

        params = self._prompt_followup_params()
        if params is None:
            return
        trials, iti, red_stabilization_s = params

        seq_err = self._trial_sequence_error_message(trials)
        if seq_err:
            QMessageBox.warning(self, "Invalid trial sequence", seq_err)
            return

        old_session_dir = self._stage_results["session_dir"]
        new_session_dir = self._prepare_followup_session_dir()
        if new_session_dir is None:
            return

        argv = self._build_final_stage_argv(
            trials=trials, iti=iti, red_stabilization_s=red_stabilization_s,
            session_dir=new_session_dir,
        )
        # Advance the cached session dir so a later follow-up (or the plain
        # "Start Trials" step, if pressed again) builds on this new folder
        # rather than the one it was just based on.
        self._stage_results["session_dir"] = str(new_session_dir)
        self._begin_stage_launch("trials")
        self._reset_trials_run_state(trials, iti)
        self._log.appendPlainText(
            f"Follow-up session: new session folder {new_session_dir}\n"
            f"  (green reference copied from {old_session_dir})"
        )
        if self._interleave_check.isChecked():
            self._log_trial_sequence(n_trials=trials)
        self._launch_stage("trials", _CALIBRATED_IMAGING_SCRIPT, argv)

    def _prepare_followup_session_dir(self) -> Path | None:
        """Create a brand-new session folder for a follow-up run and copy the
        cached session's green-reference frames into it, so the follow-up's
        trials land in their own folder (never colliding with the session
        they're based on) while --final-stage still finds a precaptured
        green reference to reuse instead of recapturing it."""
        old_session_dir = Path(self._stage_results["session_dir"])
        old_green_dir = old_session_dir / "session_green_reference"
        if not old_green_dir.is_dir() or not any(old_green_dir.iterdir()):
            QMessageBox.warning(
                self, "Cannot start",
                f"No session green reference found in:\n{old_green_dir}\n\n"
                "Complete step 3 (Capture Green Reference) first.",
            )
            return None

        new_session_dir = Path(self._dated_output()) / datetime.now().strftime("session_%Y%m%d_%H%M%S")
        try:
            new_session_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                old_green_dir, new_session_dir / "session_green_reference",
                dirs_exist_ok=True,
            )
        except OSError as exc:
            QMessageBox.warning(
                self, "Could not prepare follow-up session",
                f"Could not set up the new session folder:\n{exc}",
            )
            return None

        return new_session_dir

    def _prompt_followup_params(self) -> tuple[int, float, float] | None:
        """Small dialog collecting trials/ITI/red-warm-up for a follow-up
        session, independent of the main form so the standard session
        defaults (a full-length warm-up for a fresh animal) aren't disturbed."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Follow-up session — reuse calibration")
        dlg.resize(420, 0)
        lay = QVBoxLayout(dlg)

        info = QLabel(
            "Reuses the green reference, exposures, and ROI/crop already "
            "captured above — no new calibration runs. Set a shortened "
            "red warm-up below if the LED is still warm from the session "
            "just finished."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        form = QFormLayout()

        trials_spin = QSpinBox()
        trials_spin.setRange(1, 9999)
        trials_spin.setValue(self._trials_spin.value())
        trials_spin.setSuffix("  trial(s)")
        form.addRow("Trials:", trials_spin)

        iti_spin = QDoubleSpinBox()
        iti_spin.setRange(0.0, 3600.0)
        iti_spin.setDecimals(1)
        iti_spin.setValue(self._iti_spin.value())
        iti_spin.setSuffix(" s")
        form.addRow("Inter-trial interval:", iti_spin)

        default_warmup = self._warmup_spin.value()
        warmup_spin = QDoubleSpinBox()
        warmup_spin.setRange(0.0, 7200.0)
        warmup_spin.setDecimals(0)
        warmup_spin.setValue(min(60.0, default_warmup))
        warmup_spin.setSuffix(" s")
        form.addRow("Red warm-up (this run only):", warmup_spin)

        lay.addLayout(form)

        est_lbl = QLabel()
        est_lbl.setWordWrap(True)
        est_lbl.setStyleSheet(f"color:{text_rgba(0.7)};")
        lay.addWidget(est_lbl)

        def _update_followup_estimate() -> None:
            try:
                secs = self._estimate_trials_stage_seconds(
                    trials=trials_spin.value(), iti=iti_spin.value(),
                    warmup_s=warmup_spin.value(),
                    analyze=self._analyze_check.isChecked(),
                )
                finish = datetime.now() + timedelta(seconds=secs)
                clock = format_finish_clock(finish, with_weekday=secs >= 86400)
                est_lbl.setText(
                    f"{format_duration_hm(secs)} total  —  finishes ~ {clock}"
                )
            except Exception:
                est_lbl.setText("")

        for _w in (trials_spin, iti_spin, warmup_spin):
            _w.valueChanged.connect(_update_followup_estimate)
        _update_followup_estimate()

        warn = QLabel(
            "⚠ A shorter warm-up than the standard session default "
            f"({default_warmup:.0f} s) assumes the red LED is already warm. "
            "If it isn't, signal levels may differ from a fully-settled run."
        )
        warn.setWordWrap(True)
        warn.setStyleSheet(WARNING_STYLE)
        lay.addWidget(warn)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return trials_spin.value(), iti_spin.value(), warmup_spin.value()

    def request_stop(self) -> None:
        """Public entry point for the header SafetyBar's 'Stop Session'
        button, so it goes through the exact same graceful-stop + 15 s
        Force-Kill-escalation path as the in-tab Stop button below — one
        code path, one "Stop timed out" dialog, regardless of which button
        triggered it. No-op if nothing is currently running."""
        if self._runner.is_running:
            self._on_stop()

    def _on_stop(self) -> None:
        self._phase_label.setText("Stopping (graceful — waiting up to 15 s)…")
        self._stop_btn.setEnabled(False)
        self._runner.send_stop_signal()
        self._stop_timer.start()

    def _on_skip_stabilization_clicked(self) -> None:
        session_dir = self._stage_results.get("session_dir")
        if self._current_stage != "trials" or not session_dir:
            return
        reply = QMessageBox.question(
            self, "Skip stabilization?",
            "This ends the red-LED warm-up wait immediately and starts trials "
            "now instead of after the full warm-up period. If the LED hasn't "
            "fully thermally settled, signal levels may differ from a "
            "fully-warmed run.\n\nSkip the remaining wait?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            (Path(session_dir) / _SKIP_RED_STABILIZATION_FILENAME).touch()
        except OSError as exc:
            QMessageBox.warning(self, "Could not signal skip", f"Could not write skip signal:\n{exc}")
            return
        self._skip_stabilization_btn.setEnabled(False)
        self._log.appendPlainText("Skip requested — waiting for the running process to notice…")

    # ── Log line parsing ──────────────────────────────────────────────────────

    def _on_line(self, line: str) -> None:
        if self._current_stage == "trials":
            self._on_trials_line(line)
            return

        if not line.strip():
            return
        self._log.appendPlainText(line)

        if self._current_stage == "green_cal":
            m = _RE_GREEN_CAL_DONE.search(line)
            if m:
                self._stage_results["green_exposure_us"] = float(m.group(1))

        elif self._current_stage == "green_ref":
            m = _RE_GREEN_REF_SESSION.search(line)
            if m:
                self._stage_results["session_dir"] = m.group(1).strip()

        elif self._current_stage == "red_cal":
            m = _RE_RED_CAL_DONE.search(line)
            if m:
                self._stage_results["red_exposure_us"] = float(m.group(1))
            m = _RE_ROI_CONFIG.search(line)
            if m:
                self._stage_results["roi_json_path"] = m.group(1).strip()

    def _on_trials_line(self, line: str) -> None:
        m = _RE_COUNTDOWN.search(line)
        if m:
            mins, secs = int(m.group(1)), int(m.group(2))
            elapsed_s = max(0.0, self._warmup_total_s - (mins * 60 + secs))
            self._progress.setValue(int(elapsed_s))
            self._progress.setFormat(f"Warm-up: {mins:02d}:{secs:02d} remaining")
            # Re-sync the live ETA to the script's own countdown each second.
            self._eta_phase = "warmup"
            self._eta_warmup_left = float(mins * 60 + secs)
            self._eta_secs_in_phase = 0
            return

        m = _RE_ITI_COUNTDOWN.search(line)
        if m:
            mins, secs = int(m.group(1)), int(m.group(2))
            self._eta_phase = "iti"
            self._eta_iti_left = float(mins * 60 + secs)
            self._eta_secs_in_phase = 0
            # Mirror the _RE_COUNTDOWN early-return: intrinsic_imaging.py prints
            # this with a bare '\r' (_wait_with_marker_polling), so it reaches
            # us split out as its own line — progress chatter, not log content.
            return

        if not line.strip():
            return

        self._log.appendPlainText(line)

        m = _RE_DAILY.search(line)
        if m:
            self._daily_folder = m.group(1).strip()

        m = _RE_SETTLING_SAMPLE.search(line)
        if m:
            self._settling_plot.add_point(float(m.group(1)), float(m.group(3)))
            session_dir = self._stage_results.get("session_dir")
            if session_dir:
                self._settling_image.set_image_path(str(Path(session_dir) / "settling_sample.png"))

        m = _RE_WARMUP_START.search(line)
        if m:
            self._warmup_total_s = float(m.group(1))
            self._phase_label.setText(
                f"Phase: red LED warm-up ({int(self._warmup_total_s)} s)"
            )
            self._progress.setRange(0, int(self._warmup_total_s))
            self._progress.setValue(0)
            self._progress.setFormat("Warm-up: starting…")
            self._skip_stabilization_btn.setEnabled(True)
            if self._settling_monitor_check.isChecked():
                self._settling_image.show()
            # Authoritative warm-up total for the live ETA.
            self._eta_phase = "warmup"
            self._eta_warmup_left = float(self._warmup_total_s)
            self._eta_secs_in_phase = 0

        if _RE_WARMUP_DONE.search(line):
            self._phase_label.setText("Phase: trials")
            self._progress.setRange(0, self._num_trials)
            self._progress.setValue(0)
            self._progress.setFormat(f"Trial 0 / {self._num_trials}")
            self._skip_stabilization_btn.setEnabled(False)
            # The settling snapshot is only meaningful while sampling is
            # actually running (the wait above); this fires whether the wait
            # ended naturally or the user hit "Skip stabilization".
            self._settling_image.hide()
            self._settling_image.clear_image()
            # Fires whether the wait ended naturally or via Skip stabilization.
            self._eta_phase = "trials"
            self._eta_warmup_left = 0.0
            self._eta_secs_in_phase = 0

        m = _RE_TRIAL.search(line)
        if m:
            current, total = int(m.group(1)), int(m.group(2))
            self._num_trials = total
            self._progress.setRange(0, total)
            self._progress.setValue(current)
            self._progress.setFormat(f"Trial {current} / {total}")
            # Live ETA anchor + per-trial self-correction.
            self._eta_trial_k = current
            self._eta_total_trials = total
            self._eta_phase = "trial-running"
            self._eta_secs_in_phase = 0
            now = time.monotonic()
            if self._eta_trial1_monotonic is None:
                self._eta_trial1_monotonic = now
            elif self._eta_measured_trial_cost is None:
                # Wall gap between the first two "Sending … for trial k/N"
                # lines minus this run's ITI ≈ the real per-trial cost.
                # Mitigates _HW_TRIAL_S going stale if the .ino is reflashed.
                measured = (now - self._eta_trial1_monotonic) - self._eta_iti_s
                literal = _HW_TRIAL_S + _TRAILING_DRAIN_S + _PER_TRIAL_IO_S
                if 0.0 < measured <= 3.0 * literal:
                    self._eta_measured_trial_cost = measured

        if _RE_DONE.search(line):
            self._phase_label.setText("Phase: complete")
            self._progress.setRange(0, 1)
            self._progress.setValue(1)
            self._progress.setFormat("Done")
            self._eta_timer.stop()
            self._eta_label.setText("")

        m = _RE_TRIAL_COMPLETE.search(line)
        if m:
            self._trials_completed = int(m.group(1))

        m = _RE_ANALYSIS_PEAK.search(line)
        if m:
            try:
                trough = float(m.group(3)) if m.group(3) is not None else None
                self._peak_signals.append((int(m.group(1)), float(m.group(2)), trough))
            except ValueError:
                pass

        m = _RE_PANEL_SAVED.search(line)
        if m:
            self._panel_image.add_trial(int(m.group(2)), m.group(1).strip())

        if _RE_WARNING.search(line):
            self._health_warnings.append(line)

    def _on_done(self, exit_code: int) -> None:
        self._stop_timer.stop()
        self._eta_timer.stop()
        self._eta_label.setText("")
        self._stop_btn.setEnabled(False)
        self._skip_stabilization_btn.setEnabled(False)
        stage = self._current_stage
        self._current_stage = ""

        if stage == "green_cal":
            if exit_code == 0 and self._stage_results["green_exposure_us"] is not None:
                self._mark_stage_status(
                    "green_cal", f"Done — {self._stage_results['green_exposure_us']:.0f} µs", done=True,
                )
            else:
                self._mark_stage_status("green_cal", f"Failed (exit {exit_code}) — see log", done=False, failed=True)
            self._set_progress_idle()

        elif stage == "green_ref":
            if exit_code == 0 and self._stage_results["session_dir"] is not None:
                self._mark_stage_status(
                    "green_ref", f"Done — {self._stage_results['session_dir']}", done=True,
                )
            else:
                self._mark_stage_status("green_ref", f"Failed (exit {exit_code}) — see log", done=False, failed=True)
            self._set_progress_idle()

        elif stage == "red_cal":
            if (
                exit_code == 0
                and self._stage_results["red_exposure_us"] is not None
                and self._stage_results["roi_json_path"] is not None
            ):
                self._mark_stage_status(
                    "red_cal", f"Done — {self._stage_results['red_exposure_us']:.0f} µs", done=True,
                )
            else:
                self._mark_stage_status("red_cal", f"Failed (exit {exit_code}) — see log", done=False, failed=True)
            self._set_progress_idle()

        elif stage == "trials":
            if exit_code == 0:
                self._mark_stage_status("trials", "Done", done=True)
                self._phase_label.setText("Session finished successfully.")
            else:
                self._mark_stage_status("trials", f"Failed (exit {exit_code}) — see log", done=False, failed=True)
                self._phase_label.setText(f"Session ended (exit code {exit_code}). Check rig LEDs.")

            if self._daily_folder:
                self._post_run_label.setText(f"Output: {self._daily_folder}")
                self._post_run_label.show()
                self._open_folder_btn.show()

            self._populate_health_panel()

        self._reenable_stage_buttons()

        # Steps 1 and 2 are one combined action from the user's perspective
        # (see the manual-override short-circuit in _on_stage_run for the
        # other path into this same chain) -- a successful calibration
        # walks straight into capturing the green reference instead of
        # waiting for a second click. _reenable_stage_buttons() above just
        # set every stage button's enabled state from _stage_results; this
        # launch immediately supersedes that for "green_ref" and "trials".
        if stage == "green_cal" and exit_code == 0 and self._stage_results["green_exposure_us"] is not None:
            self._on_stage_run("green_ref")

    def _on_stop_timeout(self) -> None:
        msg = QMessageBox(self)
        msg.setWindowTitle("Stop timed out")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            "The current stage did not exit within 15 seconds after the stop signal.\n\n"
            "Force-killing will skip the teardown. "
            "LEDs may remain on — check the rig before continuing."
        )
        force_btn = msg.addButton("Force Kill", QMessageBox.ButtonRole.DestructiveRole)
        wait_btn = msg.addButton("Keep Waiting", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(wait_btn)
        msg.exec()

        if msg.clickedButton() is force_btn:
            self._runner.force_kill()
            self._phase_label.setText("Force-killed. LEDs may still be on — check the rig.")
            self._eta_timer.stop()
            self._eta_label.setText("")
            if self._current_stage:
                self._mark_stage_status(self._current_stage, "Force-killed", done=False, failed=True)
                self._current_stage = ""
            self._reenable_stage_buttons()
            self._stop_btn.setEnabled(False)
        else:
            self._stop_btn.setEnabled(True)
            self._phase_label.setText("Still running — waiting for process to exit…")

    # ── Light automation ──────────────────────────────────────────────────────

    def _send_light_command(
        self,
        command: str,
        label: str,
        on_success: Callable[[], None],
        on_failure: Callable[[], None] | None = None,
    ) -> None:
        """Send a one-shot Arduino light command directly over serial (open ->
        send -> close, same as red.py/green.py). Only safe while no stage
        subprocess owns the COM port. Calls on_success()/on_failure() once
        the command finishes (marshalled back via _lights_bridge)."""
        port = self._safety_bar.selected_port()
        if not port:
            QMessageBox.warning(self, "No port selected", "Select an Arduino COM port at the top of the window.")
            if on_failure:
                on_failure()
            return
        if self._runner.is_running:
            QMessageBox.warning(
                self, "Stage running",
                "A staged-session subprocess currently owns the COM port. Wait for it to finish first.",
            )
            if on_failure:
                on_failure()
            return
        if self._lights_busy:
            QMessageBox.information(self, "Please wait", "Still talking to the Arduino — try again in a moment.")
            if on_failure:
                on_failure()
            return

        self._lights_busy = True
        self._lights_pending_success = on_success
        self._lights_pending_failure = on_failure
        self._log.appendPlainText(f"Turning {label} on…")
        send_command_async(
            port, command,
            on_message=self._lights_bridge.message.emit,
            on_done=self._lights_bridge.finished.emit,
        )

    def _on_lights_message(self, msg: str) -> None:
        self._log.appendPlainText(msg)

    def _on_lights_done(self, success: bool, detail: str) -> None:
        self._lights_busy = False
        on_success = self._lights_pending_success
        on_failure = self._lights_pending_failure
        self._lights_pending_success = None
        self._lights_pending_failure = None
        if success:
            if on_success:
                on_success()
        else:
            QMessageBox.warning(self, "Light command failed", f"Could not reach the Arduino:\n{detail}")
            if on_failure:
                on_failure()

    # ── Live camera feed shortcuts ────────────────────────────────────────────

    def _set_live_feed_running(self, running: bool) -> None:
        for b in self._live_feed_open_buttons:
            b.setEnabled(not running)
        for b in self._live_feed_stop_buttons:
            b.setEnabled(running)

    def _on_live_feed_start(self) -> None:
        if self._live_runner.is_running:
            QMessageBox.information(self, "Already open", "The live camera feed is already open.")
            return
        self._send_light_command(CAL_GREEN_ON_CMD, "green illumination", self._launch_live_feed)

    def _launch_live_feed(self) -> None:
        args = [
            "--camera-index", "0",
            "--exposure-us", str(_LIVE_FEED_EXPOSURE_US),
            "--pixel-format", "Mono16",
            "--fps", str(_LIVE_FEED_FPS),
        ]
        self._log.appendPlainText("\n=== Live Camera Feed ===")
        try:
            self._live_runner.start(_LIVE_PREVIEW_SCRIPT, args)
        except (RuntimeError, OSError, FileNotFoundError) as exc:
            QMessageBox.warning(self, "Launch failed", str(exc))
            return
        self._set_live_feed_running(True)

    def _on_live_feed_stop(self) -> None:
        self._log.appendPlainText("Stopping live camera feed (graceful — waiting up to 15 s)…")
        for b in self._live_feed_stop_buttons:
            b.setEnabled(False)
        self._live_runner.send_stop_signal()
        self._live_stop_timer.start()

    def _on_live_line(self, line: str) -> None:
        self._log.appendPlainText(line)

    def _on_live_done(self, exit_code: int) -> None:
        self._live_stop_timer.stop()
        label = "closed" if exit_code == 0 else f"exit code {exit_code}"
        self._log.appendPlainText(f"Live camera feed {label}.")
        self._set_live_feed_running(False)

    def _on_live_stop_timeout(self) -> None:
        msg = QMessageBox(self)
        msg.setWindowTitle("Stop timed out")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            "The live camera feed did not exit within 15 seconds after the "
            "stop signal.\n\nForce-killing it will skip camera cleanup."
        )
        force_btn = msg.addButton("Force Kill", QMessageBox.ButtonRole.DestructiveRole)
        wait_btn = msg.addButton("Keep Waiting", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(wait_btn)
        msg.exec()

        if msg.clickedButton() is force_btn:
            self._live_runner.force_kill()
            self._log.appendPlainText("Live camera feed force-killed.")
            self._set_live_feed_running(False)
        else:
            self._live_stop_timer.start()

    # ── ROI time-course "peek" ────────────────────────────────────────────────

    def _on_run_peek(self) -> None:
        if self._peek_runner.is_running:
            QMessageBox.information(self, "Already running", "A peek analysis is already running.")
            return
        session_dir = self._stage_results.get("session_dir")
        if not session_dir:
            QMessageBox.warning(
                self, "Cannot run",
                "No session folder yet — complete step 3 (Capture Green Reference) first.",
            )
            return

        args = [str(session_dir), "--n-perm", str(self._peek_nperm.value())]
        if self._peek_neg_only.isChecked():
            args.append("--neg-only")
        peek_condition = self._peek_condition_combo.currentData()
        if peek_condition != "all":
            args += ["--condition", peek_condition]
        # Subfolder inside the session folder rather than a sibling
        # "<session>_reanalysis" folder (matches the Statistics tab's
        # default). Condition-suffixed so a stim peek and a catch peek of
        # the same session don't overwrite each other.
        peek_subfolder = _PEEK_OUT_SUBFOLDER
        if peek_condition != "all":
            peek_subfolder = f"{peek_subfolder}_{peek_condition}"
        args += ["--out-dir", str(Path(session_dir) / peek_subfolder)]
        self._peek_output_dir = None
        self._peek_open_folder_btn.hide()
        self._peek_status_label.setText("Running…")
        self._peek_btn.setEnabled(False)
        try:
            self._peek_runner.start(_PEEK_ROI_SCRIPT, args)
        except (RuntimeError, OSError, FileNotFoundError) as exc:
            self._peek_status_label.setText(f"Launch failed: {exc}")
            self._peek_btn.setEnabled(True)

    def _on_peek_line(self, line: str) -> None:
        if not line.strip():
            return
        m = _RE_PEEK_DONE.search(line)
        if m:
            self._peek_output_dir = m.group(1).strip()

    def _on_peek_done(self, exit_code: int) -> None:
        if exit_code == 0 and self._peek_output_dir:
            self._peek_status_label.setText(f"Done — {self._peek_output_dir}")
            self._peek_open_folder_btn.show()
        elif exit_code == 0:
            self._peek_status_label.setText("Finished, but no output produced — likely too few usable trials so far.")
        else:
            self._peek_status_label.setText(f"Peek analysis ended (exit code {exit_code}).")
        self._peek_btn.setEnabled(self._stage_results.get("session_dir") is not None)

    def _open_peek_output_folder(self) -> None:
        if self._peek_output_dir:
            try:
                subprocess.Popen(["explorer", self._peek_output_dir])
            except OSError:
                pass

    # ── Post-run folder ───────────────────────────────────────────────────────

    def _open_output_folder(self) -> None:
        if self._daily_folder:
            try:
                subprocess.Popen(["explorer", self._daily_folder])
            except OSError:
                pass
