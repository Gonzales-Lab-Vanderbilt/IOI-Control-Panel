# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Pre-Session Diagnostics — exercises the rig's hardware paths before a real
session starts: Arduino connection + red/green LED on/off, camera
responsiveness, and a live camera preview for animal placement/focus checks.

Each check/feed launches an existing utility script as a subprocess (never
imports PySpin or serial hardware code into the GUI process):
    red.py / green.py           — Arduino connection, LED on then off
    reset_blackfly_roi.py       — camera opens and responds
    live_preview.py             — free-running live camera feed (own window)

Checks run sequentially on one ScriptRunner, with a short pause between an
LED's ON and OFF commands so the light is actually visible before the check
turns it off again — this is a visual test, not just a command-sent check.
The live feed runs on its own separate ScriptRunner since it's a long-running,
user-stopped process independent of the pass/fail check sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.paths import project_root
from gui.port_indicator import PortIndicator
from gui.runner_bridge import RunnerBridge
from gui.safety_bar import SafetyBar
from gui.script_runner import ScriptRunner
from gui.theme import CHROME_ACCENT_HEX, DANGER_HEX, GREEN_HEX, HINT_STYLE, text_rgba

_PROJECT_ROOT = project_root()
_LIVE_PREVIEW_SCRIPT = str(_PROJECT_ROOT / "live_preview.py")
_GRACEFUL_STOP_TIMEOUT_MS = 10_000
_LED_VISUAL_PAUSE_MS = 2_000
_LED_BAUD = 115_200

# Colored-circle status glyphs (HTML spans, same technique as safety_bar.py's
# _DOT dict) — hollow for not-yet-run, filled for the 3 active states.
_ICON = {
    "pending": f"<span style='color:{text_rgba(0.55)};'>○</span>",
    "running": f"<span style='color:{CHROME_ACCENT_HEX};'>●</span>",
    "pass":    f"<span style='color:{GREEN_HEX};'>●</span>",
    "fail":    f"<span style='color:{DANGER_HEX};'>●</span>",
}


@dataclass
class _Action:
    label: str
    script: str
    args: list[str]
    group: str
    post_delay_ms: int = 0


@dataclass
class _Group:
    name: str
    label: QLabel
    total_actions: int
    results: list[bool] = field(default_factory=list)


class DiagnosticsWidget(QWidget):
    """Self-contained widget with its own utility ScriptRunner."""

    def __init__(self, safety_bar: SafetyBar, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._safety_bar = safety_bar
        self._runner = ScriptRunner()
        self._bridge = RunnerBridge(self._runner)
        self._bridge.line_received.connect(self._append_line)
        self._bridge.run_finished.connect(self._on_action_done)

        self._stop_timer = QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.setInterval(_GRACEFUL_STOP_TIMEOUT_MS)
        self._stop_timer.timeout.connect(self._on_stop_timeout)

        self._actions: list[_Action] = []
        self._groups: dict[str, _Group] = {}
        self._current_index = 0
        self._aborted = False

        # Separate runner for the live feed — long-running and user-stopped,
        # independent of the sequential pass/fail check runner above.
        self._live_runner = ScriptRunner()
        self._live_bridge = RunnerBridge(self._live_runner)
        self._live_bridge.line_received.connect(self._on_live_line)
        self._live_bridge.run_finished.connect(self._on_live_done)

        self._live_stop_timer = QTimer(self)
        self._live_stop_timer.setSingleShot(True)
        self._live_stop_timer.setInterval(_GRACEFUL_STOP_TIMEOUT_MS)
        self._live_stop_timer.timeout.connect(self._on_live_stop_timeout)

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setSpacing(10)
        outer.setContentsMargins(12, 12, 12, 12)

        # Left column: forms and controls, in a scroll area so that growth
        # (e.g. a future taller Checks/Live Feed group) scrolls instead of
        # forcing the window to grow to fit (see gui/run_session.py's
        # _build_ui for the full rationale).
        left = QWidget()
        left_outer = QVBoxLayout(left)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_w = QWidget()
        layout = QVBoxLayout(scroll_w)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(8)
        scroll.setWidget(scroll_w)
        left_outer.addWidget(scroll)

        header = QLabel(
            "<b>Pre-Session Diagnostics</b> — checks the Arduino connection, "
            "red/green LED response, and camera responsiveness before you "
            "start a real session. Run this any time the rig has been "
            "power-cycled, reconnected, or is behaving oddly."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Arduino port:"))
        port_row.addWidget(PortIndicator(self._safety_bar))
        layout.addLayout(port_row)

        checks_box = QGroupBox("Checks")
        form = QFormLayout(checks_box)
        self._status_labels: dict[str, QLabel] = {}
        for group_name in ("Arduino / Red LED", "Arduino / Green LED", "Camera"):
            lbl = QLabel(f"{_ICON['pending']} not checked yet")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            self._status_labels[group_name] = lbl
            form.addRow(group_name + ":", lbl)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera index:"))
        self._camera_idx = QSpinBox()
        self._camera_idx.setRange(0, 9)
        cam_row.addWidget(self._camera_idx)
        cam_row.addStretch()
        form.addRow("", cam_row)
        layout.addWidget(checks_box)

        layout.addWidget(self._build_live_feed_group())

        self._summary = QLabel("Not run yet.")
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("Run Diagnostics")
        self._btn_run.clicked.connect(self._on_run)
        btn_row.addWidget(self._btn_run)

        self._btn_stop = QPushButton("Stop checks")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self._btn_stop)

        btn_row.addStretch()
        layout.addLayout(btn_row)
        # Trailing stretch: collects leftover vertical space here instead of
        # Qt spreading it evenly between the widgets above.
        layout.addStretch()

        outer.addWidget(left, stretch=3)
        outer.addWidget(self._build_log_panel(), stretch=2)

    def _build_log_panel(self) -> QGroupBox:
        box = QGroupBox("Diagnostics output")
        vl = QVBoxLayout(box)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Output from each check will appear here…")
        vl.addWidget(self._log)
        return box

    def _build_live_feed_group(self) -> QGroupBox:
        box = QGroupBox("Live Camera Feed")
        vl = QVBoxLayout(box)

        hint = QLabel(
            "Preview only: opens a live-updating camera window and does NOT turn "
            "any LEDs on or off. To check animal placement and focus, turn green "
            "light on first (Utilities → Green ON). The focus steps in Run "
            "Session turn green light on for you."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(HINT_STYLE)
        vl.addWidget(hint)

        form = QFormLayout()

        self._live_exposure_us = QDoubleSpinBox()
        self._live_exposure_us.setRange(100.0, 5_000.0)
        self._live_exposure_us.setValue(5_000.0)
        self._live_exposure_us.setSuffix(" µs")
        self._live_exposure_us.setDecimals(0)
        form.addRow("Exposure:", self._live_exposure_us)

        self._live_pixel_format = QComboBox()
        self._live_pixel_format.addItems(["Mono16", "Mono12", "Mono8"])
        form.addRow("Pixel format:", self._live_pixel_format)

        self._live_fps = QDoubleSpinBox()
        self._live_fps.setRange(1.0, 60.0)
        self._live_fps.setValue(15.0)
        self._live_fps.setSuffix(" fps")
        self._live_fps.setDecimals(1)
        self._live_fps.setToolTip(
            "Display redraw rate (separate from the camera's own capture rate, "
            "which is shown live in the preview window)."
        )
        form.addRow("Preview rate:", self._live_fps)

        snap_row = QHBoxLayout()
        self._live_snapshot_dir = QLineEdit()
        self._live_snapshot_dir.setPlaceholderText("Default: live_preview_snapshots/ next to the scripts")
        snap_row.addWidget(self._live_snapshot_dir, stretch=1)
        btn_snap_browse = QPushButton("Browse…")
        btn_snap_browse.clicked.connect(self._browse_snapshot_dir)
        snap_row.addWidget(btn_snap_browse)
        form.addRow("Snapshot folder:", snap_row)

        vl.addLayout(form)

        snap_hint = QLabel("Press 's' in the preview window to save a snapshot (.npy + quick-look .png).")
        snap_hint.setWordWrap(True)
        snap_hint.setStyleSheet(HINT_STYLE)
        vl.addWidget(snap_hint)

        btn_row = QHBoxLayout()
        self._live_start_btn = QPushButton("Start preview (LEDs unchanged)")
        self._live_start_btn.clicked.connect(self._on_live_start)
        btn_row.addWidget(self._live_start_btn)

        self._live_stop_btn = QPushButton("Stop preview")
        self._live_stop_btn.setEnabled(False)
        self._live_stop_btn.clicked.connect(self._on_live_stop)
        btn_row.addWidget(self._live_stop_btn)

        btn_row.addStretch()
        vl.addLayout(btn_row)

        self._live_status = QLabel("Idle.")
        vl.addWidget(self._live_status)

        return box

    # ── Live feed ─────────────────────────────────────────────────────────────

    def _browse_snapshot_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select snapshot folder")
        if path:
            self._live_snapshot_dir.setText(path)

    def _on_live_start(self) -> None:
        args = [
            "--camera-index", str(self._camera_idx.value()),
            "--exposure-us", str(self._live_exposure_us.value()),
            "--pixel-format", self._live_pixel_format.currentText(),
            "--fps", str(self._live_fps.value()),
        ]
        snapshot_dir = self._live_snapshot_dir.text().strip()
        if snapshot_dir:
            args += ["--snapshot-dir", snapshot_dir]
        self._log.appendPlainText("\n=== Live Camera Feed ===")
        self._live_status.setText("Starting… (a separate preview window will open)")
        try:
            self._live_runner.start(_LIVE_PREVIEW_SCRIPT, args)
        except (RuntimeError, OSError, FileNotFoundError) as exc:
            self._live_status.setText(f"Launch failed: {exc}")
            return
        self._live_start_btn.setEnabled(False)
        self._live_stop_btn.setEnabled(True)

    def _on_live_line(self, line: str) -> None:
        self._log.appendPlainText(line)

    def _on_live_done(self, exit_code: int) -> None:
        self._live_stop_timer.stop()
        label = "closed" if exit_code == 0 else f"exit code {exit_code}"
        self._live_status.setText(f"Live feed {label}.")
        self._live_start_btn.setEnabled(True)
        self._live_stop_btn.setEnabled(False)

    def _on_live_stop(self) -> None:
        self._live_status.setText("Stopping (graceful — waiting up to 10 s)…")
        self._live_stop_btn.setEnabled(False)
        self._live_runner.send_stop_signal()
        self._live_stop_timer.start()

    def _on_live_stop_timeout(self) -> None:
        msg = QMessageBox(self)
        msg.setWindowTitle("Stop timed out")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            "The live feed did not exit within 10 seconds after the stop signal.\n\n"
            "Force-killing it will skip camera cleanup — you may need to "
            "power-cycle the camera or restart the GUI if it stops responding."
        )
        force_btn = msg.addButton("Force Kill", QMessageBox.ButtonRole.DestructiveRole)
        wait_btn = msg.addButton("Keep Waiting", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(wait_btn)
        msg.exec()

        if msg.clickedButton() is force_btn:
            self._live_runner.force_kill()
            self._live_status.setText("Force-killed.")
            self._live_start_btn.setEnabled(True)
            self._live_stop_btn.setEnabled(False)
        else:
            self._live_status.setText("Still waiting for the process to exit…")
            self._live_stop_timer.start()

    # ── Run sequence ──────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        port = self._safety_bar.selected_port()
        if not port:
            QMessageBox.warning(self, "No port", "Select an Arduino COM port at the top of the window.")
            return

        self._log.clear()
        self._aborted = False
        self._actions = self._build_actions(port)
        self._groups = {}
        for action in self._actions:
            grp = self._groups.setdefault(
                action.group,
                _Group(name=action.group, label=self._status_labels[action.group], total_actions=0),
            )
            grp.total_actions += 1
        for grp in self._groups.values():
            grp.label.setText(f"{_ICON['pending']} not checked yet")
        self._summary.setText("Running…")
        self._safety_bar.set_hardware_state("running")

        self._current_index = 0
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._start_next_action()

    def _build_actions(self, port: str) -> list[_Action]:
        red = str(_PROJECT_ROOT / "red.py")
        green = str(_PROJECT_ROOT / "green.py")
        camera = str(_PROJECT_ROOT / "reset_blackfly_roi.py")
        return [
            _Action("Red LED ON", red, ["--port", port, "--baud", str(_LED_BAUD), "on"],
                    "Arduino / Red LED", post_delay_ms=_LED_VISUAL_PAUSE_MS),
            _Action("Red LED OFF", red, ["--port", port, "--baud", str(_LED_BAUD), "off"],
                    "Arduino / Red LED"),
            _Action("Green LED ON", green, ["--port", port, "--baud", str(_LED_BAUD), "on"],
                    "Arduino / Green LED", post_delay_ms=_LED_VISUAL_PAUSE_MS),
            _Action("Green LED OFF", green, ["--port", port, "--baud", str(_LED_BAUD), "off"],
                    "Arduino / Green LED"),
            _Action("Camera responsiveness", camera, ["--camera-index", str(self._camera_idx.value())],
                    "Camera"),
        ]

    def _start_next_action(self) -> None:
        if self._aborted:
            self._finish(aborted=True)
            return
        if self._current_index >= len(self._actions):
            self._finish(aborted=False)
            return

        action = self._actions[self._current_index]
        grp = self._groups[action.group]
        grp.label.setText(f"{_ICON['running']} {action.label}…")
        self._log.appendPlainText(f"\n=== {action.label} ===")

        try:
            self._runner.start(action.script, action.args)
        except (RuntimeError, OSError, FileNotFoundError) as exc:
            self._log.appendPlainText(f"Launch failed: {exc}")
            self._record_result(action, success=False)
            self._current_index += 1
            self._start_next_action()

    def _on_action_done(self, exit_code: int) -> None:
        self._stop_timer.stop()
        if self._current_index >= len(self._actions):
            return  # stray signal after abort
        action = self._actions[self._current_index]
        self._record_result(action, success=(exit_code == 0))
        self._current_index += 1

        if self._aborted:
            self._finish(aborted=True)
            return

        if action.post_delay_ms:
            QTimer.singleShot(action.post_delay_ms, self._start_next_action)
        else:
            self._start_next_action()

    def _record_result(self, action: _Action, success: bool) -> None:
        grp = self._groups[action.group]
        grp.results.append(success)
        if len(grp.results) == grp.total_actions:
            passed = all(grp.results)
            grp.label.setText(f"{_ICON['pass'] if passed else _ICON['fail']} "
                               f"{'passed' if passed else 'FAILED — see log'}")
        elif not success:
            grp.label.setText(f"{_ICON['fail']} FAILED — see log")

    def _finish(self, aborted: bool) -> None:
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        if aborted:
            self._summary.setText("Stopped before all checks finished.")
            self._safety_bar.set_hardware_state("incomplete")
            return

        all_passed = all(all(g.results) for g in self._groups.values() if g.results)
        self._safety_bar.set_hardware_state("passed" if all_passed else "failed")
        if all_passed:
            self._summary.setText(
                f"{_ICON['pass']} <b>All checks passed</b> — safe to start a session."
            )
        else:
            self._summary.setText(
                f"{_ICON['fail']} <b>One or more checks failed</b> — resolve the "
                "issue above before starting a session."
            )

    # ── Stop ──────────────────────────────────────────────────────────────────

    def _on_stop(self) -> None:
        self._aborted = True
        self._summary.setText("Stopping (graceful — waiting up to 10 s)…")
        self._btn_stop.setEnabled(False)
        self._runner.send_stop_signal()
        self._stop_timer.start()

    def _on_stop_timeout(self) -> None:
        msg = QMessageBox(self)
        msg.setWindowTitle("Stop timed out")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            "The subprocess did not exit within 10 seconds after CTRL_BREAK_EVENT.\n\n"
            "Force-killing it will skip cleanup. If this happened during an LED "
            "check, the light may still be on — check the rig before continuing."
        )
        force_btn = msg.addButton("Force Kill", QMessageBox.ButtonRole.DestructiveRole)
        wait_btn = msg.addButton("Keep Waiting", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(wait_btn)
        msg.exec()

        if msg.clickedButton() is force_btn:
            self._runner.force_kill()
            self._summary.setText("Force-killed. LEDs may still be on — check the rig.")
            self._safety_bar.set_hardware_state("incomplete")
            self._btn_run.setEnabled(True)
        else:
            self._summary.setText("Still waiting for the process to exit…")
            self._stop_timer.start()

    # ── Log ───────────────────────────────────────────────────────────────────

    def show_environment_report(self, text: str) -> None:
        """Drop the launch-time verify_install.py report into the log pane, so
        it is already there whenever the user opens this tab."""
        self._log.appendPlainText("=== Environment check (verify_install.py) ===")
        self._log.appendPlainText(text.rstrip())
        self._log.appendPlainText("")

    def _append_line(self, line: str) -> None:
        self._log.appendPlainText(line)
