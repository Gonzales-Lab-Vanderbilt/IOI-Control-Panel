# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
SafetyBar — the always-visible strip at the top of the main window.

Contains:
  • COM port dropdown (populated from serial.tools.list_ports) + refresh button
  • Stop Session button — emits stopSessionRequested; MainWindow connects it
    to RunSessionWidget.request_stop(), reusing that tab's existing graceful-
    stop + 15 s-timeout + Force-Kill-escalation path (SafetyBar deliberately
    does not duplicate that logic — one escalation dialog, not two).
  • Lights Off button — context-aware:
      - Session running  → relabeled "Stop session & turn lights off": the
        same graceful stop as Stop Session, then LIGHTS_OFF over serial once
        the process has exited (its own teardown may leave lights on, e.g.
        with "Leave lights on when session exits")
      - Session idle     → open COM port directly, send LIGHTS_OFF
  • Readiness chips, kept separate so they can't read as contradictory:
    software installed (verify_install.py), hardware checked (Pre-Session
    Diagnostics), and visual stimulus (UDP probe to 127.0.0.1:55000, or
    "not required" when Run Session's stimulus isn't visual)

The bar must remain functional regardless of which tab is open, and both
Stop Session and Lights Off must work even mid-run.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QWidget,
)

try:
    import serial.tools.list_ports as _list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False

from gui.lights_controller import send_lights_off_async
from gui.paths import bundled_asset_path
from gui.runner_bridge import RunnerBridge
from gui.script_runner import ScriptRunner
from gui.stim_probe import StimState, probe_async
from gui.theme import BG_HEX, DANGER_HEX, STYLE_STOP, text_rgba

_STIM_PROBE_INTERVAL_MS = 5_000
_STIM_PROBE_INITIAL_DELAY_MS = 500

# Animated background: solid dark through the controls/status text (kept
# stable for guaranteed contrast -- see paintEvent), fading into a slowly
# hue-cycling pastel underneath the previously-empty right side and the
# lab logo. Slow and desaturated on purpose -- a persistent toolbar, not
# a highlight reel; nothing about it should compete for attention with
# the Lights Off button.
_GRADIENT_TICK_MS = 50
_GRADIENT_CYCLE_S = 45.0
_GRADIENT_HUE_STEP = 360.0 / (_GRADIENT_CYCLE_S * 1000.0 / _GRADIENT_TICK_MS)
_PASTEL_SATURATION = 0.28
_PASTEL_VALUE = 0.92
_LOGO_HEIGHT_PX = 28
# The logo is dark line art: it must always sit fully on pastel. Pastel
# starts this far before the logo, after a fade of at least _MIN_FADE_PX, and
# the layout reserves both (_LOGO_PLATE_PX) so status text can never run into
# the fade however narrow the window gets.
_LOGO_PASTEL_MARGIN_PX = 12
_MIN_FADE_PX = 16
_LOGO_PLATE_PX = _LOGO_PASTEL_MARGIN_PX + _MIN_FADE_PX

_DOT: dict[StimState | None, str] = {
    None: f"<span style='color:{text_rgba(0.55)};'>●</span>",
    StimState.UP: "<span style='color:#27ae60;'>●</span>",
    StimState.DOWN: f"<span style='color:{DANGER_HEX};'>●</span>",
    StimState.UNKNOWN: f"<span style='color:{text_rgba(0.55)};'>●</span>",
}


_ENV_DOT = {
    "checking": f"<span style='color:{text_rgba(0.55)};'>●</span>",
    "ready":    "<span style='color:#27ae60;'>●</span>",
    "issues":   f"<span style='color:{DANGER_HEX};'>●</span>",
    "unknown":  f"<span style='color:{text_rgba(0.55)};'>●</span>",
}

_HW_CAPTION = {
    "not_checked": (_ENV_DOT["unknown"], "<i>not checked</i>"),
    "running":     (_ENV_DOT["checking"], "<i>checking…</i>"),
    "passed":      (_ENV_DOT["ready"], "<b>passed</b>"),
    "failed":      (_ENV_DOT["issues"], "<b>failed</b>"),
    "incomplete":  (_ENV_DOT["unknown"], "<i>incomplete</i>"),
}

_LIGHTS_OFF_IDLE_TEXT = "Lights Off"
# "&&" is a literal ampersand in a QPushButton label ("&" alone is a mnemonic).
_LIGHTS_OFF_RUNNING_TEXT = "Stop session && turn lights off"
_LIGHTS_OFF_AFTER_EXIT_DELAY_MS = 500


class _ClickableLabel(QLabel):
    """QLabel that emits clicked() — used for the environment chip, which
    doubles as a shortcut to the Pre-Session Diagnostics tab."""

    clicked = Signal()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override name)
        self.clicked.emit()
        event.accept()


# ── Thread-safe bridges ───────────────────────────────────────────────────────

class _LightsOffBridge(QObject):
    """Marshals lights-controller callbacks to the main thread."""
    message = Signal(str)
    finished = Signal(bool, str)


class _StimBridge(QObject):
    """Marshals stim probe result to the main thread."""
    probed = Signal(StimState)


# ── SafetyBar ────────────────────────────────────────────────────────────────

# Exit codes the launched scripts actually produce. 0 and 130 are the only two
# that mean the script reached its own finally/teardown, so those are the only
# two where the LEDs are known to have been commanded off. Everything else ends
# with "check rig LEDs" -- see install_break_handler() in each script for why a
# raw NTSTATUS means teardown never ran.
_STATUS_CONTROL_C_EXIT = -1073741510  # 0xC000013A, as a signed 32-bit int


def describe_exit_code(code: int) -> str:
    """Plain-language outcome for a finished session subprocess. 0 and 130
    don't claim the lights are off: several stages deliberately leave
    illumination on at exit (red after calibration, "Leave lights on when
    session exits" after trials)."""
    if code == 0:
        return "Session step finished."
    if code == 130:
        return "Session step stopped."
    if code == _STATUS_CONTROL_C_EXIT:
        return "Session was killed before teardown could run — check rig LEDs."
    if code == 2:
        return "Session did not start: the script rejected its arguments."
    if code == 1:
        return "Session ended with an error — check the log, and check rig LEDs."
    if code < 0:
        return (f"Session was terminated by the OS (0x{code & 0xFFFFFFFF:08X}) "
                "— check rig LEDs.")
    return f"Session ended (exit {code}) — check rig LEDs."


class SafetyBar(QFrame):
    """
    Always-on safety strip.  Pass the session ScriptRunner and its bridge so
    the bar can check running state and react to run completion.

    Owns the single "Arduino port" selector for the whole application — other
    tabs read selected_port() and subscribe to portChanged instead of keeping
    their own dropdowns, so there is exactly one place to pick the port.
    """

    portChanged = Signal(str)
    # Emitted when Stop Session is clicked and a session is actually running.
    # SafetyBar has no reference to RunSessionWidget (it's built first) — the
    # owner (MainWindow) connects this to RunSessionWidget.request_stop().
    stopSessionRequested = Signal()
    environmentClicked = Signal()

    def __init__(
        self,
        session_runner: ScriptRunner,
        session_bridge: RunnerBridge,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = session_runner
        self._session_bridge = session_bridge

        self._lights_bridge = _LightsOffBridge(self)
        self._lights_bridge.message.connect(self._set_status)
        self._lights_bridge.finished.connect(self._on_lights_done)

        self._stim_bridge = _StimBridge(self)
        self._stim_bridge.probed.connect(self._on_stim_probed)

        self._stim_timer = QTimer(self)
        self._stim_timer.setInterval(_STIM_PROBE_INTERVAL_MS)
        self._stim_timer.timeout.connect(self._fire_stim_probe)

        self._stim_state: StimState | None = None
        self._stim_required = False
        self._session_active = False
        self._lights_off_after_stop = False

        self._gradient_hue = 0.0
        self._gradient_timer = QTimer(self)
        self._gradient_timer.setInterval(_GRADIENT_TICK_MS)
        self._gradient_timer.timeout.connect(self._on_gradient_tick)

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._build_ui()

        # Connect to session runner so we know when a run finishes.
        self._session_bridge.run_finished.connect(self._on_session_done)

        # Start stim probe after a short delay (let the window settle first).
        QTimer.singleShot(_STIM_PROBE_INITIAL_DELAY_MS, self._fire_stim_probe)
        self._stim_timer.start()
        self._gradient_timer.start()

        # Populate port list.
        self._refresh_ports()

    # ── Background gradient ──────────────────────────────────────────────────

    def _on_gradient_tick(self) -> None:
        self._gradient_hue = (self._gradient_hue + _GRADIENT_HUE_STEP) % 360.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override name)
        dark = QColor(BG_HEX)
        pastel = QColor.fromHsvF(self._gradient_hue / 360.0, _PASTEL_SATURATION, _PASTEL_VALUE)

        width = max(1, self.width())
        # The dark zone holds through every control and the status text;
        # only the stretch this bar already had going spare (now ending at
        # the logo) carries the animated fade -- computed from real widget
        # geometry so it tracks window resizes instead of a fixed fraction.
        # Pastel is anchored to the logo, not the text, so shrinking the
        # window can't slide the dark zone under the logo.
        if hasattr(self, "_status_label") and hasattr(self, "_logo_label"):
            status = self._status_label
            text_w = status.fontMetrics().horizontalAdvance(status.text())
            text_end = status.geometry().left() + min(status.width(), text_w)
            pastel_start = self._logo_label.geometry().left() - _LOGO_PASTEL_MARGIN_PX
            dark_end = min(text_end + 20, pastel_start - _MIN_FADE_PX)
        else:
            dark_end = width * 0.5
            pastel_start = width * 0.75
        frac_dark_end = min(0.998, dark_end / width)
        # Strictly after frac_dark_end: QGradient.setColorAt() at an existing
        # position replaces that stop, which would drop the dark stop and fade
        # the whole bar.
        frac_pastel_start = min(0.999, max(frac_dark_end + 0.001, pastel_start / width))

        gradient = QLinearGradient(0, 0, width, 0)
        gradient.setColorAt(0.0, dark)
        gradient.setColorAt(frac_dark_end, dark)
        gradient.setColorAt(frac_pastel_start, pastel)
        gradient.setColorAt(1.0, pastel)

        painter = QPainter(self)
        painter.fillRect(self.rect(), gradient)
        painter.end()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # COM port selector
        layout.addWidget(QLabel("Arduino port:"))
        self._port_combo = QComboBox()
        self._port_combo.setMinimumWidth(160)
        self._port_combo.setToolTip(
            "Serial port connected to the Arduino.\n"
            "This is the single Arduino port control — every tab follows it."
        )
        self._port_combo.currentIndexChanged.connect(self._emit_port_changed)
        layout.addWidget(self._port_combo)

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
        btn_refresh.setToolTip("Refresh COM port list")
        btn_refresh.clicked.connect(self._refresh_ports)
        layout.addWidget(btn_refresh)

        layout.addSpacing(12)

        # Stop Session — reachable from any tab. Graceful stop (CTRL_BREAK)
        # with the same 15 s timeout / Force-Kill escalation as the in-tab
        # Stop button; a no-op (with a status message) if nothing is running.
        self._btn_stop_session = QPushButton("Stop Session")
        self._btn_stop_session.setStyleSheet(STYLE_STOP)
        self._btn_stop_session.setToolTip(
            "Gracefully stop the running session step (calibration or "
            "trials), from any tab.\n"
            "Sends CTRL_BREAK_EVENT so the script's own teardown runs; offers "
            "Force Kill if it doesn't exit within 15 s.\n"
            "Does not by itself turn the lights off — use Lights Off for that.\n"
            "Does not stop live previews, analysis jobs, or utilities.\n"
            "No effect if nothing is currently running."
        )
        self._btn_stop_session.clicked.connect(self._on_stop_session_clicked)
        layout.addWidget(self._btn_stop_session)

        layout.addSpacing(8)

        # Lights Off
        self._btn_lights_off = QPushButton(_LIGHTS_OFF_IDLE_TEXT)
        self._btn_lights_off.setStyleSheet(STYLE_STOP)
        self._btn_lights_off.setToolTip(
            "Turn all LEDs off.\n"
            "If a session step is running, this STOPS the session first "
            "(graceful stop, same as Stop Session), then sends LIGHTS_OFF "
            "once it has exited.\n"
            "If idle: opens the COM port and sends LIGHTS_OFF directly."
        )
        self._btn_lights_off.clicked.connect(self._on_lights_off_clicked)
        layout.addWidget(self._btn_lights_off)

        layout.addSpacing(12)

        # Software check (verify_install.py), run once at launch by
        # MainWindow. Click to jump to the full report in Diagnostics.
        self._env_label = _ClickableLabel()
        self._env_label.setTextFormat(Qt.TextFormat.RichText)
        self._env_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._env_label.clicked.connect(self.environmentClicked.emit)
        self.set_environment_state("checking")
        layout.addWidget(self._env_label)

        layout.addSpacing(12)

        self._hw_label = _ClickableLabel()
        self._hw_label.setTextFormat(Qt.TextFormat.RichText)
        self._hw_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hw_label.setToolTip(
            "Result of the last Pre-Session Diagnostics run (Arduino, red/green "
            "LEDs, camera) in this window. Click to open Diagnostics."
        )
        self._hw_label.clicked.connect(self.environmentClicked.emit)
        self.set_hardware_state("not_checked")
        layout.addWidget(self._hw_label)

        layout.addSpacing(12)

        # Visual-stim server indicator
        self._stim_label = QLabel()
        self._stim_label.setTextFormat(Qt.TextFormat.RichText)
        self._stim_label.setToolTip(
            f"UDP probe to the visual-stimulus server (127.0.0.1:{55000}) every 5 s.\n"
            "Only needed when Run Session's Stimulus is Visual — shows "
            "'not required' otherwise.\n"
            "Green = server reachable, Red = nothing listening, Grey = unknown."
        )
        self._update_stim_label()
        layout.addWidget(self._stim_label)

        layout.addSpacing(8)

        # Transient status message. Kept here, inline with the other
        # controls -- NOT right-aligned out into the animated-gradient tail
        # below -- so it always sits on the guaranteed-dark, guaranteed-
        # readable part of the bar regardless of what pastel the gradient
        # is cycling through at the moment.
        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        # Ignored width: a long status message must never force the window
        # wider; it's clipped instead, with the full text in the tooltip.
        self._status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        # Stretches through the previously-empty right side: the animated
        # gradient (see paintEvent) fades to pastel after its text, ending
        # under the logo.
        layout.addWidget(self._status_label, stretch=1)
        layout.addSpacing(_LOGO_PLATE_PX)

        self._logo_label = QLabel()
        logo_pixmap = QPixmap(str(bundled_asset_path("gonzales_lab_logo.png")))
        if not logo_pixmap.isNull():
            self._logo_label.setPixmap(
                logo_pixmap.scaledToHeight(_LOGO_HEIGHT_PX, Qt.TransformationMode.SmoothTransformation)
            )
        layout.addWidget(self._logo_label)

    # ── Port list ─────────────────────────────────────────────────────────────

    def _refresh_ports(self) -> None:
        prev = self._port_combo.currentData()
        self._port_combo.clear()

        if not _SERIAL_AVAILABLE:
            self._port_combo.addItem("pyserial not installed", None)
            return

        ports = sorted(_list_ports.comports(), key=lambda p: p.device)
        if not ports:
            self._port_combo.addItem("No COM ports found", None)
            return

        for p in ports:
            label = f"{p.device} — {p.description}" if p.description else p.device
            self._port_combo.addItem(label, p.device)

        # Restore previous selection if still present.
        if prev:
            idx = self._port_combo.findData(prev)
            if idx >= 0:
                self._port_combo.setCurrentIndex(idx)

    def selected_port(self) -> str | None:
        """Return the currently selected device string (e.g. 'COM4'), or None."""
        return self._port_combo.currentData()

    def try_select_port(self, port: str) -> bool:
        """Select *port* if it's currently in the list. Returns whether it was found."""
        idx = self._port_combo.findData(port)
        if idx >= 0:
            self._port_combo.setCurrentIndex(idx)
            return True
        return False

    def _emit_port_changed(self) -> None:
        self.portChanged.emit(self.selected_port() or "")

    # ── Stop Session ──────────────────────────────────────────────────────────

    def _on_stop_session_clicked(self) -> None:
        if not self._runner.is_running:
            self._set_status("No session is running.")
            return
        self._set_status("Stopping session (graceful — waiting up to 15 s)…")
        # Disabled until the run actually ends (_on_session_done) — avoids
        # re-arming RunSessionWidget's 15 s escalation timer on a repeat
        # click while a stop is already in flight.
        self._btn_stop_session.setEnabled(False)
        # The actual send_stop_signal() + 15 s timeout + Force-Kill dialog
        # lives in RunSessionWidget (request_stop -> _on_stop) so there is
        # exactly one escalation path and one "Stop timed out" dialog,
        # whichever button triggered it.
        self.stopSessionRequested.emit()

    # ── Lights Off logic ──────────────────────────────────────────────────────

    def _on_lights_off_clicked(self) -> None:
        if self._runner.is_running:
            # The session owns the COM port, so LIGHTS_OFF can't be sent yet.
            # Its teardown can't be relied on either (it may be told to leave
            # lights on) -- stop it, then send LIGHTS_OFF from _on_session_done.
            self._lights_off_after_stop = True
            self._set_status("Stopping session, then turning lights off…")
            self._btn_lights_off.setEnabled(False)
            # A disabled Stop Session means a stop is already in flight;
            # requesting another would re-arm RunSessionWidget's 15 s timer.
            if self._btn_stop_session.isEnabled():
                self._btn_stop_session.setEnabled(False)
                self.stopSessionRequested.emit()
        else:
            self._send_lights_off()

    def _send_lights_off(self) -> None:
        port = self.selected_port()
        if not port:
            self._set_status("Select a COM port first — lights were NOT turned off.")
            self._btn_lights_off.setEnabled(True)
            return
        self._btn_lights_off.setEnabled(False)
        send_lights_off_async(
            port,
            on_message=self._lights_bridge.message.emit,
            on_done=self._lights_bridge.finished.emit,
        )

    def _on_lights_done(self, success: bool, detail: str) -> None:
        if success:
            self._set_status("Lights off.")
        else:
            self._set_status(f"Lights-off failed: {detail} — check rig LEDs.")
        self._btn_lights_off.setEnabled(True)

    def set_session_active(self, active: bool) -> None:
        """Called by RunSessionWidget when a session-step subprocess starts,
        so Lights Off can say it will stop the session."""
        self._session_active = active
        self._btn_lights_off.setText(
            _LIGHTS_OFF_RUNNING_TEXT if active else _LIGHTS_OFF_IDLE_TEXT
        )

    def _on_session_done(self, exit_code: int) -> None:
        """Re-enable Stop Session / Lights Off when the session subprocess
        exits (whether it ran to completion, was stopped, or was force-killed —
        this fires after every stage regardless of how it ended)."""
        self.set_session_active(False)
        self._btn_stop_session.setEnabled(True)
        self._set_status(describe_exit_code(exit_code))
        if self._lights_off_after_stop:
            self._lights_off_after_stop = False
            self._set_status(describe_exit_code(exit_code) + " Turning lights off…")
            # Brief pause so Windows has released the exited process's COM handle.
            QTimer.singleShot(_LIGHTS_OFF_AFTER_EXIT_DELAY_MS, self._send_lights_off)
        else:
            self._btn_lights_off.setEnabled(True)

    # ── Stim indicator ────────────────────────────────────────────────────────

    def _fire_stim_probe(self) -> None:
        probe_async(self._stim_bridge.probed.emit)

    def _on_stim_probed(self, state: StimState) -> None:
        self._stim_state = state
        self._update_stim_label()

    def set_visual_stim_required(self, required: bool) -> None:
        self._stim_required = required
        self._update_stim_label()

    def _update_stim_label(self) -> None:
        if not self._stim_required:
            self._stim_label.setText(f"{_DOT[None]} visual stimulus: <i>not required</i>")
            return
        dot = _DOT.get(self._stim_state, _DOT[None])
        if self._stim_state is StimState.UP:
            text = f"{dot} visual stim server: <b>up</b>"
        elif self._stim_state is StimState.DOWN:
            text = f"{dot} visual stim server: <b>down</b>"
        else:
            text = f"{dot} visual stim server: <i>?</i>"
        self._stim_label.setText(text)

    def set_environment_state(self, state: str, detail: str = "") -> None:
        """Set the software-check chip. state is one of the _ENV_DOT keys."""
        dot = _ENV_DOT.get(state, _ENV_DOT["unknown"])
        caption = {
            "checking": "<i>checking…</i>",
            "ready": "<b>installed</b>",
            "issues": "<b>issues</b>",
        }.get(state, "<i>?</i>")
        self._env_label.setText(f"{dot} software: {caption}")
        self._env_label.setToolTip(
            (detail + "\n\n" if detail else "")
            + "Software check (verify_install.py): the Python packages the "
            "scripts need. Does not test any hardware. Click for the full report."
        )

    def set_hardware_state(self, state: str) -> None:
        """state is one of the _HW_CAPTION keys."""
        dot, caption = _HW_CAPTION.get(state, _HW_CAPTION["not_checked"])
        self._hw_label.setText(f"{dot} hardware: {caption}")

    # ── Status helpers ────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self._status_label.setToolTip(msg)
        self.update()
