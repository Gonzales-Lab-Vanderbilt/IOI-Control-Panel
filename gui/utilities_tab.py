# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
gui/utilities_tab.py

Utilities tab — one-shot diagnostic/conversion helpers:
  LED controls (red.py / green.py)
  Camera: reset_blackfly_roi.py, capture_dark_reference.py
  File conversion: convert_raw_to_png.py, npy_to_tiff.py
  Quick-look: view_npy_image.py
  Visual stim server: start/stop intrinsic_visual_stimulus.py (own long-running
    ScriptRunner — see _stim_server_runner), manual send (send_stim.py)
"""
from __future__ import annotations


from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
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

from gui.lights_controller import send_command_async, send_lights_off_async
from gui.paths import project_root
from gui.port_indicator import PortIndicator
from gui.runner_bridge import RunnerBridge
from gui.safety_bar import SafetyBar
from gui.script_runner import ScriptRunner
from gui.theme import HINT_STYLE, STYLE_START, STYLE_STOP, WARNING_STYLE, text_rgba

_PROJECT_ROOT = project_root()
_GRACEFUL_STOP_TIMEOUT_MS = 10_000


class _LraBridge(QObject):
    """Marshals lights_controller.send_command_async callbacks to the main thread."""
    message = Signal(str)
    finished = Signal(bool, str)


class _LedsOffBridge(QObject):
    """Marshals lights_controller.send_lights_off_async callbacks to the main thread."""
    message = Signal(str)
    finished = Signal(bool, str)


class UtilitiesWidget(QWidget):
    """One-shot utility controls for manual diagnostics, LED testing, and file conversion."""

    def __init__(self, safety_bar: SafetyBar, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._safety_bar = safety_bar
        self._runner = ScriptRunner()
        self._bridge = RunnerBridge(self._runner)
        self._run_buttons: list[QPushButton] = []
        self._last_cmd: str = ""
        self._kill_timer: QTimer | None = None

        self._bridge.line_received.connect(self._on_line)
        self._bridge.run_finished.connect(self._on_finished)

        # LRA bench commands are one-shot direct-serial sends (same pattern as
        # gui/lights_controller.py's LIGHTS_OFF), not subprocess launches —
        # they're for testing the actuator standalone, not during a trial.
        self._lra_bridge = _LraBridge(self)
        self._lra_bridge.message.connect(self._on_lra_message)
        self._lra_bridge.finished.connect(self._on_lra_done)
        self._lra_busy = False
        self._lra_buttons: list[QPushButton] = []

        # "LEDs OFF" also sends a direct-serial command (LIGHTS_OFF) rather
        # than launching red.py off + green.py off as two subprocesses —
        # same mechanism as the top-bar Lights Off button when idle (see
        # gui/lights_controller.py / gui/safety_bar.py).
        self._leds_off_bridge = _LedsOffBridge(self)
        self._leds_off_bridge.message.connect(self._on_leds_off_message)
        self._leds_off_bridge.finished.connect(self._on_leds_off_done)
        self._leds_off_busy = False

        # The stim server is long-running (up to a whole session), unlike every
        # other button here which is one-shot. It gets its OWN runner so
        # starting it doesn't disable the rest of this tab (dark reference
        # capture, file conversion, etc.) for the whole time it's up.
        self._stim_server_runner = ScriptRunner()
        self._stim_server_bridge = RunnerBridge(self._stim_server_runner)
        self._stim_server_bridge.line_received.connect(self._on_stim_server_line)
        self._stim_server_bridge.run_finished.connect(self._on_stim_server_finished)
        self._stim_server_kill_timer: QTimer | None = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        # Left column: forms and controls.
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_w = QWidget()
        scroll_vl = QVBoxLayout(scroll_w)
        scroll_vl.setContentsMargins(0, 0, 0, 0)
        scroll_vl.setSpacing(8)
        scroll.setWidget(scroll_w)
        left_layout.addWidget(scroll, stretch=1)

        scroll_vl.addWidget(self._build_led_group())
        scroll_vl.addWidget(self._build_lra_group())
        scroll_vl.addWidget(self._build_camera_group())
        scroll_vl.addWidget(self._build_conversion_group())
        scroll_vl.addWidget(self._build_quicklook_group())
        scroll_vl.addWidget(self._build_stim_server_group())
        scroll_vl.addWidget(self._build_stim_group())
        scroll_vl.addStretch()

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        left_layout.addWidget(sep)

        footer_row = QHBoxLayout()
        self._show_cmd_btn = QPushButton("Show command")
        self._show_cmd_btn.setEnabled(False)
        self._show_cmd_btn.clicked.connect(self._show_command)
        footer_row.addWidget(self._show_cmd_btn)
        footer_row.addStretch()
        self._stop_btn = QPushButton("Stop utility")
        self._stop_btn.setToolTip(
            "Stops the one-shot utility running from this tab (LED script, camera "
            "reset, dark reference, conversion, viewer). Does not stop a session, "
            "a live preview, or the stim server."
        )
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        footer_row.addWidget(self._stop_btn)
        left_layout.addLayout(footer_row)

        # Right column: the command-line output box.
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._build_log_panel())

        outer.addWidget(left, stretch=3)
        outer.addWidget(right, stretch=2)

    def _build_log_panel(self) -> QGroupBox:
        box = QGroupBox("Command output")
        vl = QVBoxLayout(box)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Command output appears here…")
        vl.addWidget(self._log)
        return box

    # ── Section builders ──────────────────────────────────────────────────────

    def _build_led_group(self) -> QGroupBox:
        box = QGroupBox("LED Controls (Arduino serial)")
        vl = QVBoxLayout(box)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("COM port:"))
        port_row.addWidget(PortIndicator(self._safety_bar))
        port_row.addWidget(QLabel("  Baud:"))
        self._led_baud = QSpinBox()
        self._led_baud.setRange(9_600, 2_000_000)
        self._led_baud.setValue(115_200)
        port_row.addWidget(self._led_baud)
        vl.addLayout(port_row)

        btn_row = QHBoxLayout()
        for label, script, state in [
            ("Red ON",   "red.py",   "on"),
            ("Green ON", "green.py", "on"),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, s=script, st=state: self._run_led(s, st))
            btn_row.addWidget(btn)
            self._run_buttons.append(btn)

        self._leds_off_btn = QPushButton("LEDs OFF")
        self._leds_off_btn.setToolTip("Turns off both LEDs in one command (LIGHTS_OFF).")
        self._leds_off_btn.clicked.connect(self._run_leds_off)
        btn_row.addWidget(self._leds_off_btn)

        vl.addLayout(btn_row)

        return box

    def _build_lra_group(self) -> QGroupBox:
        box = QGroupBox("LRA Bench Controls (Arduino serial)")
        vl = QVBoxLayout(box)

        hint = QLabel(
            "Standalone actuator test/tune, independent of a trial — e.g. against "
            "the MPU-6050 characterization rig. Verify wiring direction and "
            "amplitude before trusting these as calibrated (see intrinsic_arduino.ino)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(HINT_STYLE)
        vl.addWidget(hint)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("COM port:"))
        port_row.addWidget(PortIndicator(self._safety_bar))
        port_row.addWidget(QLabel("  Baud:"))
        self._lra_baud = QSpinBox()
        self._lra_baud.setRange(9_600, 2_000_000)
        self._lra_baud.setValue(115_200)
        port_row.addWidget(self._lra_baud)
        vl.addLayout(port_row)

        onoff_row = QHBoxLayout()
        lra_on_btn = QPushButton("LRA ON")
        lra_on_btn.clicked.connect(lambda: self._send_lra_command("CAL_LRA_ON"))
        onoff_row.addWidget(lra_on_btn)
        self._lra_buttons.append(lra_on_btn)

        lra_off_btn = QPushButton("LRA OFF")
        lra_off_btn.clicked.connect(lambda: self._send_lra_command("CAL_LRA_OFF"))
        onoff_row.addWidget(lra_off_btn)
        self._lra_buttons.append(lra_off_btn)
        vl.addLayout(onoff_row)

        form = QFormLayout()

        freq_row = QHBoxLayout()
        self._lra_freq = QDoubleSpinBox()
        self._lra_freq.setRange(1.0, 500.0)
        self._lra_freq.setValue(100.0)
        self._lra_freq.setSuffix(" Hz")
        self._lra_freq.setDecimals(1)
        freq_row.addWidget(self._lra_freq)
        freq_btn = QPushButton("Set Frequency")
        freq_btn.clicked.connect(self._run_lra_set_freq)
        freq_row.addWidget(freq_btn)
        self._lra_buttons.append(freq_btn)
        form.addRow("Drive frequency:", freq_row)

        amp_row = QHBoxLayout()
        self._lra_amp = QSpinBox()
        self._lra_amp.setRange(0, 255)
        self._lra_amp.setValue(128)
        self._lra_amp.setToolTip(
            "Drive strength as a raw 0–255 step of the amplifier's digital "
            "potentiometer (LRA_SET_AMP). Higher drives the actuator harder. Not "
            "calibrated to physical units."
        )
        amp_row.addWidget(self._lra_amp)
        amp_btn = QPushButton("Set Amplitude")
        amp_btn.clicked.connect(self._run_lra_set_amp)
        amp_row.addWidget(amp_btn)
        self._lra_buttons.append(amp_btn)
        form.addRow("Drive amplitude (0–255 step):", amp_row)

        vl.addLayout(form)

        self._lra_log = QLabel()
        self._lra_log.setWordWrap(True)
        self._lra_log.setStyleSheet(f"color:{text_rgba(0.70)};")
        vl.addWidget(self._lra_log)

        return box

    def _build_camera_group(self) -> QGroupBox:
        box = QGroupBox("Camera")
        vl = QVBoxLayout(box)

        roi_lbl = QLabel("Reset ROI")
        roi_lbl.setStyleSheet("font-weight:bold;")
        vl.addWidget(roi_lbl)

        roi_row = QHBoxLayout()
        roi_row.addWidget(QLabel("Camera index:"))
        self._roi_idx = QSpinBox()
        self._roi_idx.setRange(0, 9)
        roi_row.addWidget(self._roi_idx)
        roi_row.addStretch()
        roi_btn = QPushButton("Reset camera ROI to full frame")
        roi_btn.clicked.connect(self._run_reset_roi)
        self._run_buttons.append(roi_btn)
        roi_row.addWidget(roi_btn)
        vl.addLayout(roi_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        vl.addWidget(sep)

        dark_lbl = QLabel("Capture dark reference")
        dark_lbl.setStyleSheet("font-weight:bold;")
        dark_lbl.setToolTip("Cap the lens and turn off all lights first.")
        vl.addWidget(dark_lbl)

        dark_form = QFormLayout()

        dark_out_row = QHBoxLayout()
        self._dark_output = QLineEdit()
        self._dark_output.setPlaceholderText("Output folder (required)")
        dark_out_row.addWidget(self._dark_output)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse_folder(self._dark_output))
        dark_out_row.addWidget(b)
        dark_form.addRow("Output:", dark_out_row)

        self._dark_frames = QSpinBox()
        self._dark_frames.setRange(1, 10_000)
        self._dark_frames.setValue(1_000)
        dark_form.addRow("Frames:", self._dark_frames)

        self._dark_exposure = QDoubleSpinBox()
        self._dark_exposure.setRange(1.0, 5_000.0)
        self._dark_exposure.setValue(5_000.0)
        self._dark_exposure.setDecimals(1)
        self._dark_exposure.setSuffix(" µs")
        dark_form.addRow("Exposure:", self._dark_exposure)

        self._dark_fmt = QComboBox()
        self._dark_fmt.addItems(["Mono16", "Mono12", "Mono8"])
        dark_form.addRow("Pixel format:", self._dark_fmt)

        self._dark_save_frames = QCheckBox("Save individual frames as TIFFs")
        dark_form.addRow("", self._dark_save_frames)

        vl.addLayout(dark_form)

        dark_btn = QPushButton("Capture dark reference")
        dark_btn.clicked.connect(self._run_dark_ref)
        self._run_buttons.append(dark_btn)
        vl.addWidget(dark_btn)

        return box

    def _build_conversion_group(self) -> QGroupBox:
        box = QGroupBox("File Conversion")
        vl = QVBoxLayout(box)

        raw_hdr = QLabel("RAW → PNG")
        raw_hdr.setStyleSheet("font-weight:bold;")
        vl.addWidget(raw_hdr)

        raw_form = QFormLayout()

        raw_in_row = QHBoxLayout()
        self._raw_input = QLineEdit()
        self._raw_input.setPlaceholderText("Input folder with .raw files (required)")
        raw_in_row.addWidget(self._raw_input)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse_folder(self._raw_input))
        raw_in_row.addWidget(b)
        raw_form.addRow("Input:", raw_in_row)

        raw_out_row = QHBoxLayout()
        self._raw_output = QLineEdit()
        self._raw_output.setPlaceholderText("Output folder (required)")
        raw_out_row.addWidget(self._raw_output)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse_folder(self._raw_output))
        raw_out_row.addWidget(b)
        raw_form.addRow("Output:", raw_out_row)

        self._raw_width = QSpinBox()
        self._raw_width.setRange(1, 10_000)
        self._raw_width.setValue(1_920)
        raw_form.addRow("Width (px):", self._raw_width)

        self._raw_height = QSpinBox()
        self._raw_height.setRange(1, 10_000)
        self._raw_height.setValue(1_200)
        raw_form.addRow("Height (px):", self._raw_height)

        self._raw_dtype = QComboBox()
        self._raw_dtype.addItems(["uint16", "uint8"])
        raw_form.addRow("Dtype:", self._raw_dtype)

        vl.addLayout(raw_form)

        raw_btn = QPushButton("Convert RAW → PNG")
        raw_btn.clicked.connect(self._run_raw_to_png)
        self._run_buttons.append(raw_btn)
        vl.addWidget(raw_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        vl.addWidget(sep)

        npy_hdr = QLabel("NPY → TIFF")
        npy_hdr.setStyleSheet("font-weight:bold;")
        vl.addWidget(npy_hdr)

        npy_form = QFormLayout()

        npy_in_row = QHBoxLayout()
        self._npy_input = QLineEdit()
        self._npy_input.setPlaceholderText("Source .npy file (required)")
        npy_in_row.addWidget(self._npy_input)
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_npy_file)
        npy_in_row.addWidget(b)
        npy_form.addRow("Input:", npy_in_row)

        npy_out_row = QHBoxLayout()
        self._npy_output = QLineEdit()
        self._npy_output.setPlaceholderText("Output folder (required)")
        npy_out_row.addWidget(self._npy_output)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse_folder(self._npy_output))
        npy_out_row.addWidget(b)
        npy_form.addRow("Output:", npy_out_row)

        self._npy_scale = QComboBox()
        self._npy_scale.addItems(["none", "signed_symmetric", "minmax", "percentile"])
        npy_form.addRow("Scale:", self._npy_scale)

        self._npy_bit_depth = QComboBox()
        self._npy_bit_depth.addItems(["16", "8"])
        npy_form.addRow("Bit depth:", self._npy_bit_depth)

        vl.addLayout(npy_form)

        npy_btn = QPushButton("Export NPY → TIFF")
        npy_btn.clicked.connect(self._run_npy_to_tiff)
        self._run_buttons.append(npy_btn)
        vl.addWidget(npy_btn)

        return box

    def _build_quicklook_group(self) -> QGroupBox:
        box = QGroupBox("Quick-Look")
        vl = QVBoxLayout(box)

        form = QFormLayout()
        self._view_mode = QComboBox()
        self._view_mode.addItems(["auto", "gray", "signed", "hot"])
        form.addRow("Display mode:", self._view_mode)

        self._view_save_png = QCheckBox("Also save PNG next to .npy file")
        form.addRow("", self._view_save_png)
        vl.addLayout(form)

        view_btn = QPushButton("Launch NPY Viewer")
        view_btn.setToolTip("Opens a file picker in a subprocess.")
        view_btn.clicked.connect(self._run_view_npy)
        self._run_buttons.append(view_btn)
        vl.addWidget(view_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        vl.addWidget(sep)

        open_img_btn = QPushButton("Open Image  (TIFF / PNG / JPEG)")
        open_img_btn.setToolTip(
            "Opens a file picker in a subprocess. Replaces SpinView/MS Photos for "
            "a quick look at a saved image:\n"
            "raw-linear and percentile-scaled views, plus basic pixel stats."
        )
        open_img_btn.clicked.connect(self._run_open_image)
        self._run_buttons.append(open_img_btn)
        vl.addWidget(open_img_btn)

        return box

    def _build_stim_server_group(self) -> QGroupBox:
        box = QGroupBox("Visual Stim Server — start/stop (Python)")
        vl = QVBoxLayout(box)

        hint = QLabel(
            "Runs intrinsic_visual_stimulus.py as a subprocess. "
            "Start this before enabling Visual stimulus on the Run Session tab; the "
            "'visual stim' indicator at the top of the window turns green once it's "
            "reachable (checked every 5 s). Uses the server's built-in defaults "
            "(spatial frequency, orientation sweep, etc.) — run the script "
            "directly from a terminal for that level of tuning. Run Session's "
            "\"Will run\" line shows what the running server will actually display."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(HINT_STYLE)
        vl.addWidget(hint)

        form = QFormLayout()

        self._stim_srv_monitor = QSpinBox()
        self._stim_srv_monitor.setRange(1, 9)
        self._stim_srv_monitor.setValue(2)
        self._stim_srv_monitor.setToolTip(
            "1-based monitor index for the stimulus display (matches the .m "
            "file's default stimMon = 2 — the external monitor in the usual "
            "setup). Use \"Detect monitors\" below if unsure which number maps "
            "to which physical screen."
        )
        form.addRow("Monitor:", self._stim_srv_monitor)

        self._stim_srv_port = QSpinBox()
        self._stim_srv_port.setRange(1, 65_535)
        self._stim_srv_port.setValue(55_000)
        self._stim_srv_port.setToolTip(
            "Must match the acquisition script's --stim-port (Run Session tab, "
            "default 55000)."
        )
        form.addRow("UDP port:", self._stim_srv_port)

        self._stim_srv_windowed = QCheckBox("Windowed (desk testing)")
        self._stim_srv_windowed.setToolTip(
            "Runs in a normal window instead of covering the stimulus monitor. "
            "For checking the server starts and responds — not for real sessions."
        )
        form.addRow("", self._stim_srv_windowed)

        vl.addLayout(form)

        detect_btn = QPushButton("Detect monitors")
        detect_btn.setToolTip(
            "Lists detected monitors, resolutions, and positions in the "
            "Command output panel on the right."
        )
        detect_btn.clicked.connect(self._run_detect_monitors)
        self._run_buttons.append(detect_btn)
        vl.addWidget(detect_btn)

        btn_row = QHBoxLayout()
        self._stim_srv_start_btn = QPushButton("Start Stim Server")
        self._stim_srv_start_btn.setStyleSheet(STYLE_START)
        self._stim_srv_start_btn.clicked.connect(self._run_stim_server)
        btn_row.addWidget(self._stim_srv_start_btn)

        self._stim_srv_stop_btn = QPushButton("Stop Stim Server")
        self._stim_srv_stop_btn.setStyleSheet(STYLE_STOP)
        self._stim_srv_stop_btn.setEnabled(False)
        self._stim_srv_stop_btn.clicked.connect(self._stop_stim_server)
        btn_row.addWidget(self._stim_srv_stop_btn)
        vl.addLayout(btn_row)

        self._stim_srv_status = QLabel("Not running.")
        self._stim_srv_status.setStyleSheet(f"color:{text_rgba(0.70)};")
        vl.addWidget(self._stim_srv_status)

        self._stim_server_log = QPlainTextEdit()
        self._stim_server_log.setReadOnly(True)
        self._stim_server_log.setMaximumHeight(140)
        self._stim_server_log.setPlaceholderText("Stim server output appears here…")
        vl.addWidget(self._stim_server_log)

        return box

    def _build_stim_group(self) -> QGroupBox:
        box = QGroupBox("Visual Stim Server — manual send")
        vl = QVBoxLayout(box)

        hint = QLabel(
            "⚠ With default settings the stim server ignores both values below: it "
            "uses the orientation only if started with --single-orientation, and the "
            "duration only with --respect-requested-duration. Useful for a quick "
            "BLACK/QUIT/up-down check regardless."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(WARNING_STYLE)
        vl.addWidget(hint)

        form = QFormLayout()

        self._stim_cmd = QComboBox()
        self._stim_cmd.addItems(["STIM", "BLACK", "QUIT"])
        form.addRow("Command:", self._stim_cmd)

        # Positional STIM args — hidden for BLACK / QUIT
        self._stim_args_widget = QWidget()
        stim_inner = QFormLayout(self._stim_args_widget)
        stim_inner.setContentsMargins(0, 0, 0, 0)

        self._stim_orientation = QDoubleSpinBox()
        self._stim_orientation.setRange(0.0, 359.9)
        self._stim_orientation.setValue(180.0)
        self._stim_orientation.setSuffix("°")
        stim_inner.addRow("Orientation:", self._stim_orientation)

        self._stim_duration = QDoubleSpinBox()
        self._stim_duration.setRange(0.1, 60.0)
        self._stim_duration.setValue(5.0)
        self._stim_duration.setSuffix(" s")
        stim_inner.addRow("Duration:", self._stim_duration)

        self._stim_trial = QSpinBox()
        self._stim_trial.setRange(1, 9_999)
        self._stim_trial.setValue(1)
        stim_inner.addRow("Trial index:", self._stim_trial)

        form.addRow(self._stim_args_widget)

        self._stim_host_edit = QLineEdit("127.0.0.1")
        form.addRow("Host:", self._stim_host_edit)

        self._stim_port_spin = QSpinBox()
        self._stim_port_spin.setRange(1, 65_535)
        self._stim_port_spin.setValue(55_000)
        form.addRow("Port:", self._stim_port_spin)

        vl.addLayout(form)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._run_send_stim)
        self._run_buttons.append(send_btn)
        vl.addWidget(send_btn)

        self._stim_cmd.currentTextChanged.connect(self._on_stim_cmd_changed)

        return box

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _browse_folder(self, target: QLineEdit) -> None:
        start = target.text().strip() or ""
        folder = QFileDialog.getExistingDirectory(self, "Select folder", start)
        if folder:
            target.setText(folder)

    def _browse_npy_file(self) -> None:
        start = self._npy_input.text().strip() or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select .npy file", start,
            "NumPy arrays (*.npy);;All files (*.*)",
        )
        if path:
            self._npy_input.setText(path)

    def _on_stim_cmd_changed(self, cmd: str) -> None:
        self._stim_args_widget.setVisible(cmd == "STIM")

    # ── Launchers ─────────────────────────────────────────────────────────────

    def _run_led(self, script: str, state: str) -> None:
        port = self._safety_bar.selected_port()
        if not port:
            QMessageBox.warning(self, "No port", "Select an Arduino COM port at the top of the window.")
            return
        self._launch(script, ["--port", port, "--baud", str(self._led_baud.value()), state])

    def _run_leds_off(self) -> None:
        port = self._safety_bar.selected_port()
        if not port:
            QMessageBox.warning(self, "No port", "Select an Arduino COM port at the top of the window.")
            return
        if self._leds_off_busy:
            QMessageBox.information(self, "Please wait", "Still talking to the Arduino — try again in a moment.")
            return
        self._leds_off_busy = True
        self._leds_off_btn.setEnabled(False)
        self._log.clear()
        send_lights_off_async(
            port,
            baud=self._led_baud.value(),
            on_message=self._leds_off_bridge.message.emit,
            on_done=self._leds_off_bridge.finished.emit,
        )

    def _on_leds_off_message(self, msg: str) -> None:
        self._log.appendPlainText(msg)

    def _on_leds_off_done(self, success: bool, detail: str) -> None:
        self._leds_off_busy = False
        self._leds_off_btn.setEnabled(True)
        if success:
            self._log.appendPlainText("\n[LEDs off]")
        else:
            self._log.appendPlainText(f"\n[LEDs off failed: {detail}]")
            QMessageBox.warning(self, "LEDs OFF failed", f"Could not reach the Arduino:\n{detail}")

    def _send_lra_command(self, command: str) -> None:
        port = self._safety_bar.selected_port()
        if not port:
            QMessageBox.warning(self, "No port", "Select an Arduino COM port at the top of the window.")
            return
        if self._lra_busy:
            QMessageBox.information(self, "Please wait", "Still talking to the Arduino — try again in a moment.")
            return
        self._lra_busy = True
        for btn in self._lra_buttons:
            btn.setEnabled(False)
        self._lra_log.setText(f"Sending {command}…")
        send_command_async(
            port, command,
            baud=self._lra_baud.value(),
            on_message=self._lra_bridge.message.emit,
            on_done=self._lra_bridge.finished.emit,
        )

    def _run_lra_set_freq(self) -> None:
        self._send_lra_command(f"LRA_SET_FREQ,{self._lra_freq.value():g}")

    def _run_lra_set_amp(self) -> None:
        self._send_lra_command(f"LRA_SET_AMP,{self._lra_amp.value()}")

    def _on_lra_message(self, msg: str) -> None:
        self._lra_log.setText(msg)

    def _on_lra_done(self, success: bool, detail: str) -> None:
        self._lra_busy = False
        for btn in self._lra_buttons:
            btn.setEnabled(True)
        if success:
            self._lra_log.setText("Done.")
        else:
            self._lra_log.setText(f"Failed: {detail}")
            QMessageBox.warning(self, "LRA command failed", f"Could not reach the Arduino:\n{detail}")

    def _run_reset_roi(self) -> None:
        self._launch("reset_blackfly_roi.py", ["--camera-index", str(self._roi_idx.value())])

    def _run_dark_ref(self) -> None:
        out = self._dark_output.text().strip()
        if not out:
            QMessageBox.warning(self, "Missing output", "Select an output folder for the dark reference.")
            return
        args = [
            "--output", out,
            "--frames", str(self._dark_frames.value()),
            "--exposure-us", f"{self._dark_exposure.value():.1f}",
            "--pixel-format", self._dark_fmt.currentText(),
        ]
        if self._dark_save_frames.isChecked():
            args.append("--save-frames")
        self._launch("capture_dark_reference.py", args)

    def _run_raw_to_png(self) -> None:
        in_f = self._raw_input.text().strip()
        out_f = self._raw_output.text().strip()
        if not in_f or not out_f:
            QMessageBox.warning(self, "Missing paths", "Select both input and output folders.")
            return
        self._launch("convert_raw_to_png.py", [
            "--input", in_f,
            "--output", out_f,
            "--width", str(self._raw_width.value()),
            "--height", str(self._raw_height.value()),
            "--dtype", self._raw_dtype.currentText(),
        ])

    def _run_npy_to_tiff(self) -> None:
        in_f = self._npy_input.text().strip()
        out_f = self._npy_output.text().strip()
        if not in_f or not out_f:
            QMessageBox.warning(self, "Missing paths", "Select an input .npy file and an output folder.")
            return
        self._launch("npy_to_tiff.py", [
            "--input", in_f,
            "--output", out_f,
            "--scale", self._npy_scale.currentText(),
            "--bit-depth", self._npy_bit_depth.currentText(),
        ])

    def _run_view_npy(self) -> None:
        args = ["--mode", self._view_mode.currentText()]
        if self._view_save_png.isChecked():
            args.append("--save-png")
        self._launch("view_npy_image.py", args)

    def _run_open_image(self) -> None:
        self._launch("raw_image_opening.py", [])

    def _run_detect_monitors(self) -> None:
        # --list-monitors is genuinely one-shot (enumerates and exits
        # immediately) so it fits the shared runner/log, unlike Start/Stop
        # Stim Server below which is long-running and gets its own.
        self._launch("intrinsic_visual_stimulus.py", ["--list-monitors"])

    def _run_send_stim(self) -> None:
        cmd = self._stim_cmd.currentText()
        host = self._stim_host_edit.text().strip() or "127.0.0.1"
        port = self._stim_port_spin.value()
        args = [cmd]
        if cmd == "STIM":
            args += [
                f"{self._stim_orientation.value():g}",
                f"{self._stim_duration.value():g}",
                str(self._stim_trial.value()),
            ]
        args += ["--host", host, "--port", str(port)]
        self._launch("send_stim.py", args)

    # ── Common launch / signal handling ───────────────────────────────────────

    def _launch(self, script: str, args: list[str]) -> None:
        script_path = str(_PROJECT_ROOT / script)
        self._last_cmd = " ".join(self._runner.launcher + [script_path] + args)
        self._show_cmd_btn.setEnabled(True)
        self._log.clear()
        try:
            self._runner.start(script_path, args)
        except RuntimeError as exc:
            QMessageBox.critical(self, "Cannot start", str(exc))
            return
        self._set_run_buttons_enabled(False)
        self._stop_btn.setEnabled(True)

    def _stop(self) -> None:
        if not self._runner.is_running:
            return
        self._stop_btn.setEnabled(False)
        self._runner.send_stop_signal()
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill)
        self._kill_timer.start(_GRACEFUL_STOP_TIMEOUT_MS)

    def _force_kill(self) -> None:
        if self._runner.is_running:
            self._runner.force_kill()
            self._log.appendPlainText("\n[force-killed — process did not exit in time]")

    def _show_command(self) -> None:
        if self._last_cmd:
            QMessageBox.information(self, "Last command", self._last_cmd)

    def _on_line(self, line: str) -> None:
        self._log.appendPlainText(line)

    def _on_finished(self, retcode: int) -> None:
        if self._kill_timer is not None:
            self._kill_timer.stop()
            self._kill_timer = None
        self._set_run_buttons_enabled(True)
        self._stop_btn.setEnabled(False)
        status = "OK" if retcode == 0 else f"exit {retcode}"
        self._log.appendPlainText(f"\n[process finished — {status}]")

    def _set_run_buttons_enabled(self, enabled: bool) -> None:
        for btn in self._run_buttons:
            btn.setEnabled(enabled)

    # ── Stim server start/stop (own runner — see __init__) ─────────────────────

    def _run_stim_server(self) -> None:
        if self._stim_server_runner.is_running:
            return

        monitor = self._stim_srv_monitor.value()
        if monitor == 1 and not self._stim_srv_windowed.isChecked():
            reply = QMessageBox.question(
                self, "Cover primary screen?",
                "Monitor 1 is usually the primary screen — this control panel is "
                "probably on it. A fullscreen stimulus window there will cover "
                "the GUI itself, including this Stop button.\n\n"
                "Start anyway? (Escape or Q closes the stimulus window directly.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        args = [
            "--monitor", str(monitor),
            "--udp-port", str(self._stim_srv_port.value()),
        ]
        if self._stim_srv_windowed.isChecked():
            args.append("--no-fullscreen")

        script_path = str(_PROJECT_ROOT / "intrinsic_visual_stimulus.py")
        self._stim_server_log.clear()
        self._stim_server_log.appendPlainText(
            "$ " + " ".join(self._stim_server_runner.launcher + [script_path] + args)
        )
        try:
            self._stim_server_runner.start(script_path, args)
        except RuntimeError as exc:
            QMessageBox.critical(self, "Cannot start", str(exc))
            return

        self._stim_srv_start_btn.setEnabled(False)
        self._stim_srv_stop_btn.setEnabled(True)
        self._stim_srv_status.setText(f"Running — monitor {monitor}, port {self._stim_srv_port.value()}.")

    def _stop_stim_server(self) -> None:
        if not self._stim_server_runner.is_running:
            return
        self._stim_srv_stop_btn.setEnabled(False)
        self._stim_srv_status.setText("Stopping…")
        self._stim_server_runner.send_stop_signal()
        self._stim_server_log.appendPlainText("\n[stopping — sent CTRL_BREAK_EVENT]")
        self._stim_server_kill_timer = QTimer(self)
        self._stim_server_kill_timer.setSingleShot(True)
        self._stim_server_kill_timer.timeout.connect(self._force_kill_stim_server)
        self._stim_server_kill_timer.start(_GRACEFUL_STOP_TIMEOUT_MS)

    def _force_kill_stim_server(self) -> None:
        if self._stim_server_runner.is_running:
            self._stim_server_runner.force_kill()
            self._stim_server_log.appendPlainText("\n[force-killed — process did not exit in time]")

    def _on_stim_server_line(self, line: str) -> None:
        self._stim_server_log.appendPlainText(line)

    def _on_stim_server_finished(self, retcode: int) -> None:
        if self._stim_server_kill_timer is not None:
            self._stim_server_kill_timer.stop()
            self._stim_server_kill_timer = None
        self._stim_srv_start_btn.setEnabled(True)
        self._stim_srv_stop_btn.setEnabled(False)
        status = "stopped" if retcode == 0 else f"exit {retcode}"
        self._stim_srv_status.setText(f"Not running ({status}).")
        self._stim_server_log.appendPlainText(f"\n[stim server stopped — {status}]")
