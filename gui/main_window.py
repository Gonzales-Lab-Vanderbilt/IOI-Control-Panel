# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
MainWindow — top-level application window.

Layout (permanent across all steps):
    ┌─────────────────────────────────────────────────────────────┐
    │  SafetyBar  [port] [Stop Session] [Lights Off]  ● visual stim │  ← always visible
    ├─────────────────────────────────────────────────────────────┤
    │  QTabWidget                                      │
    │    (tabs added by later build steps)             │
    └─────────────────────────────────────────────────┘

The session ScriptRunner lives here so the SafetyBar can always reach it,
regardless of which tab is open.
"""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.diagnostics import DiagnosticsWidget
from gui.run_session import RunSessionWidget
from gui.statistics_tab import StatisticsWidget
from gui.title_tips import random_window_title
from gui.ui_scale import ScaleController
from gui.utilities_tab import UtilitiesWidget
from gui.paths import project_root
from gui.runner_bridge import RunnerBridge
from gui.safety_bar import SafetyBar
from gui.script_runner import ScriptRunner


class MainWindow(QMainWindow):
    def __init__(self, scale_controller: ScaleController) -> None:
        super().__init__()
        self.setWindowTitle(random_window_title())
        self.resize(1280, 800)
        self._scale = scale_controller

        # The one session-class runner.  Step 3 will wire it to the Run Session
        # tab; for now only the safety bar (graceful-stop path) uses it.
        self.session_runner = ScriptRunner()
        self.session_bridge = RunnerBridge(self.session_runner)

        # Launch-time environment check (verify_install.py). Its own runner so
        # it can never collide with the session runner above.
        self._env_runner = ScriptRunner()
        self._env_bridge = RunnerBridge(self._env_runner)
        self._env_lines: list[str] = []
        self._env_bridge.line_received.connect(self._env_lines.append)
        self._env_bridge.run_finished.connect(self._on_env_check_done)

        self._build_ui()
        self._build_menu()

        # Deferred so the window is on screen first -- the check takes ~2 s in a
        # subprocess and must never delay first paint.
        QTimer.singleShot(400, self._start_env_check)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Safety bar (step 2)
        self._safety_bar = SafetyBar(self.session_runner, self.session_bridge)
        layout.addWidget(self._safety_bar)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Tab area — steps 3+ will add Run Session, Analyze, Utilities tabs.
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, stretch=1)

        # Pre-Session Diagnostics — hardware checks before starting a session.
        # Left-most tab: the first thing to run before anything else.
        self._diagnostics = DiagnosticsWidget(self._safety_bar)
        self._tabs.addTab(self._diagnostics, "Pre-Session Diagnostics")
        self._safety_bar.environmentClicked.connect(
            lambda: self._tabs.setCurrentWidget(self._diagnostics)
        )

        # Run Session tab (step 3) — wired to the shared session runner.
        self._run_session = RunSessionWidget(self.session_runner, self.session_bridge, self._safety_bar)
        self._tabs.addTab(self._run_session, "Run Session")

        # SafetyBar's header-level "Stop Session" button has no reference of
        # its own to RunSessionWidget (it's built first, above) — route its
        # click through here so it reuses RunSessionWidget's existing
        # graceful-stop + Force-Kill-escalation path instead of a second,
        # duplicate one living in SafetyBar.
        self._safety_bar.stopSessionRequested.connect(self._run_session.request_stop)

        # Analysis tab — statistics (statistical_analyses.py); owns its own
        # job queue (see gui/statistics_tab.py, gui/analysis_queue.py). Kept
        # as an instance attribute (not just added to the tab widget) so
        # closeEvent() below can check for jobs still queued/running.
        self._statistics_widget = StatisticsWidget()
        self._tabs.addTab(self._statistics_widget, "Analysis")
        self._run_session.analyzeSessionRequested.connect(self._open_session_in_analysis)

        # Utilities tab (step 7) — one-shot helpers, owns its own runner.
        self._tabs.addTab(UtilitiesWidget(self._safety_bar), "Utilities")

        # The Analysis tab's orientation preview is the one truly-fixed-size
        # widget in the app (not just a layout minimum) -- push the current
        # zoom into it now and on every future change.
        self._statistics_widget.set_zoom(self._scale.zoom)
        self._scale.zoom_changed.connect(self._statistics_widget.set_zoom)

    def _build_menu(self) -> None:
        view_menu = self.menuBar().addMenu("&View")

        zoom_in = QAction("Zoom In", self)
        zoom_in.setShortcuts(["Ctrl+=", "Ctrl++"])
        zoom_in.triggered.connect(self._scale.zoom_in)
        view_menu.addAction(zoom_in)

        zoom_out = QAction("Zoom Out", self)
        zoom_out.setShortcut("Ctrl+-")
        zoom_out.triggered.connect(self._scale.zoom_out)
        view_menu.addAction(zoom_out)

        zoom_reset = QAction("Reset Zoom", self)
        zoom_reset.setShortcut("Ctrl+0")
        zoom_reset.triggered.connect(self._scale.reset)
        view_menu.addAction(zoom_reset)

        self._zoom_label = QLabel(f"Zoom: {round(self._scale.zoom * 100)}%")
        self.statusBar().addPermanentWidget(self._zoom_label)
        self._zoom_label.hide()

        # Only a transient HUD-style readout while actively zooming, not a
        # permanent fixture -- re-armed on every change, hides itself once
        # the user stops.
        self._zoom_hide_timer = QTimer(self)
        self._zoom_hide_timer.setSingleShot(True)
        self._zoom_hide_timer.setInterval(1500)
        self._zoom_hide_timer.timeout.connect(self._zoom_label.hide)

        self._scale.zoom_changed.connect(self._update_zoom_label)

    def _open_session_in_analysis(self, session_dir: str) -> None:
        self._statistics_widget.set_session_folder(session_dir)
        self._tabs.setCurrentWidget(self._statistics_widget)

    def _update_zoom_label(self, zoom: float) -> None:
        self._zoom_label.setText(f"Zoom: {round(zoom * 100)}%")
        self._zoom_label.show()
        self._zoom_hide_timer.start()

    # ── Launch-time environment check ─────────────────────────────────────────

    def _start_env_check(self) -> None:
        script = str(project_root() / "verify_install.py")
        try:
            self._env_runner.start(script)
        except Exception as exc:
            # Almost always "py -3.10 not found" -- which is itself the single
            # most useful thing this check can report, so say so plainly.
            msg = (
                f"Could not launch verify_install.py: {exc}\n"
                "The GUI itself is fine, but the acquisition and analysis scripts\n"
                "need a Python 3.10 environment reachable as 'py -3.10'.\n"
                "See INSTALL.md."
            )
            self._safety_bar.set_environment_state("issues", msg)
            self._diagnostics.show_environment_report(msg)

    def _on_env_check_done(self, exit_code: int) -> None:
        report = "\n".join(self._env_lines)
        self._env_lines.clear()
        # verify_install.py exits 0 iff the CORE analysis packages are present;
        # camera and stimulus readiness are reported but deliberately do not
        # fail it, since plenty of machines legitimately need neither.
        # verify_install.py ends with a "Summary" block of one indented line per
        # readiness tier. Lift that verbatim rather than pattern-matching the
        # wording, which varies ("READY", "ready (1 camera detected)", ...).
        lines = report.splitlines()
        detail = ""
        if "Summary" in lines:
            after = lines[lines.index("Summary") + 2:]
            detail = "\n".join(l.strip() for l in after if l.startswith("  ") and l.strip())
        self._safety_bar.set_environment_state("ready" if exit_code == 0 else "issues", detail)
        self._diagnostics.show_environment_report(report)

    # ── Public helpers (used by future tabs) ──────────────────────────────────

    def add_tab(self, widget: QWidget, label: str) -> int:
        return self._tabs.addTab(widget, label)

    def selected_port(self) -> str | None:
        """Shared COM port selection from the safety bar."""
        return self._safety_bar.selected_port()

    # ── Close guard ───────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override name)
        """Analysis jobs can now be queued and run in the background, so a
        quit while some are still queued/running can silently drop work
        (queued jobs never start; running ones get no graceful-stop window).
        Warn and let the user cancel."""
        if self._statistics_widget.has_active_jobs():
            n = self._statistics_widget.active_job_count()
            reply = QMessageBox.question(
                self,
                "Analysis jobs still active",
                f"{n} analysis job(s) are still queued or running.\n\n"
                "Quitting now will stop running jobs and drop any still queued. "
                "Quit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._statistics_widget.stop_all_jobs()
        event.accept()
