# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Qt signal bridge for ScriptRunner.

Translates the runner's background-thread callbacks into queued Qt signals
so slots connected to them are always invoked on the main thread.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from gui.script_runner import ScriptRunner


class RunnerBridge(QObject):
    line_received: Signal = Signal(str)
    run_finished: Signal = Signal(int)

    def __init__(self, runner: ScriptRunner) -> None:
        super().__init__()
        runner.on_line = self.line_received.emit
        runner.on_done = self.run_finished.emit
