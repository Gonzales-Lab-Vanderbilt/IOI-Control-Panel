# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
ScriptRunner — launches an external Python script as a subprocess, streams its
combined stdout/stderr line-by-line, and supports graceful stop via
CTRL_BREAK_EVENT (Windows).

Framework-agnostic: callbacks fire on a background thread.  UI code must
marshal to the main thread (e.g. via Qt signals) before touching widgets.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from typing import Callable


class ScriptRunner:
    """
    Wraps a single long-running subprocess.

    Usage::

        runner = ScriptRunner()
        runner.on_line = lambda line: print(line)
        runner.on_done = lambda code: print("exit", code)
        runner.start("path/to/script.py", ["--flag", "value"])
        # later:
        runner.send_stop_signal()   # graceful
        runner.force_kill()         # last resort
    """

    def __init__(self, launcher: list[str] | None = None) -> None:
        # Configurable so the rig can use a different Python install.
        self.launcher: list[str] = launcher if launcher is not None else ["py", "-3.10"]

        # Assign before calling start().  Both fire on the pump thread.
        self.on_line: Callable[[str], None] = lambda _line: None
        self.on_done: Callable[[int], None] = lambda _code: None

        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, script: str, args: list[str] | None = None) -> None:
        """Launch *script* as a subprocess; raises RuntimeError if already running."""
        if self.is_running:
            raise RuntimeError("A subprocess is already running.")

        argv = self.launcher + [script] + (args or [])
        # PYTHONUNBUFFERED forces the subprocess to flush stdout on every print,
        # so countdown lines (\r-delimited) arrive in real time rather than
        # accumulating in the 8 KB default buffer.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        # CREATE_NEW_PROCESS_GROUP is required so CTRL_BREAK_EVENT reaches only
        # this process group and the script's KeyboardInterrupt fires. Since
        # our parent (a windowed GUI app) has no console of its own, Windows
        # auto-allocates a new one for this console-subsystem child — that's
        # the black popup. Hiding it via STARTUPINFO/SW_HIDE (rather than
        # CREATE_NO_WINDOW or explicit CREATE_NEW_CONSOLE — both measured to
        # silently break CTRL_BREAK_EVENT delivery entirely, verified against
        # real hardware) hides the window while keeping graceful stop working.
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            startupinfo=startupinfo,
        )
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def send_stop_signal(self) -> None:
        """
        Send CTRL_BREAK_EVENT (non-blocking).

        The script's KeyboardInterrupt handler runs its teardown (LEDs off)
        and exits.  Monitor on_done; call force_kill() if the process does
        not exit within an acceptable timeout.
        """
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.send_signal(signal.CTRL_BREAK_EVENT)
            except OSError:
                pass

    def force_kill(self) -> None:
        """
        Last-resort hard kill.  Only call this after send_stop_signal() has
        timed out.  Caller must warn the user that LEDs may still be on.
        """
        if self._proc is not None:
            try:
                self._proc.kill()
            except OSError:
                pass

    # ── Internal ──────────────────────────────────────────────────────────────

    def _pump(self) -> None:
        """Background thread: drain stdout, then report exit code."""
        proc = self._proc  # local ref so a new start() can't clobber it mid-pump
        assert proc is not None
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                self.on_line(line.rstrip("\n"))
        finally:
            code = proc.wait()
            # A process terminated by CTRL_BREAK_EVENT exits with an unsigned
            # NTSTATUS like 0xC000013A (3221225786) — that overflows Qt's
            # signed 32-bit Signal(int) and silently raises OverflowError in
            # this background thread, meaning on_done() (and therefore the
            # UI's "stage finished" handling) would never fire after ANY
            # graceful stop. Reinterpret as signed 32-bit to match it.
            if code > 0x7FFFFFFF:
                code -= 0x100000000
            self.on_done(code)
