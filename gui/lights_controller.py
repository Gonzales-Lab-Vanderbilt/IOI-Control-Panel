# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Sends LIGHTS_OFF to the Arduino over serial when no session subprocess owns
the COM port.

Mirrors what red.py / green.py do: open port → wait 2 s for Arduino reset →
clear input buffer → send "LIGHTS_OFF\n" → read any response → close.

All I/O happens on the calling thread.  Use send_lights_off_async() to keep
the GUI responsive during the ~3 s operation.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import serial

LIGHTS_OFF_CMD = "LIGHTS_OFF"
CAL_GREEN_ON_CMD = "CAL_GREEN_ON"
CAL_RED_ON_CMD = "CAL_RED_ON"
DEFAULT_BAUD = 115200
_RESET_DELAY_S = 2.0
_RESPONSE_WINDOW_S = 1.0


def send_command(
    port: str,
    command: str,
    baud: int = DEFAULT_BAUD,
    on_message: Callable[[str], None] | None = None,
    on_done: Callable[[bool, str], None] | None = None,
) -> None:
    """
    Block until *command* has been sent (or failed). Mirrors what red.py /
    green.py do: open port -> wait for Arduino reset -> clear input buffer ->
    send "<command>\\n" -> read any response -> close.

    on_message(line) — status/response strings for display.
    on_done(success, detail) — called exactly once when finished.
    Both callbacks fire on the calling thread; marshal to the UI thread
    (e.g. via Qt signals) before touching widgets.
    """
    log = on_message or (lambda _: None)
    done = on_done or (lambda *_: None)

    try:
        log(f"Opening {port} at {baud} baud…")
        ser = serial.Serial(port=port, baudrate=baud, timeout=0.5)
        try:
            # Arduino resets on DTR toggle when the port opens; wait for it.
            log("Waiting for Arduino reset (2 s)…")
            time.sleep(_RESET_DELAY_S)
            ser.reset_input_buffer()

            log(f"Sending {command}")
            ser.write((command + "\n").encode("utf-8"))
            ser.flush()

            t_end = time.time() + _RESPONSE_WINDOW_S
            while time.time() < t_end:
                line = ser.readline()
                if line:
                    log("[arduino] " + line.decode("utf-8", errors="ignore").strip())

            log("Done.")
            done(True, "OK")
        finally:
            ser.close()

    except serial.SerialException as exc:
        log(f"Serial error: {exc}")
        done(False, str(exc))
    except Exception as exc:
        log(f"Unexpected error: {exc}")
        done(False, str(exc))


def send_command_async(
    port: str,
    command: str,
    baud: int = DEFAULT_BAUD,
    on_message: Callable[[str], None] | None = None,
    on_done: Callable[[bool, str], None] | None = None,
) -> threading.Thread:
    """Non-blocking: runs send_command on a daemon thread."""
    t = threading.Thread(
        target=send_command,
        kwargs={"port": port, "command": command, "baud": baud, "on_message": on_message, "on_done": on_done},
        daemon=True,
    )
    t.start()
    return t


def send_lights_off(
    port: str,
    baud: int = DEFAULT_BAUD,
    on_message: Callable[[str], None] | None = None,
    on_done: Callable[[bool, str], None] | None = None,
) -> None:
    """Block until LIGHTS_OFF has been sent (or failed). See send_command()."""
    send_command(port, LIGHTS_OFF_CMD, baud=baud, on_message=on_message, on_done=on_done)


def send_lights_off_async(
    port: str,
    baud: int = DEFAULT_BAUD,
    on_message: Callable[[str], None] | None = None,
    on_done: Callable[[bool, str], None] | None = None,
) -> threading.Thread:
    """Non-blocking: runs send_lights_off on a daemon thread."""
    return send_command_async(port, LIGHTS_OFF_CMD, baud=baud, on_message=on_message, on_done=on_done)
