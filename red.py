#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
red.py

Simple red-light control script for the Arduino running lights_camera_action.ino.

Usage:
    py -3.10 red.py --port COM4 on
    py -3.10 red.py --port COM4 off

Optional:
    py -3.10 red.py --port COM4 on --baud 115200
"""

from __future__ import annotations

import argparse
import time
import serial


def send_command(port: str, baud: int, command: str) -> None:
    print(f"Opening {port} at {baud} baud...")
    ser = serial.Serial(port=port, baudrate=baud, timeout=0.5)

    try:
        # Many Arduinos reset when serial opens.
        time.sleep(2.0)

        # Clear old startup messages.
        ser.reset_input_buffer()

        print(f"Sending command: {command}")
        ser.write((command + "\n").encode("utf-8"))
        ser.flush()

        # Briefly print any Arduino response.
        t_end = time.time() + 1.0
        while time.time() < t_end:
            line = ser.readline()
            if line:
                print("[arduino]", line.decode("utf-8", errors="ignore").strip())

        print("Done.")

    finally:
        ser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn red illumination on/off via Arduino serial.")
    parser.add_argument("state", choices=["on", "off"], help="Turn red light on or off.")
    parser.add_argument("--port", required=True, help="Arduino serial port, e.g. COM4")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate.")

    args = parser.parse_args()

    if args.state == "on":
        command = "CAL_RED_ON"
    else:
        command = "LIGHTS_OFF"

    send_command(args.port, args.baud, command)


if __name__ == "__main__":
    main()