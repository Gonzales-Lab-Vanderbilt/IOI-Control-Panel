#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
send_stim.py - manual UDP trigger for the intrinsic_visual_stimulus.m server.

The stimulus server listens for UDP commands on a port (default 55000).
Normally intrinsic_imaging.py sends it commands when it sees Arduino markers.
This little tool lets YOU send those same commands by hand for troubleshooting,
no Arduino or acquisition pipeline required.

The stimulus server must already be running and waiting before you send anything.

Usage examples:
    python send_stim.py STIM                 # STIM 180 7 1   (defaults filled in)
    python send_stim.py STIM 90 5            # STIM 90 5 1
    python send_stim.py STIM 90 5 3          # orientation 90, duration 5 s, trial 3
    python send_stim.py BLACK                # blank to black (also interrupts a running grating)
    python send_stim.py QUIT                 # tell the server to close and save its log
    python send_stim.py STIM --host 127.0.0.1 --port 55000

Heads-up on the server side: when useMultiOrientation = true (the default), the
orientation and duration you pass are IGNORED. The server sweeps the fixed
orientationsPerStimDeg list for nOrient * gratingDurationS seconds regardless.
You still have to send 3 valid tokens or the server's validation rejects the
command, but their values only matter when useMultiOrientation = false.
"""

import argparse
import socket
import sys

DEFAULTS = {"orientation": 180.0, "duration": 7.0, "trial": 1}


def build_message(command, extra):
    """Assemble the wire string the stimulus server expects."""
    command = command.upper()

    if command in ("BLACK", "QUIT"):
        if extra:
            print(f"note: {command} takes no arguments; ignoring {extra}", file=sys.stderr)
        return command

    if command == "STIM":
        orientation = float(extra[0]) if len(extra) >= 1 else DEFAULTS["orientation"]
        duration = float(extra[1]) if len(extra) >= 2 else DEFAULTS["duration"]
        trial = int(float(extra[2])) if len(extra) >= 3 else DEFAULTS["trial"]
        if duration <= 0:
            sys.exit(f"error: duration must be > 0 (got {duration}); the server will reject it")
        # %g keeps it clean: 180 not 180.0, but 1.75 stays 1.75
        return f"STIM {orientation:g} {duration:g} {trial}"

    sys.exit(f"error: unknown command '{command}'. Use STIM, BLACK, or QUIT.")


def main():
    parser = argparse.ArgumentParser(
        description="Send a UDP command to the visual stimulus server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Reminder: in multi-orientation mode the STIM orientation/duration args are decorative.",
    )
    parser.add_argument("command", help="STIM, BLACK, or QUIT (case-insensitive)")
    parser.add_argument("args", nargs="*", help="for STIM: orientation_deg duration_s [trial_index]")
    parser.add_argument("--host", default="127.0.0.1", help="server IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=55000, help="server UDP port (default: 55000)")
    opts = parser.parse_args()

    message = build_message(opts.command, opts.args)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(message.encode("ascii"), (opts.host, opts.port))
    finally:
        sock.close()

    print(f'sent "{message}" -> {opts.host}:{opts.port}')


if __name__ == "__main__":
    main()
