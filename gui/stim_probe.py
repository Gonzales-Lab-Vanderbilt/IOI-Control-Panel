# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Periodic UDP probe for the visual stimulus server (default 127.0.0.1:55000).

On Windows, sending a UDP datagram to a closed local port triggers an ICMP
"port unreachable" message that arrives as ConnectionResetError on the next
recv().  Absence of that error (either a response or a timeout) means the
port is open and something is listening.

The probe message ("PROBE") is not a stimulus command, so the server never
starts or stops a stimulus in response. intrinsic_visual_stimulus.py replies
with "IOI_STIM_READY {json}", where the JSON says which STIM fields it will
honor (see probe_reply() there) -- parsed into StimServerConfig below.
"""
from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

PROBE_HOST = "127.0.0.1"
PROBE_PORT = 55000
_PROBE_MSG = b"PROBE"
_REPLY_PREFIX = b"IOI_STIM_READY"
_SOCKET_TIMEOUT_S = 0.3


class StimState(Enum):
    UNKNOWN = auto()
    UP = auto()
    DOWN = auto()


@dataclass(frozen=True)
class StimServerConfig:
    multi_orientation: bool
    orientations_deg: tuple[float, ...]
    grating_duration_s: float
    respect_requested_duration: bool
    randomize_orientations: bool

    @property
    def honors_orientation(self) -> bool:
        return not self.multi_orientation

    @property
    def honors_duration(self) -> bool:
        return self.respect_requested_duration


def parse_server_config(reply: bytes | None) -> StimServerConfig | None:
    """None for no reply, a foreign reply, or a server too old to report."""
    if not reply or not reply.startswith(_REPLY_PREFIX):
        return None
    payload = reply[len(_REPLY_PREFIX):].strip()
    if not payload:
        return None
    try:
        d = json.loads(payload.decode("utf-8"))
        return StimServerConfig(
            multi_orientation=bool(d["multi_orientation"]),
            orientations_deg=tuple(float(o) for o in d["orientations_deg"]),
            grating_duration_s=float(d["grating_duration_s"]),
            respect_requested_duration=bool(d["respect_requested_duration"]),
            randomize_orientations=bool(d.get("randomize_orientations", False)),
        )
    except (ValueError, KeyError, TypeError):
        return None


def probe_with_config(
    host: str = PROBE_HOST, port: int = PROBE_PORT,
) -> tuple[StimState, StimServerConfig | None]:
    """
    Send one UDP probe and return (server state, reported config).

    State is UP if no ICMP error (port accepted the packet), DOWN if ICMP
    port-unreachable was received (ConnectionResetError), UNKNOWN on socket
    creation failure or other OS errors. Config is None unless the server
    replied with one.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(_SOCKET_TIMEOUT_S)
            sock.connect((host, port))  # sets remote addr; required for ICMP feedback
            sock.send(_PROBE_MSG)
            try:
                reply = sock.recv(4096)
                return StimState.UP, parse_server_config(reply)
            except socket.timeout:
                return StimState.UP, None  # no ICMP → port open, server just didn't reply
            except ConnectionResetError:
                return StimState.DOWN, None  # ICMP port unreachable → nothing listening
    except OSError:
        return StimState.UNKNOWN, None


def probe_once(host: str = PROBE_HOST, port: int = PROBE_PORT) -> StimState:
    return probe_with_config(host, port)[0]


def probe_async(
    callback: Callable[[StimState], None],
    host: str = PROBE_HOST,
    port: int = PROBE_PORT,
) -> None:
    """Run probe_once() on a daemon thread and pass the result to callback."""
    def _run() -> None:
        callback(probe_once(host, port))

    threading.Thread(target=_run, daemon=True).start()


def probe_config_async(
    callback: Callable[[StimState, object], None],
    host: str = PROBE_HOST,
    port: int = PROBE_PORT,
) -> None:
    """Like probe_async, but also passes the reported StimServerConfig (or None)."""
    def _run() -> None:
        state, config = probe_with_config(host, port)
        callback(state, config)

    threading.Thread(target=_run, daemon=True).start()
