# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
trial_sequence.py — randomized and manual stim/catch trial sequence
generation.

Pure Python, no Qt dependency: independently testable and reusable outside
the GUI. Used by the Run Session tab to interleave stim and no-stim ("catch")
trials within a single acquisition session instead of running them as
separate sessions.
"""
from __future__ import annotations

import random
import re

_MAX_SHUFFLE_ATTEMPTS = 500

_STIM_TOKENS = {"s", "1", "stim"}
_CATCH_TOKENS = {"c", "n", "0", "catch", "no-stim", "nostim"}
_RUN_TOKEN_RE = re.compile(r"^(\d+)\s*([A-Za-z0-9\-]+)$")


def _longest_run(seq: list[bool]) -> int:
    best = cur = 0
    prev = None
    for v in seq:
        cur = cur + 1 if v == prev else 1
        prev = v
        best = max(best, cur)
    return best


def generate_trial_conditions(
    n_trials: int,
    stim_fraction: float,
    seed: int | None = None,
    max_consecutive: int = 3,
) -> tuple[list[bool], int]:
    """
    Return (conditions, seed_used).

    conditions[i] is True (stim) or False (catch) for trial i+1..n_trials.
    round(n_trials * stim_fraction) trials are stim, the rest catch. The
    sequence is shuffled with a seeded RNG, retrying up to
    _MAX_SHUFFLE_ATTEMPTS times for one with no run of same-condition trials
    longer than max_consecutive; if no attempt satisfies that (e.g. an
    unsatisfiable combination like max_consecutive=1 with a lopsided
    fraction), the attempt with the shortest longest-run found is returned
    rather than looping forever.

    seed=None auto-generates a fresh seed (via random.SystemRandom) and
    returns it, so the caller can log/record exactly what was used. Passing
    the same seed + parameters always reproduces the same sequence.
    """
    if n_trials <= 0:
        return [], (seed if seed is not None else 0)
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)

    n_stim = max(0, min(n_trials, round(n_trials * stim_fraction)))
    n_catch = n_trials - n_stim
    base = [True] * n_stim + [False] * n_catch

    rng = random.Random(seed)
    best_seq, best_run = list(base), _longest_run(base)

    for _ in range(_MAX_SHUFFLE_ATTEMPTS):
        if max_consecutive <= 0 or best_run <= max_consecutive:
            break
        candidate = list(base)
        rng.shuffle(candidate)
        run = _longest_run(candidate)
        if run < best_run:
            best_seq, best_run = candidate, run

    return best_seq, seed


def format_sequence(conditions: list[bool]) -> str:
    """Compact display form, e.g. 'S S N S N S N N S S'."""
    return " ".join("S" if c else "N" for c in conditions)


def conditions_to_argv_value(conditions: list[bool]) -> str:
    """Comma list of 1/0 for --trial-conditions, e.g. '1,1,0,1,0'."""
    return ",".join("1" if c else "0" for c in conditions)


def parse_manual_pattern(pattern: str, n_trials: int) -> list[bool]:
    """
    Parse a compact manual trial-condition pattern into a list[bool] (one
    entry per trial, True = stim), for an explicit sequence instead of a
    randomly generated one — e.g. a block design ('25S,25C': 25 stim trials
    followed by 25 catch trials), which generate_trial_conditions() can't
    reliably produce (its max_consecutive constraint actively fights a long
    same-condition run).

    Accepts run-length tokens ('25S', '10C') and bare single-trial tokens
    ('S', 'N', '1', '0'), comma- and/or whitespace-separated, freely mixed,
    e.g. '25S,25C' or 'S S S N N' or '10S 5N 15S'. Case-insensitive. 'N' and
    'C' are both catch (matches format_sequence()'s 'N' display and the
    common written shorthand).

    Raises ValueError with a message meant to be shown to the user directly
    on an unparseable token or a total length that doesn't match n_trials —
    mirrors intrinsic_calibrated_imaging.py's own --trial-conditions length
    check (SystemExit there), so a mismatch is caught here in the GUI
    instead of failing deep inside the subprocess.
    """
    raw = pattern.strip()
    if not raw:
        raise ValueError("Pattern is empty.")

    tokens = [t for t in re.split(r"[,\s]+", raw) if t]
    conditions: list[bool] = []
    for tok in tokens:
        m = _RUN_TOKEN_RE.match(tok)
        count, label = (int(m.group(1)), m.group(2)) if m else (1, tok)
        key = label.strip().lower()
        if key in _STIM_TOKENS:
            is_stim = True
        elif key in _CATCH_TOKENS:
            is_stim = False
        else:
            raise ValueError(
                f"Can't parse '{tok}' — use S/stim (or 1) or C/N/catch (or 0), "
                f"optionally prefixed with a repeat count (e.g. '25S')."
            )
        conditions.extend([is_stim] * count)

    if len(conditions) != n_trials:
        raise ValueError(
            f"Pattern has {len(conditions)} trial(s) but Trials is set to "
            f"{n_trials} — they must match exactly."
        )
    return conditions


def compress_to_pattern(conditions: list[bool]) -> str:
    """Run-length-encode a condition list into a compact pattern string,
    e.g. [True, True, False] -> '2S,1C'. Parsing the output with
    parse_manual_pattern always reproduces the same list — used to prefill
    the manual field with a concrete starting point when switching modes."""
    groups: list[list] = []
    for c in conditions:
        if groups and groups[-1][0] == c:
            groups[-1][1] += 1
        else:
            groups.append([c, 1])
    return ",".join(f"{count}{'S' if is_stim else 'C'}" for is_stim, count in groups)
