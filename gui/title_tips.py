# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Title-bar tips — Terraria/Minecraft-style, one picked at random each launch
and appended to the window title (see gui/main_window.py). Window titles
are plain OS chrome text on every platform; no italics/rich text is
possible there no matter what draws it, so keep entries plain and short.

Add your own here freely -- this list is the only thing that needs editing.
"""
from __future__ import annotations

import random

APP_NAME = "IOI Control Panel"
TIPS = [
    "did you remember to shut the oxygen valve off?",
    "no brain, no pain",
    "the Arduino remembers what you forgot",
    "hemodynamics: it's a flow state",
    "p < 0.05 or it didn't happen",
    "bubbles are for baths, not cranial windows",
    "have you done today's Minute Cryptic?",
    "the isoflurane doesn't fill itself",
    "hard at work or hardly working?",
    "smile today! you deserve it :)",
    "neuron activation",
    "intrinsic optical imaging... or something like that",
    "meowdy partner",
    "red light green light"
]

def random_tip() -> str:
    return random.choice(TIPS)

def random_window_title() -> str:
    if random.random() < 0.5:
        return APP_NAME
    return random_tip()
