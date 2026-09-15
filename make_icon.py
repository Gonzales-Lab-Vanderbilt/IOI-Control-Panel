# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Generate ioi_icon.ico for the IOI Control Panel executable.

Source: the glyph at the left of gonzales_lab_logo.png. The full logo is a
wide wordmark (500x167) -- squeezed into a 16 px icon the lettering turns to
mush, so only the roughly-square glyph is used. Two adjustments keep it
readable as an icon: the glyph is black line art on transparency, which
vanishes against a dark taskbar, so it sits on a light rounded plate; and its
strokes are hairline-thin, which greys out to a smudge below ~48 px, so each
icon size is rendered from its own master with the strokes dilated in
proportion to how far down it is being scaled.

Run once:  py -3.10 make_icon.py
Output:    ioi_icon.ico  (multi-resolution: 16, 32, 48, 64, 128, 256 px)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SOURCE = Path(__file__).parent / "gonzales_lab_logo.png"
OUT = Path(__file__).parent / "ioi_icon.ico"
SIZES = [256, 128, 64, 48, 32, 16]

GLYPH_MAX_X = 192      # right edge of the glyph, before the wordmark starts
SUPER = 1024           # supersampled canvas every size is rendered from
PAD_FRAC = 0.12        # breathing room between glyph and plate edge
PLATE_RGB = (247, 247, 245)
PLATE_RADIUS_FRAC = 0.18

# Stroke dilation, in supersampled px, per target size. Hairlines survive a
# 4x downscale but not a 64x one, so the small sizes get progressively
# fatter strokes; 256 and 128 are left alone.
DILATION = {256: 0, 128: 0, 64: 3, 48: 5, 32: 9, 16: 16}


def _glyph() -> Image.Image:
    """The logo's left glyph, cropped tight to its own ink."""
    logo = Image.open(SOURCE).convert("RGBA")
    alpha = np.array(logo)[:, :, 3]
    alpha[:, GLYPH_MAX_X:] = 0          # drop the wordmark and its traces
    ys, xs = np.where(alpha > 10)
    return logo.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def _render(glyph: Image.Image, size: int) -> Image.Image:
    """One icon size, rendered at SUPER and downsampled in a single step."""
    plate = Image.new("RGBA", (SUPER, SUPER), (0, 0, 0, 0))
    mask = Image.new("L", (SUPER, SUPER), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, SUPER - 1, SUPER - 1), radius=int(SUPER * PLATE_RADIUS_FRAC), fill=255
    )
    plate.paste(Image.new("RGBA", (SUPER, SUPER), (*PLATE_RGB, 255)), mask=mask)

    inner = int(SUPER * (1 - 2 * PAD_FRAC))
    scale = min(inner / glyph.width, inner / glyph.height)
    fitted = glyph.resize((round(glyph.width * scale), round(glyph.height * scale)), Image.LANCZOS)

    grow = DILATION[size]
    if grow:
        # MaxFilter on the alpha channel thickens every stroke outward by
        # (grow // 2) px; recolouring from the dilated alpha rather than
        # compositing keeps the ink a flat black instead of a smear.
        ink = fitted.getchannel("A").filter(ImageFilter.MaxFilter(grow | 1))
        fitted = Image.merge("RGBA", (*Image.new("RGB", fitted.size, (0, 0, 0)).split(), ink))

    plate.paste(fitted, ((SUPER - fitted.width) // 2, (SUPER - fitted.height) // 2), fitted)
    return plate.resize((size, size), Image.LANCZOS)


def main() -> None:
    glyph = _glyph()
    renders = [_render(glyph, s) for s in SIZES]
    renders[0].save(OUT, format="ICO", sizes=[(s, s) for s in SIZES], append_images=renders[1:])
    print(f"Written: {OUT}  ({OUT.stat().st_size} bytes)")
    print(f"Sizes:   {SIZES}")


if __name__ == "__main__":
    main()
