# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Application-wide "Dark Modernist" theme — a flat, architectural dark skin
(slightly rounded corners, strong dividers, one chrome accent) applied on
top of Qt's Fusion style, plus the shared green/red button styles every
Start/Stop control in the app uses.

Three-tier depth, all deliberate (do not collapse tiers back together —
see history below):
  _BG      — outermost window / tab-pane / inactive-tab background.
  _SURFACE — one step lighter: group boxes, the toolbar (SafetyBar),
             the selected tab, and input fields (line edit/combo/spin box).
  _WELL    — near-black recessed background, used ONLY by QPlainTextEdit
             (the log/output boxes) via a type-selector QSS rule below —
             NOT the palette's Base role, so it doesn't also darken
             QLineEdit/QComboBox/QSpinBox back into the "sharp dark
             rectangle" a previous session deliberately fixed by setting
             Base == Window. If you ever need the well darker/lighter,
             change the QPlainTextEdit QSS rule, not _BASE/_SURFACE.

Accent color is split in two, on purpose:
  _DANGER        (red)  — reserved for things that mean stop/danger/failure:
                  the Stop/Lights-Off buttons, the "fail" status dot, the
                  stim "down" indicator. Never used for decoration.
  _CHROME_ACCENT (blue) — everything else the design calls "accent": the
                  selected-tab indicator, focus rings, links, the
                  "running/in-progress" status dot. Kept separate from
                  _DANGER so red stays a reliable danger signal instead of
                  being diluted into ordinary UI chrome — this app drives
                  real LEDs/a camera, and Start buttons stay green / Stop
                  buttons stay red regardless of this reskin.

QPalette alone can't express two background tiers (each role is one color
app-wide), so styling is split: the palette carries the tier that maps
1:1 to a built-in role (Window=bg, Base/Button=surface, Text, Highlight),
and an app-level QSS stylesheet (_build_stylesheet(), applied via
app.setStyleSheet() right after app.setPalette()) carries everything that
needs to diverge from its parent's palette role or add shape rules
(QGroupBox fill, the tab bar's bg/surface split, the uniform corner
radius, the recessed output well, flush-left button text). Per-widget
setStyleSheet
calls (STYLE_START/STYLE_STOP below, or any one-off label color) only need
to set what's semantically special to them — anything they don't set
(radius, text-align, disabled dimming) falls through from this app-level
sheet, since Qt merges app-level and widget-level stylesheets per-property
rather than replacing wholesale.

Call apply_dark_theme() once, right after constructing QApplication and
before building any widgets (see ioi_control_panel.py). In practice this
happens indirectly, through gui/ui_scale.py's ScaleController, which also
re-invokes it on every runtime zoom change (same palette/QSS, scaled font).
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from gui.paths import bundled_asset_path

# ── Color tokens ─────────────────────────────────────────────────────────────
# Public (no leading underscore): safe for other gui/ modules to import.
# Private (leading underscore): internal to this module only (palette/QSS
# construction), not meant to be imported elsewhere.

_BG = QColor(0x16, 0x14, 0x14)
_SURFACE = QColor(0x21, 0x1f, 0x1f)
_WELL = QColor(0x0c, 0x0b, 0x0b)
_TEXT = QColor(0xf2, 0xef, 0xee)

_BG_HEX = "#161414"
_SURFACE_HEX = "#211f1f"
_WELL_HEX = "#0c0b0b"
_TEXT_HEX = "#f2efee"

# Public alias of _BG_HEX -- for widgets outside this module that need to
# paint the app's own background color directly (e.g. SafetyBar's animated
# gradient), rather than relying on inherited palette/QSS fill.
BG_HEX = _BG_HEX

# Uniform corner radius — applied everywhere (buttons, inputs, group boxes,
# the output well) so the whole app reads as one consistent shape language.
# User picked 4px from a set of side-by-side options (0/2/4/6/8px).
_RADIUS = "4px"

# Chrome accent (blue) — tabs, focus rings, links, in-progress status.
_CHROME_ACCENT = QColor(0x3a, 0x72, 0xc4)
CHROME_ACCENT_HEX = "#3a72c4"

# Danger accent (red) — Stop/Lights-Off buttons, fail status, stim "down".
DANGER_HEX = "#ec3013"
DANGER_HOVER_HEX = "#dd2b0f"
DANGER_PRESSED_HEX = "#ae1800"
_DANGER_RGB = "236,48,19"

# Kept unchanged from the existing green/red safety convention.
GREEN_HEX = "#27ae60"
GREEN_HOVER_HEX = "#2ecc71"
GREEN_PRESSED_HEX = "#1e8449"
_GREEN_RGB = "39,174,96"

# Base app font point size at 100% zoom. gui/ui_scale.py multiplies this by
# the persisted zoom factor and re-calls apply_dark_theme() -- see there for
# the runtime zoom entry point; this module stays a pure, idempotent "paint
# the theme at this size" function.
_BASE_FONT_PT = 10.0


def text_rgba(alpha: float) -> str:
    """CSS-style rgba() string at *alpha* of the theme's text color, for QSS."""
    return f"rgba({_TEXT.red()},{_TEXT.green()},{_TEXT.blue()},{alpha})"


def text_qcolor(alpha: float) -> QColor:
    """QColor at *alpha* of the theme's text color, for palette/QColor use."""
    c = QColor(_TEXT)
    c.setAlphaF(alpha)
    return c


# Explanatory text under/next to controls. Not italic and not dimmer than
# 0.72: at the 80% zoom most rigs run, dim italic hints were hard to read.
HINT_STYLE = f"color:{text_rgba(0.72)};"
WARNING_HEX = "#f0a04b"
WARNING_STYLE = f"color:{WARNING_HEX};"


# ── Shared button styles (Start/Run = green, Stop/Lights-Off = red) ─────────
# Previously duplicated byte-for-byte across analyze_tab.py/run_session.py
# (_STYLE_START/_STYLE_STOP) and safety_bar.py (_STYLE_LIGHTS_OFF, same red
# family) — consolidated here as the one source of truth. border-radius is
# deliberately omitted: the app-level QPushButton rule below already sets
# 0px, and border:none here overrides the app-level rule's divider border
# so these colored buttons don't get a stray outline on top of their fill.

STYLE_START = (
    f"QPushButton{{background:{GREEN_HEX};color:white;font-weight:bold;"
    f"padding:4px 18px;border:none;}}"
    f"QPushButton:hover{{background:{GREEN_HOVER_HEX}}}"
    f"QPushButton:pressed{{background:{GREEN_PRESSED_HEX}}}"
    f"QPushButton:disabled{{background:rgba({_GREEN_RGB},0.45);color:rgba(255,255,255,0.6)}}"
)
STYLE_STOP = (
    f"QPushButton{{background:{DANGER_HEX};color:white;font-weight:bold;"
    f"padding:4px 18px;border:none;}}"
    f"QPushButton:hover{{background:{DANGER_HOVER_HEX}}}"
    f"QPushButton:pressed{{background:{DANGER_PRESSED_HEX}}}"
    f"QPushButton:disabled{{background:rgba({_DANGER_RGB},0.45);color:rgba(255,255,255,0.6)}}"
)


# ── App-level QSS ─────────────────────────────────────────────────────────────

def _icon_url(name: str) -> str:
    return bundled_asset_path(f"gui/icons/{name}").as_posix()


def _build_stylesheet(zoom: float = 1.0) -> str:
    divider = text_rgba(0.22)
    disabled_text = text_rgba(0.45)
    disabled_border = text_rgba(0.10)
    hover_tint = text_rgba(0.07)
    pressed_tint = text_rgba(0.14)

    # QSS px sizes don't follow the app font, so scale them with zoom.
    indicator_px = max(10, round(15 * zoom))
    arrow_w = max(12, round(18 * zoom))
    chevron_px = max(7, round(10 * zoom))

    return f"""
QCheckBox {{
    spacing: {max(4, round(7 * zoom))}px;
}}
QCheckBox::indicator {{
    width: {indicator_px}px;
    height: {indicator_px}px;
    border: 2px solid {text_rgba(0.65)};
    border-radius: 3px;
    background: {_WELL_HEX};
}}
QCheckBox::indicator:hover {{
    border: 2px solid {CHROME_ACCENT_HEX};
}}
QCheckBox::indicator:checked {{
    background: {CHROME_ACCENT_HEX};
    border: 2px solid {CHROME_ACCENT_HEX};
    image: url("{_icon_url('check.svg')}");
}}
QCheckBox::indicator:disabled {{
    border: 2px solid {text_rgba(0.22)};
    background: transparent;
}}
QCheckBox::indicator:checked:disabled {{
    background: rgba(58,114,196,0.40);
    border: 2px solid transparent;
    image: url("{_icon_url('check.svg')}");
}}

QGroupBox {{
    background: {_SURFACE_HEX};
    border: 1px solid {divider};
    border-radius: {_RADIUS};
    margin-top: 16px;
    padding: 10px 6px 6px 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    top: 2px;
    padding: 0 4px;
    color: {_TEXT_HEX};
    font-weight: 800;
}}

QTabWidget::pane {{
    background: {_BG_HEX};
    border: none;
    border-top: 2px solid {divider};
}}
QTabBar::tab {{
    background: {_BG_HEX};
    color: {_TEXT_HEX};
    padding: 8px 18px;
    border: none;
    border-right: 1px solid {divider};
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    background: {_SURFACE_HEX};
    color: {CHROME_ACCENT_HEX};
    border-bottom: 2px solid {CHROME_ACCENT_HEX};
}}
QTabBar::tab:!selected:hover {{
    color: {CHROME_ACCENT_HEX};
}}

QLineEdit, QComboBox, QAbstractSpinBox {{
    background: {_SURFACE_HEX};
    color: {_TEXT_HEX};
    border: 2px solid {divider};
    border-radius: {_RADIUS};
    padding: 3px 6px;
}}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{
    border: 2px solid {CHROME_ACCENT_HEX};
}}
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled {{
    color: {disabled_text};
    border: 2px solid {disabled_border};
}}
QComboBox, QAbstractSpinBox {{
    padding-right: {arrow_w + 4}px;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: {arrow_w}px;
    border: none;
    border-left: 1px solid {divider};
    background: transparent;
}}
QComboBox::down-arrow, QAbstractSpinBox::down-arrow {{
    image: url("{_icon_url('chevron_down.svg')}");
    width: {chevron_px}px;
    height: {chevron_px}px;
}}
QAbstractSpinBox::up-arrow {{
    image: url("{_icon_url('chevron_up.svg')}");
    width: {chevron_px}px;
    height: {chevron_px}px;
}}
QComboBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:disabled {{
    image: url("{_icon_url('chevron_down_disabled.svg')}");
}}
QAbstractSpinBox::up-arrow:disabled {{
    image: url("{_icon_url('chevron_up_disabled.svg')}");
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    width: {arrow_w}px;
    border: none;
    border-left: 1px solid {divider};
    background: transparent;
}}
QAbstractSpinBox::up-button {{
    subcontrol-position: top right;
}}
QAbstractSpinBox::down-button {{
    subcontrol-position: bottom right;
}}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {hover_tint};
}}

QPlainTextEdit {{
    background: {_WELL_HEX};
    color: {_TEXT_HEX};
    border: 1px solid {divider};
    border-radius: {_RADIUS};
}}

QPushButton {{
    text-align: left;
    padding: 5px 14px;
    border-radius: {_RADIUS};
    border: 1px solid {divider};
    background: transparent;
}}
QPushButton:hover {{
    background: {hover_tint};
}}
QPushButton:pressed {{
    background: {pressed_tint};
}}
QPushButton:disabled {{
    color: {disabled_text};
    border: 1px solid {disabled_border};
}}
""".strip()


def _resolve_font_family() -> str:
    """Prefer Segoe UI (no Archivo download/bundling — see plan); fall back
    to whatever Fusion's system default resolves to on non-Windows/offscreen
    environments."""
    families = QFontDatabase.families()
    for candidate in ("Segoe UI", "Segoe UI Variable Text"):
        if candidate in families:
            return candidate
    return QApplication.font().family()


def apply_dark_theme(app: QApplication, zoom: float = 1.0) -> None:
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, _BG)
    palette.setColor(QPalette.ColorRole.WindowText, _TEXT)
    palette.setColor(QPalette.ColorRole.Base, _SURFACE)
    palette.setColor(QPalette.ColorRole.AlternateBase, _BG)
    palette.setColor(QPalette.ColorRole.ToolTipBase, _SURFACE)
    palette.setColor(QPalette.ColorRole.ToolTipText, _TEXT)
    palette.setColor(QPalette.ColorRole.Text, _TEXT)
    palette.setColor(QPalette.ColorRole.Button, _SURFACE)
    palette.setColor(QPalette.ColorRole.ButtonText, _TEXT)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(DANGER_HEX))
    palette.setColor(QPalette.ColorRole.Link, _CHROME_ACCENT)
    palette.setColor(QPalette.ColorRole.Highlight, _CHROME_ACCENT)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.PlaceholderText, text_qcolor(0.55))

    disabled = QPalette.ColorGroup.Disabled
    disabled_text = text_qcolor(0.45)
    palette.setColor(disabled, QPalette.ColorRole.WindowText, disabled_text)
    palette.setColor(disabled, QPalette.ColorRole.Text, disabled_text)
    palette.setColor(disabled, QPalette.ColorRole.ButtonText, disabled_text)

    app.setPalette(palette)
    app.setStyleSheet(_build_stylesheet(zoom))

    apply_zoom_font(app, zoom)


def apply_zoom_font(app: QApplication, zoom: float) -> None:
    """Set the app-wide font, scaled by *zoom*, for a runtime zoom change on
    an app whose widgets already exist (see gui/ui_scale.py).

    app.setFont() alone does NOT visibly relayout already-constructed
    widgets under Fusion+QSS (confirmed empirically -- text stayed pinned at
    its original size through several zoom-in steps). QStyleSheetStyle only
    repolishes on a stylesheet change, so the stylesheet has to be re-applied
    too. But re-calling app.setStyle("Fusion") on every tick (as
    apply_dark_theme() does at startup) does NOT reverse cleanly -- zooming
    in then back to 100% left the layout measurably larger than a fresh 100%
    launch, confirmed by side-by-side screenshots. setPalette() only touches
    colors, not sizing, so it's skipped here too. Re-applying just the
    stylesheet (no setStyle, no setPalette) is what actually reproduces a
    clean, reversible relayout."""
    font = QFont(_resolve_font_family())
    font.setPointSizeF(_BASE_FONT_PT * zoom)
    app.setFont(font)
    app.setStyleSheet(_build_stylesheet(zoom))
