# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
CropSelectorWidget — interactive replacement for a free-text "x,y,w,h"
shared-crop field: drag a rectangle directly on a green-reference preview
image, or type exact pixel coordinates in the four spinboxes (the actual
source of truth crop_rect() reads from) -- both stay two-way synced.

No numpy/PIL/matplotlib in the GUI process: the preview image is rendered
by render_crop_reference.py as a subprocess (own ScriptRunner/RunnerBridge
pair, same pattern as every other script this app launches) into a fresh OS
temp file, and this widget only ever loads that PNG via QPixmap -- exactly
like gui/panel_image_view.py's PanelImageView already does.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRubberBand,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.paths import project_root
from gui.runner_bridge import RunnerBridge
from gui.script_runner import ScriptRunner
from gui.theme import text_rgba

_FULL_W, _FULL_H = 1920, 1200
_MIN_DRAG_PX = 4  # drags smaller than this in full-res px are treated as an accidental click
_RENDER_SCRIPT = str(project_root() / "render_crop_reference.py")

_DEFAULT_PLACEHOLDER = "Select a session folder above to load a crop preview."
_PLACEHOLDER_STYLE = f"color:{text_rgba(0.55)}; font-style:italic;"


class _ImageCropLabel(QLabel):
    """Displays the reference image at its true aspect ratio and lets the
    user drag a QRubberBand over it. All drag/rect math is done in full-res
    (1920x1200) image coordinates; this label only knows how to translate
    between those and its own widget coordinates."""

    region_dragged = Signal(int, int, int, int)  # full-res px: x, y, w, h

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setStyleSheet(_PLACEHOLDER_STYLE)
        self.setText(_DEFAULT_PLACEHOLDER)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(360, 225)  # 1920:1200 ratio

        self._orig_pixmap: QPixmap | None = None
        self._scale: float = 0.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._current_rect: tuple[int, int, int, int] | None = None
        self._drag_start: QPoint | None = None

        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_pixmap_full_res(self, pixmap: QPixmap) -> None:
        self._orig_pixmap = pixmap
        self.setText("")
        self.setStyleSheet("")
        self._rescale()
        self._sync_rubber_band()

    def clear_pixmap(self, text: str = _DEFAULT_PLACEHOLDER) -> None:
        self._orig_pixmap = None
        self._scale = 0.0
        self.setPixmap(QPixmap())
        self.setStyleSheet(_PLACEHOLDER_STYLE)
        self.setText(text)
        self._rubber_band.hide()

    def set_rect(self, rect: tuple[int, int, int, int] | None) -> None:
        self._current_rect = rect
        self._sync_rubber_band()

    # ── Qt overrides ──────────────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override name)
        super().resizeEvent(event)
        self._rescale()
        self._sync_rubber_band()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        img_pt = self._widget_to_image(event.position().toPoint())
        if img_pt is None:
            return
        self._drag_start = img_pt
        self._rubber_band.setGeometry(self._image_rect_to_widget(img_pt.x(), img_pt.y(), 0, 0))
        self._rubber_band.show()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is None:
            return
        img_pt = self._widget_to_image(event.position().toPoint())
        if img_pt is None:
            return
        x, y, w, h = self._normalized_rect(self._drag_start, img_pt)
        self._rubber_band.setGeometry(self._image_rect_to_widget(x, y, w, h))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is None:
            return
        start = self._drag_start
        self._drag_start = None
        self._rubber_band.hide()

        img_pt = self._widget_to_image(event.position().toPoint())
        if img_pt is None:
            return
        x, y, w, h = self._normalized_rect(start, img_pt)
        if w < _MIN_DRAG_PX or h < _MIN_DRAG_PX:
            return  # accidental click, not an intentional crop drag
        self.region_dragged.emit(x, y, w, h)

    # ── Coordinate mapping ───────────────────────────────────────────────────

    @staticmethod
    def _normalized_rect(a: QPoint, b: QPoint) -> tuple[int, int, int, int]:
        x = min(a.x(), b.x())
        y = min(a.y(), b.y())
        w = abs(b.x() - a.x())
        h = abs(b.y() - a.y())
        return x, y, w, h

    def _rescale(self) -> None:
        if self._orig_pixmap is None:
            return
        target = self.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled = self._orig_pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        if scaled.width() <= 0:
            self._scale = 0.0
            return
        self._scale = scaled.width() / _FULL_W
        self._offset_x = (target.width() - scaled.width()) / 2.0
        self._offset_y = (target.height() - scaled.height()) / 2.0

    def _sync_rubber_band(self) -> None:
        if self._orig_pixmap is None or self._current_rect is None:
            self._rubber_band.hide()
            return
        x, y, w, h = self._current_rect
        self._rubber_band.setGeometry(self._image_rect_to_widget(x, y, w, h))
        self._rubber_band.show()

    def _widget_to_image(self, pt: QPoint) -> QPoint | None:
        if self._orig_pixmap is None or self._scale <= 0:
            return None
        ix = round((pt.x() - self._offset_x) / self._scale)
        iy = round((pt.y() - self._offset_y) / self._scale)
        ix = max(0, min(_FULL_W - 1, ix))
        iy = max(0, min(_FULL_H - 1, iy))
        return QPoint(ix, iy)

    def _image_rect_to_widget(self, x: int, y: int, w: int, h: int) -> QRect:
        return QRect(
            round(self._offset_x + x * self._scale),
            round(self._offset_y + y * self._scale),
            round(w * self._scale),
            round(h * self._scale),
        )


class CropSelectorWidget(QWidget):
    # Mirrors this widget's own preview render for read-only display
    # elsewhere (see gui/statistics_tab.py's orientation-row preview) --
    # not re-rendered there, just the same PNG path / a "gone back to
    # placeholder" notice.
    preview_changed = Signal(str)  # full-res green-reference PNG path
    preview_cleared = Signal()
    rect_changed = Signal()  # crop_rect() may now return something different

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(
            "Analysis rectangle (--shared-crop), in full-frame pixels, clipped to this "
            "session's own acquisition crop. It always restricts the pixelwise t-map; "
            "whether it also restricts ROI selection is set by \"Rectangle applies to\" "
            "below. Drag on the image or type exact pixel coordinates. Use to exclude "
            "an artifact, or to compare two same-day/same-mouse sessions over the same "
            "field of view (pass the intersection of their crops). \"Use session crop\" "
            "removes the rectangle."
        )

        self._session_dir: str = ""
        self._enabled = False
        self._syncing = False
        self._temp_png_path: str | None = None
        self._pending_session_dir: str | None = None
        self._render_target: str | None = None
        self._render_error_lines: list[str] = []

        self._runner = ScriptRunner()
        self._bridge = RunnerBridge(self._runner)
        self._bridge.line_received.connect(self._on_render_line)
        self._bridge.run_finished.connect(self._on_render_done)

        self._build_ui()

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._cleanup_temp_file)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self._label = _ImageCropLabel()
        self._label.region_dragged.connect(self._on_region_dragged)
        outer.addWidget(self._label, stretch=1)

        self._status_label = QLabel(_DEFAULT_PLACEHOLDER)
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(f"color:{text_rgba(0.55)}; font-size:11px;")
        outer.addWidget(self._status_label)

        coords = QHBoxLayout()
        coords.addWidget(QLabel("X:"))
        self._x_spin = self._make_coord_spin(_FULL_W - 1)
        coords.addWidget(self._x_spin)
        coords.addWidget(QLabel("Y:"))
        self._y_spin = self._make_coord_spin(_FULL_H - 1)
        coords.addWidget(self._y_spin)
        coords.addWidget(QLabel("W:"))
        self._w_spin = self._make_coord_spin(_FULL_W)
        coords.addWidget(self._w_spin)
        coords.addWidget(QLabel("H:"))
        self._h_spin = self._make_coord_spin(_FULL_H)
        coords.addWidget(self._h_spin)
        coords.addStretch()
        outer.addLayout(coords)

        self._x_spin.setValue(0)
        self._y_spin.setValue(0)
        self._w_spin.setValue(_FULL_W)
        self._h_spin.setValue(_FULL_H)
        for spin in (self._x_spin, self._y_spin, self._w_spin, self._h_spin):
            spin.valueChanged.connect(self._on_spin_changed)
            spin.editingFinished.connect(self._on_spin_editing_finished)

        btn_row = QHBoxLayout()
        self._refresh_btn = QPushButton("↻ Load / Refresh preview")
        self._refresh_btn.clicked.connect(self._trigger_render)
        btn_row.addWidget(self._refresh_btn)
        self._clear_btn = QPushButton("Use session crop")
        self._clear_btn.setToolTip(
            "Remove the rectangle: analyze this session's own acquisition crop "
            "(the one saved at recording time). This does not un-crop the session."
        )
        self._clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self._clear_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

    def _make_coord_spin(self, maximum: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, maximum)
        spin.setFixedWidth(70)
        return spin

    # ── Public API ────────────────────────────────────────────────────────────

    def crop_rect(self) -> tuple[int, int, int, int] | None:
        if not self._enabled:
            return None
        return (
            self._x_spin.value(),
            self._y_spin.value(),
            self._w_spin.value(),
            self._h_spin.value(),
        )

    def set_session_dir(self, session_dir: str) -> None:
        session_dir = session_dir.strip()
        if session_dir == self._session_dir:
            return
        self._session_dir = session_dir
        # Revert to placeholder BEFORE the async render starts -- otherwise a
        # slow/failed render for a newly-selected session would leave the
        # PREVIOUS session's picture on screen, silently mislabeled.
        self._label.clear_pixmap()
        self.preview_cleared.emit()
        if not session_dir:
            self._status_label.setText(_DEFAULT_PLACEHOLDER)
            return
        self._trigger_render()

    # ── Spinbox <-> drag sync ────────────────────────────────────────────────

    def _on_spin_changed(self, _value: int) -> None:
        if self._syncing:
            return
        self._enabled = True
        self._label.set_rect(self.crop_rect())
        self.rect_changed.emit()

    def _on_spin_editing_finished(self) -> None:
        # Cross-clamp so X+W <= 1920 and Y+H <= 1200, only once editing
        # finishes (not on every keystroke) so it doesn't fight the user
        # mid-edit. setMaximum() auto-clamps a too-large current value down.
        self._w_spin.setMaximum(_FULL_W - self._x_spin.value())
        self._h_spin.setMaximum(_FULL_H - self._y_spin.value())
        self._x_spin.setMaximum(_FULL_W - self._w_spin.value())
        self._y_spin.setMaximum(_FULL_H - self._h_spin.value())

    def _on_region_dragged(self, x: int, y: int, w: int, h: int) -> None:
        self._syncing = True
        try:
            # Widen limits before narrowing values so setValue() below can't
            # get silently clamped by a stale, too-small maximum left over
            # from the previous rectangle.
            self._x_spin.setMaximum(_FULL_W - 1)
            self._y_spin.setMaximum(_FULL_H - 1)
            self._w_spin.setMaximum(_FULL_W)
            self._h_spin.setMaximum(_FULL_H)
            self._x_spin.setValue(x)
            self._y_spin.setValue(y)
            self._w_spin.setValue(w)
            self._h_spin.setValue(h)
            self._w_spin.setMaximum(_FULL_W - x)
            self._h_spin.setMaximum(_FULL_H - y)
            self._x_spin.setMaximum(_FULL_W - w)
            self._y_spin.setMaximum(_FULL_H - h)
        finally:
            self._syncing = False
        self._enabled = True
        self._label.set_rect((x, y, w, h))
        self.rect_changed.emit()

    def _on_clear(self) -> None:
        self._syncing = True
        try:
            self._x_spin.setMaximum(_FULL_W - 1)
            self._y_spin.setMaximum(_FULL_H - 1)
            self._w_spin.setMaximum(_FULL_W)
            self._h_spin.setMaximum(_FULL_H)
            self._x_spin.setValue(0)
            self._y_spin.setValue(0)
            self._w_spin.setValue(_FULL_W)
            self._h_spin.setValue(_FULL_H)
        finally:
            self._syncing = False
        self._enabled = False
        self._label.set_rect(None)
        self.rect_changed.emit()

    # ── Preview render lifecycle ─────────────────────────────────────────────

    def _trigger_render(self) -> None:
        session_dir = self._session_dir
        if not session_dir:
            self._status_label.setText(_DEFAULT_PLACEHOLDER)
            return
        if not Path(session_dir).is_dir():
            self._status_label.setText("Session folder not found — check the path above.")
            return
        if self._runner.is_running:
            # Rapid re-triggers (e.g. fast folder edits) dedupe to just the
            # latest request instead of queuing up N renders.
            self._pending_session_dir = session_dir
            return

        self._status_label.setText("Loading crop preview…")
        fd, out_path = tempfile.mkstemp(prefix="ioi_crop_ref_", suffix=".png")
        os.close(fd)
        self._render_target = out_path
        self._render_error_lines = []
        try:
            self._runner.start(_RENDER_SCRIPT, ["--session-dir", session_dir, "--out", out_path])
        except (RuntimeError, OSError, FileNotFoundError) as exc:
            self._status_label.setText(f"Could not load crop preview: {exc}")
            self._render_target = None
            self._discard_temp_file(out_path)

    def _on_render_line(self, line: str) -> None:
        if line.strip():
            self._render_error_lines.append(line)

    def _on_render_done(self, exit_code: int) -> None:
        out_path = self._render_target
        self._render_target = None

        if exit_code == 0 and out_path:
            pixmap = QPixmap(out_path)
            if not pixmap.isNull():
                self._label.set_pixmap_full_res(pixmap)
                self._status_label.setText(f"Preview loaded — {Path(self._session_dir).name}")
                self._replace_temp_file(out_path)
                self.preview_changed.emit(out_path)
            else:
                self._status_label.setText("Preview render produced an unreadable image.")
                self._discard_temp_file(out_path)
                self.preview_cleared.emit()
        else:
            message = self._render_error_lines[-1] if self._render_error_lines else f"exit code {exit_code}"
            self._status_label.setText(f"No preview available — {message}")
            self._discard_temp_file(out_path)
            self.preview_cleared.emit()

        if self._pending_session_dir is not None:
            pending = self._pending_session_dir
            self._pending_session_dir = None
            if pending == self._session_dir:
                self._trigger_render()

    def _replace_temp_file(self, new_path: str) -> None:
        old = self._temp_png_path
        self._temp_png_path = new_path
        if old and old != new_path:
            self._discard_temp_file(old)

    @staticmethod
    def _discard_temp_file(path: str | None) -> None:
        if not path:
            return
        try:
            os.remove(path)
        except OSError:
            pass

    def _cleanup_temp_file(self) -> None:
        if self._temp_png_path:
            self._discard_temp_file(self._temp_png_path)
            self._temp_png_path = None
