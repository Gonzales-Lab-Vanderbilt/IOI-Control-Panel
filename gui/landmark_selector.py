# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
LandmarkSelectorWidget -- click bregma, lambda, the sagittal midline and an
optional surgically-known reference point on a session's green reference, set
the head orientation and this session's calibration, and save it all as
<session>/landmarks.json (format and maths: ioi_landmarks.py).

Lives as the Statistics tab's "Landmarks" sub-tab. It does not render
anything itself: it shows the same full-resolution green-reference PNG the
Region / Crop tab already renders (CropSelectorWidget.preview_changed), so
the GUI process still never touches numpy (see README, "What it is not").

Landmarks that are off-frame (e.g. bregma anterior of a posterior window)
can still be placed: "Room around image" adds a margin of blank canvas, in
mm, to click into, and every point also has x/y boxes that accept values
beyond the frame (negative, or past 1920 / 1200). Off-image points are
always Estimated, and the canvas grows on its own to show any point that
is already outside.

The saved file is read by session_poster_figures.py and session_timelapse.py
(landmark markers, per-session calibration, stereotaxic ROI coordinates), and
its orientation fills in the Figures tab's compass (orientation_loaded).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import ioi_landmarks as lmk
from gui.theme import HINT_STYLE, WARNING_STYLE, text_rgba

_FULL_W, _FULL_H = lmk.SENSOR_W, lmk.SENSOR_H
_PLACEHOLDER = "Select a session folder above. The green reference renders in the Region / Crop tab and appears here."

# what a click places, in order; (key, button text, marker colour)
_TARGETS = (
    ("bregma", "Bregma", "#FFD400"),
    ("lambda", "Lambda", "#00E5FF"),
    ("mid0", "Midline point 1", "#FFFFFF"),
    ("mid1", "Midline point 2", "#FFFFFF"),
    ("ref", "Reference point", "#FF7AF5"),
)
_STATUS_ITEMS = (("Visible", "visible"), ("Estimated", "estimated"))
_MARGIN_MM = (0.0, 1.0, 2.0, 3.0, 5.0)     # "Room around image" choices
_NOMINAL_UM = 3.682                        # only sizes the margin when no calibration is set
_FIT_PX = 40.0                             # breathing room kept around an off-image point
# x/y boxes: one frame beyond each edge (ioi_landmarks.MAX_OUTSIDE_FRAMES);
# the value one below the minimum shows as "—" (not placed)
_X_MIN, _X_MAX = -_FULL_W, 2 * _FULL_W - 1
_Y_MIN, _Y_MAX = -_FULL_H, 2 * _FULL_H - 1
_EDGE_WORDS = {"top": "above the top edge", "bottom": "below the bottom edge",
               "left": "left of the left edge", "right": "right of the right edge"}


class _LandmarkImage(QLabel):
    """Green reference at true aspect ratio, optionally inset in a margin of
    blank canvas (`pad`, full-res px on every side) so off-image landmarks can
    be clicked. Clicks come back in full-res sensor pixels (negative or past
    the frame in the margin); markers are painted over it. Same geometry
    maths as gui/crop_selector.py's _ImageCropLabel, plus the margin."""

    clicked = Signal(int, int)
    hovered = Signal(int, int)
    left = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(360, 225)
        self._orig: QPixmap | None = None
        self._scaled: QPixmap | None = None
        self._scale = 0.0
        self._ox = self._oy = 0.0     # widget position of image pixel (0, 0)
        self._pad = 0.0               # margin, full-res px
        self._grid_px = 0.0           # 1 mm in full-res px, for the margin grid (0 = none)
        self.marks: dict[str, tuple[float, float, str, bool]] = {}   # key -> (x, y, colour, dashed)
        self.clear_image()

    def set_image(self, pixmap: QPixmap) -> None:
        self._orig = pixmap
        self.setText("")
        self.setStyleSheet("")
        self._rescale()

    def clear_image(self, text: str = _PLACEHOLDER) -> None:
        self._orig = None
        self._scaled = None
        self._scale = 0.0
        self.setPixmap(QPixmap())
        self.setStyleSheet(f"color:{text_rgba(0.55)}; font-style:italic;")
        self.setText(text)

    def has_image(self) -> bool:
        return self._orig is not None

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def set_margin(self, pad_px: float, grid_px: float = 0.0) -> None:
        pad_px, grid_px = max(0.0, float(pad_px)), max(0.0, float(grid_px))
        if (pad_px, grid_px) != (self._pad, self._grid_px):
            self._pad, self._grid_px = pad_px, grid_px
            self._rescale()

    def margin(self) -> float:
        return self._pad

    def _rescale(self) -> None:
        if self._orig is None or self.width() <= 0:
            return
        vw, vh = _FULL_W + 2 * self._pad, _FULL_H + 2 * self._pad
        self._scale = min(self.width() / vw, self.height() / vh)
        self._scaled = self._orig.scaled(max(1, round(_FULL_W * self._scale)), max(1, round(_FULL_H * self._scale)),
                                         Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
        self._ox = (self.width() - vw * self._scale) / 2.0 + self._pad * self._scale
        self._oy = (self.height() - vh * self._scale) / 2.0 + self._pad * self._scale
        self.update()

    def to_image(self, pt: QPoint) -> tuple[int, int] | None:
        """Full-res pixel under a widget point, anywhere on the canvas
        (image or margin); None off the canvas."""
        if self._orig is None or self._scale <= 0:
            return None
        ix = round((pt.x() - self._ox) / self._scale)
        iy = round((pt.y() - self._oy) / self._scale)
        p = self._pad
        if not (-p <= ix < _FULL_W + p and -p <= iy < _FULL_H + p):
            return None
        return ix, iy

    def to_widget(self, x: float, y: float) -> QPointF:
        return QPointF(self._ox + x * self._scale, self._oy + y * self._scale)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            p = self.to_image(event.position().toPoint())
            if p is not None:
                self.clicked.emit(*p)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        p = self.to_image(event.position().toPoint())
        if p is not None:
            self.hovered.emit(*p)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.left.emit()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self._orig is None or self._scale <= 0 or self._scaled is None:
            return
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        img = QRectF(self._ox, self._oy, _FULL_W * self._scale, _FULL_H * self._scale)
        if self._pad > 0:
            m = self._pad * self._scale
            canvas = img.adjusted(-m, -m, m, m)
            qp.fillRect(canvas, QColor(28, 28, 30))
            if self._grid_px * self._scale >= 6:     # 1 mm grid in the margin, aligned to the image edges
                margin_only = QPainterPath()
                margin_only.addRect(canvas)
                inner = QPainterPath()
                inner.addRect(img)
                qp.save()
                qp.setClipPath(margin_only.subtracted(inner))
                qp.setPen(QPen(QColor(255, 255, 255, 38), 1.0))
                g = self._grid_px
                k = -int(self._pad // g)
                while k * g <= _FULL_W + self._pad:
                    qp.drawLine(self.to_widget(k * g, -self._pad), self.to_widget(k * g, _FULL_H + self._pad))
                    k += 1
                k = -int(self._pad // g)
                while k * g <= _FULL_H + self._pad:
                    qp.drawLine(self.to_widget(-self._pad, k * g), self.to_widget(_FULL_W + self._pad, k * g))
                    k += 1
                qp.restore()
        qp.drawPixmap(img.topLeft(), self._scaled)
        if self._pad > 0:
            qp.setPen(QPen(QColor(255, 255, 255, 110), 1.0))
            qp.drawRect(img)
        # midline: a line through both points, extended across the image
        if "mid0" in self.marks and "mid1" in self.marks:
            (x0, y0, *_), (x1, y1, *_) = self.marks["mid0"], self.marks["mid1"]
            dx, dy = x1 - x0, y1 - y0
            n = max((dx * dx + dy * dy) ** 0.5, 1e-9)
            L = 4000.0
            a = self.to_widget(x0 - dx / n * L, y0 - dy / n * L)
            b = self.to_widget(x0 + dx / n * L, y0 + dy / n * L)
            qp.setPen(QPen(QColor(255, 255, 255, 170), 1.5, Qt.PenStyle.DashLine))
            qp.drawLine(a, b)
        font = QFont()
        font.setBold(True)
        qp.setFont(font)
        for key, (x, y, colour, dashed) in self.marks.items():
            c = self.to_widget(x, y)
            r = 9.0
            qp.setPen(QPen(QColor(0, 0, 0, 200), 4.0))
            qp.drawLine(QPointF(c.x() - r, c.y()), QPointF(c.x() + r, c.y()))
            qp.drawLine(QPointF(c.x(), c.y() - r), QPointF(c.x(), c.y() + r))
            qp.setPen(QPen(QColor(colour), 2.0, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine))
            qp.drawLine(QPointF(c.x() - r, c.y()), QPointF(c.x() + r, c.y()))
            qp.drawLine(QPointF(c.x(), c.y() - r), QPointF(c.x(), c.y() + r))
            tag = {"bregma": "B", "lambda": "L", "mid0": "M1", "mid1": "M2", "ref": "R"}[key] + ("?" if dashed else "")
            qp.setPen(QPen(QColor(0, 0, 0)))
            qp.drawText(QPointF(c.x() + r + 3, c.y() - r + 1), tag)
            qp.setPen(QPen(QColor(colour)))
            qp.drawText(QPointF(c.x() + r + 2, c.y() - r), tag)
        qp.end()


class LandmarkSelectorWidget(QWidget):
    """See module docstring. The caller supplies the session folder
    (set_session_dir), the rendered green reference (set_image_path /
    clear_image) and the rig calibration as a default (set_default_um_per_px)."""

    orientation_loaded = Signal(dict)   # ioi_landmarks.compass_settings() of a loaded/saved file
    saved = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session_dir = ""
        self._lm: dict = lmk.empty()
        self._dirty = False
        self._loading = False
        self._default_um: float | None = None
        self._xy: dict[str, tuple[QSpinBox, QSpinBox]] = {}
        self._build_ui()
        self._refresh()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        hint = QLabel(
            "Pick what the next click places, then click it on the image. Mark bregma and lambda as "
            "Estimated unless the suture junction is actually visible. With neither visible, click a "
            "point whose position you know from the surgical record (e.g. the window centre) as the "
            "Reference point and enter its stereotaxic coordinates. If bregma or lambda is off the image, "
            "add room around it and click where it would be, or type x/y beyond the frame. "
            "Saved per session as landmarks.json."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(HINT_STYLE)
        v.addWidget(hint)

        pick = QHBoxLayout()
        pick.addWidget(QLabel("Next click places:"))
        self._target_group = QButtonGroup(self)
        for i, (key, text, colour) in enumerate(_TARGETS):
            rb = QRadioButton(text)
            rb.setProperty("target", key)
            rb.setStyleSheet(f"QRadioButton::indicator:checked {{ background:{colour}; border-radius:6px; }}")
            self._target_group.addButton(rb, i)
            pick.addWidget(rb)
        self._target_group.button(0).setChecked(True)
        pick.addStretch()
        pick.addWidget(QLabel("Room around image:"))
        self._margin = QComboBox()
        for mm in _MARGIN_MM:
            self._margin.addItem("None" if mm == 0 else f"{mm:g} mm", mm)
        self._margin.setToolTip("Blank canvas around the image, so an off-image bregma/lambda can be clicked "
                                "where it would be. Grid lines in the margin are 1 mm apart. Grows by itself "
                                "to show points already placed outside.")
        self._margin.currentIndexChanged.connect(lambda _i: self._apply_margin())
        pick.addWidget(self._margin)
        v.addLayout(pick)

        self._image = _LandmarkImage()
        self._image.clicked.connect(self._on_click)
        self._image.hovered.connect(self._on_hover)
        self._image.left.connect(lambda: self._hover_label.setText(""))
        v.addWidget(self._image, stretch=1)

        self._hover_label = QLabel("")
        self._hover_label.setStyleSheet(HINT_STYLE)
        v.addWidget(self._hover_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        self._status_combos: dict[str, QComboBox] = {}
        for row, (key, text) in enumerate((("bregma", "Bregma"), ("lambda", "Lambda"))):
            grid.addWidget(QLabel(f"{text}:"), row, 0)
            xs, ys = self._xy_boxes(key)
            grid.addWidget(xs, row, 1)
            grid.addWidget(ys, row, 2)
            cb = QComboBox()
            for t, d in _STATUS_ITEMS:
                cb.addItem(t, d)
            cb.setCurrentIndex(1)   # Estimated unless the user says otherwise
            cb.currentIndexChanged.connect(lambda _i, k=key: self._on_status(k))
            self._status_combos[key] = cb
            grid.addWidget(cb, row, 3)
            clr = QPushButton("Clear")
            clr.clicked.connect(lambda _c=False, k=key: self._clear_point(k))
            grid.addWidget(clr, row, 4)
            lab = QLabel("not placed")
            lab.setObjectName(f"{key}_pos")
            lab.setStyleSheet(HINT_STYLE)
            grid.addWidget(lab, row, 5)
        grid.addWidget(QLabel("Midline:"), 2, 0)
        self._mid_label = QLabel("not placed")
        grid.addWidget(self._mid_label, 2, 1, 1, 3)
        clr_mid = QPushButton("Clear")
        clr_mid.clicked.connect(lambda: self._clear_point("midline"))
        grid.addWidget(clr_mid, 2, 4)
        grid.setColumnStretch(5, 1)
        v.addLayout(grid)

        ref = QHBoxLayout()
        ref.addWidget(QLabel("Reference point:"))
        for box in self._xy_boxes("ref"):
            ref.addWidget(box)
        self._ref_label = QLabel("")
        self._ref_label.setStyleSheet(HINT_STYLE)
        ref.addWidget(self._ref_label)
        ref.addSpacing(8)
        ref.addWidget(QLabel("AP"))
        self._ref_ap = self._mm_spin()
        self._ref_ap.setToolTip("Stereotaxic AP from bregma; + = anterior.")
        self._ref_ap.valueChanged.connect(lambda _v: self._on_ref_fields())
        ref.addWidget(self._ref_ap)
        ref.addWidget(QLabel("ML"))
        self._ref_ml = self._mm_spin()
        self._ref_ml.setToolTip("Stereotaxic ML from bregma; + = animal's right, - = animal's left.")
        self._ref_ml.valueChanged.connect(lambda _v: self._on_ref_fields())
        ref.addWidget(self._ref_ml)
        self._ref_what = QLineEdit()
        self._ref_what.setPlaceholderText("what it is, e.g. window centre (surgical record)")
        self._ref_what.textEdited.connect(lambda _t: self._on_ref_fields())
        ref.addWidget(self._ref_what, stretch=1)
        clr_ref = QPushButton("Clear")
        clr_ref.clicked.connect(lambda: self._clear_point("ref"))
        ref.addWidget(clr_ref)
        v.addLayout(ref)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        orient = QHBoxLayout()
        self._anterior = QComboBox()
        self._anterior.addItem("—", None)
        for s in lmk.SIDES:
            self._anterior.addItem(s.capitalize(), s)
        self._anterior.currentIndexChanged.connect(self._on_anterior)
        orient.addWidget(QLabel("Anterior edge:"))
        orient.addWidget(self._anterior)
        orient.addSpacing(10)
        orient.addWidget(QLabel("Animal's left is image:"))
        self._left_side = QComboBox()
        self._left_side.currentIndexChanged.connect(self._on_orientation_fields)
        orient.addWidget(self._left_side)
        orient.addSpacing(10)
        orient.addWidget(QLabel("In view:"))
        self._hemi = QComboBox()
        self._hemi.addItem("—", None)
        for t, d in (("Left hemisphere", "left"), ("Right hemisphere", "right"), ("Both hemispheres", "both")):
            self._hemi.addItem(t, d)
        self._hemi.currentIndexChanged.connect(self._on_orientation_fields)
        orient.addWidget(self._hemi)
        orient.addStretch()
        form.addRow("Orientation:", orient)

        cal = QHBoxLayout()
        self._um = QDoubleSpinBox()
        self._um.setRange(0.0, 100.0)
        self._um.setDecimals(4)
        self._um.setSingleStep(0.01)
        self._um.setSpecialValueText("Not set")
        self._um.setSuffix("  µm/px")
        self._um.valueChanged.connect(self._on_cal)
        cal.addWidget(self._um)
        self._um_source = QLineEdit()
        self._um_source.setPlaceholderText("source, e.g. caliper image 2026-09-03, 271.6 px/mm")
        self._um_source.textEdited.connect(lambda _t: self._on_cal())
        cal.addWidget(self._um_source, stretch=1)
        form.addRow("This session's calibration:", cal)

        who = QHBoxLayout()
        self._annotator = QLineEdit()
        self._annotator.setPlaceholderText("initials")
        self._annotator.setMaximumWidth(90)
        self._annotator.textEdited.connect(lambda _t: self._mark_dirty())
        who.addWidget(self._annotator)
        self._notes = QLineEdit()
        self._notes.setPlaceholderText("notes: how bregma/lambda were judged, source of surgical coordinates, …")
        self._notes.textEdited.connect(lambda _t: self._mark_dirty())
        who.addWidget(self._notes, stretch=1)
        form.addRow("Annotator / notes:", who)
        v.addLayout(form)

        self._check_label = QLabel("")
        self._check_label.setWordWrap(True)
        v.addWidget(self._check_label)

        btns = QHBoxLayout()
        self._save_btn = QPushButton("Save landmarks.json")
        self._save_btn.clicked.connect(self.save)
        btns.addWidget(self._save_btn)
        self._revert_btn = QPushButton("Revert to saved")
        self._revert_btn.clicked.connect(lambda: self._load(force=True))
        btns.addWidget(self._revert_btn)
        self._file_label = QLabel("")
        self._file_label.setStyleSheet(HINT_STYLE)
        btns.addWidget(self._file_label, stretch=1)
        v.addLayout(btns)

    def _xy_boxes(self, key: str) -> tuple[QSpinBox, QSpinBox]:
        """x and y entry for a point, full-res px; accepts values beyond the
        frame for off-image landmarks. Commits on Enter / focus-out / arrows."""
        boxes = []
        for axis, lo, hi, full in (("x", _X_MIN, _X_MAX, _FULL_W), ("y", _Y_MIN, _Y_MAX, _FULL_H)):
            sb = QSpinBox()
            sb.setRange(lo - 1, hi)          # lo - 1 is the "not placed" value
            sb.setSpecialValueText(f"{axis} —")
            sb.setPrefix(f"{axis} ")
            sb.setKeyboardTracking(False)
            sb.setValue(lo - 1)
            sb.setToolTip(f"{axis} in full-resolution pixels (image is 0–{full - 1}). "
                          f"Negative or ≥ {full} places the point outside the image.")
            sb.valueChanged.connect(lambda _v, k=key: self._on_xy_edit(k))
            boxes.append(sb)
        self._xy[key] = (boxes[0], boxes[1])
        return boxes[0], boxes[1]

    @staticmethod
    def _mm_spin() -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(-10.0, 10.0)
        s.setDecimals(2)
        s.setSingleStep(0.1)
        s.setSuffix(" mm")
        return s

    # ── public API ───────────────────────────────────────────────────────────

    def set_session_dir(self, session_dir: str) -> None:
        if session_dir == self._session_dir:
            return
        if self._dirty and self._session_dir:
            answer = QMessageBox.question(
                self, "Unsaved landmarks",
                f"Save the landmarks you placed for {Path(self._session_dir).name} before switching?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard,
                QMessageBox.StandardButton.Save,
            )
            if answer == QMessageBox.StandardButton.Save:
                self.save()
        self._session_dir = session_dir
        self._image.clear_image()
        self._load(force=True)

    def set_image_path(self, png_path: str) -> None:
        pm = QPixmap(png_path)
        if pm.isNull():
            self._image.clear_image("The green-reference preview could not be read.")
            return
        self._image.set_image(pm)
        self._refresh()

    def clear_image(self) -> None:
        self._image.clear_image()

    def set_default_um_per_px(self, um: float | None) -> None:
        """The rig calibration, offered only when this session has none saved."""
        self._default_um = um
        if not self._dirty and not (self._lm.get("calibration") or {}).get("um_per_px") and um:
            self._loading = True
            self._um.setValue(um)
            self._um_source.setText("rig calibration default: confirm it applies to this session's optics")
            self._loading = False
            self._write_cal_into_lm()
            self._refresh()

    def landmarks(self) -> dict:
        return self._lm

    # ── load / save ──────────────────────────────────────────────────────────

    def _load(self, force: bool = False) -> None:
        self._loading = True
        try:
            loaded = lmk.load(self._session_dir) if self._session_dir else None
            err = ""
        except lmk.LandmarkError as exc:
            loaded, err = None, str(exc)
        self._lm = loaded or lmk.empty(Path(self._session_dir).name if self._session_dir else "")
        # copy: widget change-signals must never write back into what we're loading from
        o = dict(self._lm.get("orientation") or {})
        self._set_combo(self._anterior, o.get("anterior_side"))
        self._fill_left_sides()
        self._set_combo(self._left_side, o.get("animal_left_side"))
        self._set_combo(self._hemi, o.get("hemispheres"))
        pts = self._lm.get("points") or {}
        for key in ("bregma", "lambda"):
            p = pts.get(key)
            self._set_combo(self._status_combos[key], p.get("status") if p else "estimated")
        rp = self._lm.get("reference_point") or {}
        self._ref_ap.setValue(float(rp.get("ap_mm", 0.0) or 0.0))
        self._ref_ml.setValue(float(rp.get("ml_mm", 0.0) or 0.0))
        self._ref_what.setText(rp.get("what", ""))
        cal = self._lm.get("calibration") or {}
        if cal.get("um_per_px"):
            self._um.setValue(float(cal["um_per_px"]))
            self._um_source.setText(cal.get("source", ""))
        elif self._default_um:
            self._um.setValue(self._default_um)
            self._um_source.setText("rig calibration default: confirm it applies to this session's optics")
        else:
            self._um.setValue(0.0)
            self._um_source.setText("")
        self._annotator.setText(self._lm.get("annotator", ""))
        self._notes.setText(self._lm.get("notes", ""))
        self._loading = False
        self._write_cal_into_lm()
        self._dirty = False
        if err:
            self._file_label.setStyleSheet(WARNING_STYLE)
            self._file_label.setText(f"Existing file not loaded: {err}")
        elif loaded:
            self._file_label.setStyleSheet(HINT_STYLE)
            self._file_label.setText(f"Loaded {lmk.FILENAME} (saved {loaded.get('annotated_at') or '?'}).")
            cs = lmk.compass_settings(self._lm)
            if cs:
                self.orientation_loaded.emit(cs)
        else:
            self._file_label.setStyleSheet(HINT_STYLE)
            self._file_label.setText("No landmarks saved for this session yet." if self._session_dir else "")
        self._refresh()

    def save(self) -> None:
        if not self._session_dir:
            return
        self._lm["annotator"] = self._annotator.text().strip()
        self._lm["notes"] = self._notes.text().strip()
        self._lm["session"] = Path(self._session_dir).name
        try:
            path = lmk.save(self._session_dir, self._lm)
        except (lmk.LandmarkError, OSError) as exc:
            QMessageBox.warning(self, "Landmarks not saved", str(exc))
            return
        self._lm = lmk.load(self._session_dir) or self._lm
        self._dirty = False
        self._file_label.setStyleSheet(HINT_STYLE)
        self._file_label.setText(f"Saved {path.name}.")
        cs = lmk.compass_settings(self._lm)
        if cs:
            self.orientation_loaded.emit(cs)
        self.saved.emit(str(path))
        self._refresh()

    # ── edits ────────────────────────────────────────────────────────────────

    def _current_target(self) -> str:
        return self._target_group.checkedButton().property("target")

    def _on_click(self, x: int, y: int) -> None:
        if not self._session_dir:
            return
        pts = self._lm.setdefault("points", {"bregma": None, "lambda": None, "midline": []})
        t = self._current_target()
        if t in ("bregma", "lambda"):
            self._place(t, x, y)
        elif t in ("mid0", "mid1"):
            mid = list(pts.get("midline") or [])
            while len(mid) < 2:
                mid.append(None)
            mid[0 if t == "mid0" else 1] = {"x": x, "y": y}
            # a half-placed midline stays in memory; validate() blocks saving until both exist
            pts["midline"] = mid
        elif t == "ref":
            self._place("ref", x, y)
        # auto-advance within the midline pair; otherwise stay on the same target
        if t == "mid0":
            self._target_group.button(3).setChecked(True)
        self._mark_dirty()

    def _place(self, key: str, x: int, y: int) -> None:
        """Put bregma / lambda / the reference point at (x, y). A point
        outside the image can only be an estimate, so its status follows."""
        if key == "ref":
            self._lm["reference_point"] = {"x": x, "y": y, "ap_mm": self._ref_ap.value(),
                                           "ml_mm": self._ref_ml.value(), "what": self._ref_what.text().strip()}
            return
        status = self._status_combos[key].currentData()
        if not lmk.in_field({"x": x, "y": y}) and status == "visible":
            status = "estimated"
            cb = self._status_combos[key]
            cb.blockSignals(True)
            self._set_combo(cb, status)
            cb.blockSignals(False)
        self._lm.setdefault("points", {})[key] = {"x": x, "y": y, "status": status}

    def _on_xy_edit(self, key: str) -> None:
        if self._loading or not self._session_dir:
            return
        xs, ys = self._xy[key]
        if xs.value() == xs.minimum() or ys.value() == ys.minimum():
            return                       # wait until both are filled in
        self._place(key, xs.value(), ys.value())
        self._mark_dirty()

    def _clear_point(self, key: str) -> None:
        pts = self._lm.setdefault("points", {})
        if key in ("bregma", "lambda"):
            pts[key] = None
        elif key == "midline":
            pts["midline"] = []
        elif key == "ref":
            self._lm["reference_point"] = None
        self._mark_dirty()

    def _on_status(self, key: str) -> None:
        if self._loading:
            return
        p = (self._lm.get("points") or {}).get(key)
        if p:
            p["status"] = self._status_combos[key].currentData()
        self._mark_dirty()

    def _on_ref_fields(self) -> None:
        if self._loading:
            return
        rp = self._lm.get("reference_point")
        if rp:
            rp.update(ap_mm=self._ref_ap.value(), ml_mm=self._ref_ml.value(), what=self._ref_what.text().strip())
        self._mark_dirty()

    def _on_anterior(self) -> None:
        self._fill_left_sides()
        self._on_orientation_fields()

    def _fill_left_sides(self) -> None:
        keep = self._left_side.currentData()
        self._left_side.blockSignals(True)
        self._left_side.clear()
        self._left_side.addItem("—", None)
        a = self._anterior.currentData()
        if a:
            for s in lmk.perpendicular_sides(a):
                self._left_side.addItem(s.capitalize(), s)
        self._set_combo(self._left_side, keep)
        self._left_side.blockSignals(False)

    def _on_orientation_fields(self, *_a) -> None:
        if self._loading:      # _load() sets the combos one at a time; don't record half-set states
            return
        o = self._lm.setdefault("orientation", {})
        o["anterior_side"] = self._anterior.currentData()
        o["animal_left_side"] = self._left_side.currentData()
        o["hemispheres"] = self._hemi.currentData()
        self._mark_dirty()

    def _on_cal(self, *_a) -> None:
        if self._loading:
            return
        self._write_cal_into_lm()
        self._mark_dirty()     # _refresh() re-sizes the margin and the "mm off-image" notes

    def _write_cal_into_lm(self) -> None:
        um = self._um.value()
        self._lm["calibration"] = {"um_per_px": um if um > 0 else None, "source": self._um_source.text().strip()}

    def _mark_dirty(self) -> None:
        if self._loading:
            return
        self._dirty = True
        self._file_label.setStyleSheet(WARNING_STYLE)
        self._file_label.setText("Unsaved changes.")
        self._refresh()

    # ── display ──────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        pts = self._lm.get("points") or {}
        marks: dict[str, tuple[float, float, str, bool]] = {}
        colours = {k: c for k, _t, c in _TARGETS}
        for key in ("bregma", "lambda"):
            p = pts.get(key)
            lab = self.findChild(QLabel, f"{key}_pos")
            self._show_xy(key, p)
            if p:
                marks[key] = (p["x"], p["y"], colours[key], p.get("status") != "visible")
            if lab:
                lab.setText(self._where_text(p) if p else "not placed")
        mid = [m for m in (pts.get("midline") or []) if m]
        for i, m in enumerate(pts.get("midline") or []):
            if m:
                marks[f"mid{i}"] = (m["x"], m["y"], colours[f"mid{i}"], False)
        self._mid_label.setText({0: "not placed", 1: "1 of 2 points placed"}.get(len(mid), "2 points placed"))
        rp = self._lm.get("reference_point")
        self._show_xy("ref", rp)
        if rp:
            marks["ref"] = (rp["x"], rp["y"], colours["ref"], True)
            self._ref_label.setText(self._where_text(rp) if not lmk.in_field(rp) else "")
        else:
            self._ref_label.setText("")
        self._image.marks = marks
        self._apply_margin()
        self._image.update()

        problems = [p for p in lmk.validate(self._lm) if not p.startswith("schema")]
        if len(mid) == 1:
            problems.append("place the second midline point")
        if problems:
            self._check_label.setStyleSheet(WARNING_STYLE)
            self._check_label.setText("Can't save yet: " + "; ".join(problems) + ".")
        else:
            self._check_label.setStyleSheet(HINT_STYLE)
            self._check_label.setText(self._coverage_text())
        self._save_btn.setEnabled(bool(self._session_dir) and not problems)
        self._revert_btn.setEnabled(bool(self._session_dir))

    def _show_xy(self, key: str, p: dict | None) -> None:
        for sb, v in zip(self._xy[key], (p["x"], p["y"]) if p else (None, None)):
            sb.blockSignals(True)
            sb.setValue(sb.minimum() if v is None else int(round(v)))
            sb.blockSignals(False)

    def _um_now(self) -> float | None:
        um = self._um.value()
        return um if um > 0 else self._default_um

    def _where_text(self, p: dict) -> str:
        """'' on the image, else how far past which edge(s), in mm when calibrated."""
        out = lmk.outside_edges(p["x"], p["y"])
        if not out:
            return ""
        um = self._um_now()
        parts = [(f"{d * um / 1000:.2f} mm " if um else f"{d:.0f} px ") + _EDGE_WORDS[side]
                 for side, d in out.items()]
        return "off-image: " + ", ".join(parts)

    def _apply_margin(self) -> None:
        """Canvas margin = the chosen room, grown to show every placed point."""
        um = self._um_now() or _NOMINAL_UM
        px_per_mm = 1000.0 / um
        pad = (self._margin.currentData() or 0.0) * px_per_mm
        for x, y, *_ in self._image.marks.values():
            if key_out := lmk.outside_edges(x, y):
                pad = max(pad, max(key_out.values()) + _FIT_PX)
        self._image.set_margin(pad, px_per_mm if self._um_now() else 0.0)

    def _coverage_text(self) -> str:
        try:
            probe = lmk.stereotaxic(self._lm, _FULL_W / 2, _FULL_H / 2)
        except lmk.LandmarkError as exc:
            cs = lmk.compass_settings(self._lm)
            base = "Orientation is enough for figure compasses. " if cs else ""
            return f"{base}No stereotaxic coordinates yet: {exc}."
        return (f"Coordinates anchored at {probe['origin']} ({probe['origin_status'].replace('_', ' ')}); "
                f"ML measured from the {probe['ml_measured_from']}; axes from {probe['axes_from']}. "
                f"Image centre is AP {probe['ap_mm']:+.2f}, ML {probe['ml_mm']:+.2f} mm.")

    def _on_hover(self, x: int, y: int) -> None:
        where = self._where_text({"x": x, "y": y})
        where = f"   ({where})" if where else ""
        try:
            s = lmk.stereotaxic(self._lm, x, y)
            self._hover_label.setText(f"x {x}, y {y}   →   AP {s['ap_mm']:+.2f} mm, ML {s['ml_mm']:+.2f} mm "
                                      f"(from {s['origin']}){where}")
        except lmk.LandmarkError:
            self._hover_label.setText(f"x {x}, y {y}{where}")

    @staticmethod
    def _set_combo(combo: QComboBox, data) -> None:
        i = combo.findData(data)
        combo.setCurrentIndex(i if i >= 0 else 0)
