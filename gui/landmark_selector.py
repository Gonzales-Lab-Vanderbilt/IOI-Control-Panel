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

The saved file is read by session_poster_figures.py and session_timelapse.py
(landmark markers, per-session calibration, stereotaxic ROI coordinates), and
its orientation fills in the Figures tab's compass (orientation_loaded).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
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


class _LandmarkImage(QLabel):
    """Green reference at true aspect ratio; clicks come back in full-res
    sensor pixels; markers are painted over it. Same geometry maths as
    gui/crop_selector.py's _ImageCropLabel."""

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
        self._scale = 0.0
        self._ox = self._oy = 0.0
        self.marks: dict[str, tuple[float, float, str, bool]] = {}   # key -> (x, y, colour, dashed)
        self.clear_image()

    def set_image(self, pixmap: QPixmap) -> None:
        self._orig = pixmap
        self.setText("")
        self.setStyleSheet("")
        self._rescale()

    def clear_image(self, text: str = _PLACEHOLDER) -> None:
        self._orig = None
        self._scale = 0.0
        self.setPixmap(QPixmap())
        self.setStyleSheet(f"color:{text_rgba(0.55)}; font-style:italic;")
        self.setText(text)

    def has_image(self) -> bool:
        return self._orig is not None

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._orig is None or self.width() <= 0:
            return
        scaled = self._orig.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled)
        self._scale = scaled.width() / _FULL_W if scaled.width() else 0.0
        self._ox = (self.width() - scaled.width()) / 2.0
        self._oy = (self.height() - scaled.height()) / 2.0
        self.update()

    def to_image(self, pt: QPoint) -> tuple[int, int] | None:
        if self._orig is None or self._scale <= 0:
            return None
        ix = round((pt.x() - self._ox) / self._scale)
        iy = round((pt.y() - self._oy) / self._scale)
        if not (0 <= ix < _FULL_W and 0 <= iy < _FULL_H):
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
        if self._orig is None or self._scale <= 0:
            return
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
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
            "Reference point and enter its stereotaxic coordinates. Saved per session as landmarks.json."
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
            lab = QLabel("not placed")
            lab.setObjectName(f"{key}_pos")
            grid.addWidget(lab, row, 1)
            cb = QComboBox()
            for t, d in _STATUS_ITEMS:
                cb.addItem(t, d)
            cb.setCurrentIndex(1)   # Estimated unless the user says otherwise
            cb.currentIndexChanged.connect(lambda _i, k=key: self._on_status(k))
            self._status_combos[key] = cb
            grid.addWidget(cb, row, 2)
            clr = QPushButton("Clear")
            clr.clicked.connect(lambda _c=False, k=key: self._clear_point(k))
            grid.addWidget(clr, row, 3)
        grid.addWidget(QLabel("Midline:"), 2, 0)
        self._mid_label = QLabel("not placed")
        grid.addWidget(self._mid_label, 2, 1, 1, 2)
        clr_mid = QPushButton("Clear")
        clr_mid.clicked.connect(lambda: self._clear_point("midline"))
        grid.addWidget(clr_mid, 2, 3)
        grid.setColumnStretch(1, 1)
        v.addLayout(grid)

        ref = QHBoxLayout()
        ref.addWidget(QLabel("Reference point:"))
        self._ref_label = QLabel("not placed")
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
            pts[t] = {"x": x, "y": y, "status": self._status_combos[t].currentData()}
        elif t in ("mid0", "mid1"):
            mid = list(pts.get("midline") or [])
            while len(mid) < 2:
                mid.append(None)
            mid[0 if t == "mid0" else 1] = {"x": x, "y": y}
            # a half-placed midline stays in memory; validate() blocks saving until both exist
            pts["midline"] = mid
        elif t == "ref":
            self._lm["reference_point"] = {"x": x, "y": y, "ap_mm": self._ref_ap.value(),
                                           "ml_mm": self._ref_ml.value(), "what": self._ref_what.text().strip()}
        # auto-advance within the midline pair; otherwise stay on the same target
        if t == "mid0":
            self._target_group.button(3).setChecked(True)
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
        self._mark_dirty()

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
            if p:
                marks[key] = (p["x"], p["y"], colours[key], p.get("status") != "visible")
                if lab:
                    lab.setText(f"x {p['x']}, y {p['y']}")
            elif lab:
                lab.setText("not placed")
        mid = [m for m in (pts.get("midline") or []) if m]
        for i, m in enumerate(pts.get("midline") or []):
            if m:
                marks[f"mid{i}"] = (m["x"], m["y"], colours[f"mid{i}"], False)
        self._mid_label.setText({0: "not placed", 1: "1 of 2 points placed"}.get(len(mid), "2 points placed"))
        rp = self._lm.get("reference_point")
        if rp:
            marks["ref"] = (rp["x"], rp["y"], colours["ref"], True)
            self._ref_label.setText(f"x {rp['x']}, y {rp['y']}")
        else:
            self._ref_label.setText("not placed")
        self._image.marks = marks
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
        try:
            s = lmk.stereotaxic(self._lm, x, y)
            self._hover_label.setText(f"x {x}, y {y}   →   AP {s['ap_mm']:+.2f} mm, ML {s['ml_mm']:+.2f} mm "
                                      f"(from {s['origin']})")
        except lmk.LandmarkError:
            self._hover_label.setText(f"x {x}, y {y}")

    @staticmethod
    def _set_combo(combo: QComboBox, data) -> None:
        i = combo.findData(data)
        combo.setCurrentIndex(i if i >= 0 else 0)
