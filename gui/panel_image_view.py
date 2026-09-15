# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
PanelImageView / PanelImageBrowser — scaled, live-updating display of PNGs
saved by intrinsic_imaging.py: the post-trial three/four-panel running-
average summary (_save_trial_vs_running_average_panel — trial red
subtraction, running average(s), green reference) and the red-LED settling
sample snapshot (_sample_red_settling).

Pure QPixmap loading -- no numpy/matplotlib in the GUI process (see
requirements-gui.txt: "DO NOT add PySpin, numpy, or anything from the
acquisition stack"). Every PNG here is already fully rendered by the
acquisition subprocess; these widgets only display it, rescaling to fit the
available width whenever the pane or window is resized.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.theme import text_rgba

_DEFAULT_PLACEHOLDER_TEXT = (
    "Running-average panel image will appear here after the first trial completes…"
)
_PLACEHOLDER_STYLE = f"color:{text_rgba(0.55)}; font-style:italic;"


class PanelImageView(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        placeholder_text: str = _DEFAULT_PLACEHOLDER_TEXT,
    ) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._placeholder_text = placeholder_text

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(self._placeholder_text)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setStyleSheet(_PLACEHOLDER_STYLE)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setMinimumHeight(120)
        layout.addWidget(self._label)

    def clear_image(self) -> None:
        self._pixmap = None
        self._label.setPixmap(QPixmap())
        self._label.setText(self._placeholder_text)
        self._label.setStyleSheet(_PLACEHOLDER_STYLE)

    def set_image_path(self, path: str) -> bool:
        """Load and display the PNG at path. Returns False (leaving any
        previously displayed image in place) if it can't be read yet."""
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return False
        self._pixmap = pixmap
        self._label.setStyleSheet("")
        self._rescale()
        return True

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override name)
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        target = self._label.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled = self._pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)


class PanelImageBrowser(QWidget):
    """PanelImageView plus Prev/Next/Latest navigation across every trial
    panel image seen so far this session, so a user mid-session (or
    reviewing right after) can step back to an earlier trial without losing
    the running display.

    Auto-follows the newest trial as it arrives -- unless the user has
    stepped back to look at an earlier one, in which case a new trial just
    extends the history without yanking the view away from what's on
    screen. "Latest" jumps back to following.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._history: list[tuple[int, str]] = []  # (trial_index, path), chronological
        self._cursor: int = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._view = PanelImageView()
        layout.addWidget(self._view, stretch=1)

        nav = QHBoxLayout()
        self._prev_btn = QPushButton("◀ Prev")
        self._prev_btn.setEnabled(False)
        self._prev_btn.clicked.connect(self._on_prev)
        nav.addWidget(self._prev_btn)

        self._position_label = QLabel("")
        self._position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(self._position_label, stretch=1)

        self._next_btn = QPushButton("Next ▶")
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._on_next)
        nav.addWidget(self._next_btn)

        self._latest_btn = QPushButton("⏭ Latest")
        self._latest_btn.setEnabled(False)
        self._latest_btn.setToolTip("Jump to the most recently completed trial's panel image.")
        self._latest_btn.clicked.connect(self._on_latest)
        nav.addWidget(self._latest_btn)

        layout.addLayout(nav)

    def clear_all(self) -> None:
        self._history = []
        self._cursor = -1
        self._view.clear_image()
        self._refresh_nav()

    def add_trial(self, trial_index: int, path: str) -> None:
        was_following_latest = self._cursor == len(self._history) - 1
        self._history.append((trial_index, path))
        if was_following_latest:
            self._cursor = len(self._history) - 1
            self._view.set_image_path(path)
        self._refresh_nav()

    def _on_prev(self) -> None:
        if self._cursor > 0:
            self._cursor -= 1
            self._show_current()

    def _on_next(self) -> None:
        if self._cursor < len(self._history) - 1:
            self._cursor += 1
            self._show_current()

    def _on_latest(self) -> None:
        if self._history:
            self._cursor = len(self._history) - 1
            self._show_current()

    def _show_current(self) -> None:
        if 0 <= self._cursor < len(self._history):
            _, path = self._history[self._cursor]
            self._view.set_image_path(path)
        self._refresh_nav()

    def _refresh_nav(self) -> None:
        n = len(self._history)
        if n == 0:
            self._position_label.setText("")
        else:
            trial_index, _ = self._history[self._cursor]
            self._position_label.setText(f"Trial {trial_index:03d}   ({self._cursor + 1} of {n})")
        self._prev_btn.setEnabled(self._cursor > 0)
        self._next_btn.setEnabled(0 <= self._cursor < n - 1)
        self._latest_btn.setEnabled(n > 0 and self._cursor != n - 1)
