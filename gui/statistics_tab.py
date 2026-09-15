# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
StatisticsWidget — content of the Analysis tab: the leave-one-out ROI
time-course / pixelwise t-map / sign-flip permutation cluster test pipeline.

Wraps statistical_analyses.py via subprocess. This is the lab's canonical
post-acquisition analysis pipeline (used for the poster and all reported
results). Consumes per-trial arrays (baseline_reference.npy /
post_mean_analysis_window.npy) and produces the ROI time course and the
statistical significance test.

Runs go through gui/analysis_queue.py's AnalysisQueue instead of running one
at a time inline: "Run analysis" snapshots the form into an AnalysisJob
(gui/analysis_job.py) and hands it to the queue, which starts it right away
(only same-output-folder jobs wait for each other).
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui import analysis_argv
from gui.analysis_argv import PosterFormSnapshot, StatsFormSnapshot
from gui.analysis_job import AnalysisJob, JobStatus, TERMINAL_STATUSES
from gui.analysis_queue import AnalysisQueue
from gui.crop_selector import CropSelectorWidget
from gui.panel_image_view import PanelImageView
from gui.rig_settings import SpatialCalibration, load_spatial_calibration, save_spatial_calibration
from gui.script_runner import ScriptRunner
from gui.theme import (
    CHROME_ACCENT_HEX,
    DANGER_HEX,
    GREEN_HEX,
    HINT_STYLE,
    STYLE_START,
    STYLE_STOP,
    WARNING_STYLE,
    text_rgba,
)

_DEFAULT_N_PERM = 1000
_DEFAULT_FIXED_VMAX = 0.05

_STATUS_ICON = {
    JobStatus.QUEUED:    f"<span style='color:{text_rgba(0.55)};'>○</span>",
    JobStatus.RUNNING:   f"<span style='color:{CHROME_ACCENT_HEX};'>●</span>",
    JobStatus.STOPPING:  f"<span style='color:{DANGER_HEX};'>◐</span>",
    JobStatus.SUCCEEDED: f"<span style='color:{GREEN_HEX};'>●</span>",
    JobStatus.FAILED:    f"<span style='color:{DANGER_HEX};'>●</span>",
    JobStatus.STOPPED:   f"<span style='color:{DANGER_HEX};'>●</span>",
}

_COL_STATUS, _COL_SESSION, _COL_PHASE, _COL_RESULT, _COL_ACTION = range(5)


class StatisticsWidget(QWidget):
    """
    Leave-one-out ROI time course + pixelwise t-map / permutation cluster
    test. Owns an AnalysisQueue so several runs can be queued and, if the
    user raises the concurrency limit, run at the same time.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._queue = AnalysisQueue()
        self._queue.jobs_changed.connect(self._refresh_table)

        self._selected_job: AnalysisJob | None = None

        self._trial_checks: dict[int, QCheckBox] = {}
        self._trials_scanned_session_dir: str = ""
        self._detected_conditions: set[str] = set()

        self._build_ui()
        self._clear_detail_panel()

        # Mirror the Region / Crop tab's rendered green-reference preview
        # next to the Figures tab's orientation controls -- same image, not
        # re-rendered, so both widgets need to exist first (built above).
        self._crop_selector.preview_changed.connect(self._orientation_preview.set_image_path)
        self._crop_selector.preview_cleared.connect(self._orientation_preview.clear_image)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setSpacing(10)
        outer.setContentsMargins(14, 10, 14, 10)

        left = QWidget()
        left_outer = QVBoxLayout(left)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_w = QWidget()
        layout = QVBoxLayout(scroll_w)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(8)
        scroll.setWidget(scroll_w)
        left_outer.addWidget(scroll)

        layout.addWidget(self._build_common_form())
        layout.addWidget(self._build_calibration_group())
        layout.addWidget(self._build_advanced_section())
        layout.addLayout(self._build_button_row())
        layout.addStretch()

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(self._build_queue_panel())
        right_layout.addWidget(self._build_progress_area())
        right_layout.addLayout(self._build_result_row())
        right_layout.addLayout(self._build_post_run_row())
        right_layout.addWidget(self._build_log_panel(), stretch=1)

        # Equal halves: the advanced panel's widest rows need roughly half of a
        # 1920 px window, and an uneven split put a horizontal scrollbar under
        # the parameters column.
        outer.addWidget(left, stretch=1)
        outer.addWidget(right, stretch=1)

    def _build_common_form(self) -> QGroupBox:
        box = QGroupBox("Final statistics && figures")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(7)

        intro = QLabel(
            "For a finished session: leave-one-out ROI time course, pixelwise "
            "t-map with a permutation cluster test, then figures. Uses the "
            "per-trial response maps made during acquisition (Run Session → "
            "\"Make response maps after each trial\")."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(HINT_STYLE)
        form.addRow(intro)

        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select session folder to analyze…")
        self._folder_edit.editingFinished.connect(self._rescan_trials)
        folder_row.addWidget(self._folder_edit, stretch=1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(btn_browse)
        form.addRow("Session folder:", folder_row)

        self._nperm_spin = QSpinBox()
        self._nperm_spin.setRange(10, 20_000)
        self._nperm_spin.setValue(_DEFAULT_N_PERM)
        self._nperm_spin.setToolTip(
            "Sign-flip permutations for the cluster-corrected p-value.\n"
            f"{_DEFAULT_N_PERM} is the pipeline's own default, used for reported results.\n"
            "Lower values run faster for a quick look but give a coarser p-value."
        )
        form.addRow("Permutations:", self._nperm_spin)

        self._neg_only_check = QCheckBox("Negative-going clusters only")
        self._neg_only_check.setChecked(True)
        self._neg_only_check.setToolTip(
            "Cluster-forming threshold t < -3.0.\n"
            "Activation at 630 nm is a reflectance DECREASE. A two-sided test can let a "
            "positive-going artifact (e.g. edge vignetting) win as the 'largest cluster' "
            "and hide the real response — this happened on 0724_s1fl_stim. Leave checked "
            "unless you specifically need the two-sided test."
        )
        form.addRow("Cluster sign:", self._neg_only_check)

        self._condition_combo = QComboBox()
        self._condition_combo.addItem("All trials", "all")
        self._condition_combo.addItem("Stim only", "stim")
        self._condition_combo.addItem("Catch (no-stim) only", "catch")
        self._condition_combo.setToolTip(
            "For sessions with interleaved stim/catch trials (see Run Session's "
            "'Interleave stim / catch trials'): restrict the analysis to one "
            "condition. 'All trials' matches every session's behavior before "
            "interleaving existed. Requires trial_condition in each trial's "
            "trial_metadata.json — older sessions only support 'All trials'."
        )
        form.addRow("Condition:", self._condition_combo)

        self._interleaved_hint = QLabel()
        self._interleaved_hint.setWordWrap(True)
        self._interleaved_hint.setStyleSheet(f"color:{CHROME_ACCENT_HEX};")
        self._interleaved_hint.hide()
        form.addRow("", self._interleaved_hint)

        self._roi_mode_combo = QComboBox()
        self._roi_mode_combo.addItem("Strongest-responding 5% of pixels (leave-one-out)", False)
        self._roi_mode_combo.addItem("Whole analysis region", True)
        self._roi_mode_combo.setToolTip(
            "Strongest-responding 5% (default): for each trial, the ROI is the 5% of "
            "pixels with the most negative ΔR/R averaged over the OTHER trials "
            "(leave-one-out), so a trial never selects its own ROI. Use for a session "
            "measured on its own merits.\n"
            "Whole analysis region (--full-frame): the ROI is the entire session crop, "
            "or the analysis rectangle if one is set in Advanced → Region / Crop. Use "
            "for a control/no-stim session compared to a stim session on equal footing, "
            "not for the stim session itself."
        )
        self._roi_mode_combo.currentIndexChanged.connect(self._update_crop_effect_label)
        form.addRow("ROI selection:", self._roi_mode_combo)

        self._out_subfolder_edit = QLineEdit(analysis_argv.DEFAULT_OUT_SUBFOLDER)
        self._out_subfolder_edit.setToolTip(
            "Subfolder created INSIDE the session folder: <session folder>/<name>. "
            "For a non-'all' condition above, '_stim' or '_catch' is appended so a "
            "stim run and a catch run of the same session don't overwrite each "
            "other. Ignored if an output folder override is set in "
            "Advanced → Output / Performance."
        )
        form.addRow("Output subfolder:", self._out_subfolder_edit)

        return box

    # ── Spatial calibration (per computer, not per run) ───────────────────────

    def _build_calibration_group(self) -> QGroupBox:
        box = QGroupBox("Spatial calibration for your rig")
        v = QVBoxLayout(box)
        v.setSpacing(6)

        row = QHBoxLayout()
        self._um_per_px_spin = QDoubleSpinBox()
        self._um_per_px_spin.setRange(0.0, 100.0)
        self._um_per_px_spin.setDecimals(4)
        self._um_per_px_spin.setSingleStep(0.01)
        self._um_per_px_spin.setSpecialValueText("Not set")
        self._um_per_px_spin.setSuffix("  µm/px")
        self._um_per_px_spin.setMinimumWidth(150)
        self._um_per_px_spin.setToolTip(
            "Micrometers per full-frame (1920 px) camera pixel, used only for the "
            "figures' 1 mm scale bar. Measure it by imaging a ruler or calibration "
            "slide through your own objective, aperture, and working distance; "
            "re-measure whenever any of them change. 0 = not set (no scale bar)."
        )
        row.addWidget(self._um_per_px_spin)

        self._verify_cal_btn = QPushButton("Mark as verified")
        self._verify_cal_btn.setToolTip(
            "Confirm this value was measured on this rig in its current configuration."
        )
        self._verify_cal_btn.clicked.connect(self._on_verify_calibration)
        row.addWidget(self._verify_cal_btn)
        row.addStretch()
        v.addLayout(row)

        self._cal_status_label = QLabel()
        self._cal_status_label.setWordWrap(True)
        v.addWidget(self._cal_status_label)

        cal = load_spatial_calibration()
        self._cal_verified = cal.verified
        self._um_per_px_spin.setValue(cal.um_per_px or 0.0)
        # Connected after the initial setValue so loading doesn't un-verify.
        self._um_per_px_spin.valueChanged.connect(self._on_calibration_edited)
        self._refresh_calibration_status()
        return box

    def _calibration_um_per_px(self) -> float | None:
        value = self._um_per_px_spin.value()
        return value if value > 0 else None

    def _on_calibration_edited(self, _value: float) -> None:
        self._cal_verified = False
        self._save_calibration()

    def _on_verify_calibration(self) -> None:
        if self._calibration_um_per_px() is None:
            return
        self._cal_verified = True
        self._save_calibration()

    def _save_calibration(self) -> None:
        save_spatial_calibration(SpatialCalibration(self._calibration_um_per_px(), self._cal_verified))
        self._refresh_calibration_status()

    def _refresh_calibration_status(self) -> None:
        value = self._calibration_um_per_px()
        if value is None:
            self._cal_status_label.setStyleSheet(HINT_STYLE)
            self._cal_status_label.setText(
                "Not set: figures will be made without a scale bar. Saved on this "
                "computer, not in presets."
            )
            self._verify_cal_btn.setEnabled(False)
            self._verify_cal_btn.setText("Mark as verified")
        elif not self._cal_verified:
            self._cal_status_label.setStyleSheet(WARNING_STYLE)
            self._cal_status_label.setText(
                f"⚠ Unverified: scale bars will assume {value:g} µm/px. Confirm it was "
                "measured on this rig as currently set up, then mark it verified."
            )
            self._verify_cal_btn.setEnabled(True)
            self._verify_cal_btn.setText("Mark as verified")
        else:
            self._cal_status_label.setStyleSheet(f"color:{GREEN_HEX};")
            self._cal_status_label.setText(
                f"✓ Verified for this rig: {value:g} µm/px. Editing the value clears "
                "verification."
            )
            self._verify_cal_btn.setEnabled(False)
            self._verify_cal_btn.setText("Verified")

    def _confirm_unverified_calibration(self) -> bool:
        """True if the run should go ahead."""
        value = self._calibration_um_per_px()
        if value is None or self._cal_verified:
            return True
        msg = QMessageBox(self)
        msg.setWindowTitle("Unverified spatial calibration")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            f"The figures' scale bars will use an unverified calibration of "
            f"{value:g} µm/px.\n\nIf this wasn't measured on this rig as it is set up "
            "now, the scale bars will be wrong."
        )
        verify_btn = msg.addButton("Mark verified && run", QMessageBox.ButtonRole.AcceptRole)
        anyway_btn = msg.addButton("Run anyway", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(cancel_btn)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is verify_btn:
            self._on_verify_calibration()
            return True
        return clicked is anyway_btn

    # ── Advanced section ──────────────────────────────────────────────────────

    def _build_advanced_section(self) -> QWidget:
        container = QWidget()
        vlay = QVBoxLayout(container)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(4)

        self._adv_toggle = QPushButton("▶  Advanced settings")
        self._adv_toggle.setCheckable(True)
        self._adv_toggle.setChecked(False)
        self._adv_toggle.setStyleSheet(
            "QPushButton{padding:3px 8px;}"
            "QPushButton:checked{font-weight:bold;}"
        )
        self._adv_toggle.toggled.connect(self._on_adv_toggled)
        vlay.addWidget(self._adv_toggle)

        self._adv_panel = self._build_advanced_panel()
        self._adv_panel.hide()
        vlay.addWidget(self._adv_panel)

        return container

    def _on_adv_toggled(self, checked: bool) -> None:
        self._adv_toggle.setText(
            "▼  Advanced settings" if checked else "▶  Advanced settings"
        )
        self._adv_panel.setVisible(checked)

    def _on_midline_centered_toggled(self, checked: bool) -> None:
        # Medial edge choice is meaningless once medial is the centre line.
        self._medial_label.setEnabled(not checked)
        self._medial_combo.setEnabled(not checked)

    def _build_advanced_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 4, 0, 4)

        tabs = QTabWidget()
        tabs.addTab(self._build_poster_figures_tab(), "Figures")
        tabs.addTab(self._build_trials_tab(), "Trials")
        tabs.addTab(self._build_region_tab(), "Region / Crop")
        tabs.addTab(self._build_log_override_tab(), "Log override")
        tabs.addTab(self._build_output_perf_tab(), "Output / Performance")
        layout.addWidget(tabs)

        return panel

    def _make_form(self) -> tuple[QWidget, QFormLayout]:
        w = QWidget()
        f = QFormLayout(w)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f.setSpacing(6)
        f.setContentsMargins(8, 8, 8, 8)
        return w, f

    def _build_trials_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        hint = QLabel(
            "Uncheck a trial to exclude it from this run — e.g. a trial contaminated "
            "by a bubble under the cranial window. Excluded trials are dropped before "
            "ROI selection, the t-map, and the figures; nothing else about the run "
            "changes. Scanned from the session folder above — click Refresh if trials "
            "have landed since you picked it (e.g. mid-session)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(HINT_STYLE)
        v.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("↻ Refresh trial list")
        btn_refresh.clicked.connect(self._rescan_trials)
        btn_row.addWidget(btn_refresh)
        btn_check_all = QPushButton("Check all")
        btn_check_all.clicked.connect(self._check_all_trials)
        btn_row.addWidget(btn_check_all)
        btn_uncheck_all = QPushButton("Uncheck all")
        btn_uncheck_all.clicked.connect(self._uncheck_all_trials)
        btn_row.addWidget(btn_uncheck_all)
        btn_row.addStretch()
        v.addLayout(btn_row)

        self._trials_empty_label = QLabel("Select a session folder above, then click Refresh.")
        self._trials_empty_label.setStyleSheet(f"color:{text_rgba(0.55)};")
        v.addWidget(self._trials_empty_label)

        self._trials_grid_widget = QWidget()
        self._trials_grid = QGridLayout(self._trials_grid_widget)
        self._trials_grid.setSpacing(4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._trials_grid_widget)
        scroll.setMaximumHeight(180)
        self._trials_grid_widget.hide()
        v.addWidget(scroll)

        self._trials_count_label = QLabel("")
        self._trials_count_label.setStyleSheet(f"color:{text_rgba(0.6)}; font-size:11px;")
        v.addWidget(self._trials_count_label)

        v.addStretch()
        return w

    def _build_region_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        crop_hint = QLabel(
            "Optional analysis rectangle: drag on the image below or type pixel "
            "coordinates, e.g. to exclude an artifact such as a bubble. With no "
            "rectangle, this session's own acquisition crop is used."
        )
        crop_hint.setWordWrap(True)
        crop_hint.setStyleSheet(HINT_STYLE)
        v.addWidget(crop_hint)

        self._crop_selector = CropSelectorWidget()
        self._crop_selector.rect_changed.connect(self._update_crop_effect_label)
        v.addWidget(self._crop_selector, stretch=1)

        form_widget, f = self._make_form()
        f.setContentsMargins(0, 4, 0, 0)
        v.addWidget(form_widget)

        self._crop_scope_combo = QComboBox()
        self._crop_scope_combo.addItem("t-map only", False)
        self._crop_scope_combo.addItem("t-map and ROI selection", True)
        self._crop_scope_combo.setToolTip(
            "t-map only (default): the rectangle restricts the pixelwise t-map, but the "
            "strongest-5% ROI is still chosen from the whole session crop, so it can "
            "land outside the rectangle (e.g. inside the artifact you meant to "
            "exclude), and the ROI time course matches an un-rectangled run.\n"
            "t-map and ROI selection (--roi-within-region): ROI candidates are also "
            "limited to the rectangle.\n"
            "No effect without a rectangle."
        )
        self._crop_scope_combo.currentIndexChanged.connect(self._update_crop_effect_label)
        f.addRow("Rectangle applies to:", self._crop_scope_combo)

        self._crop_effect_label = QLabel()
        self._crop_effect_label.setWordWrap(True)
        f.addRow("", self._crop_effect_label)

        self._no_crop_check = QCheckBox("Use the full 1920×1200 frame instead of the session crop")
        self._no_crop_check.setToolTip(
            "Ignore the acquisition crop saved with this session (--no-crop).\n"
            "Requires an array source directory (below) pointing at pre-computed, "
            "uncropped arrays — the arrays under trial_XXX/analysis/ are already "
            "cropped and can't be un-cropped after the fact."
        )
        self._no_crop_check.toggled.connect(self._update_crop_effect_label)
        f.addRow("", self._no_crop_check)

        array_row = QHBoxLayout()
        self._array_source_edit = QLineEdit()
        self._array_source_edit.setPlaceholderText(
            "Optional: trial_XXX/{baseline_reference,post_mean_analysis_window}.npy source"
        )
        array_row.addWidget(self._array_source_edit, stretch=1)
        btn_array = QPushButton("Browse…")
        btn_array.clicked.connect(self._browse_array_source)
        array_row.addWidget(btn_array)
        f.addRow("Array source dir:", array_row)

        self._update_crop_effect_label()
        return w

    def _update_crop_effect_label(self, *_args) -> None:
        """Spell out exactly what the analysis rectangle restricts, since it
        affects the t-map and ROI selection differently."""
        if not hasattr(self, "_crop_effect_label"):
            return
        rect = self._crop_selector.crop_rect()
        base = "the full 1920×1200 frame" if self._no_crop_check.isChecked() else "this session's own crop"
        whole_region_roi = bool(self._roi_mode_combo.currentData())
        if rect is None:
            text = f"No rectangle: the t-map and ROI selection both use {base}."
            style = HINT_STYLE
        else:
            x, y, w, h = rect
            where = f"Rectangle x={x}, y={y}, {w}×{h} px"
            if whole_region_roi:
                text = f"{where} restricts the t-map, and the ROI is the whole rectangle."
                style = HINT_STYLE
            elif self._crop_scope_combo.currentData():
                text = f"{where} restricts both the t-map and ROI selection."
                style = HINT_STYLE
            else:
                text = (
                    f"{where} restricts the t-map only. The ROI is still chosen from "
                    f"{base} and can fall outside the rectangle."
                )
                style = WARNING_STYLE
        self._crop_effect_label.setStyleSheet(style)
        self._crop_effect_label.setText(text)

    def _build_log_override_tab(self) -> QWidget:
        w, f = self._make_form()

        log_row = QHBoxLayout()
        self._log_dir_edit = QLineEdit()
        self._log_dir_edit.setPlaceholderText(
            "Folder with session_log.csv/marker_log.csv, if not the session folder itself"
        )
        log_row.addWidget(self._log_dir_edit, stretch=1)
        btn_log = QPushButton("Browse…")
        btn_log.clicked.connect(self._browse_log_dir)
        log_row.addWidget(btn_log)
        f.addRow("Log dir:", log_row)

        self._log_time_min_edit = QLineEdit()
        self._log_time_min_edit.setPlaceholderText("ISO timestamp, inclusive lower bound (optional)")
        f.addRow("Log time min:", self._log_time_min_edit)

        self._log_time_max_edit = QLineEdit()
        self._log_time_max_edit.setPlaceholderText("ISO timestamp, exclusive upper bound (optional)")
        f.addRow("Log time max:", self._log_time_max_edit)

        hint = QLabel(
            "Use when a session's own log files are missing and its rows were appended "
            "into a different session's log instead — point Log dir at that session and "
            "use the time bounds to select this session's rows."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(HINT_STYLE)
        f.addRow("", hint)

        return w

    def _build_output_perf_tab(self) -> QWidget:
        w, f = self._make_form()

        out_row = QHBoxLayout()
        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setPlaceholderText("Override output folder (optional; ignores output suffix)")
        out_row.addWidget(self._out_dir_edit, stretch=1)
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._browse_out_dir)
        out_row.addWidget(btn_out)
        f.addRow("Output dir:", out_row)

        self._reuse_cache_check = QCheckBox("Reuse cached time-course")
        self._reuse_cache_check.setToolTip(
            "If roi_timecourse_raw.npz / loo_roi_amplitude_summary.csv already exist in "
            "the output folder, reuse them instead of re-streaming raw frames — only the "
            "t-map/permutation section is rebuilt. Useful for re-running with a different "
            "permutation count or cluster sign without repeating the slow part."
        )
        f.addRow("", self._reuse_cache_check)

        self._save_loo_masks_check = QCheckBox("Save per-fold LOO ROI masks")
        self._save_loo_masks_check.setToolTip(
            "Saves every fold's LOO ROI mask, a fold-selection-frequency heatmap, and an "
            "example-folds figure into out_dir/loo_masks/. Fast (no raw-frame streaming)."
        )
        f.addRow("", self._save_loo_masks_check)

        return w

    def _build_poster_figures_tab(self) -> QWidget:
        w, f = self._make_form()

        hint = QLabel(
            "session_poster_figures.py runs automatically after every successful "
            "Statistics run, into the same output folder: an annotated cortical "
            "panel (autoscaled + fixed ±scale below), a green-reference image "
            "with the analysis region marked, and a styled ROI time course with a "
            "self-contained out-of-activation trace — the same visual language as "
            "the poster figures. Not a separate step to opt into. The scale bar "
            "uses \"Spatial calibration for your rig\" above."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(HINT_STYLE)
        f.addRow(hint)

        compare_row = QHBoxLayout()
        self._compare_session_edit = QLineEdit()
        self._compare_session_edit.setPlaceholderText(
            "Optional: a second session (e.g. control/catch), measured in this session's ROI"
        )
        compare_row.addWidget(self._compare_session_edit, stretch=1)
        btn_compare = QPushButton("Browse…")
        btn_compare.clicked.connect(self._browse_compare_session)
        compare_row.addWidget(btn_compare)
        f.addRow("Compare session:", compare_row)

        self._compare_label_edit = QLineEdit("Comparison session")
        self._compare_label_edit.setToolTip(
            "Legend label for the compare-session trace, e.g. 'No stimulus' or 'Interleaved catch'."
        )
        f.addRow("Compare label:", self._compare_label_edit)

        orient_row = QHBoxLayout()
        orient_row.addWidget(QLabel("Anterior:"))
        self._anterior_combo = QComboBox()
        for side in ("left", "right", "top", "bottom"):
            self._anterior_combo.addItem(side.capitalize(), side)
        orient_row.addWidget(self._anterior_combo)
        orient_row.addSpacing(12)
        self._medial_label = QLabel("Medial:")
        orient_row.addWidget(self._medial_label)
        self._medial_combo = QComboBox()
        for side in ("left", "right", "top", "bottom"):
            self._medial_combo.addItem(side.capitalize(), side)
        self._medial_combo.setCurrentIndex(self._medial_combo.findData("bottom"))
        orient_row.addWidget(self._medial_combo)
        orient_row.addSpacing(12)

        # Bilateral image with the midline down the centre of the frame: both
        # edges perpendicular to Anterior are lateral, medial is the centre
        # line -- so the Medial edge choice no longer applies.
        self._midline_centered_check = QCheckBox("Midline-centered")
        self._midline_centered_check.setToolTip(
            "Both hemispheres in frame with the cortical midline running down the "
            "centre of the image. Both edges perpendicular to Anterior are then "
            "lateral and medial is marked at the compass centre; the Medial "
            "dropdown no longer applies (A/P still come from Anterior)."
        )
        self._midline_centered_check.toggled.connect(self._on_midline_centered_toggled)

        # Same green-reference preview the Region / Crop tab renders (see
        # gui/crop_selector.py's CropSelectorWidget) -- mirrored here, not
        # re-rendered, so picking anterior/medial has the vasculature image
        # to judge orientation against without switching tabs.
        self._orientation_preview = PanelImageView(
            placeholder_text="Green reference preview\n(set a session folder above)"
        )
        self._orientation_preview.setFixedSize(224, 140)  # 1920:1200 ratio
        orient_row.addStretch()

        # The preview sits UNDER the dropdowns rather than beside them. Side by
        # side, this row's minimum width was the controls plus the image (~970 px),
        # which is what forced a horizontal scrollbar under the parameters column;
        # stacked, the minimum is just the wider of the two. Both are still visible
        # at once, which is the point of mirroring the preview here.
        #
        # Midline-centered rides with the preview rather than the dropdowns: it
        # describes how the image is framed, and keeping it out of the dropdown
        # row is what brings that row's minimum width inside half the window.
        preview_row = QHBoxLayout()
        preview_row.addWidget(self._orientation_preview)
        preview_row.addSpacing(12)
        preview_row.addWidget(self._midline_centered_check)
        preview_row.addStretch()

        orient_col = QVBoxLayout()
        orient_col.setContentsMargins(0, 0, 0, 0)
        orient_col.setSpacing(6)
        orient_col.addLayout(orient_row)
        orient_col.addLayout(preview_row)

        orient_widget = QWidget()
        orient_widget.setLayout(orient_col)
        orient_widget.setToolTip(
            "Compass orientation for the panel figures. Default (anterior=left, "
            "medial=bottom) is a carried-forward default assumption, not per-session "
            "metadata — update if this rig's mounting differs. The preview image "
            "is the same green reference rendered in the Region / Crop tab."
        )
        f.addRow("Orientation:", orient_widget)

        self._fixed_vmax_spin = QDoubleSpinBox()
        self._fixed_vmax_spin.setRange(0.001, 1.0)
        self._fixed_vmax_spin.setDecimals(3)
        self._fixed_vmax_spin.setValue(_DEFAULT_FIXED_VMAX)
        self._fixed_vmax_spin.setSuffix(" %")
        self._fixed_vmax_spin.setToolTip(
            "Shared color scale for the _v05 panel variants, for comparing sessions "
            "side by side rather than each autoscaled to its own signal."
        )
        f.addRow("Fixed scale:", self._fixed_vmax_spin)

        self._reuse_extraction_cache_check = QCheckBox(
            "Reuse cached out-region / compare-session extraction"
        )
        self._reuse_extraction_cache_check.setChecked(True)
        self._reuse_extraction_cache_check.setToolTip(
            "If <label>_extraction_cache.npz already exists in the output folder, "
            "reuse it instead of re-streaming raw frames for the out-region (and "
            "compare-session, if used) time course. Useful when re-running just to "
            "tweak the scale bar/orientation/fixed scale."
        )
        f.addRow("", self._reuse_extraction_cache_check)

        self._suppress_title_check = QCheckBox("Suppress auto-generated header")
        self._suppress_title_check.setToolTip(
            "Leave <label>_timecourse.png's title blank instead of auto-filling "
            "'<label>: leave-one-out ROI dR/R(t)' -- useful if you're adding your "
            "own caption elsewhere (a poster, a figure panel)."
        )
        f.addRow("", self._suppress_title_check)

        self._show_amplitude_labels_check = QCheckBox("Show numerical dR/R value on each mean line")
        self._show_amplitude_labels_check.setToolTip(
            "Print each series' amplitude-window mean (e.g. '-0.068%') next to its "
            "horizontal mean line on <label>_timecourse.png, colored to match and "
            "nudged apart automatically if two series' means land close together."
        )
        f.addRow("", self._show_amplitude_labels_check)

        return w

    # ── Job queue panel ──────────────────────────────────────────────────────

    def _build_queue_panel(self) -> QGroupBox:
        box = QGroupBox("Analysis jobs")
        v = QVBoxLayout(box)

        hint = QLabel(
            "Each \"Run analysis\" starts right away, alongside anything already "
            "running. The one exception: a job that would write to the same output "
            "folder as a running job waits for it to finish."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(HINT_STYLE)
        v.addWidget(hint)

        self._job_table = QTableWidget(0, 5)
        self._job_table.setHorizontalHeaderLabels(["", "Session", "Phase / Progress", "Result", ""])
        self._job_table.verticalHeader().setVisible(False)
        self._job_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._job_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._job_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._job_table.setAlternatingRowColors(True)
        self._job_table.setMinimumHeight(150)
        header = self._job_table.horizontalHeader()
        header.setSectionResizeMode(_COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_COL_SESSION, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_PHASE, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_RESULT, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_ACTION, QHeaderView.ResizeMode.ResizeToContents)
        self._job_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        v.addWidget(self._job_table)

        return box

    def _refresh_table(self) -> None:
        # Status label + action button are rebuilt from scratch every call
        # rather than cached per-job across calls -- setRowCount() shrinking
        # (e.g. after a Remove) deletes whatever cell widgets occupy the
        # rows being trimmed off the end, but jobs that SURVIVE a removal
        # still shift up to new row indices. A per-job widget cache handed
        # those shifted rows the SAME (now possibly already-deleted, for
        # whichever job used to sit in a trimmed row) widget instances,
        # corrupting the table -- confirmed by reproduction: removing one
        # job left an unrelated surviving row with no button at all. Rows
        # are few (a handful of queued/running jobs), so rebuilding is cheap.
        jobs = self._queue.jobs
        self._job_table.setRowCount(len(jobs))

        for row, job in enumerate(jobs):
            status_label = QLabel()
            status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_label.setText(_STATUS_ICON[job.status])
            status_label.setToolTip(job.status.name.replace("_", " ").title())
            self._job_table.setCellWidget(row, _COL_STATUS, status_label)

            self._set_table_text(row, _COL_SESSION, job.label, tooltip=job.session_dir)
            self._set_table_text(row, _COL_PHASE, job.phase_text)
            self._set_table_text(row, _COL_RESULT, job.result_summary())

            btn = QPushButton()
            btn.clicked.connect(lambda _checked=False, j=job: self._on_row_action(j))
            if job.status is JobStatus.QUEUED:
                btn.setText("Cancel")
                btn.setEnabled(True)
            elif job.status is JobStatus.RUNNING:
                btn.setText("Stop")
                btn.setEnabled(True)
            elif job.status is JobStatus.STOPPING:
                btn.setText("Stopping…")
                btn.setEnabled(False)
            else:
                btn.setText("Remove")
                btn.setEnabled(True)
            self._job_table.setCellWidget(row, _COL_ACTION, btn)

        if self._selected_job is not None and self._selected_job not in jobs:
            self._select_job(None)
        elif self._selected_job is not None:
            row = jobs.index(self._selected_job)
            if self._job_table.currentRow() != row:
                self._job_table.selectRow(row)
            self._refresh_detail_panel()

    def _set_table_text(self, row: int, col: int, text: str, tooltip: str | None = None) -> None:
        item = self._job_table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._job_table.setItem(row, col, item)
        item.setText(text)
        item.setToolTip(tooltip if tooltip is not None else text)

    def _on_row_action(self, job: AnalysisJob) -> None:
        # Deferred to the next event-loop tick: this handler runs
        # synchronously inside `job`'s own row button's `clicked` signal.
        # Any of the three actions below ends up rebuilding the job table's
        # cell widgets (see _refresh_table -- rows are rebuilt from scratch
        # on every change, including a bare status tick), which can delete
        # the very button whose click is still on the call stack -- Qt
        # doesn't tolerate that. QTimer.singleShot(0, ...) lets the click
        # fully unwind first.
        QTimer.singleShot(0, lambda j=job: self._perform_row_action(j))

    def _perform_row_action(self, job: AnalysisJob) -> None:
        if job.status is JobStatus.QUEUED:
            self._queue.cancel_queued(job)
        elif job.status is JobStatus.RUNNING:
            job.request_stop()
        elif job.status in TERMINAL_STATUSES:
            was_selected = job is self._selected_job
            if self._queue.remove_job(job) and was_selected:
                self._select_job(None)

    def _on_table_selection_changed(self) -> None:
        rows = self._job_table.selectionModel().selectedRows()
        jobs = self._queue.jobs
        row = rows[0].row() if rows else -1
        job = jobs[row] if 0 <= row < len(jobs) else None
        if job is self._selected_job:
            return
        self._select_job(job)

    def _disconnect_selected_log(self) -> None:
        if self._selected_job is not None:
            try:
                self._selected_job.log_line.disconnect(self._on_selected_job_log_line)
            except (RuntimeError, TypeError):
                pass

    def _select_job(self, job: AnalysisJob | None) -> None:
        self._disconnect_selected_log()
        self._selected_job = job
        self._log.clear()

        if job is None:
            self._job_table.clearSelection()
            self._clear_detail_panel()
            return

        job.log_line.connect(self._on_selected_job_log_line)
        self._log.setPlainText("\n".join(job.log_lines))

        jobs = self._queue.jobs
        if job in jobs:
            row = jobs.index(job)
            if self._job_table.currentRow() != row:
                self._job_table.selectRow(row)
        self._refresh_detail_panel()

    def _on_selected_job_log_line(self, line: str) -> None:
        self._log.appendPlainText(line)

    def _on_job_stop_timeout(self, job: AnalysisJob) -> None:
        msg = QMessageBox(self)
        msg.setWindowTitle("Stop timed out")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            f"The run for \"{job.label}\" did not exit within 30 seconds after the "
            "stop signal.\n\nForce-killing may leave partial output in the output folder."
        )
        force_btn = msg.addButton("Force Kill", QMessageBox.ButtonRole.DestructiveRole)
        wait_btn = msg.addButton("Keep Waiting", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(wait_btn)
        msg.exec()

        if msg.clickedButton() is force_btn:
            job.force_kill()
        else:
            job.keep_waiting()

    # ── Progress / result / log (detail panel for the selected job) ─────────────

    def _build_progress_area(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        self._phase_label = QLabel("Idle.")
        v.addWidget(self._phase_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("")
        v.addWidget(self._progress)

        return w

    def _build_result_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._result_label = QLabel()
        self._result_label.setWordWrap(True)
        self._result_label.hide()
        row.addWidget(self._result_label)
        row.addStretch()
        return row

    def _build_log(self) -> QPlainTextEdit:
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Select a job above to see its output here…")
        return self._log

    def _build_log_panel(self) -> QGroupBox:
        box = QGroupBox("Statistics output")
        vl = QVBoxLayout(box)
        vl.addWidget(self._build_log())
        return box

    def _build_post_run_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._post_run_label = QLabel()
        self._post_run_label.hide()
        row.addWidget(self._post_run_label)

        self._open_folder_btn = QPushButton("Open folder")
        self._open_folder_btn.hide()
        self._open_folder_btn.clicked.connect(self._open_output_folder)
        row.addWidget(self._open_folder_btn)

        row.addStretch()
        return row

    def _refresh_detail_panel(self) -> None:
        job = self._selected_job
        if job is None:
            return

        self._phase_label.setText(job.phase_text)
        lo, hi = job.progress_range
        self._progress.setRange(lo, hi)
        self._progress.setValue(job.progress_value)
        self._progress.setFormat(job.progress_format)

        result = job.result_summary()
        if result:
            self._result_label.setText(result)
            self._result_label.show()
        else:
            self._result_label.hide()

        if job.post_run_text:
            self._post_run_label.setText(job.post_run_text)
            self._post_run_label.show()
        else:
            self._post_run_label.hide()
        self._open_folder_btn.setVisible(job.output_dir is not None)

        self._stop_btn.setEnabled(job.status is JobStatus.RUNNING)

    def _clear_detail_panel(self) -> None:
        self._phase_label.setText("No job selected — run an analysis to see progress here.")
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("")
        self._result_label.hide()
        self._post_run_label.hide()
        self._open_folder_btn.hide()
        self._stop_btn.setEnabled(False)

    # ── Button row ────────────────────────────────────────────────────────────

    def _build_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._show_cmd_btn = QPushButton("Show command…")
        self._show_cmd_btn.clicked.connect(self._show_command)
        row.addWidget(self._show_cmd_btn)

        row.addStretch()

        self._start_btn = QPushButton("▶  Run analysis")
        self._start_btn.setToolTip(
            "Starts statistics (then figures) for this form right away, as a new "
            "entry in Analysis jobs."
        )
        self._start_btn.setStyleSheet(STYLE_START)
        self._start_btn.clicked.connect(self._on_add_to_queue)
        row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop selected job")
        self._stop_btn.setToolTip("Gracefully stop the job selected in Analysis jobs.")
        self._stop_btn.setStyleSheet(STYLE_STOP)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_selected)
        row.addWidget(self._stop_btn)

        return row

    # ── Browse helpers ────────────────────────────────────────────────────────

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select session folder")
        if path:
            self.set_session_folder(path)

    def set_session_folder(self, path: str) -> None:
        self._folder_edit.setText(path)
        self._rescan_trials()

    def _browse_array_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select array source folder")
        if path:
            self._array_source_edit.setText(path)

    def _browse_log_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select log folder")
        if path:
            self._log_dir_edit.setText(path)

    def _browse_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._out_dir_edit.setText(path)

    def _browse_compare_session(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select compare session folder")
        if path:
            self._compare_session_edit.setText(path)

    # ── Trial selection ──────────────────────────────────────────────────────

    @staticmethod
    def _scan_trial_ids(session_dir: str) -> list[int]:
        p = Path(session_dir)
        if not p.is_dir():
            return []
        ids = []
        for d in sorted(p.glob("trial_[0-9]*")):
            try:
                ids.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(ids)

    @staticmethod
    def _scan_trial_conditions(session_dir: str) -> set[str]:
        """Distinct trial_condition values recorded across this session's
        trial_metadata.json files -- set by --trial-conditions at
        acquisition time (see statistical_analyses.py's own --condition
        docs). A session that never used interleaved trials, or predates
        that support, scans to an empty set. Scoped to every trial on disk
        (not just currently-checked ones in the Trials tab) -- whether a
        session's *structure* is interleaved doesn't depend on which
        trials happen to be excluded from a given run."""
        p = Path(session_dir)
        if not p.is_dir():
            return set()
        conditions: set[str] = set()
        for meta_path in p.glob("trial_[0-9]*/meta/trial_metadata.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, ValueError):
                continue
            cond = meta.get("trial_condition")
            if cond:
                conditions.add(cond)
        return conditions

    def _update_interleaved_hint(self) -> None:
        if {"stim", "catch"} <= self._detected_conditions:
            self._interleaved_hint.setText(
                "Interleaved session detected (stim + catch trials present). "
                "\"All trials\" would pool them into one ROI/t-map, which "
                "statistical_analyses.py's own docs warn can let one condition's "
                "response contaminate the other's ROI selection -- Run analysis "
                "will start separate Stim and Catch jobs instead. The Stim job's "
                "figures will wait for and compare against the Catch job's "
                "statistics, same as a manual Compare session, unless one is "
                "set below."
            )
            self._interleaved_hint.show()
        else:
            self._interleaved_hint.hide()

    def _rescan_trials(self) -> None:
        session_dir = self._folder_edit.text().strip()
        self._crop_selector.set_session_dir(session_dir)
        self._detected_conditions = self._scan_trial_conditions(session_dir) if session_dir else set()
        self._update_interleaved_hint()
        trial_ids = self._scan_trial_ids(session_dir) if session_dir else []

        # Only carry unchecked trials forward across a rescan of the SAME
        # session folder (e.g. Refresh after more trials land mid-session).
        # Switching to a different session must start from all-checked --
        # trial numbers restart at trial_001 in every session, so without this
        # scoping a bubble-trial exclusion from one session would silently
        # follow the same trial number into an unrelated session.
        norm_dir = str(Path(session_dir)).casefold() if session_dir else ""
        same_session = bool(norm_dir) and norm_dir == self._trials_scanned_session_dir
        previously_unchecked = (
            {t for t, cb in self._trial_checks.items() if not cb.isChecked()}
            if same_session else set()
        )
        self._trials_scanned_session_dir = norm_dir

        while self._trials_grid.count():
            item = self._trials_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._trial_checks = {}

        if not trial_ids:
            self._trials_empty_label.setText(
                "No trial_XXX folders found in this session folder."
                if session_dir else
                "Select a session folder above, then click Refresh."
            )
            self._trials_empty_label.show()
            self._trials_grid_widget.hide()
        else:
            self._trials_empty_label.hide()
            self._trials_grid_widget.show()
            cols = 8
            for i, t in enumerate(trial_ids):
                cb = QCheckBox(f"{t:03d}")
                # Preserve exclusions across a rescan for trials still present
                # (e.g. re-scanning mid-session after more trials have landed).
                cb.setChecked(t not in previously_unchecked)
                cb.toggled.connect(self._update_trial_count_label)
                self._trial_checks[t] = cb
                self._trials_grid.addWidget(cb, i // cols, i % cols)

        self._update_trial_count_label()

    def _check_all_trials(self) -> None:
        for cb in self._trial_checks.values():
            cb.setChecked(True)

    def _uncheck_all_trials(self) -> None:
        for cb in self._trial_checks.values():
            cb.setChecked(False)

    def _update_trial_count_label(self) -> None:
        if not self._trial_checks:
            self._trials_count_label.setText("")
            return
        total = len(self._trial_checks)
        included = sum(1 for cb in self._trial_checks.values() if cb.isChecked())
        if included == total:
            self._trials_count_label.setText(f"All {total} trial(s) selected for analysis.")
        else:
            self._trials_count_label.setText(f"{included} of {total} trial(s) selected for analysis.")

    def _excluded_trial_ids(self) -> list[int]:
        return sorted(t for t, cb in self._trial_checks.items() if not cb.isChecked())

    # ── Form snapshots ────────────────────────────────────────────────────────

    def _snapshot_stats_form(self) -> StatsFormSnapshot:
        return StatsFormSnapshot(
            session_dir=self._folder_edit.text().strip(),
            full_frame=bool(self._roi_mode_combo.currentData()),
            neg_only=self._neg_only_check.isChecked(),
            condition=self._condition_combo.currentData(),
            excluded_trial_ids=tuple(self._excluded_trial_ids()),
            n_perm=self._nperm_spin.value(),
            crop_rect=self._crop_selector.crop_rect(),
            no_crop=self._no_crop_check.isChecked(),
            roi_within_region=bool(self._crop_scope_combo.currentData()),
            array_source_dir=self._array_source_edit.text().strip(),
            log_dir=self._log_dir_edit.text().strip(),
            log_time_min=self._log_time_min_edit.text().strip(),
            log_time_max=self._log_time_max_edit.text().strip(),
            out_dir_override=self._out_dir_edit.text().strip(),
            out_subfolder=self._out_subfolder_edit.text().strip(),
            reuse_timecourse_cache=self._reuse_cache_check.isChecked(),
            save_loo_masks=self._save_loo_masks_check.isChecked(),
        )

    def _snapshot_poster_form(self) -> PosterFormSnapshot:
        return PosterFormSnapshot(
            compare_session=self._compare_session_edit.text().strip(),
            compare_label=self._compare_label_edit.text().strip(),
            um_per_px=self._calibration_um_per_px(),
            anterior_side=self._anterior_combo.currentData(),
            medial_side=self._medial_combo.currentData(),
            midline_centered=self._midline_centered_check.isChecked(),
            fixed_vmax=self._fixed_vmax_spin.value(),
            reuse_extraction_cache=self._reuse_extraction_cache_check.isChecked(),
            suppress_title=self._suppress_title_check.isChecked(),
            show_amplitude_labels=self._show_amplitude_labels_check.isChecked(),
        )

    def _stats_snapshots_to_queue(self) -> list[StatsFormSnapshot]:
        """Normally just the one snapshot the form currently describes. But
        if this session is interleaved (both stim and catch trial_condition
        values detected) and "All trials" is still selected, pooling them
        would contaminate each condition's own ROI selection (see
        statistical_analyses.py's --condition docs) -- split into one stim
        + one catch snapshot instead, everything else about the form
        (n_perm, crop, exclusions, ...) carried over unchanged."""
        base = self._snapshot_stats_form()
        if base.condition == "all" and {"stim", "catch"} <= self._detected_conditions:
            return [
                dataclasses.replace(base, condition="stim"),
                dataclasses.replace(base, condition="catch"),
            ]
        return [base]

    @staticmethod
    def _default_stats_snapshot(session_dir: str) -> StatsFormSnapshot:
        """Plain "just analyze this session" parameters -- all trials, the
        pipeline's own defaults -- used for an auto-generated compare-session
        stats prerequisite, where nobody has opened that session's own form
        to choose anything."""
        return StatsFormSnapshot(
            session_dir=session_dir,
            full_frame=False,
            neg_only=True,
            condition="all",
            excluded_trial_ids=(),
            n_perm=_DEFAULT_N_PERM,
            crop_rect=None,
            no_crop=False,
            roi_within_region=False,
            array_source_dir="",
            log_dir="",
            log_time_min="",
            log_time_max="",
            out_dir_override="",
            out_subfolder=analysis_argv.DEFAULT_OUT_SUBFOLDER,
            reuse_timecourse_cache=False,
            save_loo_masks=False,
        )

    def _ensure_compare_session_stats(self, compare_session: str) -> AnalysisJob | None:
        """session_poster_figures.py's --compare-session needs
        <compare_session>/session_stats/analysis_summary.json to already
        exist, or it errors with "Run Statistics on the compare session
        first." Rather than surface that error, queue a stats-only
        prerequisite job for the compare session (default parameters) when
        that file is missing, and return it so the caller's job can depend
        on it. Returns None if no prerequisite is needed."""
        compare_session = compare_session.strip()
        if not compare_session:
            return None

        summary_path = Path(compare_session) / analysis_argv.DEFAULT_OUT_SUBFOLDER / "analysis_summary.json"
        if summary_path.exists():
            return None

        compare_snapshot = self._default_stats_snapshot(compare_session)
        target_key = None
        resolved = analysis_argv.resolve_out_dir(compare_snapshot)
        if resolved is not None:
            target_key = str(resolved).casefold()

        # Already queued/running for this exact session+out-dir (e.g. the
        # user queued the compare session directly, or queued two jobs that
        # both compare against it)? Depend on that instead of launching a
        # redundant duplicate -- the collision guard would only serialize
        # them, not merge them.
        for existing in self._queue.jobs:
            if target_key is not None and existing.out_dir_key == target_key \
                    and existing.status not in TERMINAL_STATUSES:
                return existing

        prereq = AnalysisJob(compare_snapshot, self._snapshot_poster_form(), chain_poster=False)
        prereq.label = f"{prereq.label}  (compare-session stats)"
        prereq.stop_timed_out.connect(lambda j=prereq: self._on_job_stop_timeout(j))
        return prereq

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> str | None:
        folder = self._folder_edit.text().strip()
        if not folder:
            return "Session folder is required."
        if not Path(folder).exists():
            return f"Session folder not found:\n{folder}"
        if self._trial_checks and not any(cb.isChecked() for cb in self._trial_checks.values()):
            return "No trials selected — check at least one trial in the Trials tab before running."
        compare_session = self._compare_session_edit.text().strip()
        if compare_session and not Path(compare_session).exists():
            return f"Compare session folder not found:\n{compare_session}"
        return None

    # ── Show command ──────────────────────────────────────────────────────────

    def _show_command(self) -> None:
        # Always previews what "Run analysis" would start right now, not
        # whatever job happens to be selected in the table below -- this
        # button sits next to Run analysis, not next to the job list. Shows
        # two commands when the interleaved-session split (see
        # _stats_snapshots_to_queue()) would apply.
        launcher = ScriptRunner().launcher
        commands = []
        for snap in self._stats_snapshots_to_queue():
            argv = launcher + [analysis_argv.STATS_SCRIPT] + analysis_argv.build_stats_argv(snap)
            commands.append(" ".join(argv))
        text = "\n\n".join(commands)
        dlg = QDialog(self)
        dlg.setWindowTitle("Full command")
        dlg.resize(680, 140 if len(commands) == 1 else 240)
        lay = QVBoxLayout(dlg)
        edit = QPlainTextEdit(text)
        edit.setReadOnly(True)
        edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        lay.addWidget(edit)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        dlg.exec()

    # ── Queue / stop ──────────────────────────────────────────────────────────

    def _on_add_to_queue(self) -> None:
        err = self._validate()
        if err:
            QMessageBox.warning(self, "Cannot run analysis", err)
            return
        if not self._confirm_unverified_calibration():
            return

        poster_snapshot = self._snapshot_poster_form()

        # Normally one job. For an interleaved session still set to "All
        # trials", this is two (stim + catch) -- see
        # _stats_snapshots_to_queue()'s docstring for why pooling them
        # would be wrong. Both share whatever compare-session prerequisite
        # is needed, computed once below.
        jobs = [AnalysisJob(snap, poster_snapshot) for snap in self._stats_snapshots_to_queue()]
        for job in jobs:
            job.stop_timed_out.connect(lambda j=job: self._on_job_stop_timeout(j))

        # If a compare session is set and hasn't been analyzed yet, queue a
        # stats-only prerequisite job for it first, so the figures phase's
        # --compare-session lookup doesn't fail with "Run Statistics on the
        # compare session first." -- the whole point being this shouldn't
        # be something the user has to do by hand in a separate step.
        prereq = self._ensure_compare_session_stats(poster_snapshot.compare_session)
        prereq_is_new = prereq is not None and prereq not in self._queue.jobs
        if prereq is not None:
            for job in jobs:
                job.depends_on = prereq
        else:
            # Interleaved auto-split (see _stats_snapshots_to_queue()) and no
            # manual Compare session set: treat the stim job's figures the
            # same as any other paired-session comparison -- it depends on
            # the catch job (its own statistics prerequisite, exactly like
            # _ensure_compare_session_stats' external case above) and
            # compares against the catch job's stats once that dependency
            # clears. One-directional only: the catch job runs plain.
            stim_job = next((j for j in jobs if j.condition == "stim"), None)
            catch_job = next((j for j in jobs if j.condition == "catch"), None)
            if stim_job is not None and catch_job is not None:
                stim_job.depends_on = catch_job
                stim_job.compare_job = catch_job

        # Select the first new job before it's even in the queue's list --
        # once enqueue() below triggers a table refresh, the refresh's tail
        # finds it already selected and picks the right row automatically.
        self._disconnect_selected_log()
        self._selected_job = jobs[0]
        self._log.clear()
        jobs[0].log_line.connect(self._on_selected_job_log_line)

        if prereq_is_new:
            self._queue.enqueue(prereq)
        for job in jobs:
            self._queue.enqueue(job)

    def _on_stop_selected(self) -> None:
        if self._selected_job is not None:
            self._selected_job.request_stop()

    # ── MainWindow close-guard passthroughs ──────────────────────────────────

    def has_active_jobs(self) -> bool:
        return self._queue.has_active_jobs()

    def active_job_count(self) -> int:
        return self._queue.active_job_count()

    def stop_all_jobs(self) -> None:
        self._queue.stop_all()

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def set_zoom(self, zoom: float) -> None:
        """Rescale the orientation preview -- the app's only truly fixed-size
        widget (setFixedSize, not just a layout minimum), so it doesn't stay
        pinned at 224x140 while the rest of the UI grows/shrinks around it."""
        self._orientation_preview.setFixedSize(round(224 * zoom), round(140 * zoom))

    # ── Post-run folder ───────────────────────────────────────────────────────

    def _open_output_folder(self) -> None:
        if self._selected_job is not None and self._selected_job.output_dir:
            try:
                subprocess.Popen(["explorer", self._selected_job.output_dir])
            except OSError:
                pass
