# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
AnalysisJob — one queued/running statistical_analyses.py (+ chained
session_poster_figures.py) run, as its own object.

Ported off what used to be StatisticsWidget's single set of instance fields
(one _phase/_output_dir/_loo_progress/... per widget, one ScriptRunner) so
gui/analysis_queue.py can run several of these at once. Each job owns its
own ScriptRunner/RunnerBridge pair -- proven safe to do N times over by
gui/diagnostics.py, which already runs two independent pairs (the check
sequence and the live camera feed) inside one widget.

No QMessageBox here: this class only emits signals. The 30s-graceful-stop
Force Kill / Keep Waiting dialog stays owned by StatisticsWidget, same as
ScriptRunner/RunnerBridge never touching UI today.
"""
from __future__ import annotations

import re
from enum import Enum, auto
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from gui import analysis_argv
from gui.analysis_argv import PosterFormSnapshot, StatsFormSnapshot
from gui.runner_bridge import RunnerBridge
from gui.script_runner import ScriptRunner

_GRACEFUL_STOP_TIMEOUT_MS = 30_000

# ── Log line patterns (see statistical_analyses.py's own print()s) ──────────
_RE_SESSION       = re.compile(r"^Session:\s*(.+?)\s+trial folders on disk:\s*(\d+)")
_RE_USABLE        = re.compile(r"^Usable trials \(frames \+ STIM_START matched\):\s*(\d+)")
_RE_TOO_FEW       = re.compile(r"^Too few usable trials, aborting\.$")
_RE_REUSE         = re.compile(r"^Reusing cached LOO time-course results")
_RE_STREAMING     = re.compile(r"^Running leave-one-out ROI \+ time course extraction")
_RE_LOO_TRIAL     = re.compile(r"trial\s+(\d+):\s+ROI=")
_RE_PERM_START    = re.compile(r"^Running (\d+) sign-flip permutations")
_RE_PEAK_SUMMARY  = re.compile(r"^LOO peak dR/R%: mean=([+\-\d.]+)\s+SEM=([+\-\d.]+)")
_RE_CLUSTER_RESULT = re.compile(
    r"^Observed max cluster.*?:\s*(\d+)px,\s*null mean=([\d.]+),\s*"
    r"p=([\d.]+),\s*peak\|t\|=([\d.]+)\s*\(df=(\d+)\)"
)
_RE_DONE          = re.compile(r"^Done -> (.+)$")

# ── Log line patterns for the chained session_poster_figures.py phase ───────
_RE_POSTER_MASKS      = re.compile(r"^Building masks")
_RE_POSTER_GREEN_REF  = re.compile(r"^Rendering (green reference|targeting reference)")
_RE_POSTER_PANELS     = re.compile(r"^Rendering cortical panels")
_RE_POSTER_EXTRACTING = re.compile(r"^Extracting (out-region|compare-session ROI) timecourse")
_RE_POSTER_TIMECOURSE = re.compile(r"^Rendering timecourse")

# Compare-trace label for AnalysisJob.compare_job, keyed by the partner
# job's own condition -- see _launch_poster_figures().
_CONDITION_COMPARE_LABELS = {"catch": "Catch (no-stim)", "stim": "Stim"}


class JobStatus(Enum):
    QUEUED = auto()
    RUNNING = auto()
    STOPPING = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    STOPPED = auto()


TERMINAL_STATUSES = (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED)


class AnalysisJob(QObject):
    """One statistics run (+ auto-chained figures). Snapshots are frozen at
    construction time -- built from the form when the user hits "Add to
    Queue", not re-read later, so a job started well after being queued
    runs with the parameters it was queued with."""

    status_changed = Signal()   # phase/progress/result changed -> refresh table row + detail panel if selected
    log_line = Signal(str)      # one stdout line -> live-append to the detail panel's log if this job is selected
    finished = Signal()         # terminal status reached -> AnalysisQueue can start the next queued job
    stop_timed_out = Signal()   # 30s after a graceful stop request, the subprocess still hasn't exited

    def __init__(
        self,
        stats_snapshot: StatsFormSnapshot,
        poster_snapshot: PosterFormSnapshot,
        parent: QObject | None = None,
        *,
        chain_poster: bool = True,
    ) -> None:
        super().__init__(parent)
        self._stats_snapshot = stats_snapshot
        self._poster_snapshot = poster_snapshot
        # False for an auto-generated "compute stats for the compare session
        # first" prerequisite job (gui/statistics_tab.py) -- it only needs to
        # produce analysis_summary.json, not figures.
        self.chain_poster = chain_poster

        self.session_dir = stats_snapshot.session_dir
        self.condition = stats_snapshot.condition
        self.label = Path(stats_snapshot.session_dir).name or stats_snapshot.session_dir
        if stats_snapshot.condition != "all":
            self.label += f"  [{stats_snapshot.condition}]"

        # Resolved once, up front -- the collision guard needs this BEFORE
        # the subprocess ever runs, not after it prints "Done -> ...".
        resolved = analysis_argv.resolve_out_dir(stats_snapshot)
        self.out_dir_key: str | None = str(resolved).casefold() if resolved is not None else None

        # Set by the caller (gui/statistics_tab.py) when this job needs
        # another job's compare-session stats to exist before it can start
        # -- see AnalysisQueue._pump()'s dependency handling and
        # mark_dependency_failed() below.
        self.depends_on: "AnalysisJob | None" = None

        # Set alongside depends_on (above) for an interleaved session's stim
        # job: its figures compare against this catch job's own statistics
        # instead of the form's (blank, in that case) Compare session field.
        # A SEPARATE attribute from depends_on because AnalysisQueue._pump()
        # clears depends_on to None once satisfied -- this one needs to
        # survive until _launch_poster_figures() actually runs.
        self.compare_job: "AnalysisJob | None" = None

        self.status: JobStatus = JobStatus.QUEUED
        self.phase_text: str = "Queued."
        self.post_run_text: str = ""
        self.progress_range: tuple[int, int] = (0, 1)
        self.progress_value: int = 0
        self.progress_format: str = ""
        self.output_dir: str | None = None
        self.log_lines: list[str] = []

        self._phase: str = "stats"  # "stats" or "poster" -- which subprocess is (about to be) running
        self._usable_total: int | None = None
        self._loo_progress = 0
        self._result_mean: str | None = None
        self._result_sem: str | None = None
        self._result_cluster_px: str | None = None
        self._result_cluster_p: str | None = None
        self._result_peak_t: str | None = None
        self._stop_requested = False
        self._force_killed = False

        self._runner = ScriptRunner()
        self._bridge = RunnerBridge(self._runner)
        self._bridge.line_received.connect(self._on_line)
        self._bridge.run_finished.connect(self._on_done)

        self._stop_timer = QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.setInterval(_GRACEFUL_STOP_TIMEOUT_MS)
        self._stop_timer.timeout.connect(self._on_stop_timer_elapsed)

    # ── Public API ────────────────────────────────────────────────────────────

    def command_preview(self) -> list[str]:
        snap = self._stats_snapshot
        return self._runner.launcher + [analysis_argv.STATS_SCRIPT] + analysis_argv.build_stats_argv(snap)

    def result_summary(self) -> str:
        if self._result_mean is None:
            return ""
        parts = [f"dR/R = {self._result_mean}% ± {self._result_sem}%"]
        if self._result_cluster_px is not None:
            parts.append(
                f"cluster {self._result_cluster_px}px  ·  "
                f"cluster p = {self._result_cluster_p}  ·  "
                f"peak|t| = {self._result_peak_t}"
            )
        return "   ".join(parts)

    def start(self) -> None:
        """Called by AnalysisQueue when a concurrency slot is free. Not for
        UI code to call directly -- go through AnalysisQueue.enqueue()."""
        self.status = JobStatus.RUNNING
        self._phase = "stats"
        self.phase_text = "Phase: starting…"
        self.progress_range = (0, 0)
        self.progress_value = 0
        self.progress_format = ""
        self.output_dir = None
        self.post_run_text = ""
        self._usable_total = None
        self._loo_progress = 0
        self._result_mean = self._result_sem = None
        self._result_cluster_px = self._result_cluster_p = self._result_peak_t = None
        self._stop_requested = False
        self._force_killed = False

        try:
            self._runner.start(analysis_argv.STATS_SCRIPT, analysis_argv.build_stats_argv(self._stats_snapshot))
        except (RuntimeError, OSError, FileNotFoundError) as exc:
            self.phase_text = f"Launch failed: {exc}"
            self._finish(JobStatus.FAILED)
            return
        self.status_changed.emit()

    def request_stop(self) -> None:
        if self.status is not JobStatus.RUNNING:
            return
        self._stop_requested = True
        self.status = JobStatus.STOPPING
        phase_name = "figures" if self._phase == "poster" else "statistics"
        self.phase_text = f"Stopping {phase_name}…"
        self._runner.send_stop_signal()
        self._stop_timer.start()
        self.status_changed.emit()

    def keep_waiting(self) -> None:
        """"Keep Waiting" branch of the stop-timeout dialog: recheck again
        in another 30s rather than going silent (the original single-job
        dialog only ever checked once)."""
        self.phase_text = "Still running — waiting for process to exit…"
        self._stop_timer.start()
        self.status_changed.emit()

    def force_kill(self) -> None:
        """Last resort. Deliberately does NOT mark the job terminal here --
        the real exit confirmation arrives via _on_done() once the OS has
        actually torn the process down. Freeing this job's concurrency slot
        (and its out-dir collision guard) before that would let a new job
        start writing into the same folder while the killed process might
        still be alive."""
        self._force_killed = True
        self.phase_text = "Force-killing…"
        self._runner.force_kill()
        self.status_changed.emit()

    def mark_dependency_failed(self, message: str) -> None:
        """Called by AnalysisQueue when a job this one depends on (e.g. a
        compare-session stats prerequisite) ended without succeeding -- this
        job can never start, so finish it as FAILED without ever launching
        a subprocess."""
        self.depends_on = None
        self._finish(JobStatus.FAILED, message)

    # ── Log line parsing ──────────────────────────────────────────────────────

    def _on_line(self, line: str) -> None:
        if not line.strip():
            return
        self.log_lines.append(line)
        self.log_line.emit(line)

        if self._phase == "poster":
            self._on_poster_line(line)
            self.status_changed.emit()
            return

        m = _RE_SESSION.search(line)
        if m:
            self.phase_text = f"Phase: loading session ({m.group(2)} trial folders on disk)…"

        m = _RE_USABLE.search(line)
        if m:
            self._usable_total = int(m.group(1))
            self.phase_text = f"Phase: {self._usable_total} usable trials matched…"

        if _RE_TOO_FEW.search(line):
            self.phase_text = "Too few usable trials (need 3+) — aborting."

        if _RE_REUSE.search(line):
            self.phase_text = "Phase: reusing cached time-course (fast path)…"
            self.progress_range = (0, 0)
            self.progress_format = ""

        if _RE_STREAMING.search(line):
            self._loo_progress = 0
            self.phase_text = "Phase: extracting per-trial time courses (streaming raw frames)…"
            if self._usable_total:
                self.progress_range = (0, self._usable_total)
                self.progress_value = 0
            else:
                self.progress_range = (0, 0)

        if _RE_LOO_TRIAL.search(line):
            self._loo_progress += 1
            total = self._usable_total
            if total:
                self.progress_range = (0, total)
                self.progress_value = self._loo_progress
                self.progress_format = f"Trial {self._loo_progress} / {total}"
            self.phase_text = (
                f"Phase: extracting per-trial time courses "
                f"({self._loo_progress}/{total or '?'})…"
            )

        m = _RE_PERM_START.search(line)
        if m:
            self.phase_text = f"Phase: running {m.group(1)} sign-flip permutations…"
            self.progress_range = (0, 0)
            self.progress_format = ""

        m = _RE_PEAK_SUMMARY.search(line)
        if m:
            self._result_mean, self._result_sem = m.group(1), m.group(2)

        m = _RE_CLUSTER_RESULT.search(line)
        if m:
            self._result_cluster_px = m.group(1)
            self._result_cluster_p = m.group(3)
            self._result_peak_t = m.group(4)

        m = _RE_DONE.search(line)
        if m:
            self.output_dir = m.group(1).strip()
            self.phase_text = "Phase: complete"
            self.progress_range = (0, 1)
            self.progress_value = 1
            self.progress_format = "Done"

        self.status_changed.emit()

    def _on_poster_line(self, line: str) -> None:
        if _RE_POSTER_MASKS.search(line):
            self.phase_text = "Phase: figures — building masks…"
        elif _RE_POSTER_GREEN_REF.search(line):
            self.phase_text = "Phase: figures — rendering green/targeting reference…"
        elif _RE_POSTER_PANELS.search(line):
            self.phase_text = "Phase: figures — rendering cortical panels…"
        elif _RE_POSTER_EXTRACTING.search(line):
            self.phase_text = "Phase: figures — extracting timecourse (streaming raw frames)…"
        elif _RE_POSTER_TIMECOURSE.search(line):
            self.phase_text = "Phase: figures — rendering timecourse…"

    # ── Phase transitions ────────────────────────────────────────────────────

    def _on_done(self, exit_code: int) -> None:
        if self._phase == "poster":
            self._on_poster_done(exit_code)
        else:
            self._on_stats_done(exit_code)

    def _on_stats_done(self, exit_code: int) -> None:
        if self._force_killed:
            self._finish(JobStatus.STOPPED, "Force-killed.")
            return

        if exit_code == 0:
            self.phase_text = "Statistics complete."
        else:
            self.phase_text = (
                f"Statistics stopped (exit code {exit_code})."
                if self._stop_requested else
                f"Statistics ended (exit code {exit_code})."
            )

        if self.output_dir:
            self.post_run_text = f"Output: {self.output_dir}"
        elif exit_code == 0:
            self.post_run_text = (
                "Finished, but no output produced — likely too few usable "
                "trials (need 3+). See log."
            )

        if exit_code == 0 and self.output_dir and self.chain_poster:
            self._launch_poster_figures()
        else:
            status = (
                JobStatus.STOPPED if self._stop_requested else
                JobStatus.SUCCEEDED if exit_code == 0 else
                JobStatus.FAILED
            )
            self._finish(status)

    def _launch_poster_figures(self) -> None:
        self._phase = "poster"
        self.phase_text = "Phase: figures — starting…"
        self.progress_range = (0, 0)
        self.progress_value = 0
        self.progress_format = ""
        compare_session_override = compare_stats_dir_override = compare_label_override = None
        if self.compare_job is not None:
            compare_session_override = self.compare_job.session_dir
            compare_stats_dir_override = self.compare_job.output_dir
            compare_label_override = _CONDITION_COMPARE_LABELS.get(
                self.compare_job.condition, self.compare_job.condition.capitalize()
            )
        try:
            argv = analysis_argv.build_poster_argv(
                self._stats_snapshot.session_dir, self.output_dir, self._poster_snapshot,
                compare_session_override=compare_session_override,
                compare_stats_dir_override=compare_stats_dir_override,
                compare_label_override=compare_label_override,
            )
            self._runner.start(analysis_argv.POSTER_SCRIPT, argv)
        except (RuntimeError, OSError, FileNotFoundError) as exc:
            line = f"Figures launch failed: {exc}"
            self.log_lines.append(line)
            self.log_line.emit(line)
            self.phase_text = f"Statistics complete; figures failed to launch: {exc}"
            self._phase = "stats"
            self._finish(JobStatus.FAILED)
            return
        self.status_changed.emit()

    def _on_poster_done(self, exit_code: int) -> None:
        if self._force_killed:
            self._finish(JobStatus.STOPPED, "Force-killed.")
            return

        if exit_code == 0:
            self.phase_text = "Statistics + figures complete."
            status = JobStatus.SUCCEEDED
        elif self._stop_requested:
            self.phase_text = "Statistics complete; figures stopped — see log."
            status = JobStatus.STOPPED
        else:
            self.phase_text = f"Statistics complete; figures ended (exit code {exit_code}) — see log."
            status = JobStatus.FAILED
        self._finish(status)

    def _finish(self, status: JobStatus, phase_text: str | None = None) -> None:
        self.status = status
        if phase_text is not None:
            self.phase_text = phase_text
        self._stop_timer.stop()
        self.status_changed.emit()
        self.finished.emit()

    def _on_stop_timer_elapsed(self) -> None:
        self.stop_timed_out.emit()
