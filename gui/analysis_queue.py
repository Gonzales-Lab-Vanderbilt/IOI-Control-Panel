# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
AnalysisQueue — backs the Analysis tab's "Analysis jobs" list. Starts every
AnalysisJob immediately ("Run analysis" in the UI, not "add to queue"), with
no concurrency limit: the only thing that ever holds a job back is the
out-dir collision guard, so two jobs that would resolve to the identical
--out-dir never run at the same time (they'd race each other writing
loo_roi_amplitude_summary.csv / roi_timecourse_raw.npz / etc. into the
same folder -- see gui/analysis_argv.py's resolve_out_dir()). Everything
else queued runs right away, in parallel, unthrottled.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from gui.analysis_job import AnalysisJob, JobStatus, TERMINAL_STATUSES

_ACTIVE_STATUSES = (JobStatus.RUNNING, JobStatus.STOPPING)


class AnalysisQueue(QObject):
    jobs_changed = Signal()  # structural change or any job's progress tick -> table should refresh

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._jobs: list[AnalysisJob] = []

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def jobs(self) -> list[AnalysisJob]:
        return list(self._jobs)

    def enqueue(self, job: AnalysisJob) -> None:
        job.setParent(self)
        job.status_changed.connect(self.jobs_changed.emit)
        job.finished.connect(self._pump)
        self._jobs.append(job)
        self.jobs_changed.emit()
        self._pump()

    def remove_job(self, job: AnalysisJob) -> bool:
        """Only terminal jobs can be removed from the list -- a queued or
        running job has to be stopped first."""
        if job.status not in TERMINAL_STATUSES:
            return False
        if job not in self._jobs:
            return False
        self._jobs.remove(job)
        job.deleteLater()
        self.jobs_changed.emit()
        return True

    def cancel_queued(self, job: AnalysisJob) -> bool:
        """Drop a job that hasn't started yet -- no subprocess exists, so
        there's nothing to stop, just remove it from the list. Any other
        queued job that depends on this one (e.g. a compare-session stats
        prerequisite) can never proceed either, so it fails along with it
        instead of waiting forever."""
        if job.status is not JobStatus.QUEUED:
            return False
        if job not in self._jobs:
            return False
        self._jobs.remove(job)
        dependents = [j for j in self._jobs if j.depends_on is job]
        job.deleteLater()
        self.jobs_changed.emit()
        for dependent in dependents:
            dependent.mark_dependency_failed(
                f"Could not start — prerequisite job \"{job.label}\" was cancelled."
            )
        return True

    def has_active_jobs(self) -> bool:
        return any(j.status in _ACTIVE_STATUSES or j.status is JobStatus.QUEUED for j in self._jobs)

    def active_job_count(self) -> int:
        return sum(1 for j in self._jobs if j.status in _ACTIVE_STATUSES or j.status is JobStatus.QUEUED)

    def stop_all(self) -> None:
        """Best-effort graceful stop of every running job. Queued jobs are
        simply never started -- nothing to clean up for those."""
        for job in self._jobs:
            if job.status is JobStatus.RUNNING:
                job.request_stop()

    # ── Scheduling ────────────────────────────────────────────────────────────

    def _pump(self) -> None:
        running_out_dirs = {
            j.out_dir_key for j in self._jobs if j.status in _ACTIVE_STATUSES and j.out_dir_key
        }
        started_any = False
        # Jobs whose prerequisite ended without succeeding: failed *after*
        # this loop, not inline -- mark_dependency_failed() synchronously
        # emits `finished`, which is wired back to this same _pump(), and
        # reentering it mid-iteration here would be needlessly fragile.
        to_fail: list[tuple[AnalysisJob, str]] = []

        for job in self._jobs:
            if job.status is not JobStatus.QUEUED:
                continue

            if job.depends_on is not None:
                dep = job.depends_on
                if dep.status is JobStatus.SUCCEEDED:
                    job.depends_on = None
                elif dep.status in TERMINAL_STATUSES:
                    to_fail.append((job, dep.label))
                    continue
                else:
                    job.phase_text = f"Queued — waiting for \"{dep.label}\" to finish first."
                    job.status_changed.emit()
                    continue

            if job.out_dir_key and job.out_dir_key in running_out_dirs:
                job.phase_text = "Queued — waiting: output folder in use by another running job."
                job.status_changed.emit()
                continue
            job.start()
            if job.out_dir_key:
                running_out_dirs.add(job.out_dir_key)
            started_any = True

        if started_any:
            self.jobs_changed.emit()

        for job, dep_label in to_fail:
            job.mark_dependency_failed(
                f"Could not start — prerequisite job \"{dep_label}\" did not finish successfully."
            )
