#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
Run interactive Blackfly exposure/ROI calibration, then immediately run intrinsic imaging
using the selected exposure and ROI.

Place this file in the same folder as:
    intrinsic_calibration.py
    intrinsic_imaging.py

Example:
    py -3.10 intrinsic_calibrated_imaging.py --output .\captures --port COM4 --trials 10 --analyze --analysis-method raw_counts --save-format raw

The calibration step saves selected_exposure.json and selected_roi.json in an
exposure_calibration_YYYYMMDD_HHMMSS folder. The imaging step then uses:
    --exposure-us <selected exposure>
    --roi-config <selected_roi.json>
without you copying those values manually.
"""

from __future__ import annotations

import argparse
import importlib.util
import signal
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def _load_module(module_name: str, module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(f"Required module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Analysis-method identifiers, kept in step with intrinsic_imaging.py. Defined
# locally rather than imported because that module is only loaded (via
# importlib, in main()) after argument parsing has already happened.
_ANALYSIS_METHOD_ALIASES = {
    "fractional": "fractional_reflectance",
}


def normalize_analysis_method(value: str) -> str:
    key = str(value).strip().lower()
    return _ANALYSIS_METHOD_ALIASES.get(key, key)


def dated_output_folder(base_output: Path) -> Path:
    """Convert an output folder like captures/ into captures_MMDDYYYY/."""
    base_output = Path(base_output)
    today = datetime.now().strftime("%m%d%Y")

    # Avoid double-appending if user already passes captures_MMDDYYYY.
    if base_output.name.endswith(f"_{today}"):
        return base_output
    
    return base_output.with_name(f"{base_output.name}_{today}")


def _parse_trial_conditions(raw: Optional[str], n_trials: int) -> Optional[list]:
    """Parse --trial-conditions ('1,0,1,...') into a list[bool], one entry
    per trial (True = stim, False = catch). None/empty input means the
    feature is unused, reproducing today's uniform stim-every-trial
    behavior. Raises SystemExit on a length mismatch or a bad token, same
    pattern as the --final-stage requires ... checks in main()."""
    if raw is None or not raw.strip():
        return None
    tokens = [t.strip() for t in raw.split(",")]
    if len(tokens) != n_trials:
        raise SystemExit(
            f"--trial-conditions has {len(tokens)} entries but --trials is "
            f"{n_trials}; they must match exactly."
        )
    try:
        return [bool(int(t)) for t in tokens]
    except ValueError:
        raise SystemExit(
            "--trial-conditions must be a comma list of 1s and 0s, e.g. '1,0,1,1,0'."
        )

def build_imaging_config(
    img_mod, args: argparse.Namespace, selected_exposure_us: float,
    green_exposure_us: Optional[float], roi_json_path: Path,
    precaptured_green_reference: bool = False,
):
    return img_mod.TrialConfig(
        num_trials=args.trials,
        inter_trial_s=args.iti,
        exposure_us=selected_exposure_us,
        green_exposure_us=green_exposure_us,
        gain_db=args.gain_db if args.gain_db is not None and args.gain_db >= 0 else None,
        pixel_format=args.pixel_format,
        binning_horizontal=args.camera_binning_horizontal if args.camera_binning_horizontal is not None else args.camera_binning,
        binning_vertical=args.camera_binning_vertical if args.camera_binning_vertical is not None else args.camera_binning,
        analysis_binning=args.analysis_binning if args.analysis_binning is not None else args.binning,
        save_format=args.save_format,
        overwrite=args.overwrite,
        serial_port=args.port,
        serial_baud=args.baud,
        serial_timeout_s=args.serial_timeout_s,
        marker_timeout_s=args.marker_timeout,
        save_gap_frames=not args.no_save_gap_frames,
        green_frames=args.green_frames,
        green_reference_trim_frames=args.green_reference_trim_frames,
        baseline_frames=args.baseline_frames,
        post_frames=args.post_frames,
        trailing_timeout_s=args.trailing_timeout_s,
        post_idle_timeout_s=args.post_idle_timeout_s,
        run_analysis=args.analyze,
        analysis_start_frame=args.analysis_start_frame,
        analysis_end_frame=args.analysis_end_frame if args.analysis_end_frame > 0 else None,
        analysis_smoothing_sigma=args.analysis_smoothing_sigma,
        analysis_mask_percentile=args.analysis_mask_percentile,
        analysis_denominator_floor_percentile=args.analysis_denominator_floor_percentile,
        analysis_denominator_floor_counts=args.analysis_denominator_floor_counts,
        analysis_median_filter_size=args.analysis_median_filter_size,
        roi_config=roi_json_path,
        dark_reference_path=args.dark_reference,
        analysis_method=args.analysis_method,
        run_session_analysis=not args.no_session_analysis,
        rescale_bit_depth=args.rescale_bit_depth,
        rescale_mode=args.rescale_mode,
        invert_display_signal=args.invert_display_signal,
        save_analysis_arrays=not args.no_save_analysis_arrays,
        analyze=args.analyze,
        open_overlay=args.open_overlay,
        open_trial_overlays=args.open_trial_overlays,
        open_session_overlay=args.open_overlay,
        visual_stim=args.visual_stim,
        trial_conditions=_parse_trial_conditions(args.trial_conditions, args.trials),
        stim_host=args.stim_host,
        stim_port=args.stim_port,
        stim_orientations=args.stim_orientations,
        stim_duration_s=args.stim_duration_s,
        session_green_cmd=args.session_green_cmd,
        red_on_cmd=args.red_on_cmd,
        trial_start_cmd=args.trial_start_cmd,
        lights_off_cmd=args.lights_off_cmd,
        red_stabilization_s=args.red_stabilization_s,
        red_settling_sample_interval_s=args.red_settling_sample_interval_s,
        turn_lights_off_on_exit=not args.leave_lights_on_exit,
        precaptured_green_reference=precaptured_green_reference,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate exposure/ROI, then run Blackfly intrinsic imaging in one command."
    )

    # Shared/core
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--serial-timeout-s", type=float, default=0.1)
    parser.add_argument("--pixel-format", type=str, default="Mono16", choices=["Mono8", "Mono12", "Mono16"])
    parser.add_argument("--gain-db", type=float, default=0.0, help="Gain in dB. Use -1 to leave unchanged where supported.")

    # Calibration-specific
    parser.add_argument("--arduino-ready-marker", type=str, default="ARDUINO_READY")
    parser.add_argument("--cal-red-cmd", type=str, default="CAL_RED_ON")
    parser.add_argument("--cal-green-cmd", type=str, default="CAL_GREEN_ON")
    parser.add_argument("--skip-green-calibration", action="store_true", help="Use the red/trial exposure for green reference instead of running a separate green exposure calibration.")
    parser.add_argument("--green-exposure-us", type=float, default=None, help="Manually provide green-reference exposure and skip green exposure calibration.")
    parser.add_argument("--no-light-control", action="store_true")
    parser.add_argument("--cal-min-us", type=float, default=1000.0)
    parser.add_argument("--cal-max-us", type=float, default=40000.0)
    parser.add_argument("--cal-fps", type=float, default=10.0)
    parser.add_argument("--cal-steps", type=int, default=50)
    parser.add_argument("--cal-frames-per-exposure", type=int, default=1)
    parser.add_argument("--cal-settle-s", type=float, default=0.05)
    parser.add_argument("--cal-no-discard-first-frame", action="store_true")
    parser.add_argument("--cal-save-preview-npy", action="store_true")
    parser.add_argument("--cal-display-percentile-low", type=float, default=1.0)
    parser.add_argument("--cal-display-percentile-high", type=float, default=99.0)
    parser.add_argument("--skip-roi-selection", action="store_true", help="Use full-frame analysis ROI.")
    parser.add_argument("--cal-external-trigger", action="store_true", help="Use Arduino trigger pulses for each calibration frame instead of free-running camera capture.")
    parser.add_argument("--cal-trigger-cmd", type=str, default="CAL_TRIGGER", help="Arduino command that emits one camera trigger pulse during calibration.")
    parser.add_argument("--cal-red-max-us", type=float, default=18000.0, help="Upper exposure bound for the RED sweep only; green still uses --cal-max-us.")
    parser.add_argument("--keep-red-warm", action="store_true", help="Leave the red LED on after red calibration so it stays warm into imaging (requires the auto-reset cap). Pair with a reduced --red-stabilization-s.")

    # Imaging-specific, mostly matching intrinsic_imaging.py
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--iti", type=float, default=20.0)
    parser.add_argument("--binning", type=int, default=2)
    parser.add_argument("--analysis-binning", type=int, default=None)
    parser.add_argument("--camera-binning", type=int, default=1)
    parser.add_argument("--camera-binning-horizontal", type=int, default=None)
    parser.add_argument("--camera-binning-vertical", type=int, default=None)
    parser.add_argument("--save-format", type=str, default="raw", choices=["png", "tiff", "raw"])
    parser.add_argument("--marker-timeout", type=float, default=30.0)
    parser.add_argument("--green-frames", type=int, default=30)
    parser.add_argument("--green-reference-trim-frames", type=int, default=5)
    parser.add_argument("--baseline-frames", type=int, default=40)
    parser.add_argument("--post-frames", type=int, default=40)
    parser.add_argument("--trailing-timeout-s", type=float, default=2.0)
    parser.add_argument("--post-idle-timeout-s", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-gap-frames", action="store_true", help="Deprecated no-op: gap/debounce frames are saved by default now. Kept only for backward compatibility.")
    parser.add_argument("--no-save-gap-frames", action="store_true", help="Don't save frames captured during the inter-phase gap/debounce windows (baseline->post transition). The camera still triggers continuously through those windows; this only controls whether the frames are written to disk.")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--analysis-start-frame", type=int, default=5)
    parser.add_argument("--analysis-end-frame", type=int, default=35)
    parser.add_argument("--analysis-smoothing-sigma", type=float, default=5.0)
    parser.add_argument("--dark-reference", type=Path, default=None)
    parser.add_argument("--analysis-mask-percentile", type=float, default=20.0)
    parser.add_argument("--analysis-denominator-floor-percentile", type=float, default=5.0)
    parser.add_argument("--analysis-denominator-floor-counts", type=float, default=100.0)
    parser.add_argument("--analysis-median-filter-size", type=int, default=3)
    parser.add_argument("--analysis-method", type=normalize_analysis_method, default="raw_counts", choices=["raw_counts", "fractional_reflectance"], help="Intrinsic signal processing method. 'raw_counts' uses post mean - baseline mean; 'fractional_reflectance' uses fractional reflectance with a denominator floor. Method names used by earlier versions are still accepted.")
    parser.add_argument("--no-session-analysis", action="store_true")
    parser.add_argument("--rescale-bit-depth", type=int, default=16)
    parser.add_argument("--rescale-mode", type=str, default="signed_symmetric", choices=["signed_symmetric", "minmax"])
    parser.add_argument("--invert-display-signal", action="store_true")
    parser.add_argument("--no-save-analysis-arrays", action="store_true")
    parser.add_argument("--open-overlay", action="store_true")
    parser.add_argument("--open-trial-overlays", action="store_true")
    parser.add_argument("--visual-stim", action="store_true")
    parser.add_argument("--stim-host", type=str, default="127.0.0.1")
    parser.add_argument("--stim-port", type=int, default=55000)
    parser.add_argument("--stim-orientations", type=str, default="45")
    parser.add_argument("--stim-duration-s", type=float, default=7.0)
    parser.add_argument(
        "--trial-conditions", type=str, default=None,
        help="Comma list of 1(stim)/0(catch), one per trial, e.g. '1,0,1,1,0'. "
             "Length must equal --trials. Omit for today's uniform behavior "
             "(every trial fires stim per the configured modality).",
    )
    parser.add_argument("--session-green-cmd", type=str, default="SESSION_GREEN_REFERENCE")
    parser.add_argument("--red-on-cmd", type=str, default="SESSION_RED_ON")
    parser.add_argument("--trial-start-cmd", type=str, default="START_TRIAL")
    parser.add_argument("--lights-off-cmd", type=str, default="LIGHTS_OFF")
    parser.add_argument("--red-stabilization-s", type=float, default=600.0)
    parser.add_argument("--red-settling-sample-interval-s", type=float, default=0.0, help="If > 0, sample mean ROI brightness this often (seconds) during red stabilization for live settling monitoring. 0 disables.")
    parser.add_argument("--leave-lights-on-exit", action="store_true")

    # Staged-workflow support: skip calibration entirely and go straight to
    # red stabilization + trials, reusing exposure/ROI/green-reference results
    # already produced by earlier, separately-launched calibration stages
    # (intrinsic_calibration.py --stage green / --stage red and
    # intrinsic_imaging.py --green-reference-only). Lets a GUI insert a live
    # camera feed pause between those stages instead of running everything
    # as one uninterruptible process.
    parser.add_argument(
        "--final-stage", action="store_true",
        help="Skip calibration; requires --red-exposure-us, --roi-config, and "
             "--session-dir from earlier staged runs. Implies a precaptured "
             "session green reference in that session dir.",
    )
    parser.add_argument("--red-exposure-us", type=float, default=None, help="Required with --final-stage.")
    parser.add_argument("--roi-config", type=Path, default=None, help="Required with --final-stage.")
    parser.add_argument(
        "--session-dir", type=Path, default=None,
        help="Required with --final-stage: the session folder created by the "
             "earlier --green-reference-only run, reused instead of starting a "
             "new one.",
    )

    return parser.parse_args(argv)


def install_break_handler() -> None:
    """
    Route Windows CTRL_BREAK_EVENT into the normal KeyboardInterrupt path.

    The GUI stops a subprocess with CTRL_BREAK_EVENT (see gui/script_runner.py).
    Python maps CTRL_C_EVENT to KeyboardInterrupt on its own, but SIGBREAK
    defaults to SIG_DFL, which lets the OS terminate the process outright with
    STATUS_CONTROL_C_EXIT (0xC000013A) - no except, no finally, and therefore no
    lights-off teardown. With this installed, KeyboardInterrupt is raised
    normally, the finally blocks run, and the LEDs are commanded off on the way
    out. Same handler intrinsic_visual_stimulus.py installs, for the same reason.
    """
    handler = getattr(signal, "SIGBREAK", None)
    if handler is not None:
        signal.signal(handler, signal.default_int_handler)


def main(argv: list[str]) -> int:
    install_break_handler()
    args = parse_args(argv)

    args.output = dated_output_folder(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Daily output folder: {args.output}")

    if args.cal_steps < 2:
        raise SystemExit("--cal-steps must be at least 2")
    if args.cal_max_us <= args.cal_min_us:
        raise SystemExit("--cal-max-us must be greater than --cal-min-us")
    if args.cal_frames_per_exposure < 1:
        raise SystemExit("--cal-frames-per-exposure must be at least 1")
    if args.cal_display_percentile_high <= args.cal_display_percentile_low:
        raise SystemExit("--cal-display-percentile-high must be greater than --cal-display-percentile-low")
    if args.cal_red_max_us is not None and args.cal_red_max_us <= args.cal_min_us:
        raise SystemExit("--cal-red-max-us must be greater than --cal-min-us")

    here = Path(__file__).resolve().parent
    img_mod = _load_module("intrinsic_imaging_imported", here / "intrinsic_imaging.py")

    if args.final_stage:
        if args.red_exposure_us is None:
            raise SystemExit("--final-stage requires --red-exposure-us")
        if args.roi_config is None:
            raise SystemExit("--final-stage requires --roi-config")
        if args.session_dir is None:
            raise SystemExit("--final-stage requires --session-dir")

        selected_exposure_us = float(args.red_exposure_us)
        green_exposure_us = args.green_exposure_us
        roi_json_path = Path(args.roi_config)
        external_serial = None

        print("\nSkipping calibration (--final-stage). Using results from earlier staged runs:")
        print(f"  red/trial exposure_us  = {selected_exposure_us:.3f}")
        print(f"  green exposure_us      = {green_exposure_us if green_exposure_us is not None else selected_exposure_us:.3f}")
        print(f"  roi_config             = {roi_json_path}")
        print(f"  session_dir (reused)   = {args.session_dir}")

        cfg = build_imaging_config(
            img_mod, args, selected_exposure_us, green_exposure_us, roi_json_path,
            precaptured_green_reference=True,
        )
        capture = img_mod.BlackflyCapture(
            args.output, cfg, external_serial=external_serial, session_dir_override=args.session_dir,
        )
    else:
        cal_mod = _load_module("intrinsic_calibration_imported", here / "intrinsic_calibration.py")

        # Green-reference calibration now lives inside intrinsic_calibration.py.
        # This wrapper just asks that module for the full red+green calibration sequence.
        cal_results = cal_mod.run_red_green_calibration_from_args(args, keep_red_on_after=args.keep_red_warm)

        selected_exposure_us = cal_results["red_exposure_us"]
        green_exposure_us = cal_results["green_exposure_us"]
        roi_json_path = cal_results["roi_json_path"]
        external_serial = cal_results.get("serial_connection")

        if args.keep_red_warm:
            if external_serial is not None:
                print("Carrying the red calibration's serial connection into imaging (red LED stays on, no reset).")
            else:
                print("Warning: --keep-red-warm was set but the serial connection wasn't carried over "
                      "(calibration may have failed or lights_off_on_exit triggered). "
                      "Imaging will open a fresh connection and the red LED will be re-warmed from cold.")

        print("\nCalibration complete. Starting triggered imaging with:")
        print(f"  red/trial exposure_us  = {selected_exposure_us:.3f}")
        print(f"  green exposure_us      = {green_exposure_us if green_exposure_us is not None else selected_exposure_us:.3f}")
        print(f"  roi_config             = {roi_json_path}")
        print(f"  red calibration        = {cal_results['red_cal_dir']}")
        if cal_results["green_cal_dir"] is not None:
            print(f"  green calibration      = {cal_results['green_cal_dir']}")

        cfg = build_imaging_config(img_mod, args, selected_exposure_us, green_exposure_us, roi_json_path)
        capture = img_mod.BlackflyCapture(args.output, cfg, external_serial=external_serial)

    try:
        capture.setup()
        capture.run()
        print("Done.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        capture.teardown()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
