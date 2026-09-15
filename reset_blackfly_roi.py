#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
"""
reset_blackfly_roi.py

Reset the Blackfly camera hardware ROI to full-frame capture and do nothing else.

This script:
    - Initializes the first detected Blackfly/Spinnaker camera
    - Sets OffsetX = 0 and OffsetY = 0
    - Sets Width and Height to their maximum allowed values
    - Deinitializes and releases the camera

It does NOT:
    - Start acquisition
    - Save images
    - Talk to Arduino
    - Turn lights on/off
    - Change exposure/gain/pixel format/trigger settings

Usage:
    py -3.10 reset_blackfly_roi.py

Optional:
    py -3.10 reset_blackfly_roi.py --camera-index 0
"""

from __future__ import annotations

import argparse
import sys

try:
    import PySpin  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "PySpin is not installed or not visible to this Python environment."
    ) from exc


class CameraConfigError(RuntimeError):
    pass


def get_node(cam, name: str):
    node = cam.GetNodeMap().GetNode(name)
    if node is None or not PySpin.IsAvailable(node):
        raise CameraConfigError(f"Node not available: {name}")
    return node


def get_int(cam, name: str) -> int:
    node = PySpin.CIntegerPtr(get_node(cam, name))
    if not PySpin.IsReadable(node):
        raise CameraConfigError(f"Integer node not readable: {name}")
    return int(node.GetValue())


def set_int(cam, name: str, value: int) -> int:
    """Set an integer node, clipped/aligned to the node's allowed range/increment."""
    node = PySpin.CIntegerPtr(get_node(cam, name))
    if not PySpin.IsWritable(node):
        raise CameraConfigError(f"Integer node not writable: {name}")

    min_value = int(node.GetMin())
    max_value = int(node.GetMax())
    inc = int(node.GetInc()) if hasattr(node, "GetInc") else 1
    inc = max(1, inc)

    clipped = max(min_value, min(max_value, int(value)))
    aligned = min_value + ((clipped - min_value) // inc) * inc
    aligned = max(min_value, min(max_value, aligned))

    node.SetValue(aligned)
    return int(node.GetValue())


def get_max_int(cam, name: str) -> int:
    node = PySpin.CIntegerPtr(get_node(cam, name))
    if not PySpin.IsReadable(node):
        raise CameraConfigError(f"Integer node not readable: {name}")
    return int(node.GetMax())


def print_roi(cam, label: str) -> None:
    print(label)
    print(f"  OffsetX: {get_int(cam, 'OffsetX')}")
    print(f"  OffsetY: {get_int(cam, 'OffsetY')}")
    print(f"  Width:   {get_int(cam, 'Width')}")
    print(f"  Height:  {get_int(cam, 'Height')}")


def reset_roi_to_full_frame(cam) -> None:
    """Reset camera hardware ROI to full frame.

    The order matters on many FLIR/Blackfly cameras:
        1. Zero offsets
        2. Expand width/height to max
        3. Re-zero offsets
    """
    print_roi(cam, "Current camera ROI:")

    set_int(cam, "OffsetX", 0)
    set_int(cam, "OffsetY", 0)

    width_max = get_max_int(cam, "Width")
    height_max = get_max_int(cam, "Height")

    actual_width = set_int(cam, "Width", width_max)
    actual_height = set_int(cam, "Height", height_max)

    actual_x = set_int(cam, "OffsetX", 0)
    actual_y = set_int(cam, "OffsetY", 0)

    print("\nCamera ROI reset to full frame:")
    print(f"  OffsetX: {actual_x}")
    print(f"  OffsetY: {actual_y}")
    print(f"  Width:   {actual_width}")
    print(f"  Height:  {actual_height}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Reset Blackfly/Spinnaker camera ROI to full frame."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index to reset if multiple cameras are connected. Default: 0.",
    )
    args = parser.parse_args(argv)

    system = None
    cam_list = None
    cam = None

    try:
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()

        num_cameras = cam_list.GetSize()
        if num_cameras < 1:
            raise RuntimeError("No Blackfly/Spinnaker camera detected.")

        if args.camera_index < 0 or args.camera_index >= num_cameras:
            raise RuntimeError(
                f"Camera index {args.camera_index} is invalid. "
                f"{num_cameras} camera(s) detected."
            )

        cam = cam_list.GetByIndex(args.camera_index)
        cam.Init()

        reset_roi_to_full_frame(cam)

        cam.DeInit()
        cam = None

        print("\nDone. No images were acquired and no Arduino commands were sent.")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    finally:
        if cam is not None:
            try:
                cam.DeInit()
            except Exception:
                pass

        if cam_list is not None:
            try:
                cam_list.Clear()
            except Exception:
                pass

        if system is not None:
            try:
                system.ReleaseInstance()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
