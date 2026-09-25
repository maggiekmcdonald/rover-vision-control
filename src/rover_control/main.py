"""Closed-loop runner: vision -> navigator -> motor driver.

Selects a frame source and a motor driver based on --mode, then runs the
same control loop regardless of which pair was selected. On this dev
laptop only --mode sim is usable; --mode hardware is written to run
unmodified on a Raspberry Pi 5 with a PiCar-X once the hardware extras are
installed.
"""

import argparse
import sys
import time

from .navigator import Navigator, NavigatorConfig
from .vision import ColorObjectTracker, SyntheticFrameSource, WebcamFrameSource
from .drivers.simulated import SimulatedDriver


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vision-guided rover object-following control loop.")
    parser.add_argument(
        "--mode",
        choices=["sim", "hardware"],
        default="sim",
        help="'sim' runs against a synthetic frame source and SimulatedDriver (default, works on any "
        "machine). 'hardware' runs against a real camera and PiCarXDriver (Raspberry Pi 5 + PiCar-X only).",
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "webcam"],
        default="synthetic",
        help="Frame source for --mode hardware. Ignored in --mode sim, which always uses the synthetic source.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="cv2.VideoCapture index for --source webcam.")
    parser.add_argument("--iterations", type=int, default=200, help="Number of control loop ticks to run.")
    parser.add_argument("--dt", type=float, default=0.05, help="Seconds per control loop tick.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-tick console output.")
    return parser


def run(args) -> int:
    navigator = Navigator(NavigatorConfig())

    if args.mode == "sim":
        tracker = ColorObjectTracker()
        frame_source = SyntheticFrameSource(seed=42)
        driver = SimulatedDriver()

        # Simple scripted target motion for a standalone `main.py --mode sim`
        # run: a sine sweep across the frame at a fixed apparent distance.
        def next_frame(tick: int):
            import math

            cx = 320 + 220 * math.sin(tick * 0.05)
            cy = 240
            radius = 55
            return frame_source.generate(cx, cy, radius)

    else:
        if args.source == "webcam":
            frame_source = WebcamFrameSource(args.camera_index)

            def next_frame(tick: int):
                return frame_source.read()

        else:
            synth = SyntheticFrameSource(seed=42)

            def next_frame(tick: int):
                import math

                cx = 320 + 220 * math.sin(tick * 0.05)
                return synth.generate(cx, 240, 55)

        tracker = ColorObjectTracker()

        from .drivers.picarx_driver import PiCarXDriver

        driver = PiCarXDriver()

    for tick in range(args.iterations):
        frame = next_frame(tick)
        if frame is None:
            print("No frame available, stopping.", file=sys.stderr)
            driver.stop()
            break

        detection = tracker.detect(frame)
        command = navigator.step(detection, args.dt)

        driver.set_steering(command.steering_deg)
        driver.set_speed(command.speed_percent if command.target_visible else 0.0)

        if not args.quiet:
            status = driver.get_status()
            print(
                f"tick={tick:04d} visible={command.target_visible} "
                f"steer={status.steering_deg:6.2f} speed={status.speed_percent:6.2f}"
            )

        if args.mode == "sim":
            driver.step(args.dt)

        time.sleep(0) if args.quiet else None

    driver.stop()
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
