"""End-to-end closed-loop simulation producing the portfolio demo artifact.

A target wanders through the world along a sine-wave path. At each tick the
target's position is expressed relative to the rover's current pose (bearing
and range), projected into a synthetic camera frame, and fed through the
real vision tracker exactly as a live camera frame would be. The navigator
turns that detection into steering/speed commands, and the SimulatedDriver
integrates those commands into an updated rover pose in world coordinates.
Both the target's and the rover's world-frame paths are recorded and
plotted together -- this is the clearest single artifact for showing the
closed loop actually converges and follows.
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rover_control.vision import ColorObjectTracker, SyntheticFrameSource
from rover_control.navigator import Navigator, NavigatorConfig
from rover_control.drivers.simulated import SimulatedDriver, Pose


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DT = 0.05
NUM_TICKS = 700

IDEAL_RADIUS_PX = 60.0
# Calibration constant for apparent radius vs. range: radius_px = K / range_m.
RADIUS_CALIBRATION_K = 66.0
CAMERA_HALF_FOV_RAD = math.radians(50.0)  # target leaves the frame past this bearing


def target_world_position(t: float):
    """Ground-truth target path: the target wanders in a slow, gentle loop."""
    x = 1.2 * math.sin(0.10 * t) + 1.4
    y = 0.8 * math.sin(0.18 * t + 1.0)
    return x, y


def world_to_camera(rover_pose: Pose, target_x: float, target_y: float):
    """Express the target's world position relative to the rover as (bearing, range)."""
    dx = target_x - rover_pose.x
    dy = target_y - rover_pose.y
    range_m = math.hypot(dx, dy)
    absolute_angle = math.atan2(dy, dx)
    bearing = absolute_angle - rover_pose.heading_rad
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    return bearing, range_m


def camera_to_frame(bearing: float, range_m: float):
    """Project bearing/range into synthetic-frame (cx, radius), or None if out of view."""
    if abs(bearing) > CAMERA_HALF_FOV_RAD or range_m < 0.05:
        return None

    cx = FRAME_WIDTH / 2 - (bearing / CAMERA_HALF_FOV_RAD) * (FRAME_WIDTH / 2 - 40)
    radius = RADIUS_CALIBRATION_K / range_m
    radius = max(6.0, min(140.0, radius))
    return cx, radius


def run_simulation():
    tracker = ColorObjectTracker()
    frame_source = SyntheticFrameSource(width=FRAME_WIDTH, height=FRAME_HEIGHT, noise_std=10.0, seed=7)
    navigator = Navigator(
        NavigatorConfig(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT, ideal_radius_px=IDEAL_RADIUS_PX)
    )
    # Start the rover behind and to the side of the target's initial position
    # so the plot clearly shows it closing the gap and then following.
    driver = SimulatedDriver(initial_pose=Pose(x=0.0, y=-0.6, heading_rad=0.0), max_speed_mps=0.9)

    target_xs, target_ys = [], []
    rover_xs, rover_ys = [], []
    frames_for_gif = []

    for i in range(NUM_TICKS):
        t = i * DT
        tx, ty = target_world_position(t)
        target_xs.append(tx)
        target_ys.append(ty)

        bearing, range_m = world_to_camera(driver.pose, tx, ty)
        projection = camera_to_frame(bearing, range_m)

        if projection is not None:
            cx, radius = projection
            frame = frame_source.generate(cx, FRAME_HEIGHT / 2, radius)
        else:
            # Target outside the field of view: synthesize a frame with no target.
            frame = frame_source.generate(-1000, -1000, 1)

        detection = tracker.detect(frame)
        command = navigator.step(detection, DT)

        driver.set_steering(command.steering_deg)
        driver.set_speed(command.speed_percent if command.target_visible else 0.0)
        pose = driver.step(DT)

        rover_xs.append(pose.x)
        rover_ys.append(pose.y)

        if i % 10 == 0:
            frames_for_gif.append(frame)

    return {
        "target_xs": np.array(target_xs),
        "target_ys": np.array(target_ys),
        "rover_xs": np.array(rover_xs),
        "rover_ys": np.array(rover_ys),
        "frames_for_gif": frames_for_gif,
    }


def plot_paths(result, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(result["target_xs"], result["target_ys"], label="Target path (ground truth)",
            color="#d62728", linewidth=2)
    ax.plot(result["rover_xs"], result["rover_ys"], label="Rover path (SimulatedDriver)",
            color="#1f77b4", linewidth=2, linestyle="--")

    ax.scatter([result["target_xs"][0]], [result["target_ys"][0]], color="#d62728", marker="o", s=70,
               zorder=5, label="Target start")
    ax.scatter([result["rover_xs"][0]], [result["rover_ys"][0]], color="#1f77b4", marker="s", s=70,
               zorder=5, label="Rover start")
    ax.scatter([result["target_xs"][-1]], [result["target_ys"][-1]], color="#d62728", marker="x", s=90,
               zorder=5, label="Target end")
    ax.scatter([result["rover_xs"][-1]], [result["rover_ys"][-1]], color="#1f77b4", marker="+", s=110,
               zorder=5, label="Rover end")

    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_title("Object-Following Rover: Target Path vs. Rover Path (Closed-Loop Simulation)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_gif(frames, out_path: Path):
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not available, skipping GIF export.")
        return False

    pil_frames = [Image.fromarray(f[:, :, ::-1]) for f in frames]  # BGR -> RGB
    pil_frames = [f.resize((200, 150)).convert("P", palette=Image.ADAPTIVE, colors=64) for f in pil_frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=150,
        loop=0,
        optimize=True,
    )
    return True


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    result = run_simulation()

    png_path = RESULTS_DIR / "path_tracking.png"
    plot_paths(result, png_path)
    print(f"Wrote {png_path}")

    gif_path = RESULTS_DIR / "tracking_demo.gif"
    if save_gif(result["frames_for_gif"], gif_path):
        print(f"Wrote {gif_path}")

    final_error = math.hypot(
        result["target_xs"][-1] - result["rover_xs"][-1],
        result["target_ys"][-1] - result["rover_ys"][-1],
    )
    mean_error = float(
        np.mean(
            np.hypot(
                result["target_xs"] - result["rover_xs"],
                result["target_ys"] - result["rover_ys"],
            )
        )
    )
    print(f"Final target-rover separation: {final_error:.3f} m")
    print(f"Mean target-rover separation: {mean_error:.3f} m")


if __name__ == "__main__":
    main()
