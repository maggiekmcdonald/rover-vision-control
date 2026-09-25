"""In-process kinematic simulation standing in for the PiCar-X on a dev machine.

PiCar-X is a rear-driven, front-steered chassis, so a bicycle model is a
closer approximation of its motion than a differential-drive model would be.
Pose is tracked purely for the demo plot; no physical units beyond
"percent speed maps to some max linear speed" are meant to be exact.
"""

import math
from dataclasses import dataclass

from .base import MotorDriver, DriverStatus


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    heading_rad: float = 0.0  # 0 = facing +x axis


class SimulatedDriver(MotorDriver):
    """Bicycle-model rover simulator: integrates pose from steering/speed commands."""

    def __init__(
        self,
        wheelbase_m: float = 0.12,
        max_speed_mps: float = 0.5,
        initial_pose: Pose = None,
    ):
        self.wheelbase_m = wheelbase_m
        self.max_speed_mps = max_speed_mps

        self.pose = initial_pose if initial_pose is not None else Pose()
        self._steering_deg = 0.0
        self._speed_percent = 0.0
        self.pose_history = [Pose(self.pose.x, self.pose.y, self.pose.heading_rad)]

    def set_steering(self, angle_deg: float) -> None:
        self._steering_deg = max(-90.0, min(90.0, angle_deg))

    def set_speed(self, percent: float) -> None:
        self._speed_percent = max(-100.0, min(100.0, percent))

    def stop(self) -> None:
        self._speed_percent = 0.0

    def get_status(self) -> DriverStatus:
        return DriverStatus(
            steering_deg=self._steering_deg,
            speed_percent=self._speed_percent,
            connected=True,
        )

    def step(self, dt: float) -> Pose:
        """Advance the kinematic model by dt seconds using the current commands."""
        linear_speed = (self._speed_percent / 100.0) * self.max_speed_mps
        steering_rad = math.radians(self._steering_deg)

        # Standard bicycle-model update: heading rate depends on speed and
        # steering angle via the wheelbase; near-zero steering degenerates
        # gracefully to straight-line motion.
        heading_rate = (linear_speed / self.wheelbase_m) * math.tan(steering_rad)

        self.pose.x += linear_speed * math.cos(self.pose.heading_rad) * dt
        self.pose.y += linear_speed * math.sin(self.pose.heading_rad) * dt
        self.pose.heading_rad += heading_rate * dt
        self.pose.heading_rad = math.atan2(
            math.sin(self.pose.heading_rad), math.cos(self.pose.heading_rad)
        )

        self.pose_history.append(Pose(self.pose.x, self.pose.y, self.pose.heading_rad))
        return self.pose
