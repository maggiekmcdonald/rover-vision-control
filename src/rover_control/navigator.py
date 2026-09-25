"""Maps vision tracking error to steering/speed commands via two PID loops.

One loop drives the horizontal centroid error (target off-center in x) to
zero by commanding a steering angle. The other drives the apparent-size
error (proxy for distance, since a closer object looks bigger) to zero by
commanding a forward/reverse speed. Keeping them as independent PID loops
mirrors how the physical rover has independent steering and drive actuators.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

from .pid import PIDController
from .vision import Detection


@dataclass
class NavigatorConfig:
    frame_width: int = 640
    frame_height: int = 480

    # Apparent radius (px) of the target when the rover is at the desired
    # following distance. Smaller measured radius => target is farther away
    # => drive forward; larger => too close => back off.
    ideal_radius_px: float = 60.0

    max_steering_deg: float = 40.0
    max_speed_percent: float = 60.0

    steering_kp: float = 0.6
    steering_ki: float = 0.02
    steering_kd: float = 0.15

    speed_kp: float = 1.2
    speed_ki: float = 0.05
    speed_kd: float = 0.1

    # Number of consecutive lost-target frames tolerated before commanding
    # a full stop, to smooth over brief single-frame detection dropouts.
    lost_target_grace_frames: int = 5


class Command(NamedTuple):
    steering_deg: float
    speed_percent: float
    target_visible: bool


class Navigator:
    """Converts a vision Detection into a steering/speed Command each tick."""

    def __init__(self, config: Optional[NavigatorConfig] = None):
        self.config = config or NavigatorConfig()
        cfg = self.config

        self._steering_pid = PIDController(
            kp=cfg.steering_kp,
            ki=cfg.steering_ki,
            kd=cfg.steering_kd,
            output_min=-1.0,
            output_max=1.0,
            integral_min=-1.0,
            integral_max=1.0,
        )
        self._speed_pid = PIDController(
            kp=cfg.speed_kp,
            ki=cfg.speed_ki,
            kd=cfg.speed_kd,
            output_min=-1.0,
            output_max=1.0,
            integral_min=-1.0,
            integral_max=1.0,
        )
        self._lost_frames = 0

    def reset(self) -> None:
        self._steering_pid.reset()
        self._speed_pid.reset()
        self._lost_frames = 0

    def step(self, detection: Detection, dt: float) -> Command:
        cfg = self.config

        if not detection.found:
            self._lost_frames += 1
            if self._lost_frames >= cfg.lost_target_grace_frames:
                # Losing the target for a sustained period should not leave
                # stale integral wind-up ready to fire once it reappears.
                self.reset()
                return Command(steering_deg=0.0, speed_percent=0.0, target_visible=False)
            # Brief dropout: hold last commands implicitly by not updating
            # the PIDs, but still report not-visible so callers can decide.
            return Command(steering_deg=0.0, speed_percent=0.0, target_visible=False)

        self._lost_frames = 0

        frame_center_x = cfg.frame_width / 2.0
        x_error = (frame_center_x - detection.cx) / frame_center_x  # + => target left of center

        # + => too far away (radius smaller than ideal) => drive forward.
        size_error = (cfg.ideal_radius_px - detection.radius) / cfg.ideal_radius_px

        steering_output = self._steering_pid.update(x_error, dt)
        speed_output = self._speed_pid.update(size_error, dt)

        steering_deg = steering_output * cfg.max_steering_deg
        speed_percent = speed_output * cfg.max_speed_percent

        return Command(steering_deg=steering_deg, speed_percent=speed_percent, target_visible=True)
