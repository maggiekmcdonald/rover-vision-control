"""Hardware driver wrapping the SunFounder picar-x library.

This module is meant to run on a Raspberry Pi 5 with a SunFounder PiCar-X kit
assembled and the vendor library installed via:

    pip install picar-x

On any other machine (including this dev laptop) the `picarx` package is not
installed and is not installable, since it depends on Raspberry Pi GPIO/I2C
access. The import below is therefore wrapped in a try/except so that this
module -- and the rest of the rover_control package -- stays importable
everywhere. Attempting to actually construct a PiCarXDriver on a machine
without the library (or without the hardware) raises a clear RuntimeError
instead of a bare ImportError deep in some unrelated stack trace.

Written against the documented picar-x API (Picarx(), set_dir_servo_angle(),
forward(), backward(), stop()) but not yet exercised against the physical
rover -- development so far has been validated in simulation only.
"""

from .base import MotorDriver, DriverStatus

try:
    from picarx import Picarx

    _PICARX_AVAILABLE = True
except ImportError:
    Picarx = None
    _PICARX_AVAILABLE = False


class PiCarXDriver(MotorDriver):
    """Drives a real SunFounder PiCar-X via the vendor `picarx` library.

    Steering and speed are commanded through the same signed conventions as
    SimulatedDriver (positive steering = right, positive speed = forward),
    so the navigator and main loop need no special-casing per backend.
    """

    def __init__(self, steering_trim_deg: float = 0.0):
        if not _PICARX_AVAILABLE:
            raise RuntimeError(
                "The 'picarx' package is not installed. PiCarXDriver only runs "
                "on a Raspberry Pi with a PiCar-X kit and 'pip install picar-x' "
                "completed. Install requirements-hardware.txt on the Pi to use "
                "this driver, or use --mode sim on a dev machine."
            )

        self._px = Picarx()
        self.steering_trim_deg = steering_trim_deg
        self._steering_deg = 0.0
        self._speed_percent = 0.0

    def set_steering(self, angle_deg: float) -> None:
        # picar-x's own set_dir_servo_angle() clamps to [-30, 30] internally
        # (DIR_MIN/DIR_MAX in the vendor library); clamp here too so
        # get_status() reports the angle actually applied.
        clamped = max(-30.0, min(30.0, angle_deg))
        self._steering_deg = clamped
        self._px.set_dir_servo_angle(clamped + self.steering_trim_deg)

    def set_speed(self, percent: float) -> None:
        clamped = max(-100.0, min(100.0, percent))
        self._speed_percent = clamped
        if clamped >= 0:
            self._px.forward(clamped)
        else:
            self._px.backward(-clamped)

    def stop(self) -> None:
        self._speed_percent = 0.0
        self._px.stop()

    def get_status(self) -> DriverStatus:
        return DriverStatus(
            steering_deg=self._steering_deg,
            speed_percent=self._speed_percent,
            connected=_PICARX_AVAILABLE,
        )
