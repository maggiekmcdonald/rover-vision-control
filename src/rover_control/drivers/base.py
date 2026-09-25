"""Abstract motor driver interface shared by simulated and real hardware backends.

This is the seam that makes the same navigator/vision code usable both on a
laptop with no hardware and on the real Raspberry Pi 5 + PiCar-X: main.py
only ever talks to a MotorDriver, never to picarx or a kinematic model
directly, so swapping SimulatedDriver for PiCarXDriver changes nothing else
in the control loop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DriverStatus:
    steering_deg: float
    speed_percent: float
    connected: bool


class MotorDriver(ABC):
    """Common interface for anything that can steer and drive the rover."""

    @abstractmethod
    def set_steering(self, angle_deg: float) -> None:
        """Command a steering angle in degrees, positive = right, negative = left."""

    @abstractmethod
    def set_speed(self, percent: float) -> None:
        """Command a drive speed as a percentage, positive = forward, negative = reverse."""

    @abstractmethod
    def stop(self) -> None:
        """Immediately command zero speed (steering angle is left as-is)."""

    @abstractmethod
    def get_status(self) -> DriverStatus:
        """Return the driver's last commanded state."""
