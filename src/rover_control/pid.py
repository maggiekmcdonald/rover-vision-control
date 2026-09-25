"""Discrete PID controller with anti-windup clamping.

Used twice by the navigator: once to steer toward the target's horizontal
center, once to regulate follow distance from the target's apparent size.
"""

from dataclasses import dataclass


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float


class PIDController:
    """Standard discrete-time PID with output clamping and integral anti-windup.

    Anti-windup: the integral term only accumulates further in a direction
    that would push the output further past its clamp if the output is
    already saturated. This keeps a long-lived error (e.g. the target
    leaving frame) from building a huge integral term that then causes a
    large overshoot once the error clears.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float = -1.0,
        output_max: float = 1.0,
        integral_min: float = -1.0,
        integral_max: float = 1.0,
    ):
        if output_min >= output_max:
            raise ValueError("output_min must be < output_max")
        if integral_min >= integral_max:
            raise ValueError("integral_min must be < integral_max")

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_min = integral_min
        self.integral_max = integral_max

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_error_valid = False

    def reset(self) -> None:
        """Clear accumulated integral and derivative history."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_error_valid = False

    def update(self, error: float, dt: float) -> float:
        """Compute the next control output for the given error and timestep.

        Args:
            error: setpoint - measurement (or any signed error signal).
            dt: elapsed time in seconds since the previous update. Must be > 0.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        proportional = self.kp * error

        candidate_integral = self._integral + error * dt
        clamped_integral = min(max(candidate_integral, self.integral_min), self.integral_max)

        # Only let the integral term move toward its clamp if doing so would
        # not push a saturated output further past its own clamp.
        unclamped_output = (
            proportional
            + self.ki * clamped_integral
            + self._derivative_term(error, dt)
        )
        if unclamped_output > self.output_max and clamped_integral > self._integral:
            clamped_integral = self._integral
        elif unclamped_output < self.output_min and clamped_integral < self._integral:
            clamped_integral = self._integral

        self._integral = clamped_integral

        derivative = self._derivative_term(error, dt)
        self._prev_error = error
        self._prev_error_valid = True

        output = proportional + self.ki * self._integral + derivative
        return min(max(output, self.output_min), self.output_max)

    def _derivative_term(self, error: float, dt: float) -> float:
        if not self._prev_error_valid:
            return 0.0
        return self.kd * (error - self._prev_error) / dt
