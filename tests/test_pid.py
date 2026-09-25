import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from rover_control.pid import PIDController


def test_proportional_only_responds_immediately():
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0, output_min=-10, output_max=10)
    output = pid.update(error=2.0, dt=0.1)
    assert output == pytest.approx(2.0)


def test_step_response_converges_toward_setpoint():
    """A simple first-order plant under PI control should converge error to ~0."""
    pid = PIDController(kp=0.8, ki=0.5, kd=0.0, output_min=-5, output_max=5)

    measurement = 0.0
    setpoint = 10.0
    dt = 0.05
    errors = []

    for _ in range(400):
        error = setpoint - measurement
        errors.append(abs(error))
        control = pid.update(error, dt)
        # Simple plant: measurement moves toward control-implied velocity.
        measurement += control * dt

    assert errors[-1] < 0.5
    assert errors[-1] < errors[0]


def test_output_is_clamped_to_bounds():
    pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_min=-1.0, output_max=1.0)
    assert pid.update(error=50.0, dt=0.1) == pytest.approx(1.0)
    pid.reset()
    assert pid.update(error=-50.0, dt=0.1) == pytest.approx(-1.0)


def test_anti_windup_limits_integral_when_saturated():
    """Integral should stop growing once output is saturated, so no big overshoot follows."""
    pid = PIDController(
        kp=0.1, ki=2.0, kd=0.0, output_min=-1.0, output_max=1.0, integral_min=-1.0, integral_max=1.0
    )

    # Drive the controller with a huge, sustained error, which would push
    # naive integral accumulation far past what's needed to saturate.
    for _ in range(200):
        pid.update(error=100.0, dt=0.05)

    integral_after_saturation = pid._integral

    # Now flip the error sign; if anti-windup worked, the controller should
    # react quickly (output drops from saturation) rather than continuing
    # to command max output for many more steps while integral unwinds.
    output_after_flip = pid.update(error=-100.0, dt=0.05)

    assert integral_after_saturation <= 1.0 + 1e-9
    assert output_after_flip < 1.0


def test_reset_clears_internal_state():
    pid = PIDController(kp=1.0, ki=1.0, kd=1.0)
    pid.update(error=1.0, dt=0.1)
    pid.update(error=1.0, dt=0.1)
    pid.reset()
    assert pid._integral == 0.0
    assert pid._prev_error_valid is False


def test_invalid_dt_raises():
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
    with pytest.raises(ValueError):
        pid.update(error=1.0, dt=0.0)
    with pytest.raises(ValueError):
        pid.update(error=1.0, dt=-1.0)


def test_invalid_bounds_raise():
    with pytest.raises(ValueError):
        PIDController(kp=1, ki=0, kd=0, output_min=5, output_max=1)
    with pytest.raises(ValueError):
        PIDController(kp=1, ki=0, kd=0, integral_min=5, integral_max=1)


def test_derivative_reacts_to_rate_of_change():
    pid_no_d = PIDController(kp=0.0, ki=0.0, kd=0.0, output_min=-100, output_max=100)
    pid_with_d = PIDController(kp=0.0, ki=0.0, kd=1.0, output_min=-100, output_max=100)

    pid_no_d.update(error=0.0, dt=0.1)
    pid_with_d.update(error=0.0, dt=0.1)

    out_no_d = pid_no_d.update(error=5.0, dt=0.1)
    out_with_d = pid_with_d.update(error=5.0, dt=0.1)

    assert out_no_d == pytest.approx(0.0)
    assert out_with_d == pytest.approx(50.0)
