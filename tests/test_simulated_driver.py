import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from rover_control.drivers.simulated import SimulatedDriver, Pose


def test_zero_commands_hold_pose():
    driver = SimulatedDriver()
    driver.set_speed(0.0)
    driver.set_steering(0.0)
    pose = driver.step(dt=0.1)
    assert pose.x == pytest.approx(0.0)
    assert pose.y == pytest.approx(0.0)
    assert pose.heading_rad == pytest.approx(0.0)


def test_straight_forward_motion_moves_along_heading():
    driver = SimulatedDriver(initial_pose=Pose(0, 0, 0), max_speed_mps=1.0)
    driver.set_speed(100.0)
    driver.set_steering(0.0)

    for _ in range(10):
        driver.step(dt=0.1)

    assert driver.pose.x == pytest.approx(1.0, abs=1e-6)
    assert driver.pose.y == pytest.approx(0.0, abs=1e-6)
    assert driver.pose.heading_rad == pytest.approx(0.0, abs=1e-6)


def test_positive_steering_with_forward_speed_turns_heading():
    driver = SimulatedDriver(initial_pose=Pose(0, 0, 0), max_speed_mps=1.0)
    driver.set_speed(50.0)
    driver.set_steering(20.0)

    for _ in range(20):
        driver.step(dt=0.05)

    assert driver.pose.heading_rad != pytest.approx(0.0, abs=1e-6)


def test_negative_speed_moves_backward():
    driver = SimulatedDriver(initial_pose=Pose(0, 0, 0), max_speed_mps=1.0)
    driver.set_speed(-100.0)
    driver.set_steering(0.0)
    driver.step(dt=0.1)

    assert driver.pose.x < 0


def test_steering_is_clamped():
    driver = SimulatedDriver()
    driver.set_steering(500.0)
    assert driver.get_status().steering_deg == 90.0
    driver.set_steering(-500.0)
    assert driver.get_status().steering_deg == -90.0


def test_speed_is_clamped():
    driver = SimulatedDriver()
    driver.set_speed(500.0)
    assert driver.get_status().speed_percent == 100.0
    driver.set_speed(-500.0)
    assert driver.get_status().speed_percent == -100.0


def test_stop_zeroes_speed_but_keeps_steering():
    driver = SimulatedDriver()
    driver.set_steering(15.0)
    driver.set_speed(80.0)
    driver.stop()
    status = driver.get_status()
    assert status.speed_percent == 0.0
    assert status.steering_deg == 15.0


def test_pose_history_accumulates():
    driver = SimulatedDriver()
    driver.set_speed(30.0)
    for _ in range(5):
        driver.step(dt=0.1)
    assert len(driver.pose_history) == 6  # initial + 5 steps


def test_get_status_reports_connected_true():
    driver = SimulatedDriver()
    assert driver.get_status().connected is True


def test_turning_circle_is_consistent_with_bicycle_model():
    """At constant speed/steering, heading should change at a constant rate."""
    driver = SimulatedDriver(initial_pose=Pose(0, 0, 0), max_speed_mps=1.0, wheelbase_m=0.1)
    driver.set_speed(100.0)
    driver.set_steering(30.0)

    headings = []
    for _ in range(5):
        driver.step(dt=0.05)
        headings.append(driver.pose.heading_rad)

    deltas = [headings[i + 1] - headings[i] for i in range(len(headings) - 1)]
    for d in deltas:
        assert d == pytest.approx(deltas[0], abs=1e-9)
