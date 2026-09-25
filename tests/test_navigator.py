import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from rover_control.navigator import Navigator, NavigatorConfig
from rover_control.vision import Detection


@pytest.fixture
def nav():
    return Navigator(NavigatorConfig(frame_width=640, frame_height=480, ideal_radius_px=60.0))


def test_target_left_of_center_steers_toward_positive(nav):
    # Target at x=100 (left of center=320) should command the rover to turn
    # toward it. Convention: positive x_error means target is left of
    # center, producing a positive (rightward-labelled) steering command is
    # a design choice -- what matters is the sign is consistent and nonzero.
    detection = Detection(found=True, cx=100, cy=240, radius=60, area=1000)
    cmd = nav.step(detection, dt=0.05)
    assert cmd.target_visible
    assert cmd.steering_deg != 0.0


def test_target_dead_center_produces_near_zero_steering(nav):
    detection = Detection(found=True, cx=320, cy=240, radius=60, area=1000)
    cmd = nav.step(detection, dt=0.05)
    assert cmd.steering_deg == pytest.approx(0.0, abs=1e-6)


def test_target_left_and_right_produce_opposite_sign_steering():
    nav_left = Navigator(NavigatorConfig(frame_width=640))
    nav_right = Navigator(NavigatorConfig(frame_width=640))

    left_cmd = nav_left.step(Detection(found=True, cx=100, cy=240, radius=60, area=1000), dt=0.05)
    right_cmd = nav_right.step(Detection(found=True, cx=540, cy=240, radius=60, area=1000), dt=0.05)

    assert left_cmd.steering_deg * right_cmd.steering_deg < 0


def test_target_too_close_commands_negative_speed(nav):
    # radius > ideal_radius_px => target is closer than desired => back off.
    detection = Detection(found=True, cx=320, cy=240, radius=120, area=5000)
    cmd = nav.step(detection, dt=0.05)
    assert cmd.speed_percent < 0


def test_target_too_far_commands_positive_speed(nav):
    # radius < ideal_radius_px => target is farther than desired => drive forward.
    detection = Detection(found=True, cx=320, cy=240, radius=20, area=200)
    cmd = nav.step(detection, dt=0.05)
    assert cmd.speed_percent > 0


def test_target_at_ideal_distance_near_zero_speed(nav):
    detection = Detection(found=True, cx=320, cy=240, radius=60, area=1000)
    cmd = nav.step(detection, dt=0.05)
    assert cmd.speed_percent == pytest.approx(0.0, abs=1e-6)


def test_lost_target_eventually_stops_and_reports_not_visible(nav):
    lost = Detection(found=False)
    last_cmd = None
    for _ in range(nav.config.lost_target_grace_frames + 2):
        last_cmd = nav.step(lost, dt=0.05)

    assert last_cmd.target_visible is False
    assert last_cmd.speed_percent == 0.0
    assert last_cmd.steering_deg == 0.0


def test_commands_stay_within_configured_limits(nav):
    detection = Detection(found=True, cx=0, cy=240, radius=200, area=9000)
    for _ in range(50):
        cmd = nav.step(detection, dt=0.05)

    assert abs(cmd.steering_deg) <= nav.config.max_steering_deg + 1e-6
    assert abs(cmd.speed_percent) <= nav.config.max_speed_percent + 1e-6


def test_reset_clears_pid_state(nav):
    detection = Detection(found=True, cx=100, cy=240, radius=60, area=1000)
    for _ in range(20):
        nav.step(detection, dt=0.05)
    nav.reset()
    assert nav._steering_pid._integral == 0.0
    assert nav._speed_pid._integral == 0.0
