import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from rover_control.vision import ColorObjectTracker, SyntheticFrameSource, Detection, TrackerConfig


@pytest.fixture
def source():
    # Low noise so detection is deterministic-ish for tolerance checks.
    return SyntheticFrameSource(width=640, height=480, noise_std=3.0, seed=123)


@pytest.fixture
def tracker():
    return ColorObjectTracker()


def test_detects_centroid_near_ground_truth(source, tracker):
    frame = source.generate(cx=300, cy=200, radius=50)
    detection = tracker.detect(frame)

    assert detection.found is True
    assert detection.cx == pytest.approx(300, abs=5)
    assert detection.cy == pytest.approx(200, abs=5)


def test_detected_radius_scales_with_ground_truth(source, tracker):
    small = tracker.detect(source.generate(cx=320, cy=240, radius=20))
    large = tracker.detect(source.generate(cx=320, cy=240, radius=80))

    assert small.found and large.found
    assert large.radius > small.radius
    assert small.radius == pytest.approx(20, abs=6)
    assert large.radius == pytest.approx(80, abs=6)


def test_no_target_on_plain_background(tracker):
    source_no_target = SyntheticFrameSource(width=320, height=240, noise_std=3.0, seed=1)
    import numpy as np

    frame = np.full((240, 320, 3), 60, dtype=np.uint8)
    detection = tracker.detect(frame)
    assert detection.found is False


def test_detection_at_multiple_positions_tracks_movement(source, tracker):
    positions = [(100, 100), (320, 240), (500, 400)]
    for gx, gy in positions:
        frame = source.generate(cx=gx, cy=gy, radius=40)
        detection = tracker.detect(frame)
        assert detection.found
        assert detection.cx == pytest.approx(gx, abs=6)
        assert detection.cy == pytest.approx(gy, abs=6)


def test_small_blob_below_min_area_is_ignored():
    source_local = SyntheticFrameSource(width=200, height=200, noise_std=1.0, seed=5)
    tracker_strict = ColorObjectTracker(TrackerConfig(min_area=10_000))
    frame = source_local.generate(cx=100, cy=100, radius=5)
    detection = tracker_strict.detect(frame)
    assert detection.found is False


def test_detection_namedtuple_defaults():
    d = Detection(found=False)
    assert d.cx == 0.0
    assert d.area == 0.0
