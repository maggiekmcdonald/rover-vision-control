"""Color-based target tracking and frame sources.

The tracker itself only ever sees a BGR numpy array, so it is indifferent to
whether that frame came from a synthetic generator, a webcam, or a video
file. That indirection is what lets the whole vision->control loop be
exercised on a laptop with no camera attached: SyntheticFrameSource stands
in for cv2.VideoCapture during development and testing.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import cv2
import numpy as np


# HSV range for a saturated red/orange target ball. Two ranges are needed
# because red wraps around the hue circle at 0/180 in OpenCV's 8-bit HSV.
DEFAULT_HSV_LOWER_1 = np.array([0, 120, 80])
DEFAULT_HSV_UPPER_1 = np.array([10, 255, 255])
DEFAULT_HSV_LOWER_2 = np.array([170, 120, 80])
DEFAULT_HSV_UPPER_2 = np.array([180, 255, 255])


class Detection(NamedTuple):
    """A single tracked-target detection in one frame."""

    found: bool
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0
    area: float = 0.0


@dataclass
class TrackerConfig:
    hsv_lower_1: np.ndarray = None
    hsv_upper_1: np.ndarray = None
    hsv_lower_2: Optional[np.ndarray] = None
    hsv_upper_2: Optional[np.ndarray] = None
    min_area: float = 50.0
    blur_kernel: int = 5

    def __post_init__(self):
        if self.hsv_lower_1 is None:
            self.hsv_lower_1 = DEFAULT_HSV_LOWER_1
        if self.hsv_upper_1 is None:
            self.hsv_upper_1 = DEFAULT_HSV_UPPER_1


class ColorObjectTracker:
    """Detects a colored blob via HSV thresholding + largest-contour selection."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()

    def _mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        cfg = self.config
        blurred = cv2.GaussianBlur(frame_bgr, (cfg.blur_kernel, cfg.blur_kernel), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, cfg.hsv_lower_1, cfg.hsv_upper_1)
        if cfg.hsv_lower_2 is not None and cfg.hsv_upper_2 is not None:
            mask2 = cv2.inRange(hsv, cfg.hsv_lower_2, cfg.hsv_upper_2)
            mask = cv2.bitwise_or(mask, mask2)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        """Return the centroid/size of the largest matching blob, if any."""
        mask = self._mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return Detection(found=False)

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.config.min_area:
            return Detection(found=False)

        (x, y), radius = cv2.minEnclosingCircle(largest)
        return Detection(found=True, cx=float(x), cy=float(y), radius=float(radius), area=float(area))


class SyntheticFrameSource:
    """Generates noisy frames with a colored circle at a caller-controlled position.

    Stands in for a real camera so vision and control logic can be tested and
    demoed end-to-end without any hardware attached.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        noise_std: float = 12.0,
        target_color_bgr: tuple = (30, 30, 220),
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.noise_std = noise_std
        self.target_color_bgr = target_color_bgr
        self._rng = np.random.default_rng(seed)

    def generate(self, cx: float, cy: float, radius: float, background_gray: int = 60) -> np.ndarray:
        """Draw one synthetic frame with a filled circle target at (cx, cy, radius)."""
        frame = np.full((self.height, self.width, 3), background_gray, dtype=np.uint8)

        noise = self._rng.normal(0, self.noise_std, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), max(1, int(round(radius)))
        cv2.circle(frame, (cx_i, cy_i), r_i, self.target_color_bgr, thickness=-1)
        cv2.circle(frame, (cx_i, cy_i), r_i, (0, 0, 0), thickness=1)

        return frame


class WebcamFrameSource:
    """Optional real-camera source. Not required for the simulation demo.

    Wraps cv2.VideoCapture so main.py can select it via --source webcam, or
    a video file path, when actual camera hardware is available.
    """

    def __init__(self, device_index_or_path=0):
        self._cap = cv2.VideoCapture(device_index_or_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {device_index_or_path!r}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
