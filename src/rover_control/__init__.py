"""Vision-guided rover control stack.

Ties together color-based object tracking, a dual-loop PID navigator, and a
swappable motor driver so the same control code runs against a simulated
rover on a laptop or a real PiCar-X on a Raspberry Pi 5.
"""

__version__ = "0.1.0"
