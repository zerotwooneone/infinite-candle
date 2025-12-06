import numpy as np
from src.config import LED_COUNT, PILLAR_WRAPS

class PillarMapper:
    def __init__(self):
        # Pre-calculate coordinates for all LEDs
        # x = Azimuth (0.0 to 1.0 around the cylinder)
        # y = Height (0.0 to 1.0 from bottom to top)

        indices = np.arange(LED_COUNT)
        self.leds_per_wrap = LED_COUNT / PILLAR_WRAPS

        # Height is simple linear progression
        self.coords_y = indices / LED_COUNT

        # X is the remainder of the spiral
        # 0.0 = Front, 0.5 = Back, 0.99 = Front again
        self.coords_x = (indices % self.leds_per_wrap) / self.leds_per_wrap

        # Aspect Ratio (Height / Circumference)
        # 48 inches tall / 21 inches circumference ≈ 2.28
        # We need this so a 45-degree line actually looks 45 degrees
        self.aspect_ratio = 48.0 / 21.0