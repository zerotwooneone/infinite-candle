# src/engine/mapper.py
import numpy as np
from src.config import LED_COUNT, PILLAR_WRAPS

class PillarMapper:
    def __init__(self):
        indices = np.arange(LED_COUNT)
        self.leds_per_wrap = LED_COUNT / PILLAR_WRAPS

        # 0.0 to 1.0 (Height)
        self.coords_y = indices / LED_COUNT

        # 0.0 to 1.0 (Rotation/Azimuth)
        self.coords_x = (indices % self.leds_per_wrap) / self.leds_per_wrap

        # Aspect Ratio (Height 48" / Circumference 21" ≈ 2.28)
        self.aspect_ratio = 48.0 / 21.0