# lights-control/src/engine/mapper.py
import numpy as np
from src.config import LED_COUNT, PILLAR_WRAPS, PILLAR_FACES

class PillarMapper:
    def __init__(self):
        self.leds_per_wrap = LED_COUNT / PILLAR_WRAPS
        self.leds_per_face = self.leds_per_wrap / PILLAR_FACES

    def get_indices_for_region(self, faces: list[int], min_h: float, max_h: float):
        """
        Returns a boolean mask (numpy array) of LEDs that fall inside the requested region.
        
        faces: list of integers [0, 1, 2, 3]
        min_h: 0.0 (bottom) to 1.0 (top)
        max_h: 0.0 (bottom) to 1.0 (top)
        """
        # Create an array of all LED indices [0, 1, ... 599]
        all_indices = np.arange(LED_COUNT)

        # Calculate the "Height" of every pixel (0.0 to 1.0)
        # Pixel 0 is height 0.0, Pixel 600 is height 1.0
        pixel_heights = all_indices / LED_COUNT

        # Calculate the "Face" of every pixel
        # This is a modulo operation. 
        # If a wrap is 24 pixels, pixels 0-5 are Face 0, 6-11 are Face 1, etc.
        # We calculate "how far along the current wrap are we?"
        position_in_wrap = all_indices % self.leds_per_wrap
        pixel_faces = (position_in_wrap / self.leds_per_face).astype(int)

        # Create Boolean Masks
        height_mask = (pixel_heights >= min_h) & (pixel_heights <= max_h)
        face_mask = np.isin(pixel_faces, faces)

        # Combine
        return height_mask & face_mask