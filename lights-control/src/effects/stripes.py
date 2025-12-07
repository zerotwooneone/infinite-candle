import numpy as np
import math
from src.effects.abstract import Effect

class StripesEffect(Effect):
    def __init__(self, config):
        super().__init__(config)
        self.t = 0.0

        # Colors
        self.c_a = np.array(config.color_a, dtype=float)
        self.c_b = np.array(config.color_b, dtype=float)

        # --- SEAMLESS MATH CALCULATION ---

        # 1. Calculate ideal geometric components
        rad = math.radians(config.angle)
        nx = math.sin(rad) # Horizontal travel
        ny = math.cos(rad) # Vertical travel

        # 2. Force Horizontal Continuity (The Fix)
        # We need the pattern to repeat an INTEGER number of times around X.
        # Target Period = config.width * 2 (Width A + Width B)
        # How many periods fit in X=0..1?
        # If nx is near 0 (Vertical lines), we treat differently.

        if abs(nx) < 0.01:
            # Case: Vertical Lines (90 deg)
            # Just wrap X an integer amount
            self.freq_x = round(1.0 / (config.width * 2))
            self.freq_y = 0.0
        else:
            # Case: Diagonal/Horizontal Spirals
            # We approximate the "natural" number of spirals based on angle
            # then snap it to the nearest integer.

            # This is a heuristic to keep the visual angle close to requested
            raw_repeats = nx / (config.width * 2)

            # Ensure at least 1 repeat so we don't divide by zero
            self.freq_x = max(1.0, round(abs(raw_repeats))) * np.sign(nx)

            # Calculate corresponding Vertical frequency to maintain angle
            # aspect_ratio scales the Y coord logic
            # tan(angle) = x / y -> y = x / tan(angle)
            if abs(ny) > 0.001:
                self.freq_y = (self.freq_x * ny) / nx
            else:
                self.freq_y = 0.0

        # Speed scaling
        self.speed = config.speed

    def update(self, dt: float):
        self.t += dt * self.speed

    def render(self, buffer: np.ndarray, mapper):
        # HELIX EQUATION: sin( X*FreqX + Y*FreqY + Time )

        # 1. Calculate Phase
        # mapper.coords_x goes 0.0->1.0. We multiply by 2*PI * FreqX
        # to get N perfect sine waves around the circle.
        phase_x = mapper.coords_x * self.freq_x * (2 * np.pi)

        # mapper.coords_y goes 0.0->1.0. Multiply by FreqY and Aspect Ratio
        # We multiply by Aspect Ratio because the physical Y is longer than X circumference
        phase_y = mapper.coords_y * self.freq_y * (2 * np.pi) * mapper.aspect_ratio

        # 2. Combine for Helix
        val = np.sin(phase_x + phase_y + self.t)

        # 3. Soften the Edge (Anti-Aliasing)
        # Instead of 'val > 0', we use a Sigmoid function.
        # This removes the "Jagged" look of the hard cut.
        # sharpness=10.0 gives a crisp but smooth transition.
        sharpness = 10.0
        mix = 1.0 / (1.0 + np.exp(-sharpness * val))

        # Reshape for broadcasting
        mix = mix[:, np.newaxis]

        # 4. Color Blend
        final_colors = (self.c_a * mix) + (self.c_b * (1.0 - mix))

        buffer[:] = final_colors.astype(np.uint8)