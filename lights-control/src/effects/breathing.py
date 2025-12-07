import numpy as np
from src.effects.abstract import Effect

class BreathingEffect(Effect):
    def __init__(self, config):
        super().__init__(config)

        # Pre-convert colors to float for smooth blending
        self.c_a = np.array(config.color_a, dtype=float)
        self.c_b = np.array(config.color_b, dtype=float)

        self.speed = config.speed
        self.t = 0.0

        # Opacity from BaseLayer
        self.opacity = getattr(config, 'opacity', 1.0)

        # Height limits
        self.h_min = config.h_min
        self.h_max = config.h_max

    def update(self, dt: float):
        # Accumulate time
        self.t += dt * self.speed

    def render(self, buffer, mapper):
        # 1. Calculate Sine Wave
        # sin() gives -1.0 to 1.0. 
        # We remap it to 0.0 to 1.0 for the blend factor.
        # (sin(t) + 1) / 2
        wave = np.sin(self.t * 2 * np.pi)
        k = (wave + 1.0) / 2.0

        # 2. Interpolate Colors
        # Current Color = A * (1-k) + B * k
        # We broadcast this single color to the whole strip shape (1, 3)
        current_color = (self.c_a * (1.0 - k)) + (self.c_b * k)

        # 3. Apply Masking (Height)
        mask = (mapper.coords_y >= self.h_min) & (mapper.coords_y <= self.h_max)

        # 4. Write to Buffer (With Alpha Blending)
        # We extract the current buffer pixels where the mask is valid
        if self.opacity >= 1.0:
            buffer[mask] = current_color.astype(np.uint8)
        else:
            # Blend with existing background
            bg = buffer[mask].astype(float)

            # Final = BG * (1 - Opacity) + Color * Opacity
            blended = (bg * (1.0 - self.opacity)) + (current_color * self.opacity)

            buffer[mask] = blended.astype(np.uint8)