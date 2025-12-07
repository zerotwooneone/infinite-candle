import numpy as np
from src.effects.abstract import Effect

class SolidEffect(Effect):
    def __init__(self, config):
        super().__init__(config)
        self.color = np.array(config.color, dtype=float)
        self.h_min = config.h_min
        self.h_max = config.h_max
        self.opacity = config.opacity # 0.0 to 1.0

    def update(self, dt: float):
        pass

    def render(self, buffer: np.ndarray, mapper):
        # 1. Calculate Mask (Where do we draw?)
        mask = (mapper.coords_y >= self.h_min) & (mapper.coords_y <= self.h_max)

        # 2. Optimization: If fully opaque, just overwrite (Fast)
        if self.opacity >= 1.0:
            buffer[mask] = self.color.astype(np.uint8)
            return

        # 3. Alpha Blending (Slow but Pretty)
        # Formula: Result = (Background * (1 - Alpha)) + (Foreground * Alpha)

        # Fetch background pixels
        bg = buffer[mask].astype(float)

        # Blend
        # We need to broadcast self.color to the shape of bg
        blended = (bg * (1.0 - self.opacity)) + (self.color * self.opacity)

        # Write back
        buffer[mask] = blended.astype(np.uint8)