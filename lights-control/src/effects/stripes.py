import numpy as np
import math
from src.effects.abstract import Effect

class StripesEffect(Effect):
    def __init__(self, config):
        super().__init__(config)
        self.t = 0.0

        # Pre-convert colors to numpy arrays
        self.color_a = np.array(config.color_a, dtype=np.uint8)
        self.color_b = np.array(config.color_b, dtype=np.uint8)

        # Convert Angle to Vector (Rad)
        # We assume 0 degrees is Horizontal rings
        rad = math.radians(config.angle)
        self.nx = math.sin(rad)
        self.ny = math.cos(rad)

    def update(self, dt: float):
        self.t += dt * self.config.speed

    def render(self, buffer: np.ndarray, mapper):
        # 1. Coordinate Projection
        # Project every LED coordinate onto the Normal Vector
        # Value = (x * nx) + (y * ny * aspect_ratio)
        # We multiply Y by aspect_ratio so 45 degrees looks square

        # Vectorized math (fast!)
        projected_dist = (mapper.coords_x * self.nx) + \
                         (mapper.coords_y * self.ny * mapper.aspect_ratio)

        # 2. Add Movement
        projected_dist += self.t

        # 3. Create Wave Pattern (Sine wave or Modulo)
        # We use a repeating sawtooth (modulo) for sharp stripes
        # 0.0 -> 1.0 repeating
        period = self.config.width * 2  # Width of A + Width of B
        pattern = (projected_dist % period) / period

        # 4. Color Mixing
        # If pattern > 0.5, use Color A. Else Color B.
        # This creates a hard edge. For soft edge, use interpolation.
        mask_a = pattern > 0.5

        # Apply to buffer (Additively or overwriting?)
        # For stripes, overwriting usually makes more sense for a base layer
        # But here we add to support layering
        buffer[mask_a] = self.color_a
        buffer[~mask_a] = self.color_b