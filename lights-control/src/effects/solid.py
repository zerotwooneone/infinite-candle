import numpy as np
from src.effects.abstract import Effect

class SolidEffect(Effect):
    def __init__(self, config):
        super().__init__(config)
        self.color = np.array(config.color, dtype=np.uint8)
        self.h_min = config.h_min
        self.h_max = config.h_max

    def update(self, dt: float):
        pass

    def render(self, buffer: np.ndarray, mapper):
        # Only draw where pixels are within the height range
        # mapper.coords_y is (0.0 to 1.0)
        mask = (mapper.coords_y >= self.h_min) & (mapper.coords_y <= self.h_max)

        buffer[mask] = self.color