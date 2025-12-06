import numpy as np
from src.effects.abstract import Effect

class SolidEffect(Effect):
    def __init__(self, config):
        super().__init__(config)
        self.color = np.array(config.color, dtype=np.uint8)

    def update(self, dt: float):
        pass # Solid color doesn't change over time

    def render(self, buffer: np.ndarray, mapper):
        # Fill the entire buffer with this color
        # In the future, we can re-add 'faces' or 'height' masking here!
        buffer[:] = self.color