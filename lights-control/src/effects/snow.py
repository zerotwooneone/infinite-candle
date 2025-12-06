import numpy as np
import random
from src.effects.abstract import Effect
from src.config import LED_COUNT

class SnowEffect(Effect):
    def __init__(self, config):
        super().__init__(config)
        self.count = config.flake_count
        # State: Each flake has a position [strip_index, precise_height]
        # Since it's a 1D strip, we just track index (0-600) as float
        self.positions = np.random.uniform(0, LED_COUNT, self.count)
        self.speeds = np.random.uniform(config.fall_speed * 0.8, config.fall_speed * 1.2, self.count)
        self.color = np.array(config.color, dtype=np.uint8)

    def update(self, dt: float):
        # Physics: Move every flake down
        # Note: On a spiral, "Down" means decreasing index
        self.positions -= self.speeds * (dt * 60) # Scale for 60fps

        # Reset flakes that hit the bottom
        for i in range(self.count):
            if self.positions[i] < 0:
                self.positions[i] = LED_COUNT - 1 # Reset to top

    def render(self, buffer: np.ndarray, mapper):
        for pos in self.positions:
            idx = int(pos)
            if 0 <= idx < len(buffer):
                # Draw the flake
                buffer[idx] = self.color
                # Optional: Add a faint trail pixel above it