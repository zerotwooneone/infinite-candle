import numpy as np
from src.effects.abstract import Effect

class SnowEffect(Effect):
    def __init__(self, config):
        super().__init__(config)
        self.count = config.flake_count
        self.color = np.array(config.color, dtype=np.uint8)

        # --- PHYSICAL STATE ---
        # Instead of "Index" (0-600), we track 2D coordinates.
        # Y: 1.0 (Top) to 0.0 (Bottom)
        # X: 0.0 to 1.0 (Around the cylinder)
        self.flakes_y = np.random.uniform(0.0, 1.0, self.count)
        self.flakes_x = np.random.uniform(0.0, 1.0, self.count)

        # Random fall speeds for each flake
        # gravity 1.0 approx 20% of height per second
        self.speeds = np.random.uniform(0.1, 0.3, self.count) * config.gravity

    def update(self, dt: float):
        # 1. Fall Down (Decrease Y)
        self.flakes_y -= self.speeds * dt

        # 2. Reset flakes that pass the bottom
        # We use a mask to find flakes < 0.0 and wrap them to top (1.0 + random offset)
        reset_mask = self.flakes_y < 0
        self.flakes_y[reset_mask] = 1.0 + np.random.uniform(0, 0.2, np.sum(reset_mask))
        self.flakes_x[reset_mask] = np.random.uniform(0.0, 1.0, np.sum(reset_mask))

        # 3. Lateral Drift (Wind)
        # Small sine wave motion + random jitter
        self.flakes_x += np.random.normal(0, 0.05, self.count) * dt
        self.flakes_x %= 1.0 # Wrap around cylinder

    def render(self, buffer: np.ndarray, mapper):
        """
        Matrix Math Magic:
        We calculate the distance from every Snowflake to every LED in one go.
        """
        # LED Coordinates (Shape: 1 x 600)
        led_y = mapper.coords_y[np.newaxis, :]
        led_x = mapper.coords_x[np.newaxis, :]

        # Flake Coordinates (Shape: 50 x 1)
        flake_y = self.flakes_y[:, np.newaxis]
        flake_x = self.flakes_x[:, np.newaxis]

        # --- Distance Calculation ---

        # 1. Vertical Distance
        dy = np.abs(led_y - flake_y)
        # Scale Y by aspect ratio so '0.05' distance is a circle, not an oval
        dy *= mapper.aspect_ratio

        # 2. Horizontal Distance (Handle Cylinder Wrap)
        # Distance between 0.99 and 0.01 should be 0.02, not 0.98
        dx_raw = np.abs(led_x - flake_x)
        dx = np.minimum(dx_raw, 1.0 - dx_raw)

        # 3. Euclidean Distance Squared
        dist_sq = (dx**2) + (dy**2)

        # --- Rendering ---

        # Snow Size Radius (squared)
        # 0.002 is roughly a "tight" dot. Increase for fuzzier snow.
        radius_sq = 0.002

        # Find all pairs where distance is close enough
        # Shape: (50, 600) Boolean Matrix
        visible_mask = dist_sq < radius_sq

        # Collapse the matrix: If an LED is hit by ANY snowflake, light it up
        # Shape: (600,) Boolean Array
        leds_to_light = np.any(visible_mask, axis=0)

        # Apply color
        buffer[leds_to_light] = self.color