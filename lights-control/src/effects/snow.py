import numpy as np
from src.effects.abstract import Effect
from src.config import LED_COUNT

# --- CONFIGURATION ---
# Toggle this to False if you want hard, crisp pixels only
ENABLE_BLOOM = False
BLOOM_RADIUS = 0.02   # How wide is the soft glow? (0.0 to 1.0)
BLOOM_BRIGHTNESS = 0.3 # Brightness of the ring relative to core (0.0 to 1.0)

class SnowEffect(Effect):
    def __init__(self, config):
        super().__init__(config)
        self.count = config.flake_count
        self.color = np.array(config.color, dtype=np.uint8)

        # --- PHYSICS CONSTANTS ---
        # We default wind to 0.1 (slight rotation) if not in config
        self.gravity = getattr(config, 'gravity', 0.3)
        self.wind_speed = getattr(config, 'wind', 0.05)
        self.turbulence_magnitude = 0.2 # How chaotic is the air?

        # --- PARTICLE STATE ---
        # Y: 1.0 (Top) to 0.0 (Bottom)
        # X: 0.0 to 1.0 (Azimuth/Rotation around pillar)
        self.flakes_y = np.random.uniform(0.0, 1.0, self.count)
        self.flakes_x = np.random.uniform(0.0, 1.0, self.count)

        # Individual particle variations (Mass/Drag)
        # Some fall faster, some get pushed by wind more
        self.mass_variance = np.random.uniform(0.8, 1.2, self.count)

    def update(self, dt: float):
        # 1. Calculate Forces

        # Gravity Vector (Down)
        dy = -self.gravity * self.mass_variance * dt

        # Wind Vector (Lateral) + Brownian Motion (Random Jitter)
        # We generate a random drift for every flake, every frame
        brownian_drift = np.random.normal(0, self.turbulence_magnitude, self.count)
        dx = (self.wind_speed + brownian_drift) * dt

        # 2. Apply Movement
        self.flakes_y += dy
        self.flakes_x += dx

        # 3. Boundary Checks

        # Floor Check: If it hits bottom (y < 0), wrap to top
        # We add a random Y offset so they don't all reappear at exactly 1.0 at once
        reset_mask = self.flakes_y < 0
        reset_count = np.sum(reset_mask)
        if reset_count > 0:
            self.flakes_y[reset_mask] = 1.0 + np.random.uniform(0, 0.1, reset_count)
            # Re-randomize X so it doesn't look like the same flake looping
            self.flakes_x[reset_mask] = np.random.uniform(0.0, 1.0, reset_count)

        # Cylinder Wrap: If x goes < 0.0 or > 1.0, wrap around
        self.flakes_x %= 1.0

    def render(self, buffer: np.ndarray, mapper):
        # --- COORDINATE MAPPING ---
        # Map 1D LEDs to 2D Space
        led_y = mapper.coords_y[np.newaxis, :] # (1, 600)
        led_x = mapper.coords_x[np.newaxis, :] # (1, 600)

        flake_y = self.flakes_y[:, np.newaxis] # (50, 1)
        flake_x = self.flakes_x[:, np.newaxis] # (50, 1)

        # --- DISTANCE CALCULATION ---
        # 1. Vertical Distance (Scaled by aspect ratio for circularity)
        dy = np.abs(led_y - flake_y) * mapper.aspect_ratio

        # 2. Horizontal Distance (Handling the 0.0 <-> 1.0 wrap)
        raw_dx = np.abs(led_x - flake_x)
        dx = np.minimum(raw_dx, 1.0 - raw_dx)

        # 3. Exact Distance Squared
        dist_sq = (dx**2) + (dy**2)

        # --- RENDER: CORE (The 1-pixel flake) ---
        # Find the index of the single closest LED for each flake
        nearest_indices = np.argmin(dist_sq, axis=1)

        # We use a histogram method or simple assignment to handle collisions
        # (Two flakes hitting same pixel)
        buffer[nearest_indices] = self.color

        # --- RENDER: BLOOM (Optional) ---
        if ENABLE_BLOOM:
            # Find LEDs that are within the bloom radius (but NOT the core pixel)
            # We create a dimmer color
            dim_color = (self.color * BLOOM_BRIGHTNESS).astype(np.uint8)

            # Mask: "Close enough to glow" AND "Not the core center"
            # Note: radius is squared here to match dist_sq
            bloom_mask = (dist_sq < (BLOOM_RADIUS**2))

            # We must verify we don't overwrite the bright cores.
            # However, since 'render' runs sequentially, the simplest way is:
            # 1. Draw Bloom everywhere applicable
            # 2. Draw Cores on top (which we did above, so we reverse order)

            # Let's do it cleanly:
            # Collapse bloom mask: If an LED is in range of ANY flake
            leds_to_bloom = np.any(bloom_mask, axis=0)

            # Draw bloom (using simple addition to blend if multiple flakes overlap?)
            # For simplicity, we just set them to dim_color first
            # Use 'np.maximum' to blend so we don't dim existing bright spots if layers mix
            # But since we have direct access, let's just write:

            # 1. Clear buffer (Engine does this, but we want to mix bloom)
            # Actually, the Engine clears the buffer before calling us.

            # Optimization: Write Bloom First
            buffer[leds_to_bloom] = dim_color

            # Optimization: Write Core Second (Overwrite bloom)
            buffer[nearest_indices] = self.color