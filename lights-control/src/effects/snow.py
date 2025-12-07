import numpy as np
from src.effects.particle_system import ParticleSystem

class SnowEffect(ParticleSystem):
    def __init__(self, config):
        super().__init__(config, particle_count=config.flake_count)

        self.color = np.array(config.color, dtype=float)

        # Physics Params
        raw_gravity = getattr(config, 'gravity', 0.05)
        self.gravity = -abs(raw_gravity) # Down
        self.wind = getattr(config, 'wind', 0.02)

        # Lower turbulence for less "jittery" brownian motion
        # Was 0.2, now 0.05 (Smoother drift)
        self.turbulence = 0.05

    def update(self, dt: float):
        # 1. Physics
        self.physics_step(dt, gravity=self.gravity, wind=self.wind, turbulence=self.turbulence)

        # 2. Reset Logic (Bottom -> Top)
        reset_mask = self.y < self.config.h_min
        if np.any(reset_mask):
            self.y[reset_mask] = self.config.h_max
            self.x[reset_mask] = np.random.uniform(0.0, 1.0, np.sum(reset_mask))
            self.vy[reset_mask] = 0

    def render(self, buffer, mapper):
        """
        Specialized 'Crisp' Renderer for Snow.
        Maps each snowflake to exactly ONE pixel.
        """
        # --- 1. Calculate Distance Matrix (Same as ParticleSystem) ---
        led_y = mapper.coords_y[np.newaxis, :] # (1, 600)
        led_x = mapper.coords_x[np.newaxis, :] # (1, 600)

        p_y = self.y[:, np.newaxis] # (N, 1)
        p_x = self.x[:, np.newaxis] # (N, 1)

        # Vertical Dist
        dy = np.abs(led_y - p_y) * mapper.aspect_ratio

        # Horizontal Dist (Wrap)
        raw_dx = np.abs(led_x - p_x)
        dx = np.minimum(raw_dx, 1.0 - raw_dx)

        # Distance Squared
        dist_sq = (dx**2) + (dy**2)

        # --- 2. Find The ONE Closest LED per Flake ---
        # axis=1 means "For each ROW (flake), find the min column (LED)"
        closest_led_indices = np.argmin(dist_sq, axis=1) # Shape: (N,)

        # Get the actual distances to those closest LEDs
        # We need this to ensure we don't light up an LED if the flake 
        # is floating in the middle of a huge gap (too far to be seen)
        # Advanced NumPy indexing: pick values from dist_sq using the indices
        row_indices = np.arange(self.count)
        min_distances = dist_sq[row_indices, closest_led_indices]

        # Threshold: Only draw if reasonably close (e.g. within 1/2 spacing)
        # 0.001 is a tight tolerance roughly matching LED spacing
        valid_mask = min_distances < 0.001

        # Filter down to just the valid LEDs
        final_indices = closest_led_indices[valid_mask]

        # --- 3. Write to Buffer ---
        # Set color directly (No blending/bloom)
        # We use simple assignment. If 2 flakes hit the same pixel, it just stays the color.
        buffer[final_indices] = self.color.astype(np.uint8)