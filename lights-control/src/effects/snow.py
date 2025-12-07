import numpy as np
from src.effects.particle_system import ParticleSystem

class SnowEffect(ParticleSystem):
    def __init__(self, config):
        super().__init__(config, particle_count=config.flake_count)
        self.color = np.array(config.color, dtype=float)

        raw_gravity = getattr(config, 'gravity', 0.05)
        self.gravity = -abs(raw_gravity)
        self.wind = getattr(config, 'wind', 0.02)

        # "Random up, down, left, or right"
        # Since this is now a Velocity Magnitude, a value of 0.05 
        # means it can fight against gravity (which is also 0.05).
        self.turbulence = 0.05

        # Configure the "Every N Steps" variable
        self.jitter_interval = 0.2 # Change direction 5 times a second

    def update(self, dt: float):
        self.physics_step(dt, gravity=self.gravity, wind=self.wind, turbulence=self.turbulence)

        # Reset Logic
        reset_mask = self.y < self.config.h_min
        if np.any(reset_mask):
            self.y[reset_mask] = self.config.h_max
            self.x[reset_mask] = np.random.uniform(0.0, 1.0, np.sum(reset_mask))
            self.vy[reset_mask] = 0

    def render(self, buffer, mapper):
        # Keep the "Crisp" renderer we wrote in the previous step
        # (Nearest neighbor, no bloom)
        led_y = mapper.coords_y[np.newaxis, :]
        led_x = mapper.coords_x[np.newaxis, :]
        p_y = self.y[:, np.newaxis]
        p_x = self.x[:, np.newaxis]

        dy = np.abs(led_y - p_y) * mapper.aspect_ratio
        raw_dx = np.abs(led_x - p_x)
        dx = np.minimum(raw_dx, 1.0 - raw_dx)
        dist_sq = (dx**2) + (dy**2)

        closest_led_indices = np.argmin(dist_sq, axis=1)
        row_indices = np.arange(self.count)
        min_distances = dist_sq[row_indices, closest_led_indices]

        valid_mask = min_distances < 0.001
        final_indices = closest_led_indices[valid_mask]
        buffer[final_indices] = self.color.astype(np.uint8)