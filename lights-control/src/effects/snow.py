import numpy as np
from src.effects.particle_system import ParticleSystem

class SnowEffect(ParticleSystem):
    def __init__(self, config):
        super().__init__(config, particle_count=config.flake_count)

        self.color = np.array(config.color, dtype=float)

        # FIX 1: Negative means DOWN
        # If config.gravity is 0.05, we make it -0.05
        raw_gravity = getattr(config, 'gravity', 0.05)
        self.gravity = -abs(raw_gravity)

        self.wind = getattr(config, 'wind', 0.02)

        self.bloom_radius = 0.02
        self.particle_intensity = 0.8

    def update(self, dt: float):
        self.physics_step(dt, gravity=self.gravity, wind=self.wind, turbulence=0.2)

        # 2. Reset at Bottom
        # If Y drops below h_min (or 0.0), wrap to top (h_max)
        reset_mask = self.y < self.config.h_min

        if np.any(reset_mask):
            self.y[reset_mask] = self.config.h_max
            self.x[reset_mask] = np.random.uniform(0.0, 1.0, np.sum(reset_mask))
            self.vy[reset_mask] = 0

    def render(self, buffer, mapper):
        self.render_particles(buffer, mapper, self.color)