import numpy as np
from src.effects.particle_system import ParticleSystem

class SnowEffect(ParticleSystem):
    def __init__(self, config):
        # Init Base System
        super().__init__(config, particle_count=config.flake_count)

        self.color = np.array(config.color, dtype=float)

        # Physics Params
        self.gravity = getattr(config, 'gravity', 0.05)     # Positive = Down
        self.wind = getattr(config, 'wind', 0.02)

        # Override Base Settings
        self.bloom_radius = 0.02
        self.particle_intensity = 0.8 # Snow is soft

    def update(self, dt: float):
        # 1. Run Standard Physics
        self.physics_step(dt, gravity=self.gravity, wind=self.wind, turbulence=0.2)

        # 2. Snow Specific: Reset at Bottom
        # If Y < h_min, move to h_max
        reset_mask = self.y < self.config.h_min

        if np.any(reset_mask):
            self.y[reset_mask] = self.config.h_max
            # Randomize X so it doesn't loop perfectly
            self.x[reset_mask] = np.random.uniform(0.0, 1.0, np.sum(reset_mask))
            # Kill velocity
            self.vy[reset_mask] = 0

    def render(self, buffer, mapper):
        self.render_particles(buffer, mapper, self.color)