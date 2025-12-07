import numpy as np
from src.effects.particle_system import ParticleSystem

class FireEffect(ParticleSystem):
    def __init__(self, config):
        # Fire needs LOTS of particles to look good
        super().__init__(config, particle_count=150)

        # Physics Params
        self.cooling = config.cooling  # Used for life decay
        self.rise_speed = -0.3         # Negative Gravity (Up)

        # Color Gradients
        self.c_start = np.array(config.color_start, dtype=float)
        self.c_end = np.array(config.color_end, dtype=float)

        # Override Base Settings
        self.bloom_radius = 0.035      # Fire is "glowing" and wider than snow
        self.particle_intensity = 1.0

        # Initialize particles at bottom
        self.y = np.random.uniform(config.h_min, config.h_min + 0.05, self.count)
        self.life = np.random.uniform(0.0, 1.0, self.count)

    def update(self, dt: float):
        # 1. Run Physics (Negative Gravity = Rise)
        self.physics_step(dt, gravity=self.rise_speed, wind=0.0, turbulence=0.5)

        # 2. Age the particles
        # Cooling determines how fast they die
        self.life -= self.cooling * dt

        # 3. Fire Specific: Respawn Logic
        # Die if Life < 0 OR Height > h_max
        dead_mask = (self.life <= 0) | (self.y > self.config.h_max)

        if np.any(dead_mask):
            count = np.sum(dead_mask)
            # Respawn at bottom
            self.y[dead_mask] = self.config.h_min
            self.x[dead_mask] = np.random.uniform(0.0, 1.0, count)
            # Reset Life to 1.0
            self.life[dead_mask] = np.random.uniform(0.8, 1.2, count)
            # Reset Velocity (Explosive start)
            self.vy[dead_mask] = np.random.uniform(-0.1, -0.5, count) # Upward burst

    def render(self, buffer, mapper):
        # --- DYNAMIC COLOR CALCULATION ---
        # Map 'Life' (0.0 to 1.0) to Color (End -> Start)
        # Life 1.0 = Start (Red), Life 0.0 = End (Yellow/Smoke)

        # Prepare color array (N, 3)
        # Linear Interpolation: start * life + end * (1-life)
        # Reshape life for broadcasting
        l = np.clip(self.life, 0.0, 1.0)[:, np.newaxis]

        particle_colors = (self.c_start * l) + (self.c_end * (1.0 - l))

        self.render_particles(buffer, mapper, particle_colors)