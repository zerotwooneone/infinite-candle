import numpy as np
from src.effects.particle_system import ParticleSystem

class FireEffect(ParticleSystem):
    def __init__(self, config):
        # Fire needs LOTS of particles to look good
        super().__init__(config, particle_count=150)

        # Physics Params
        self.cooling = config.cooling

        # Physics: Positive means UP
        self.rise_speed = 0.5

        # Color Gradients
        self.c_start = np.array(config.color_start, dtype=float)
        self.c_end = np.array(config.color_end, dtype=float)

        self.bloom_radius = 0.035
        self.particle_intensity = 1.0

        # Initialize particles
        self.y = np.random.uniform(config.h_min, config.h_min + 0.05, self.count)
        self.life = np.random.uniform(0.0, 1.0, self.count)
        self.vy = np.random.uniform(0.1, 0.4, self.count)

    def update(self, dt: float):
        # 1. Run Physics (Positive Gravity = Accelerate Up)
        self.physics_step(dt, gravity=self.rise_speed, wind=0.0, turbulence=0.5)

        # 2. Age the particles
        self.life -= self.cooling * dt

        # 3. Respawn Logic
        # Die if Life < 0 OR Height > h_max
        dead_mask = (self.life <= 0) | (self.y > self.config.h_max)

        if np.any(dead_mask):
            count = np.sum(dead_mask)
            # Respawn at h_min
            self.y[dead_mask] = self.config.h_min
            self.x[dead_mask] = np.random.uniform(0.0, 1.0, count)
            # Reset Life
            self.life[dead_mask] = np.random.uniform(0.8, 1.2, count)
            # Reset Velocity
            self.vy[dead_mask] = np.random.uniform(0.1, 0.5, count)

    def render(self, buffer, mapper):
        # --- DYNAMIC COLOR & BRIGHTNESS CALCULATION ---

        # 1. Clip Life to 0.0-1.0
        l = np.clip(self.life, 0.0, 1.0)[:, np.newaxis]

        # 2. Calculate Base Color (Interpolate Start -> End)
        # Note: You might want to flip this if you want Yellow(Start) -> Red(End)
        # Currently: Start(Red) -> End(Yellow)
        base_color = (self.c_start * l) + (self.c_end * (1.0 - l))

        # 3. Calculate Fade (Brightness)
        # We square the life (l^2) to make the fade happen faster at the end
        # This prevents "Ghost particles" that are barely visible for too long
        fade = l ** 2

        # 4. Apply Fade to Color
        final_color = base_color * fade

        self.render_particles(buffer, mapper, final_color)