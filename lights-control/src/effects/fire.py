import numpy as np
from src.effects.particle_system import ParticleSystem

class FireEffect(ParticleSystem):
    def __init__(self, config):
        super().__init__(config, particle_count=150)

        # Physics Params
        self.cooling = config.cooling

        # FIX 1: Turn off "Anti-Gravity" acceleration. 
        # We rely on the initial burst (vy) instead.
        self.rise_speed = 0.05  # Very gentle lift, not a rocket booster

        self.c_start = np.array(config.color_start, dtype=float)
        self.c_end = np.array(config.color_end, dtype=float)

        self.bloom_radius = 0.035
        self.particle_intensity = 0.15 # Keep this low to prevent whiteout!

        # Init Particles
        self.y = np.random.uniform(config.h_min, config.h_min + 0.1, self.count)
        self.life = np.random.uniform(0.0, 1.0, self.count)

        # FIX 2: Stronger Initial Burst, but random
        self.vy = np.random.uniform(0.2, 0.8, self.count)

    def update(self, dt: float):
        # 1. Physics Step
        self.physics_step(dt, gravity=self.rise_speed, wind=0.0, turbulence=0.6)

        # FIX 3: Add Air Resistance (Drag)
        # Slow down the vertical speed by 50% every second
        # This prevents the "Flamethrower" rocket effect
        self.vy *= (1.0 - (0.5 * dt))

        # 2. Age (Cooling)
        self.life -= self.cooling * dt

        # 3. Respawn
        dead_mask = (self.life <= 0) | (self.y > self.config.h_max)

        if np.any(dead_mask):
            count = np.sum(dead_mask)
            self.y[dead_mask] = self.config.h_min
            self.x[dead_mask] = np.random.uniform(0.0, 1.0, count)
            self.life[dead_mask] = np.random.uniform(0.8, 1.2, count)

            # FIX 4: Respawn with high velocity (The "Pop")
            self.vy[dead_mask] = np.random.uniform(0.3, 0.8, count)

    def render(self, buffer, mapper):
        # Color Logic (Same as before)
        l = np.clip(self.life, 0.0, 1.0)[:, np.newaxis]
        base_color = (self.c_start * l) + (self.c_end * (1.0 - l))
        fade = l ** 2 # Square law for nicer fade-out
        final_color = base_color * fade

        self.render_particles(buffer, mapper, final_color)