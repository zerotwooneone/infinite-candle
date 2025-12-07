# src/effects/particle_system.py
import numpy as np
from src.effects.abstract import Effect

class ParticleSystem(Effect):
    def __init__(self, config, particle_count):
        super().__init__(config)
        self.count = particle_count

        # --- WORLD STATE ---
        self.y = np.random.uniform(config.h_min, config.h_max, self.count)
        self.x = np.random.uniform(0.0, 1.0, self.count)

        # Internal Momentum (Gravity/Drag affects this)
        self.vy = np.zeros(self.count)
        self.vx = np.zeros(self.count)

        self.life = np.random.uniform(0.0, 1.0, self.count)

        # --- TURBULENCE STATE ---
        # Instead of calculating random jitter every frame (which cancels itself out
        # or accumulates into drift), we pick a random direction and hold it
        # for a few frames.

        # 1. Current Random Vector (Velocity offsets)
        self.turb_vx = np.zeros(self.count)
        self.turb_vy = np.zeros(self.count)

        # 2. Timer for each particle
        # "Change direction every 0.1 to 0.2 seconds"
        self.jitter_interval = 0.15
        self.jitter_clock = np.random.uniform(0, self.jitter_interval, self.count)

        # Rendering Settings
        self.bloom_radius = 0.02
        self.particle_intensity = 1.0

    def physics_step(self, dt, gravity, wind, turbulence):
        """
        dt: Delta Time
        gravity: Acceleration (adds to self.vy)
        wind: Constant Velocity (adds to position)
        turbulence: Random Velocity Magnitude (adds to position)
        """

        # --- 1. Update Internal Physics (Acceleration) ---
        self.vy += gravity * dt
        # Note: We do NOT add wind/turbulence to self.vx/self.vy anymore.
        # Those are momentary external forces, not momentum.

        # --- 2. Update Turbulence State (The "Every N Steps" Logic) ---
        self.jitter_clock -= dt

        # Find particles that need a new direction
        update_mask = self.jitter_clock <= 0

        if np.any(update_mask):
            count = np.sum(update_mask)

            # Reset Timer (Add randomness so they don't sync up)
            self.jitter_clock[update_mask] = self.jitter_interval + np.random.uniform(0, 0.05, count)

            # Generate new random vector (Up/Down/Left/Right)
            # We use uniform distribution centered on 0
            self.turb_vx[update_mask] = np.random.uniform(-turbulence, turbulence, count)
            self.turb_vy[update_mask] = np.random.uniform(-turbulence, turbulence, count)

        # --- 3. Move ---
        # Total X Movement = Internal Momentum + Wind + Random Turbulence
        step_vx = self.vx + wind + self.turb_vx

        # Total Y Movement = Internal Momentum (Gravity) + Random Turbulence
        step_vy = self.vy + self.turb_vy

        self.x += step_vx * dt
        self.y += step_vy * dt

        # --- 4. Cleanup ---
        self.life += dt
        self.x %= 1.0

    def render_particles(self, buffer, mapper, colors):
        # ... (Keep existing render_particles logic exactly as is) ...
        # (If you need me to repost the render function, let me know, 
        # but the physics fix is strictly in the code above)

        # --- COORDINATE MAPPING ---
        led_y = mapper.coords_y[np.newaxis, :]
        led_x = mapper.coords_x[np.newaxis, :]

        p_y = self.y[:, np.newaxis]
        p_x = self.x[:, np.newaxis]

        dy = np.abs(led_y - p_y) * mapper.aspect_ratio
        raw_dx = np.abs(led_x - p_x)
        dx = np.minimum(raw_dx, 1.0 - raw_dx)

        dist_sq = (dx**2) + (dy**2)

        # Render Logic (Using your preferred method - keeping Generic or Crisp depending on file)
        # Since this is the Base Class, we use the Generic one:

        mask = dist_sq < (self.bloom_radius ** 2)

        if colors.ndim == 1:
            current_colors = np.tile(colors, (self.count, 1))
        else:
            current_colors = colors

        nearest_idx = np.argmin(dist_sq, axis=0)
        min_dist_sq = np.min(dist_sq, axis=0)

        visible = min_dist_sq < (self.bloom_radius ** 2)
        winning_particles = nearest_idx[visible]
        final_colors = current_colors[winning_particles]

        radius = max(self.bloom_radius, 0.001)
        dists = np.sqrt(min_dist_sq[visible])
        intensity = 1.0 - (dists / radius)
        intensity = np.clip(intensity, 0.0, 1.0)[:, np.newaxis]

        weighted_colors = (final_colors * intensity * self.particle_intensity)

        target_indices = np.where(visible)[0]
        current_vals = buffer[target_indices].astype(np.uint16)
        new_vals = current_vals + weighted_colors.astype(np.uint16)
        np.clip(new_vals, 0, 255, out=new_vals)
        buffer[target_indices] = new_vals.astype(np.uint8)