# src/effects/particle_system.py
import numpy as np
from src.effects.abstract import Effect

class ParticleSystem(Effect):
    def __init__(self, config, particle_count):
        super().__init__(config)
        self.count = particle_count

        # --- SHARED STATE (World Coordinates) ---
        # Y: 0.0 (Bottom) to 1.0 (Top)
        # X: 0.0 to 1.0 (Azimuth/Rotation)
        self.y = np.random.uniform(config.h_min, config.h_max, self.count)
        self.x = np.random.uniform(0.0, 1.0, self.count)

        # Velocity
        self.vy = np.zeros(self.count)
        self.vx = np.zeros(self.count)

        # Life (0.0 = Dead/Spawn, 1.0 = Max Age) - Useful for Fire fading
        self.life = np.random.uniform(0.0, 1.0, self.count)

        # Rendering Settings (Subclasses can override)
        self.bloom_radius = 0.02
        self.particle_intensity = 1.0  # Brightness multiplier

    def physics_step(self, dt, gravity, wind, turbulence):
        """
        Generic Euler Integration for any particle type
        """
        # 1. Apply Forces
        # Gravity (Vertical acceleration)
        self.vy += gravity * dt

        # Wind + Turbulence (Horizontal acceleration)
        # Brownian motion: Random jitter for every particle
        jitter = np.random.normal(0, turbulence, self.count)
        self.vx += (wind + jitter) * dt

        # 2. Move
        self.y += self.vy * dt
        self.x += self.vx * dt

        # 3. Update Life (Optional, assumes linear aging)
        self.life += dt

        # 4. Cylinder Wrap (X-axis)
        self.x %= 1.0

    def render_particles(self, buffer, mapper, colors):
        """
        Vectorized Renderer: Maps 2D particles to 1D LEDs
        colors: Can be a single color (Snow) or an array of colors (Fire)
        """
        # --- COORDINATE MAPPING ---
        led_y = mapper.coords_y[np.newaxis, :] # (1, 600)
        led_x = mapper.coords_x[np.newaxis, :] # (1, 600)

        p_y = self.y[:, np.newaxis] # (N, 1)
        p_x = self.x[:, np.newaxis] # (N, 1)

        # --- DISTANCE CALCULATION ---
        # Vertical (Scaled by aspect ratio)
        dy = np.abs(led_y - p_y) * mapper.aspect_ratio

        # Horizontal (Cylinder Shortest Path)
        raw_dx = np.abs(led_x - p_x)
        dx = np.minimum(raw_dx, 1.0 - raw_dx)

        # Distance Squared
        dist_sq = (dx**2) + (dy**2)

        # --- RENDER LOGIC ---
        # Find LEDs within radius
        # Shape: (N, 600) boolean mask
        mask = dist_sq < (self.bloom_radius ** 2)

        # Optimization: If colors is a single array (1, 3), expand it
        if colors.ndim == 1:
            # Broadcast to (N, 3)
            current_colors = np.tile(colors, (self.count, 1))
        else:
            current_colors = colors

        # We need to map "Which particle hit this LED?"
        # Since multiple particles can hit one LED, we take the CLOSEST one
        # or simply the LAST one for speed.

        # Fast approach: Iterate particles that are visible
        # (A fully vectorized reduction is complex with variable colors)

        # Let's use the 'any' mask for geometric clip, then iterate for color
        # Or simpler: Just loop through particles? No, too slow in Python.

        # VECTORIZED COLOR BLENDING:
        # We calculate an intensity based on distance for Soft Bloom
        # intensity = 1.0 - (dist / radius)
        # This gives us a nice "Ball" of light

        # Avoid division by zero
        radius = max(self.bloom_radius, 0.001)

        # Valid Impacts (N, 600)
        # We calculate intensity only where mask is True to save time
        # But boolean indexing flattens arrays.

        # Let's simplify: "Nearest Neighbor" coloring.
        # For every LED, find the index of the closest particle
        nearest_idx = np.argmin(dist_sq, axis=0) # (600,)
        min_dist_sq = np.min(dist_sq, axis=0)    # (600,)

        # Check if the closest particle is actually within range
        visible = min_dist_sq < (radius ** 2)

        # Get indices of particles that "won" the pixel
        winning_particles = nearest_idx[visible]

        # Get their colors
        final_colors = current_colors[winning_particles]

        # Apply Bloom Falloff (Optional: dimmer at edges)
        # intensity = 1.0 - (sqrt(min_dist) / radius)
        dists = np.sqrt(min_dist_sq[visible])
        intensity = 1.0 - (dists / radius)
        intensity = np.clip(intensity, 0.0, 1.0)[:, np.newaxis]

        # Write to buffer
        # Note: We ADD to existing buffer to support transparency/mixing
        # But we need to be careful of overflow.

        weighted_colors = (final_colors * intensity * self.particle_intensity)

        # Cast to uint16 to prevent wrap-around, then clip
        target_indices = np.where(visible)[0]

        current_vals = buffer[target_indices].astype(np.uint16)
        new_vals = current_vals + weighted_colors.astype(np.uint16)
        np.clip(new_vals, 0, 255, out=new_vals)

        buffer[target_indices] = new_vals.astype(np.uint8)