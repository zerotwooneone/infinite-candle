import numpy as np
from src.effects.abstract import Effect
from src.config import LED_COUNT

class LavaLampEffect(Effect):
    def __init__(self, config):
        super().__init__(config)

        self.blob_color = np.array(config.color, dtype=float)
        self.bg_color = np.array(config.bg_color, dtype=float)

        self.count = config.blob_count
        self.speed_scaler = config.speed

        # --- BLOB PHYSICS STATE ---
        # Y: 0.0 (Bottom) to 1.0 (Top)
        # X: 0.0 to 1.0 (Around cylinder)
        self.y = np.random.uniform(0.0, 1.0, self.count)
        self.x = np.random.uniform(0.0, 1.0, self.count)

        # Temperature: 0.0 (Cold/Heavy) -> 1.0 (Hot/Light)
        # Starts random so they don't all move together
        self.temp = np.random.uniform(0.0, 1.0, self.count)

        # Radii: Variation in blob sizes
        # 0.15 is a good base size for the pillar
        self.radii = np.random.uniform(0.1, 0.2, self.count)

    def update(self, dt: float):
        # 1. Update Temperature based on Height
        # If at bottom (y < 0.1), Heat up (+ temp)
        # If at top (y > 0.9), Cool down (- temp)
        # In the middle, maintain momentum but slowly gravitate towards matching temp

        # Physics model:
        # Target Temp is 1.0 if y < 0.5, else 0.0? 
        # Better: Classic Lava Lamp cycle.

        heating_rate = 0.5 * self.speed_scaler * dt
        cooling_rate = 0.3 * self.speed_scaler * dt

        # Heat/Cool logic
        # We vectorizing the logic:
        # Create a "Target Temperature" based on current state
        # If you are cold and at the bottom, stay there until fully heated.

        for i in range(self.count):
            if self.y[i] <= 0.1:
                self.temp[i] += heating_rate # Heating element at bottom
            elif self.y[i] >= 0.9:
                self.temp[i] -= cooling_rate # Cooling at top

            # Clamp Temp
            self.temp[i] = max(0.0, min(1.0, self.temp[i]))

            # 2. Velocity based on Temperature
            # Hot = Rise (+), Cold = Sink (-)
            # Temp 0.5 is neutral buoyancy
            # Speed range: -0.1 to +0.1
            buoyancy = (self.temp[i] - 0.5) * 0.4 * self.speed_scaler

            self.y[i] += buoyancy * dt

            # 3. Horizontal Drift (Small rotation)
            # Add slight sine wave drift based on height
            drift = np.sin(self.y[i] * 10.0 + i) * 0.05 * self.speed_scaler
            self.x[i] += drift * dt
            self.x[i] %= 1.0 # Wrap X

        # Hard clamp Y to keep them inside the lamp
        np.clip(self.y, 0.0, 1.0, out=self.y)

    def render(self, buffer, mapper):
        # --- METABALL RENDERING ---
        # Field Strength = Sum( Radius^2 / Distance^2 )

        # 1. Coordinates
        led_y = mapper.coords_y[np.newaxis, :] # (1, 600)
        led_x = mapper.coords_x[np.newaxis, :]

        blob_y = self.y[:, np.newaxis] # (6, 1)
        blob_x = self.x[:, np.newaxis]
        blob_r = self.radii[:, np.newaxis]

        # 2. Distance Squared (Vectorized)
        dy = (led_y - blob_y) * mapper.aspect_ratio

        raw_dx = np.abs(led_x - blob_x)
        dx = np.minimum(raw_dx, 1.0 - raw_dx)

        dist_sq = (dx**2) + (dy**2)

        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 0.0001)

        # 3. Calculate Influence
        # Formula: Influence = Radius^2 / Distance^2
        # This creates the "gooey" merging effect
        influence = (blob_r ** 2) / dist_sq

        # Sum influences from all blobs
        total_field = np.sum(influence, axis=0) # (600,)

        # 4. Thresholding (The "Edge" of the lava)
        # If field > 1.0, it's inside the blob.
        # We want a soft edge, so we create a mix factor.
        # Smoothstep logic:
        # Edge between 0.8 (fading in) and 1.2 (solid blob)

        edge_min = 0.5
        edge_max = 1.0

        mix = (total_field - edge_min) / (edge_max - edge_min)
        mix = np.clip(mix, 0.0, 1.0)

        # Reshape for broadcasting
        mix = mix[:, np.newaxis]

        # 5. Blend Colors
        final_color = (self.bg_color * (1.0 - mix)) + (self.blob_color * mix)

        # 6. Apply Layer Opacity (from BaseLayer)
        if hasattr(self.config, 'opacity') and self.config.opacity < 1.0:
            # Read current buffer for blending
            current = buffer.astype(float)
            blended = (current * (1.0 - self.config.opacity)) + (final_color * self.config.opacity)
            buffer[:] = blended.astype(np.uint8)
        else:
            buffer[:] = final_color.astype(np.uint8)