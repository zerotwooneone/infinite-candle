# src/effects/fireworks.py
import numpy as np
import random
from src.effects.particle_system import ParticleSystem

class FireworkEffect(ParticleSystem):
    def __init__(self, config):
        # Initialize parent with 400 particles
        super().__init__(config, particle_count=400)

        # --- CONFIGURATION ---
        self.launch_rate = config.launch_rate
        self.launch_timer = 0.0
        self.target_height = config.burst_height

        # --- STATE MANAGEMENT ---
        # 0=Dead, 1=Rocket, 2=Spark
        self.state = np.zeros(self.max_particles, dtype=int)
        self.spark_type = np.zeros(self.max_particles, dtype=int)

        # Colors (N, 3)
        self.colors = np.zeros((self.max_particles, 3), dtype=np.uint8)

    def spawn_rocket(self):
        # Find 1 dead slot
        dead_slots = np.where(self.state == 0)[0]
        if len(dead_slots) < 1: return

        idx = dead_slots[0]

        # Initialize Rocket
        self.state[idx] = 1 # Rocket
        self.y[idx] = 0.0   # Start at bottom
        self.x[idx] = np.random.uniform(0.0, 1.0)
        self.vy[idx] = 0.9  # Fast launch speed
        self.vx[idx] = 0.0
        self.life[idx] = 1.0

        # Color: Dim White
        self.colors[idx] = [100, 100, 100]
        self.spark_type[idx] = 0

    def explode(self, parent_idx):
        # Kill rocket
        self.state[parent_idx] = 0
        origin_x = self.x[parent_idx]
        origin_y = self.y[parent_idx]

        # Choose Explosion
        exp_type = random.choices([0, 1, 2, 3], weights=[0.4, 0.2, 0.2, 0.2])[0]
        base_color = np.random.randint(50, 255, 3)

        # Spawn Sparks
        count = 40 if exp_type != 3 else 60
        dead_slots = np.where(self.state == 0)[0]
        if len(dead_slots) < count: count = len(dead_slots)
        if count == 0: return

        indices = dead_slots[:count]

        # Set State
        self.state[indices] = 2
        self.life[indices] = 1.0
        self.spark_type[indices] = exp_type
        self.x[indices] = origin_x
        self.y[indices] = origin_y

        # Explosion Physics
        if exp_type == 3: # Shockwave
            angle = np.linspace(0, 2*np.pi, count)
            speed = 0.3
            self.vx[indices] = np.cos(angle) * speed
            self.vy[indices] = np.sin(angle) * (speed * 0.2)
            self.colors[indices] = [255, 255, 255]
        else:
            self.vx[indices] = np.random.uniform(-0.3, 0.3, count)
            self.vy[indices] = np.random.uniform(-0.3, 0.3, count)
            self.colors[indices] = base_color
            if exp_type == 2: self.vy[indices] -= 0.1 # Streamer drag

    def update(self, dt: float):
        # --- 1. SPAWN ---
        self.launch_timer += dt
        if self.launch_timer > (1.0 / self.launch_rate):
            self.spawn_rocket()
            self.launch_timer = 0.0
            self.launch_timer -= np.random.uniform(0.0, 0.5)

        # --- 2. PHYSICS ---
        active = self.state > 0
        is_rocket = self.state == 1
        is_spark = self.state == 2

        # Gravity
        self.vy[is_rocket] -= 0.3 * dt
        self.vy[is_spark] -= 0.5 * dt

        # Drag
        self.vx *= (1.0 - (0.5 * dt))
        self.vy *= (1.0 - (0.5 * dt))

        # Move
        self.x[active] += self.vx[active] * dt
        self.y[active] += self.vy[active] * dt
        self.x %= 1.0

        # --- 3. LIFECYCLE ---
        self.life[active] -= 0.4 * dt
        self.state[self.life <= 0] = 0

        # Detonation Check (Height OR Stall)
        ready_to_blow = (self.state == 1) & (
                (self.y > self.target_height) | (self.vy < 0)
        )

        detonators = np.where(ready_to_blow)[0]
        for idx in detonators:
            self.explode(idx)

    def render(self, buffer, mapper):
        active_indices = np.where(self.state > 0)[0]
        if len(active_indices) == 0: return

        # Coordinate Mapping
        led_y = mapper.coords_y[np.newaxis, :]
        led_x = mapper.coords_x[np.newaxis, :]

        p_y = self.y[active_indices, np.newaxis]
        p_x = self.x[active_indices, np.newaxis]

        dy = np.abs(led_y - p_y) * mapper.aspect_ratio
        raw_dx = np.abs(led_x - p_x)
        dx = np.minimum(raw_dx, 1.0 - raw_dx)
        dist_sq = (dx**2) + (dy**2)

        # Find Closest
        closest_leds = np.argmin(dist_sq, axis=1)
        min_dists = dist_sq[np.arange(len(active_indices)), closest_leds]

        # Filter
        valid_mask = min_dists < 0.0015
        final_leds = closest_leds[valid_mask]
        final_indices = active_indices[valid_mask]

        # Colors
        colors = self.colors[final_indices].astype(float)
        life_factors = self.life[final_indices, np.newaxis]
        colors *= (life_factors ** 2)

        # Crackle Effect
        crackle_mask = self.spark_type[final_indices] == 1
        if np.any(crackle_mask):
            flicker = np.random.choice([0.0, 1.0], size=np.sum(crackle_mask))
            colors[crackle_mask] *= flicker[:, np.newaxis]

        # Write
        target_indices = final_leds
        current = buffer[target_indices].astype(float)
        new_val = current + colors
        np.clip(new_val, 0, 255, out=new_val)
        buffer[target_indices] = new_val.astype(np.uint8)