import numpy as np
import random
from src.effects.abstract import Effect

class FireworkEffect(Effect):
    def __init__(self, config):
        super().__init__(config)

        # --- CONFIGURATION ---
        self.launch_rate = config.launch_rate
        self.launch_timer = 0.0
        self.target_height = config.burst_height

        # --- PARTICLE POOL (The "Memory Bank") ---
        # We allocate 400 particles total.
        # Rockets use 1 slot. Explosions use ~50 slots.
        self.max_particles = 400

        # POSITIONS
        self.y = np.zeros(self.max_particles)
        self.x = np.zeros(self.max_particles)

        # VELOCITIES
        self.vy = np.zeros(self.max_particles)
        self.vx = np.zeros(self.max_particles)

        # ATTRIBUTES
        self.life = np.zeros(self.max_particles)     # 1.0 -> 0.0
        self.state = np.zeros(self.max_particles, dtype=int) # 0=Dead, 1=Rocket, 2=Spark

        # COLORS (R, G, B)
        self.colors = np.zeros((self.max_particles, 3), dtype=np.uint8)

        # TYPE FLAGS (For special rendering like Crackle)
        # 0=Normal, 1=Crackle, 2=Streamer, 3=Shockwave
        self.spark_type = np.zeros(self.max_particles, dtype=int)

    def spawn_rocket(self):
        # Find 1 dead slot
        dead_slots = np.where(self.state == 0)[0]
        if len(dead_slots) < 1: return # Pool full

        idx = dead_slots[0]

        # Initialize Rocket
        self.state[idx] = 1 # Rocket
        self.y[idx] = 0.0   # Start at bottom
        self.x[idx] = np.random.uniform(0.0, 1.0) # Random side
        self.vy[idx] = 0.6  # Fast launch speed
        self.vx[idx] = 0.0
        self.life[idx] = 1.0

        # Color: Dim White/Smoke
        self.colors[idx] = [100, 100, 100]
        self.spark_type[idx] = 0

    def explode(self, parent_idx):
        # The rocket has reached its peak. Time to spawn sparks!

        # 1. Kill the rocket
        self.state[parent_idx] = 0
        origin_x = self.x[parent_idx]
        origin_y = self.y[parent_idx]

        # 2. Choose Explosion Type
        # 0=Starburst, 1=Crackle, 2=Streamer, 3=Shockwave
        exp_type = random.choices([0, 1, 2, 3], weights=[0.4, 0.2, 0.2, 0.2])[0]

        # 3. Choose Color
        # Random vivid color
        base_color = np.random.randint(50, 255, 3)
        # Randomize secondary color slightly for variety

        # 4. Spawn Sparks
        # How many?
        count = 40 if exp_type != 3 else 60

        dead_slots = np.where(self.state == 0)[0]
        if len(dead_slots) < count:
            count = len(dead_slots)

        if count == 0: return

        indices = dead_slots[:count]

        # Set State
        self.state[indices] = 2 # Spark
        self.life[indices] = 1.0
        self.spark_type[indices] = exp_type

        # Set Position (Start at rocket location)
        self.x[indices] = origin_x
        self.y[indices] = origin_y

        # --- EXPLOSION PHYSICS ---

        if exp_type == 3: # Shockwave (Ring)
            # Expand purely horizontally (around the pillar)
            # and slightly vertically
            angle = np.linspace(0, 2*np.pi, count)
            speed = 0.3
            # We map 2D circle to Cylinder surface
            self.vx[indices] = np.cos(angle) * speed
            self.vy[indices] = np.sin(angle) * (speed * 0.2) # Flattened ring
            self.colors[indices] = [255, 255, 255] # White shockwave

        else: # Starburst / Crackle / Streamer
            # Random spherical burst
            self.vx[indices] = np.random.uniform(-0.3, 0.3, count)
            self.vy[indices] = np.random.uniform(-0.3, 0.3, count)
            self.colors[indices] = base_color

            if exp_type == 2: # Streamer
                # Streamers fall faster (heavy)
                self.vy[indices] -= 0.1

    def update(self, dt: float):
        # --- 1. SPAWN LOGIC ---
        self.launch_timer += dt
        if self.launch_timer > (1.0 / self.launch_rate):
            self.spawn_rocket()
            self.launch_timer = 0.0
            # Add randomness to next launch
            self.launch_timer -= np.random.uniform(0.0, 0.5)

        # --- 2. PHYSICS UPDATE (Vectorized) ---
        active = self.state > 0

        # Gravity
        # Rockets (State 1): Slight drag, mostly momentum
        # Sparks (State 2): Heavy gravity
        is_rocket = self.state == 1
        is_spark = self.state == 2

        # Apply Gravity/Drag
        self.vy[is_rocket] -= 0.3 * dt  # Rockets slow down as they rise
        self.vy[is_spark] -= 0.5 * dt   # Sparks fall

        # Apply Drag (Air Resistance)
        self.vx *= (1.0 - (0.5 * dt))
        self.vy *= (1.0 - (0.5 * dt))

        # Move
        self.x[active] += self.vx[active] * dt
        self.y[active] += self.vy[active] * dt

        # Cylinder Wrap X
        self.x %= 1.0

        # --- 3. LIFECYCLE MANAGEMENT ---

        # Age the particles
        self.life[active] -= 0.4 * dt # Fade out speed

        # Kill old particles
        self.state[self.life <= 0] = 0

        # Rocket Detonation Logic
        # If rocket slows down enough (vy < 0.1) or hits target height
        ready_to_blow = (self.state == 1) & (self.y > self.target_height)

        # We need to process detonations one by one to spawn children
        # (This is the only loop, but it runs rarely)
        detonators = np.where(ready_to_blow)[0]
        for idx in detonators:
            self.explode(idx)

    def render(self, buffer, mapper):
        active_indices = np.where(self.state > 0)[0]
        if len(active_indices) == 0: return

        # --- BATCH RENDER ---
        # Map logical coordinates to visual LEDS
        # (Using same Nearest Neighbor logic as Snow for crispness)

        led_y = mapper.coords_y[np.newaxis, :]
        led_x = mapper.coords_x[np.newaxis, :]

        # Filter down to active only
        p_y = self.y[active_indices, np.newaxis]
        p_x = self.x[active_indices, np.newaxis]

        dy = np.abs(led_y - p_y) * mapper.aspect_ratio
        raw_dx = np.abs(led_x - p_x)
        dx = np.minimum(raw_dx, 1.0 - raw_dx)

        dist_sq = (dx**2) + (dy**2)

        # Find closest LEDs
        closest_leds = np.argmin(dist_sq, axis=1)
        min_dists = dist_sq[np.arange(len(active_indices)), closest_leds]

        # Filter valid hits
        valid_mask = min_dists < 0.0015
        final_leds = closest_leds[valid_mask]
        final_indices = active_indices[valid_mask]

        # --- COLOR LOGIC ---
        # 1. Get Base Colors
        colors = self.colors[final_indices].astype(float)

        # 2. Apply Fade (Life)
        life_factors = self.life[final_indices, np.newaxis]
        colors *= (life_factors ** 2) # Exponential fade

        # 3. Apply Special Effects (Crackle)
        # If type == 1 (Crackle), randomly multiply by 0 or 1
        crackle_mask = self.spark_type[final_indices] == 1
        if np.any(crackle_mask):
            # 50% chance to be invisible this frame
            flicker = np.random.choice([0.0, 1.0], size=np.sum(crackle_mask))
            colors[crackle_mask] *= flicker[:, np.newaxis]

        # 4. Write to Buffer (Additive)
        target_indices = final_leds
        current = buffer[target_indices].astype(float)
        new_val = current + colors
        np.clip(new_val, 0, 255, out=new_val)
        buffer[target_indices] = new_val.astype(np.uint8)