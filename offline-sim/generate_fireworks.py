import numpy as np

# --- CONFIGURATION ---
LED_COUNT = 600
PILLAR_WRAPS = 18.6
HEIGHT_INCHES = 48.0
CIRCUMFERENCE_INCHES = 21.0
RADIUS_INCHES = CIRCUMFERENCE_INCHES / (2 * np.pi)

# Simulation
FPS = 60
DURATION_SEC = 20.0 # Longer clip to allow for pacing
TOTAL_FRAMES = int(FPS * DURATION_SEC)
OUTPUT_FILE = "fireworks_show.npy"

# Physics Constants
GRAVITY = 6.0        # inches/sec^2
DRAG_AIR = 1.5       # Standard drag
DRAG_WILLOW = 4.0    # High drag for "hanging" effect

class ParticleSystem:
    def __init__(self, max_particles=2000):
        # State: 0=Dead, 1=Rocket, 2=Star
        self.state = np.zeros(max_particles, dtype=int)

        # Positions (x, y, z)
        self.pos = np.zeros((max_particles, 3))

        # Velocities (vx, vy, vz)
        self.vel = np.zeros((max_particles, 3))

        # Life (1.0 -> 0.0)
        self.life = np.zeros(max_particles)

        # Colors (R, G, B) - Float 0-1 for easier blending
        self.color = np.zeros((max_particles, 3))

        # Metadata (Type of explosion, etc)
        self.meta = np.zeros(max_particles)

    def spawn_rocket(self):
        # Find a dead slot
        idx = np.where(self.state == 0)[0]
        if len(idx) == 0: return
        i = idx[0]

        # Launch Parameters
        # Launch from ground (Z=0) but at a random distance OUTSIDE the pillar
        # This makes them fly "up and past" the display surface
        angle = np.random.uniform(0, 2 * np.pi)
        dist = RADIUS_INCHES + np.random.uniform(5.0, 15.0) # 5-15 inches away

        self.state[i] = 1 # Rocket
        self.pos[i] = [np.cos(angle)*dist, np.sin(angle)*dist, 0.0]

        # Aim slightly inwards or outwards? Vertical is best.
        # Launch Speed: Enough to reach 70-90% height
        # Speed ~ sqrt(2 * g * h)
        target_h = HEIGHT_INCHES * np.random.uniform(0.6, 0.95)
        launch_speed = np.sqrt(2 * GRAVITY * target_h)

        self.vel[i] = [0, 0, launch_speed]

        # Add slight wobble to launch
        self.vel[i, 0] += np.random.uniform(-1, 1)
        self.vel[i, 1] += np.random.uniform(-1, 1)

        self.life[i] = 1.0
        self.color[i] = [1.0, 0.8, 0.5] # White/Gold trail

    def explode(self, parent_idx):
        # Kill Rocket
        self.state[parent_idx] = 0
        p_pos = self.pos[parent_idx]

        # Pick Type
        # 0=Peony (Colorful, Sphere)
        # 1=Willow (Gold, High Drag, Falls)
        # 2=Ring (2D Circle)
        exp_type = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])

        # How many stars?
        count = 100 if exp_type != 2 else 60

        # Find slots
        dead = np.where(self.state == 0)[0]
        if len(dead) < count: count = len(dead)
        if count == 0: return

        idx = dead[:count]

        self.state[idx] = 2 # Star
        self.pos[idx] = p_pos
        self.life[idx] = 1.0
        self.meta[idx] = exp_type

        # Set Colors
        if exp_type == 1: # Willow (Gold)
            self.color[idx] = [1.0, 0.6, 0.2]
        else: # Random vivid color
            base = np.random.uniform(0, 1, 3)
            # Make it vibrant (normalize max to 1.0)
            base /= np.max(base)
            self.color[idx] = base

        # Set Velocities
        speed = np.random.uniform(3.0, 8.0) # Explosion force

        if exp_type == 2: # Ring
            # 2D circle on XY plane (horizontal ring)
            angles = np.linspace(0, 2*np.pi, count)
            self.vel[idx, 0] = np.cos(angles) * speed
            self.vel[idx, 1] = np.sin(angles) * speed
            self.vel[idx, 2] = 0.0 # Flat
        else:
            # Sphere
            # Random uniform vectors on sphere
            # Gaussian approach
            v = np.random.normal(0, 1, (count, 3))
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            v = (v / norms) * speed

            # Add randomness to speed for Peony (filled sphere)
            if exp_type == 0:
                v *= np.random.uniform(0.2, 1.0, (count, 1))

            self.vel[idx] = v

    def update(self, dt):
        active = self.state > 0
        if not np.any(active): return

        # Gravity
        self.vel[active, 2] -= GRAVITY * dt

        # Drag (Variable based on type)
        # Default drag
        drag_factors = np.ones(len(self.state)) * DRAG_AIR

        # Willow stars (Type 1) get higher drag
        willows = (self.state == 2) & (self.meta == 1)
        drag_factors[willows] = DRAG_WILLOW

        # Apply Drag: v -= v * drag * dt
        # Vectorized scaling
        factor = 1.0 - (drag_factors[:, np.newaxis] * dt)
        self.vel *= np.clip(factor, 0.0, 1.0) # Prevent negative velocity flip

        # Move
        self.pos[active] += self.vel[active] * dt

        # Lifecycle
        # Rockets: Explode at peak (vz < 0)
        rockets = np.where((self.state == 1) & (self.vel[:, 2] < 1.0))[0]
        for r in rockets:
            self.explode(r)

        # Stars: Fade out
        self.life[active] -= 0.5 * dt # 2 seconds life

        # Kill dead
        self.state[self.life <= 0] = 0

def generate_clip():
    print(f"Generating Fireworks Show ({DURATION_SEC}s)...")

    ps = ParticleSystem()
    clip_data = np.zeros((TOTAL_FRAMES, LED_COUNT, 3), dtype=np.uint8)

    # Pre-calc map
    led_indices = np.arange(LED_COUNT)
    led_z_norm = led_indices / LED_COUNT
    led_theta = (led_indices % (LED_COUNT / PILLAR_WRAPS)) / (LED_COUNT / PILLAR_WRAPS)

    # Timeline
    next_launch = 0.0
    dt = 1.0 / FPS

    for f in range(TOTAL_FRAMES):
        if f % 60 == 0: print(f"Frame {f}/{TOTAL_FRAMES}")
        t = f * dt

        # Launch Logic
        if t >= next_launch:
            ps.spawn_rocket()
            # Random interval 0.5 to 2.0 seconds
            next_launch = t + np.random.uniform(0.5, 2.0)

        # Physics Step
        ps.update(dt)

        # --- RENDER ---
        active = np.where(ps.state > 0)[0]
        if len(active) == 0: continue

        # Convert Particle Pos to Cylindrical Coords relative to Pillar Center
        # r, theta, z
        p_x = ps.pos[active, 0]
        p_y = ps.pos[active, 1]
        p_z = ps.pos[active, 2]

        p_r = np.sqrt(p_x**2 + p_y**2)
        p_theta = (np.arctan2(p_y, p_x) + np.pi) / (2 * np.pi) # 0..1

        # Brightness based on distance (Depth Fade)
        # Closer to pillar = Brighter. Far away = Dimmer.
        dist_from_surface = np.abs(p_r - RADIUS_INCHES)

        # 10 inches is the max visibility depth
        brightness = 1.0 - (dist_from_surface / 10.0)
        brightness = np.clip(brightness, 0.0, 1.0)

        # Multiply by Life (fade out)
        brightness *= ps.life[active]
        brightness = brightness ** 2 # Gamma curve

        # Vectorized Mapping to LEDs
        # We check every active particle against every LED? Too slow (2000 * 600).
        # Optimization: Loop over particles, find nearest LED (Forward Projection)

        # Filter valid particles (Must be Z 0..48)
        valid_mask = (p_z >= 0) & (p_z <= HEIGHT_INCHES) & (brightness > 0.01)
        valid_indices = np.where(valid_mask)[0] # Indices into 'active' array

        for idx in valid_indices:
            real_idx = active[idx] # Index into main PS arrays (for Color)

            # FIX: Use 'idx' (local index) for p_z and p_theta arrays
            nz = p_z[idx] / HEIGHT_INCHES
            nt = p_theta[idx]

            # Calculate distance to nearest LED logic
            d_z = np.abs(led_z_norm - nz) * (HEIGHT_INCHES / CIRCUMFERENCE_INCHES)
            raw_dt = np.abs(led_theta - nt)
            d_t = np.minimum(raw_dt, 1.0 - raw_dt)

            dist_sq = d_z**2 + d_t**2
            nearest = np.argmin(dist_sq)

            if dist_sq[nearest] < 0.001: # Hit
                # Color * Brightness
                r = int(ps.color[real_idx, 0] * 255 * brightness[idx])
                g = int(ps.color[real_idx, 1] * 255 * brightness[idx])
                b = int(ps.color[real_idx, 2] * 255 * brightness[idx])

                # Additive Blend
                clip_data[f, nearest, 0] = min(255, clip_data[f, nearest, 0] + r)
                clip_data[f, nearest, 1] = min(255, clip_data[f, nearest, 1] + g)
                clip_data[f, nearest, 2] = min(255, clip_data[f, nearest, 2] + b)

    print(f"Saving {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, clip_data)

if __name__ == "__main__":
    generate_clip()