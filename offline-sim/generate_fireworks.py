import numpy as np

# --- CONFIGURATION ---
LED_COUNT = 600
PILLAR_WRAPS = 19.4
HEIGHT_INCHES = 48.0
CIRCUMFERENCE_INCHES = 21.0
RADIUS_INCHES = CIRCUMFERENCE_INCHES / (2 * np.pi)

# Simulation
FPS = 60
DURATION_SEC = 25.0
TOTAL_FRAMES = int(FPS * DURATION_SEC)
OUTPUT_FILE = "fireworks_show.npy"

# Physics Constants
GRAVITY = 6.0
DRAG_ROCKET = 0.1
DRAG_AIR = 1.5
DRAG_HEAVY = 4.0

class ParticleSystem:
    def __init__(self, max_particles=3000):
        self.state = np.zeros(max_particles, dtype=int)
        self.pos = np.zeros((max_particles, 3))
        self.vel = np.zeros((max_particles, 3))
        self.life = np.zeros(max_particles)
        self.color = np.zeros((max_particles, 3))
        self.meta = np.zeros(max_particles, dtype=int)
        self.phase = np.random.uniform(0, 2*np.pi, max_particles)

    def spawn_rocket(self):
        idx = np.where(self.state == 0)[0]
        if len(idx) == 0: return
        i = idx[0]

        angle = np.random.uniform(0, 2 * np.pi)
        dist = RADIUS_INCHES + np.random.uniform(8.0, 18.0)

        self.state[i] = 1 # Rocket
        self.pos[i] = [np.cos(angle)*dist, np.sin(angle)*dist, 0.0]

        target_h = HEIGHT_INCHES * np.random.uniform(0.7, 1.1)
        launch_speed = np.sqrt(2 * GRAVITY * target_h)

        self.vel[i] = [0, 0, launch_speed]
        self.vel[i, 0] += np.random.uniform(-0.5, 0.5)
        self.vel[i, 1] += np.random.uniform(-0.5, 0.5)

        self.life[i] = 2.0
        self.color[i] = [0.8, 0.8, 0.8]

    def explode(self, parent_idx):
        self.state[parent_idx] = 0
        p_pos = self.pos[parent_idx]

        # 0=Peony, 1=Willow, 2=Saturn, 3=Strobe, 4=Palm
        exp_type = np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.2, 0.2, 0.15, 0.15])

        count = 150
        if exp_type == 2: count = 100
        if exp_type == 4: count = 60

        dead = np.where(self.state == 0)[0]
        if len(dead) < count: count = len(dead)
        if count == 0: return
        idx = dead[:count]

        self.state[idx] = 2 # Star
        self.pos[idx] = p_pos
        self.life[idx] = 1.0
        self.meta[idx] = exp_type

        speed = np.random.uniform(4.0, 9.0)

        if exp_type == 0: # Peony
            rgb = np.random.uniform(0, 1, 3)
            rgb /= np.max(rgb)
            self.color[idx] = rgb
            v = np.random.normal(0, 1, (count, 3))
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            self.vel[idx] = (v / norms) * speed * np.random.uniform(0.1, 1.0, (count, 1))

        elif exp_type == 1: # Willow
            self.color[idx] = [1.0, 0.7, 0.3]
            v = np.random.normal(0, 1, (count, 3))
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            self.vel[idx] = (v / norms) * speed

        elif exp_type == 2: # Saturn
            self.color[idx] = [0.2, 1.0, 0.2]
            n_ring = int(count * 0.7)
            angles = np.linspace(0, 2*np.pi, n_ring)
            self.vel[idx[:n_ring], 0] = np.cos(angles) * speed
            self.vel[idx[:n_ring], 1] = np.sin(angles) * speed
            self.vel[idx[:n_ring], 2] = 0.0

            v = np.random.normal(0, 1, (count - n_ring, 3))
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            self.vel[idx[n_ring:]] = (v / norms) * (speed * 0.3)
            self.color[idx[n_ring:]] = [1.0, 1.0, 1.0]

        elif exp_type == 3: # Strobe
            self.color[idx] = [1.0, 1.0, 1.0]
            v = np.random.normal(0, 1, (count, 3))
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            self.vel[idx] = (v / norms) * (speed * 1.2)

        elif exp_type == 4: # Palm
            self.color[idx] = [1.0, 0.8, 0.1]
            v = np.random.normal(0, 1, (count, 3))
            v[:, 2] = np.abs(v[:, 2]) + 0.5
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            self.vel[idx] = (v / norms) * speed

    def update(self, dt):
        active = self.state > 0
        if not np.any(active): return

        self.vel[active, 2] -= GRAVITY * dt

        drag = np.ones(len(self.state)) * DRAG_AIR
        drag[self.state == 1] = DRAG_ROCKET
        drag[(self.state == 2) & ((self.meta == 1) | (self.meta == 4))] = DRAG_HEAVY

        factor = 1.0 - (drag * dt)
        self.vel *= np.clip(factor[:, np.newaxis], 0.0, 1.0)

        self.pos[active] += self.vel[active] * dt

        rockets = np.where((self.state == 1) & (self.vel[:, 2] < 2.0))[0]
        for r in rockets:
            self.explode(r)

        long_life = (self.meta == 1) | (self.meta == 4)
        self.life[active & ~long_life] -= 0.5 * dt
        self.life[active & long_life] -= 0.3 * dt

        self.state[self.life <= 0] = 0

def generate_clip():
    print(f"Generating Fireworks V2 ({DURATION_SEC}s)...")

    ps = ParticleSystem()
    clip_data = np.zeros((TOTAL_FRAMES, LED_COUNT, 3), dtype=np.uint8)

    led_indices = np.arange(LED_COUNT)
    led_z_norm = led_indices / LED_COUNT
    led_theta = (led_indices % (LED_COUNT / PILLAR_WRAPS)) / (LED_COUNT / PILLAR_WRAPS)

    next_launch = 0.0
    dt = 1.0 / FPS

    for f in range(TOTAL_FRAMES):
        if f % 60 == 0: print(f"Frame {f}/{TOTAL_FRAMES}")
        t = f * dt

        if t >= next_launch:
            ps.spawn_rocket()
            next_launch = t + np.random.uniform(0.5, 1.5)

        ps.update(dt)

        active = np.where(ps.state > 0)[0]
        if len(active) == 0: continue

        p_x = ps.pos[active, 0]
        p_y = ps.pos[active, 1]
        p_z = ps.pos[active, 2]

        p_r = np.sqrt(p_x**2 + p_y**2)
        p_theta = (np.arctan2(p_y, p_x) + np.pi) / (2 * np.pi)

        dist_from_surface = np.abs(p_r - RADIUS_INCHES)
        brightness = 1.0 - (dist_from_surface / 12.0)
        brightness = np.clip(brightness, 0.0, 1.0)
        brightness *= (ps.life[active] ** 2)

        # --- FIX 1: STROBE LOGIC ---
        is_strobe = (ps.meta[active] == 3) & (ps.state[active] == 2)
        if np.any(is_strobe):
            # We must slice phase using [active][is_strobe] to match brightness[is_strobe]
            phases = ps.phase[active][is_strobe]
            flash = np.sin(t * 30.0 + phases)
            brightness[is_strobe] *= (flash > 0.0).astype(float)

        valid_mask = (p_z >= 0) & (p_z <= HEIGHT_INCHES) & (brightness > 0.01)
        valid_indices = np.where(valid_mask)[0]

        for idx in valid_indices:
            real_idx = active[idx]

            nz = p_z[idx] / HEIGHT_INCHES # Fix: use local idx
            nt = p_theta[idx]

            d_z = np.abs(led_z_norm - nz) * (HEIGHT_INCHES / CIRCUMFERENCE_INCHES)
            raw_dt = np.abs(led_theta - nt)
            d_t = np.minimum(raw_dt, 1.0 - raw_dt)

            dist_sq = d_z**2 + d_t**2
            nearest = np.argmin(dist_sq)

            if dist_sq[nearest] < 0.0025:
                b_val = brightness[idx]
                r = int(ps.color[real_idx, 0] * 255 * b_val)
                g = int(ps.color[real_idx, 1] * 255 * b_val)
                b = int(ps.color[real_idx, 2] * 255 * b_val)

                # --- FIX 2: EXPLICIT CAST TO INT TO PREVENT OVERFLOW ---
                cur_r = int(clip_data[f, nearest, 0])
                cur_g = int(clip_data[f, nearest, 1])
                cur_b = int(clip_data[f, nearest, 2])

                clip_data[f, nearest, 0] = min(255, cur_r + r)
                clip_data[f, nearest, 1] = min(255, cur_g + g)
                clip_data[f, nearest, 2] = min(255, cur_b + b)

    print(f"Saving {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, clip_data)

if __name__ == "__main__":
    generate_clip()