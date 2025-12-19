import numpy as np

# --- CONFIGURATION ---
LED_COUNT = 600
PILLAR_WRAPS = 19.4
HEIGHT_INCHES = 48.0
CIRCUMFERENCE_INCHES = 21.0

# Animation
FPS = 60
DURATION_SEC = 20.0 # Longer loop allows for slower background stars
TOTAL_FRAMES = int(FPS * DURATION_SEC)
OUTPUT_FILE = "hyperspace_sparse.npy"

# Star Settings
STAR_COUNT = 60 # Very sparse so individual stars stand out
UNIVERSE_HEIGHT = HEIGHT_INCHES * 1.2

def generate_clip():
    print(f"Generating Sparse Hyperspace ({DURATION_SEC}s)...")

    # 1. Initialize Depth
    # 0.0 = Very Far Away (Background)
    # 1.0 = Very Close (Foreground)
    # We bias distribution so there are more background stars than foreground ones
    depths = np.random.power(2, STAR_COUNT) # Bias towards 1.0? No, let's use uniform for now.
    depths = np.random.uniform(0.1, 1.0, STAR_COUNT)

    # 2. Assign Speeds based on Depth (Parallax)
    # To ensure looping, speed must be an integer number of "Laps"
    # Far stars (Depth 0.1) -> 1 or 2 laps
    # Near stars (Depth 1.0) -> 8 to 12 laps
    min_laps = 1
    max_laps = 12

    laps = (min_laps + (depths * (max_laps - min_laps))).astype(int)

    # Calculate Velocity (Inches per second)
    velocities = (laps * UNIVERSE_HEIGHT) / DURATION_SEC

    # 3. Initial Positions
    start_z = np.random.uniform(0, UNIVERSE_HEIGHT, STAR_COUNT)
    start_theta = np.random.uniform(0, 1, STAR_COUNT)

    # 4. Colors
    colors = np.zeros((STAR_COUNT, 3))
    for i in range(STAR_COUNT):
        # Base Blue-ish White
        c = np.array([0.7, 0.8, 1.0])

        # Rare Colors (15% chance)
        if np.random.random() < 0.15:
            type_r = np.random.choice(['red', 'deep_blue', 'purple'])
            if type_r == 'red': c = [1.0, 0.1, 0.1]
            elif type_r == 'deep_blue': c = [0.1, 0.2, 1.0]
            elif type_r == 'purple': c = [0.9, 0.0, 1.0]

        colors[i] = c

    clip_data = np.zeros((TOTAL_FRAMES, LED_COUNT, 3), dtype=np.uint8)

    # Pre-calc Mapping
    led_indices = np.arange(LED_COUNT)
    led_h_norm = led_indices / LED_COUNT
    led_z = led_h_norm * HEIGHT_INCHES
    led_theta = (led_indices % (LED_COUNT / PILLAR_WRAPS)) / (LED_COUNT / PILLAR_WRAPS)

    dt = 1.0 / FPS

    for f in range(TOTAL_FRAMES):
        if f % 60 == 0: print(f"Frame {f}/{TOTAL_FRAMES}")
        t = f * dt

        # Update Z positions (Falling Down)
        current_z = (start_z - (velocities * t)) % UNIVERSE_HEIGHT

        # Filter: Only render stars currently on the pillar
        # Buffer of 10 inches for trails
        visible_mask = (current_z < (HEIGHT_INCHES + 10.0)) & (current_z > -10.0)
        active_indices = np.where(visible_mask)[0]

        for i in active_indices:
            depth = depths[i]

            # --- DEPTH EFFECTS ---

            # 1. Trail Length
            # Close stars leave long streaks (motion blur)
            # Far stars are just points or short dashes
            trail_len = 0.5 + (depth * 6.0) # 0.5" to 6.5" trails

            # 2. Brightness
            # Far stars are dim (0.2), Near stars are bright (1.0)
            base_brightness = 0.2 + (depth * 0.8)

            # 3. Thickness (Hit Radius)
            # Far stars are sharp points. Near stars are slightly fatter.
            hit_width = 0.01 + (depth * 0.02) # Angular width

            # --- RENDER SEGMENT ---
            z_head = current_z[i]
            z_tail = z_head + trail_len
            theta = start_theta[i]

            # Angular Check
            raw_d_theta = np.abs(led_theta - theta)
            d_theta = np.minimum(raw_d_theta, 1.0 - raw_d_theta)

            # Select LEDs in the column
            col_mask = d_theta < hit_width
            if not np.any(col_mask): continue

            candidate_indices = np.where(col_mask)[0]
            lz = led_z[candidate_indices]

            # Vertical Check (Inside Streak)
            # Since falling down, Head is lower Z, Tail is higher Z
            in_streak = (lz >= z_head) & (lz <= z_tail)
            hit_indices = candidate_indices[in_streak]

            if len(hit_indices) == 0: continue

            # --- GRADIENT TAIL ---
            # Calculate position within streak (0.0 at Head, 1.0 at Tail)
            streak_lz = lz[in_streak]
            pos_in_streak = (streak_lz - z_head) / trail_len

            # Fade out the tail
            fade = 1.0 - pos_in_streak
            fade = np.clip(fade, 0.0, 1.0)

            # Apply all brightness factors
            final_b = base_brightness * fade

            # Color Math
            c = colors[i]
            r = (c[0] * 255 * final_b).astype(int)
            g = (c[1] * 255 * final_b).astype(int)
            b = (c[2] * 255 * final_b).astype(int)

            # Additive Blend
            cur = clip_data[f, hit_indices].astype(int)
            cur[:, 0] += r
            cur[:, 1] += g
            cur[:, 2] += b
            np.clip(cur, 0, 255, out=cur)
            clip_data[f, hit_indices] = cur.astype(np.uint8)

    print(f"Saving {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, clip_data)

if __name__ == "__main__":
    generate_clip()