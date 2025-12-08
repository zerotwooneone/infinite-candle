import numpy as np

# --- CONFIGURATION (Must match your Pi Setup) ---
LED_COUNT = 600
PILLAR_WRAPS = 19.4
HEIGHT_INCHES = 48.0
CIRCUMFERENCE_INCHES = 21.0
RADIUS_INCHES = CIRCUMFERENCE_INCHES / (2 * np.pi)

# Simulation Settings
FPS = 60
DURATION_SEC = 10.0
TOTAL_FRAMES = int(FPS * DURATION_SEC)
OUTPUT_FILE = "snow_heavy.npy"

# --- REFINED PHYSICS ---
FLAKE_COUNT = 800  # More flakes for better density
GRAVITY = 5.0      # Reduced gravity (was 9.8)
DRAG_COEFF = 5.0   # High Drag = Feather-like falling
WIND_X = 1.0       # Gentle breeze
DEPTH_FADE = 6.0   # Deeper fade range (particles visible further back)

def generate_clip():
    print(f"Generating {DURATION_SEC}s of Refined 3D Snow...")

    # 1. Spawn Volume
    # We expand the spawn area significantly so we see "Deep" snow
    spawn_radius = RADIUS_INCHES + DEPTH_FADE + 2.0

    # Position: (N, 3) -> [x, y, z]
    pos = np.random.uniform(-spawn_radius, spawn_radius, (FLAKE_COUNT, 3))
    pos[:, 2] = np.random.uniform(0, HEIGHT_INCHES * 1.5, FLAKE_COUNT)

    # Unique "Phase" for each particle (used for turbulence)
    # This ensures every flake wobbles differently
    phases = np.random.uniform(0, 2 * np.pi, FLAKE_COUNT)

    # Velocities start at 0
    vel = np.zeros((FLAKE_COUNT, 3))

    # Output Buffer
    clip_data = np.zeros((TOTAL_FRAMES, LED_COUNT, 3), dtype=np.uint8)

    # Pre-calc LED map
    led_z, led_theta = get_led_cylindrical_coords()

    dt = 1.0 / FPS

    for f in range(TOTAL_FRAMES):
        if f % 60 == 0: print(f"Rendering Frame {f}/{TOTAL_FRAMES}")

        # --- A. PHYSICS STEP ---

        # 1. Gravity (Down on Z)
        vel[:, 2] -= GRAVITY * dt

        # 2. Turbulence / Meander
        # Instead of just constant wind, we add a Sine wave "wobble"
        # based on time and the particle's unique phase.
        # This makes them drift like falling leaves.
        t = f * dt
        wobble_x = np.sin(t * 2.0 + phases) * 2.0 # Amplitude 2.0
        wobble_y = np.cos(t * 1.5 + phases) * 2.0

        # Apply forces to velocity
        # We nudge velocity towards the target wind + wobble
        # "Air Control" factor of 2.0
        vel[:, 0] += ((WIND_X + wobble_x) - vel[:, 0]) * 2.0 * dt
        vel[:, 1] += (wobble_y - vel[:, 1]) * 2.0 * dt

        # 3. Strong Air Resistance (The key to "Floating")
        # Drag Force = -Velocity * Coeff
        vel -= vel * DRAG_COEFF * dt

        # 4. Move
        pos += vel * dt

        # 5. Reset Particles
        reset_mask = pos[:, 2] < -5.0 # Let them fall a bit below 0 before resetting
        if np.any(reset_mask):
            count = np.sum(reset_mask)
            # Spawn high above
            pos[reset_mask, 2] = HEIGHT_INCHES + np.random.uniform(0, 10, count)
            # Random X/Y
            pos[reset_mask, 0] = np.random.uniform(-spawn_radius, spawn_radius, count)
            pos[reset_mask, 1] = np.random.uniform(-spawn_radius, spawn_radius, count)
            # Reset velocity
            vel[reset_mask] = 0

        # --- B. RENDER STEP ---

        # Convert to Cylindrical
        p_r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        p_theta = (np.arctan2(pos[:, 1], pos[:, 0]) + np.pi) / (2 * np.pi)
        p_z = pos[:, 2]

        # Filter Visibilty
        # Must be outside the tube, but within fade range
        dist_from_surface = p_r - RADIUS_INCHES

        # We only care about particles in the valid height range 0..Height
        valid_mask = (dist_from_surface > 0) & (dist_from_surface < DEPTH_FADE) & \
                     (p_z >= 0) & (p_z <= HEIGHT_INCHES)

        visible_indices = np.where(valid_mask)[0]

        if len(visible_indices) == 0: continue

        # --- OPTIMIZED MAPPING ---
        # Instead of brute force loop, let's vectorizing the Nearest Neighbor search.
        # For offline sim, we can just loop over visible particles (e.g. 200) vs 600 LEDs.

        for i in visible_indices:
            # Normalized Height
            norm_z = p_z[i] / HEIGHT_INCHES

            # Distance Calculation (Same as before but tuned)
            # Scale Z diff by Aspect Ratio so the "search circle" is round on the cylinder
            d_z = np.abs(led_z - norm_z) * (HEIGHT_INCHES / CIRCUMFERENCE_INCHES)

            # Angular distance
            raw_d_theta = np.abs(led_theta - p_theta[i])
            d_theta = np.minimum(raw_d_theta, 1.0 - raw_d_theta)

            metric = d_z**2 + d_theta**2

            # Find closest LED
            nearest_led = np.argmin(metric)
            min_dist = metric[nearest_led]

            # Radius of influence for a single flake (how "big" it looks)
            # 0.0003 is roughly a point. 0.001 is a soft blob.
            # We scale this radius based on distance!
            # Far away flakes should appear smaller/sharper.

            depth = dist_from_surface[i]
            # Max brightness drops with distance
            brightness = 1.0 - (depth / DEPTH_FADE)
            brightness = brightness ** 2 # Quadratic falloff looks more realistic

            if min_dist < 0.0005 and brightness > 0.01:
                val = int(255 * brightness)

                # Additive Blend
                current = clip_data[f, nearest_led].astype(int)
                new_val = np.clip(current + val, 0, 255)
                clip_data[f, nearest_led] = new_val.astype(np.uint8)

    print(f"Saving {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, clip_data)
    print("Done!")

def get_led_cylindrical_coords():
    indices = np.arange(LED_COUNT)
    leds_per_wrap = LED_COUNT / PILLAR_WRAPS
    led_z = indices / LED_COUNT
    led_theta = (indices % leds_per_wrap) / leds_per_wrap
    return led_z, led_theta

if __name__ == "__main__":
    generate_clip()