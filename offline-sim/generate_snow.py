import numpy as np
import math

# --- CONFIGURATION (Must match your Pi Setup) ---
LED_COUNT = 600
PILLAR_WRAPS = 18.6
HEIGHT_INCHES = 48.0
CIRCUMFERENCE_INCHES = 21.0
RADIUS_INCHES = CIRCUMFERENCE_INCHES / (2 * np.pi)

# Simulation Settings
FPS = 60
DURATION_SEC = 10.0
TOTAL_FRAMES = int(FPS * DURATION_SEC)
OUTPUT_FILE = "snow_heavy.npy"

# Snow Physics
FLAKE_COUNT = 500
GRAVITY = 9.8  # inches/sec^2
TERMINAL_VELOCITY = 15.0 # inches/sec
DRAG_COEFF = 2.0  # Air resistance factor
WIND_X = 2.0      # Wind blowing "Across" the pillar
DEPTH_FADE = 4.0  # Particles > 4 inches from surface become invisible

def generate_clip():
    print(f"Generating {DURATION_SEC}s of 3D Snow...")

    # 1. Initialize Particles in 3D Space
    # Coordinate System:
    # Z = Vertical Height (0 to HEIGHT_INCHES)
    # X/Y = Horizontal Plane (0,0 is center of pillar)

    # Spawn area: A box slightly larger than the pillar
    spawn_radius = RADIUS_INCHES + DEPTH_FADE

    # Pos: (N, 3) -> [x, y, z]
    pos = np.random.uniform(-spawn_radius, spawn_radius, (FLAKE_COUNT, 3))
    pos[:, 2] = np.random.uniform(0, HEIGHT_INCHES * 1.5, FLAKE_COUNT) # Spread vertically

    # Vel: (N, 3)
    vel = np.zeros((FLAKE_COUNT, 3))

    # The Output Buffer: [Frames, LEDs, 3] (RGB)
    # Using uint8 to save space
    clip_data = np.zeros((TOTAL_FRAMES, LED_COUNT, 3), dtype=np.uint8)

    # Pre-calculate LED positions in 3D space for mapping
    # This allows us to find the "Nearest LED" for any 3D particle
    led_z, led_angle = get_led_cylindrical_coords()

    # Simulation Loop
    dt = 1.0 / FPS

    for f in range(TOTAL_FRAMES):
        if f % 60 == 0: print(f"Rendering Frame {f}/{TOTAL_FRAMES}")

        # --- A. PHYSICS STEP ---
        # 1. Gravity (Down on Z)
        vel[:, 2] -= GRAVITY * dt

        # 2. Wind (Add to X/Y velocity)
        # Simple gust model
        vel[:, 0] += (WIND_X - vel[:, 0]) * 0.5 * dt

        # 3. Air Resistance (Drag)
        # Drag Force = -C * v
        vel -= vel * DRAG_COEFF * dt

        # 4. Integrate Position
        pos += vel * dt

        # 5. Reset Particles that fall below 0
        reset_mask = pos[:, 2] < 0
        if np.any(reset_mask):
            count = np.sum(reset_mask)
            pos[reset_mask, 2] = HEIGHT_INCHES + np.random.uniform(0, 5, count)
            pos[reset_mask, 0] = np.random.uniform(-spawn_radius, spawn_radius, count)
            pos[reset_mask, 1] = np.random.uniform(-spawn_radius, spawn_radius, count)
            vel[reset_mask] = 0

        # --- B. RENDER STEP (Projection) ---

        # 1. Convert Particles to Cylindrical (r, theta, z)
        # r = sqrt(x^2 + y^2)
        p_r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        # theta = arctan2(y, x) -> Normalized to 0..1
        p_theta = (np.arctan2(pos[:, 1], pos[:, 0]) + np.pi) / (2 * np.pi)
        p_z = pos[:, 2]

        # 2. Filter: Only particles "Outside" the pillar surface are visible
        # And closer than max depth
        dist_from_surface = p_r - RADIUS_INCHES
        valid_mask = (dist_from_surface > 0) & (dist_from_surface < DEPTH_FADE) & (p_z >= 0) & (p_z <= HEIGHT_INCHES)

        visible_indices = np.where(valid_mask)[0]

        for i in visible_indices:
            # 3. Map 3D Particle to Nearest LED Index
            # We map p_z (height) and p_theta (angle) to the spiral strip.

            # Normalize Height 0..1
            norm_z = p_z[i] / HEIGHT_INCHES

            # In the spiral map: 
            # X coordinate on strip = theta
            # Y coordinate on strip = z

            # Simple Projection: Find LED with closest (z, theta)
            # Since LEDs are fixed, we can calculate the exact Strip Index
            # Index = (Height_Ratio * Total_LEDs) + (Angle_Offset?)
            # Because it's a spiral, Angle helps determine WHICH wrap we are on.

            # Spiral Math Reversal:
            # The strip wraps PILLAR_WRAPS times.
            # Total "Phase" = norm_z * PILLAR_WRAPS
            # The fractional part of the phase should match p_theta

            # Let's use a "Bucket" approach which is robust:
            # Calculate the idealized LED index for this height/angle

            # Rough loop index
            loop_idx = norm_z * PILLAR_WRAPS

            # We need to find the integer loop number where the fraction matches p_theta
            # But simpler: Map particle to "Strip Space"
            # Strip_Pos = (norm_z * LED_COUNT)
            # But we need to align the angle.

            # Let's brute force the "Nearest LED" for simulation accuracy
            # (Since this is offline, we can afford the math)

            # Distance metric on cylinder surface:
            d_z = np.abs(led_z - norm_z) * (HEIGHT_INCHES / CIRCUMFERENCE_INCHES) # Aspect corrected

            # Angular distance (shortest path around circle)
            raw_d_theta = np.abs(led_angle - p_theta[i])
            d_theta = np.minimum(raw_d_theta, 1.0 - raw_d_theta)

            metric = d_z**2 + d_theta**2
            nearest_led = np.argmin(metric)

            # Check if it's actually close enough to hit the pixel
            if metric[nearest_led] < 0.0005: # Threshold
                # 4. Calculate Depth Dimming
                depth = dist_from_surface[i]
                brightness = 1.0 - (depth / DEPTH_FADE)
                brightness = np.clip(brightness, 0.0, 1.0)

                # Add to pixel (Additive Blending)
                color_val = int(255 * brightness)

                # Assume White Snow [Val, Val, Val]
                # We add to existing value to support overlapping flakes
                current = clip_data[f, nearest_led].astype(int)
                new_val = np.clip(current + color_val, 0, 255)
                clip_data[f, nearest_led] = new_val.astype(np.uint8)

    # Save
    print(f"Saving {OUTPUT_FILE} ({clip_data.nbytes / 1024 / 1024:.2f} MB)...")
    np.save(OUTPUT_FILE, clip_data)
    print("Done!")

def get_led_cylindrical_coords():
    # Helper to generate the (z, theta) map of your physical LEDs
    indices = np.arange(LED_COUNT)
    leds_per_wrap = LED_COUNT / PILLAR_WRAPS

    # Z (0..1)
    led_z = indices / LED_COUNT

    # Theta (0..1)
    led_theta = (indices % leds_per_wrap) / leds_per_wrap

    return led_z, led_theta

if __name__ == "__main__":
    generate_clip()