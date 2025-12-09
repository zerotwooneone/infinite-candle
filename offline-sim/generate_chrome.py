import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
LED_COUNT = 600
PILLAR_WRAPS = 19.4
HEIGHT_INCHES = 48.0
CIRCUMFERENCE_INCHES = 21.0
RADIUS_INCHES = CIRCUMFERENCE_INCHES / (2 * np.pi)

FPS = 30
DURATION_SEC = 20.0
TOTAL_FRAMES = int(FPS * DURATION_SEC)
OUTPUT_FILE = "chrome_spin.npy"

# --- ENVIRONMENT MAP GENERATION ---
# We create a synthetic "World" to reflect.
# A mix of horizon lines, soft gradients, and bright neon lights.
def generate_environment_map(width=512, height=256):
    env = np.zeros((height, width, 3), dtype=float)

    # 1. Background Gradient (Sunset: Blue top -> Purple mid -> Orange bottom)
    for y in range(height):
        if y < height // 2: # Sky
            t = y / (height // 2)
            # Deep Blue -> Purple
            col = (1-t)*np.array([0,0,0.2]) + t*np.array([0.2,0,0.4])
        else: # Ground
            t = (y - height//2) / (height//2)
            # Orange -> Dark Red
            col = (1-t)*np.array([0.8,0.4,0]) + t*np.array([0.1,0,0])
        env[y, :] = col

    # 2. Horizon Line (Bright White Strip)
    h_mid = height // 2
    env[h_mid-5:h_mid+5, :] += 0.8 # Add bright light

    # 3. Random Neon Lights (Vertical Strips)
    for _ in range(20):
        x = np.random.randint(0, width)
        w = np.random.randint(5, 20)
        h_start = np.random.randint(0, height-50)
        h_len = np.random.randint(20, 100)

        # Random Neon Color
        color = np.random.uniform(0, 1, 3)
        color /= np.max(color) # Max brightness

        # Add light
        # Handle wrap around X
        for i in range(w):
            xi = (x + i) % width
            env[h_start:h_start+h_len, xi] += color * 0.5

    # Clip to valid range
    return np.clip(env, 0.0, 1.0)

def generate_clip():
    print(f"Generating Chrome Spin ({DURATION_SEC}s)...")

    # 1. Create the Environment
    print("Generating Environment Map...")
    env_map = generate_environment_map()
    env_h, env_w, _ = env_map.shape

    # 2. Pre-calculate LED Normals
    # We need to know which direction every LED is facing in 3D space.
    led_indices = np.arange(LED_COUNT)

    # Theta: Angle around the cylinder (0 to 2pi)
    led_theta_norm = (led_indices % (LED_COUNT / PILLAR_WRAPS)) / (LED_COUNT / PILLAR_WRAPS)
    led_theta = led_theta_norm * 2 * np.pi

    # Normal Vector (Nx, Ny) for each LED (pointing out from center)
    # Z-component of normal is 0 for a vertical cylinder wall
    n_x = np.cos(led_theta)
    n_y = np.sin(led_theta)

    clip_data = np.zeros((TOTAL_FRAMES, LED_COUNT, 3), dtype=np.uint8)

    print("Rendering Reflection...")
    for f in range(TOTAL_FRAMES):
        if f % 30 == 0: print(f"Frame {f}/{TOTAL_FRAMES}")

        progress = f / TOTAL_FRAMES

        # We rotate the Cylinder OR rotate the Environment.
        # Rotating env is easier: shift texture coordinates.
        rotation_angle = progress * 2 * np.pi

        # Calculate Reflection Lookup
        # For a cylinder, the reflection vector R depends on the Normal N and View Vector V.
        # Simplified: We map the LED's angle + rotation to the X coordinate of the env map.

        # Angle of the reflection lookup
        look_angle = led_theta + rotation_angle

        # Map angle (0..2pi..) to texture X (0..env_w)
        # Use modulo to wrap
        u = ((look_angle / (2 * np.pi)) % 1.0) * (env_w - 1)

        # Map LED Height to texture Y (0..env_h)
        # LEDs at top reflect sky, LEDs at bottom reflect ground
        led_h_norm = led_indices / LED_COUNT
        # Inverse Y because images are usually top-down
        v = (1.0 - led_h_norm) * (env_h - 1)

        # WOBBLE / DISTORTION (The "Liquid" part)
        # Add sine wave distortion to the texture lookup coordinates
        # This makes the surface look uneven or liquid-like
        u_distort = u + np.sin(v * 0.1 + rotation_angle * 5) * 10.0
        v_distort = v + np.cos(u * 0.05 + rotation_angle * 3) * 5.0

        # Sample the Texture (Nearest Neighbor for speed)
        u_idx = np.clip(u_distort.astype(int) % env_w, 0, env_w - 1)
        v_idx = np.clip(v_distort.astype(int), 0, env_h - 1)

        # Lookup colors
        # (600, 3)
        colors = env_map[v_idx, u_idx]

        # Add simulated "Specular Highlight"
        # If the reflection aligns perfectly with a light source?
        # We simulate this by boosting bright pixels
        colors = colors ** 0.8 # Gamma adjustment to make brights pop

        # Write to buffer
        clip_data[f] = (colors * 255).astype(np.uint8)

    print(f"Saving {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, clip_data)

if __name__ == "__main__":
    generate_clip()