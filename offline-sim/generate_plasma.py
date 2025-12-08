import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
LED_COUNT = 600
PILLAR_WRAPS = 19.4
HEIGHT_INCHES = 48.0
CIRCUMFERENCE_INCHES = 21.0
RADIUS_INCHES = CIRCUMFERENCE_INCHES / (2 * np.pi)

# Animation
FPS = 30
DURATION_SEC = 30.0
TOTAL_FRAMES = int(FPS * DURATION_SEC)
OUTPUT_FILE = "plasma_rainbow.npy"

# Volume
MIN_R = RADIUS_INCHES - 2.5
MAX_R = RADIUS_INCHES + 4.5
SAMPLE_POINTS = 70000

# Rendering
HIT_RADIUS_SQ = 0.007
DEPTH_FADE_INCHES = 5.0

# --- COLOR PALETTE ---
def create_palette(n_steps=256):
    # 'nipy_spectral' is great for "Alien Plasma"
    # It goes Black -> Purple -> Blue -> Green -> Yellow -> Red -> White
    cmap = plt.get_cmap('nipy_spectral')
    lut = (cmap(np.linspace(0, 1, n_steps))[:, :3] * 255).astype(np.uint8)
    return lut

PLASMA_LUT = create_palette()

def generate_clip():
    print(f"Generating Rainbow Plasma ({DURATION_SEC}s)...")

    # 1. Initialize Cloud
    r_norm = np.random.power(1.5, SAMPLE_POINTS) # Bias towards outer edge
    p_r = MIN_R + r_norm * (MAX_R - MIN_R)
    p_theta = np.random.uniform(0, 2*np.pi, SAMPLE_POINTS)
    p_z = np.random.uniform(-8.0, HEIGHT_INCHES + 8.0, SAMPLE_POINTS)

    p_x = p_r * np.cos(p_theta)
    p_y = p_r * np.sin(p_theta)

    # Pre-calc map
    clip_data = np.zeros((TOTAL_FRAMES, LED_COUNT, 3), dtype=np.uint8)
    led_indices = np.arange(LED_COUNT)
    led_z_norm = led_indices / LED_COUNT
    led_theta_map = (led_indices % (LED_COUNT / PILLAR_WRAPS)) / (LED_COUNT / PILLAR_WRAPS)
    aspect = HEIGHT_INCHES / CIRCUMFERENCE_INCHES

    # Vectorized Sine Wave Offsets
    offsets = np.random.uniform(0, 100, (3, 3))

    print("Starting Render Loop...")

    for f in range(TOTAL_FRAMES):
        if f % 30 == 0: print(f"Frame {f}/{TOTAL_FRAMES}")

        progress = f / TOTAL_FRAMES
        angle = progress * 2 * np.pi

        # Move noise field in a circle
        tx = np.cos(angle) * 1.5
        ty = np.sin(angle) * 1.5

        # --- DENSITY FIELD ---
        density = np.zeros(SAMPLE_POINTS)

        # Layer 1: Large Structures (The "Body")
        density += np.sin(p_x * 0.2 + offsets[0,0] + tx)
        density += np.sin(p_y * 0.2 + offsets[0,1] + ty)
        density += np.sin(p_z * 0.1 + offsets[0,2])

        # Layer 2: Surface Detail
        density += np.sin(p_x * 0.5 + p_y * 0.5 + offsets[1,0]) * 0.5

        # Range is roughly -3.5 to 3.5.
        # We want to isolate the "peaks" (values > 1.0)

        # Thresholding
        # Values < 1.0 become invisible
        # Values 1.0 -> 2.5 become the visible range 0.0 -> 1.0
        thresh = 1.0
        max_val = 2.5

        norm_val = (density - thresh) / (max_val - thresh)

        visible_mask = (norm_val > 0.001) & (p_z >= 0) & (p_z <= HEIGHT_INCHES)
        indices = np.where(visible_mask)[0]

        if len(indices) == 0: continue

        # --- RENDER VISIBLE POINTS ---
        intensities = norm_val[indices]
        intensities = np.clip(intensities, 0.0, 1.0)

        # Color Mapping
        # Map intensity to LUT index
        lut_indices = (intensities * 255).astype(int)
        colors = PLASMA_LUT[lut_indices].astype(int)

        # Depth Fade
        dists = p_r[indices] - RADIUS_INCHES
        # Allow fading on BOTH sides (inside and outside)
        dist_fade = np.abs(dists)
        fade = 1.0 - (dist_fade / DEPTH_FADE_INCHES)
        fade = np.clip(fade, 0.0, 1.0)

        # Apply fade to color brightness
        colors = (colors * fade[:, np.newaxis]).astype(int)

        # Projection
        nz = p_z[indices] / HEIGHT_INCHES
        nt = (np.arctan2(p_y[indices], p_x[indices]) + np.pi) / (2 * np.pi)

        for i in range(len(indices)):
            if fade[i] < 0.05: continue

            d_z = np.abs(led_z_norm - nz[i]) * aspect
            raw_dt = np.abs(led_theta_map - nt[i])
            d_t = np.minimum(raw_dt, 1.0 - raw_dt)

            dist_sq = d_z**2 + d_t**2
            nearest = np.argmin(dist_sq)

            if dist_sq[nearest] < HIT_RADIUS_SQ:
                # Direct Add with capping
                c = colors[i]
                cur = clip_data[f, nearest].astype(int)
                new_col = np.minimum(cur + c, 255)
                clip_data[f, nearest] = new_col.astype(np.uint8)

    print(f"Saving {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, clip_data)

if __name__ == "__main__":
    generate_clip()