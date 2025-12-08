import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
LED_COUNT = 600
PILLAR_WRAPS = 19.4
HEIGHT_INCHES = 48.0
CIRCUMFERENCE_INCHES = 21.0
RADIUS_INCHES = CIRCUMFERENCE_INCHES / (2 * np.pi)

# Animation Settings
FPS = 30
DURATION_SEC = 30.0
TOTAL_FRAMES = int(FPS * DURATION_SEC)
OUTPUT_FILE = "plasma_loop_v1.npy"

# Plasma Volume Settings
MIN_R = RADIUS_INCHES - 2.0
MAX_R = RADIUS_INCHES + 4.0
SAMPLE_POINTS = 60000 # Increased count because Numpy is fast!

# Rendering Settings
HIT_RADIUS_SQ = 0.006
DEPTH_FADE_INCHES = 5.0

# --- COLOR PALETTE ---
def create_plasma_lut(n_steps=256):
    cmap = plt.get_cmap('plasma')
    lut = (cmap(np.linspace(0, 1, n_steps))[:, :3] * 255).astype(np.uint8)
    return lut

PLASMA_LUT = create_plasma_lut()

def generate_clip():
    print(f"Generating Fast Vectorized Plasma ({DURATION_SEC}s)...")

    # 1. Initialize Static Cloud
    # We use more points for a dense, continuous look
    r_norm = np.random.power(1, SAMPLE_POINTS)
    p_r = MIN_R + r_norm * (MAX_R - MIN_R)
    p_theta = np.random.uniform(0, 2*np.pi, SAMPLE_POINTS)
    p_z = np.random.uniform(-5.0, HEIGHT_INCHES + 5.0, SAMPLE_POINTS)

    # Convert to Cartesian
    p_x = p_r * np.cos(p_theta)
    p_y = p_r * np.sin(p_theta)

    # Pre-calc LED map
    clip_data = np.zeros((TOTAL_FRAMES, LED_COUNT, 3), dtype=np.uint8)
    led_indices = np.arange(LED_COUNT)
    led_z_norm = led_indices / LED_COUNT
    led_theta_map = (led_indices % (LED_COUNT / PILLAR_WRAPS)) / (LED_COUNT / PILLAR_WRAPS)
    aspect = HEIGHT_INCHES / CIRCUMFERENCE_INCHES

    # --- VECTORIZED NOISE FUNCTIONS ---
    # We sum multiple sine waves to create organic "Blobs"
    # This is much faster than Perlin Noise for this many points

    # Random offsets for our layers
    offsets = np.random.uniform(0, 100, (3, 3)) # 3 layers, 3 dimensions
    freqs = [0.15, 0.3, 0.6] # Frequencies (Base, Detail, Fine)
    scales = [1.0, 0.5, 0.25] # Amplitudes

    print("Starting Render Loop...")

    for f in range(TOTAL_FRAMES):
        if f % 60 == 0: print(f"Frame {f}/{TOTAL_FRAMES}")

        # Time Loop: Map linear time to a circle
        progress = f / TOTAL_FRAMES
        angle = progress * 2 * np.pi

        # Time coordinates (looping)
        # We move the noise field in a circle
        tx = np.cos(angle) * 2.0
        ty = np.sin(angle) * 2.0

        # Calculate Density Field (The "Plasma")
        # Start with zero
        density = np.zeros(SAMPLE_POINTS)

        # Layer 1: Big Blobby Shapes
        density += np.sin(p_x * freqs[0] + offsets[0,0] + tx) * scales[0]
        density += np.sin(p_y * freqs[0] + offsets[0,1] + ty) * scales[0]
        density += np.sin(p_z * (freqs[0]*0.5) + offsets[0,2]) * scales[0] # Stretch Z

        # Layer 2: Medium Details
        density += np.sin(p_x * freqs[1] + p_y * freqs[1] + offsets[1,0]) * scales[1]
        density += np.sin(p_z * freqs[1] + tx) * scales[1]

        # Layer 3: Texture
        density += np.sin(p_x * freqs[2] - p_z * freqs[2] + ty) * scales[2]

        # Normalize Density (-2.0 to 2.0) -> (0.0 to 1.0)
        # We want "islands" of plasma, so we threshold high
        # Shift range so peaks are at 1.0
        norm_val = (density + 1.5) / 3.0

        # Threshold: Only show the densest parts
        visible_mask = (norm_val > 0.5) & (p_z >= 0) & (p_z <= HEIGHT_INCHES)

        # Re-normalize visible parts to 0..1 for color lookup
        # (val - 0.5) / 0.5 -> 0..1
        intensities = (norm_val[visible_mask] - 0.5) * 2.0

        indices = np.where(visible_mask)[0]

        if len(indices) == 0: continue

        # --- PROJECTION ---
        # Get coordinates of visible points
        vis_z = p_z[indices]
        vis_x = p_x[indices]
        vis_y = p_y[indices]
        vis_r = p_r[indices]

        # Color Lookup
        lut_indices = (intensities * 255).astype(int)
        lut_indices = np.clip(lut_indices, 0, 255)
        colors = PLASMA_LUT[lut_indices].astype(int) # (N, 3)

        # Depth Fade
        dists = vis_r - RADIUS_INCHES
        fade = 1.0 - (dists / DEPTH_FADE_INCHES)
        fade = np.clip(fade, 0.0, 1.0)

        # Pre-multiply fade into colors
        colors = (colors * fade[:, np.newaxis]).astype(int)

        # Cylindrical Map
        nz = vis_z / HEIGHT_INCHES
        nt = (np.arctan2(vis_y, vis_x) + np.pi) / (2 * np.pi)

        # Optimize: Loop over visible points
        # (Since we have 60k points but maybe only 5k are visible, this is fast)
        for i in range(len(indices)):
            # Skip if dim
            if fade[i] < 0.05: continue

            # Nearest LED Search
            d_z = np.abs(led_z_norm - nz[i]) * aspect
            raw_dt = np.abs(led_theta_map - nt[i])
            d_t = np.minimum(raw_dt, 1.0 - raw_dt)

            dist_sq = d_z**2 + d_t**2
            nearest = np.argmin(dist_sq)

            if dist_sq[nearest] < HIT_RADIUS_SQ:
                # Additive Blend
                c = colors[i]

                # Careful Add
                # We fetch, add, clamp, store
                # Doing it per-channel is safer for overflow in simple loop

                # Get current
                cur = clip_data[f, nearest].astype(int)

                # Blend Factor (Average them out slightly so clumps don't white out instantly)
                # Or just simple add with a cap
                new_col = cur + (c * 0.6).astype(int) # 0.6 opacity per point

                # Clamp 255
                new_col = np.minimum(new_col, 255)

                clip_data[f, nearest] = new_col.astype(np.uint8)

    print(f"Saving {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, clip_data)

if __name__ == "__main__":
    generate_clip()