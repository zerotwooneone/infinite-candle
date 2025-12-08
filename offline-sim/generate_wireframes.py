import numpy as np

# --- CONFIGURATION ---
LED_COUNT = 600
PILLAR_WRAPS = 19.4
HEIGHT_INCHES = 48.0
CIRCUMFERENCE_INCHES = 21.0
RADIUS_INCHES = CIRCUMFERENCE_INCHES / (2 * np.pi)

FPS = 30
DURATION_SEC = 24.0 # Must match loop cycle (multiples of 2*pi ideally)
TOTAL_FRAMES = int(FPS * DURATION_SEC)
OUTPUT_FILE = "wireframes_v1.npy"

# Object Size Definition
# A 3x3x3 grid of cubes.
SUB_CUBE_SIZE = 7.0 # Each small cube is 7x7x7 inches
GRID_SPACING = 7.5  # Slightly spaced out

# Colors
COLOR_SURFACE = np.array([0, 100, 180]) # Deep Cyan/Blue
COLOR_EDGE = np.array([255, 160, 0])    # Bright Orange/Gold

# Rendering Tunables
# How "thick" the solid surface feels. Needs to match sub_cube_size roughly.
SURFACE_HIT_RADIUS_SQ = (SUB_CUBE_SIZE / 2.0 * 1.1) ** 2
EDGE_THICKNESS_SQ = 0.005 # Thickness of wireframes on surface

# --- GEOMETRY GENERATION ---

def generate_compound_geometry():
    centers = []
    edges = set() # Use set to avoid duplicate edges between adjacent cubes

    # Generate 3x3x3 grid
    offset = GRID_SPACING * 1.0 # Center the grid around 0,0,0
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                # 1. Center point for surface volumetric rendering
                cx, cy, cz = x * GRID_SPACING, y * GRID_SPACING, z * GRID_SPACING
                centers.append([cx, cy, cz])

                # 2. Generate edges for this sub-cube
                # Define 8 corners relative to center
                s = SUB_CUBE_SIZE / 2.0
                corners = [
                    (cx-s, cy-s, cz-s), (cx+s, cy-s, cz-s), (cx+s, cy+s, cz-s), (cx-s, cy+s, cz-s), # Bottom
                    (cx-s, cy-s, cz+s), (cx+s, cy-s, cz+s), (cx+s, cy+s, cz+s), (cx-s, cy+s, cz+s)  # Top
                ]
                # Indices needed to make a cube
                # (p1_idx, p2_idx)
                local_edge_indices = [
                    (0,1), (1,2), (2,3), (3,0), # Bottom ring
                    (4,5), (5,6), (6,7), (7,4), # Top ring
                    (0,4), (1,5), (2,6), (3,7)  # Pillars
                ]

                for p1_idx, p2_idx in local_edge_indices:
                    # Create sorted tuple of actual 3D coordinates to ensure uniqueness
                    # e.g. edge (A to B) is same as (B to A)
                    p1_tuple = tuple(np.round(corners[p1_idx], 3))
                    p2_tuple = tuple(np.round(corners[p2_idx], 3))
                    edge = tuple(sorted((p1_tuple, p2_tuple)))
                    edges.add(edge)

    return np.array(centers), list(edges)

# --- ROTATION HELPERS ---
def get_rotation_matrix(angle_x, angle_y, angle_z):
    # Combine Rx, Ry, Rz
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    cz, sz = np.cos(angle_z), np.sin(angle_z)

    rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return rz @ ry @ rx

# --- MAIN GENERATOR ---
def generate_clip():
    print(f"Generating Solid Compound Loop ({DURATION_SEC}s)...")
    print("Building geometry...")
    base_centers, base_edges_tuples = generate_compound_geometry()
    print(f"Geometry: {len(base_centers)} sub-cubes, {len(base_edges_tuples)} unique edges.")

    clip_data = np.zeros((TOTAL_FRAMES, LED_COUNT, 3), dtype=np.uint8)

    # --- PRE-CALCULATE LED 3D POSITIONS ---
    # Crucial for volumetric rendering. We need to know where every LED is in real space.
    led_indices = np.arange(LED_COUNT)
    # Normalized Height (0.0 to 1.0)
    led_h_norm = led_indices / LED_COUNT
    # Actual Height Z
    led_z_world = led_h_norm * HEIGHT_INCHES
    # Angle Theta (0.0 to 1.0 representing 0 to 2pi)
    led_theta_norm = (led_indices % (LED_COUNT / PILLAR_WRAPS)) / (LED_COUNT / PILLAR_WRAPS)
    led_theta_rad = led_theta_norm * 2 * np.pi

    # Cartesian coordinates of LEDs on cylinder surface
    led_x_world = RADIUS_INCHES * np.cos(led_theta_rad)
    led_y_world = RADIUS_INCHES * np.sin(led_theta_rad)

    # Stack into (600, 3) array for fast vector math
    led_pos_world = np.stack([led_x_world, led_y_world, led_z_world], axis=1)

    # Pre-calc standard mapping data for edges
    aspect = HEIGHT_INCHES / CIRCUMFERENCE_INCHES

    print("Starting Render Loop...")
    for f in range(TOTAL_FRAMES):
        if f % 10 == 0: print(f"Frame {f}/{TOTAL_FRAMES}")

        # Time progress 0.0 -> 1.0
        t_norm = f / TOTAL_FRAMES
        angle_base = t_norm * 2 * np.pi # 0 to 2pi for perfect looping

        # --- MOVEMENT ANIMATION (Looping Sine/Cos) ---
        # Rotation: Slow tumble on all axes
        rot_matrix = get_rotation_matrix(
            angle_base * 1.0, # X rot
            angle_base * 0.7, # Y rot (different speeds make it look odd)
            angle_base * 0.3  # Z rot
        )

        # Position: Pass through center.
        # Use sin/cos combinations to make a complex looping path through origin.
        # Amplitude needs to be large enough to push it entirely out of the pillar.
        amp = RADIUS_INCHES + (SUB_CUBE_SIZE * 2)
        pos_offset = np.array([
            np.sin(angle_base) * amp, # X Oscillation
            np.cos(angle_base * 2.0) * (amp * 0.5), # Y Oscillation (faster, smaller)
            (HEIGHT_INCHES/2.0) + np.sin(angle_base * 0.5) * (HEIGHT_INCHES*0.4) # Z bob up and down
        ])

        # --- TRANSFORM GEOMETRY ---
        # 1. Transform Centers (Apply Rotation then Translation)
        current_centers = (base_centers @ rot_matrix.T) + pos_offset

        # 2. Transform and Densify Edges
        edge_points = []
        points_per_edge = 40
        for p1_tup, p2_tup in base_edges_tuples:
            # Convert tuple back to array
            p1_local = np.array(p1_tup)
            p2_local = np.array(p2_tup)
            # Transform endpoints
            p1_world = (p1_local @ rot_matrix.T) + pos_offset
            p2_world = (p2_local @ rot_matrix.T) + pos_offset
            # Interpolate
            for i in range(points_per_edge):
                mix = i / (points_per_edge - 1)
                edge_points.append(p1_world * (1-mix) + p2_world * mix)
        current_edge_points = np.array(edge_points)

        # --- RENDER PASS 1: SURFACES (Volumetric) ---
        # For every LED, check distance to nearest sub-cube center.

        # Calculate distances from all LEDs to all Cube Centers
        # (600, 1, 3) - (1, 27, 3) -> Broadcast to (600, 27, 3) differences
        deltas = led_pos_world[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        # Sum of squares along last axis -> (600, 27) squared distances
        dists_sq = np.sum(deltas**2, axis=2)
        # Find distance to nearest center for each LED -> (600,)
        min_dists_sq = np.min(dists_sq, axis=1)

        # Identify LEDs inside the shape
        surface_mask = min_dists_sq < SURFACE_HIT_RADIUS_SQ

        # Apply Base Surface Color
        clip_data[f, surface_mask] = COLOR_SURFACE.astype(np.uint8)

        # --- RENDER PASS 2: EDGES (Point Projection) ---
        # Standard projection of edge points onto surface

        p_x = current_edge_points[:, 0]
        p_y = current_edge_points[:, 1]
        p_z = current_edge_points[:, 2]
        p_r = np.sqrt(p_x**2 + p_y**2)
        p_theta = (np.arctan2(p_y, p_x) + np.pi) / (2 * np.pi)

        dist_from_surface = p_r - RADIUS_INCHES
        # Only render edges that are "in front" or slightly embedded
        valid_mask = (p_z >= 0) & (p_z <= HEIGHT_INCHES) & (dist_from_surface > -2.0)

        # Depth dimming for edges
        brightness = 1.0 - (np.abs(dist_from_surface) / 8.0)
        brightness = np.clip(brightness, 0.0, 1.0)

        valid_indices = np.where(valid_mask)[0]

        for idx in valid_indices:
            nz = p_z[idx] / HEIGHT_INCHES
            nt = p_theta[idx]
            d_z = np.abs(led_h_norm - nz) * aspect
            raw_dt = np.abs(led_theta_norm - nt)
            d_t = np.minimum(raw_dt, 1.0 - raw_dt)
            dist_sq = d_z**2 + d_t**2
            nearest = np.argmin(dist_sq)

            if dist_sq[nearest] < EDGE_THICKNESS_SQ:
                b = brightness[idx]
                # Additive blend Edge color on top of Surface color
                cur = clip_data[f, nearest].astype(int)
                add = (COLOR_EDGE * b).astype(int)
                new_col = np.minimum(cur + add, 255)
                clip_data[f, nearest] = new_col.astype(np.uint8)

    print(f"Saving {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, clip_data)

if __name__ == "__main__":
    generate_clip()