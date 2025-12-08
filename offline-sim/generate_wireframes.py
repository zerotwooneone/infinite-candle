import numpy as np

# --- CONFIGURATION ---
LED_COUNT = 600
PILLAR_WRAPS = 19.4
HEIGHT_INCHES = 48.0
CIRCUMFERENCE_INCHES = 21.0
RADIUS_INCHES = CIRCUMFERENCE_INCHES / (2 * np.pi)

FPS = 30 # Lower FPS is fine for slow-moving shapes
DURATION_SEC = 30.0
TOTAL_FRAMES = int(FPS * DURATION_SEC)
OUTPUT_FILE = "wireframes_v1.npy"

# How thick the wires look. Bigger = chunkier, better for low res.
WIRE_THICKNESS_RADIUS_SQ = 0.004

# --- GEOMETRY DEFINITIONS ---
# Unit size shapes centered at 0,0,0

# CUBE
CUBE_VERTS = np.array([
    [-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1], # Bottom
    [-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1]  # Top
])
CUBE_EDGES = [
    (0,1), (1,2), (2,3), (3,0), # Bottom ring
    (4,5), (5,6), (6,7), (7,4), # Top ring
    (0,4), (1,5), (2,6), (3,7)  # Pillars
]

# OCTAHEDRON (Diamond shape)
OCTA_VERTS = np.array([
    [0,0,-1], # Bottom tip (0)
    [1,0,0], [0,1,0], [-1,0,0], [0,-1,0], # Middle ring (1,2,3,4)
    [0,0,1]   # Top tip (5)
])
OCTA_EDGES = [
    (0,1), (0,2), (0,3), (0,4), # Bottom pyr
    (5,1), (5,2), (5,3), (5,4), # Top pyr
    (1,2), (2,3), (3,4), (4,1)  # Middle ring
]

SHAPE_TYPES = [
    (CUBE_VERTS, CUBE_EDGES, 5.0), # Verts, Edges, Scale size inches
    (OCTA_VERTS, OCTA_EDGES, 7.0),
    (CUBE_VERTS, CUBE_EDGES, 4.0),
]

# --- HELPER: 3D Rotation Matrices ---
def rotate_x(a):
    c = np.cos(a)
    s = np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rotate_y(a):
    c = np.cos(a)
    s = np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rotate_z(a):
    c = np.cos(a)
    s = np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

class WireframeObject:
    def __init__(self, shape_idx):
        verts, edges, scale = SHAPE_TYPES[shape_idx]
        self.local_verts = verts * scale
        self.edges = edges

        # Orbit parameters
        self.orbit_radius = RADIUS_INCHES + np.random.uniform(4.0, 10.0)
        self.orbit_speed = np.random.uniform(0.2, 0.6) * np.random.choice([-1, 1])
        self.orbit_height = np.random.uniform(HEIGHT_INCHES*0.2, HEIGHT_INCHES*0.8)
        self.orbit_phase = np.random.uniform(0, 2*np.pi)

        # Self-tumbling parameters (rotation speeds around X, Y, Z axes)
        self.tumble_speeds = np.random.uniform(0.5, 2.0, 3)

        # Color (Neon palette)
        colors = [
            [0, 1, 1], # Cyan
            [1, 0, 1], # Magenta
            [1, 1, 0], # Yellow
            [0, 1, 0.5], # Lime
            [1, 0.5, 0]  # Orange
        ]
        self.color = np.array(colors[shape_idx % len(colors)])

    def get_world_points(self, t):
        # 1. Calculate Tumble Rotation Matrix
        rx = rotate_x(t * self.tumble_speeds[0])
        ry = rotate_y(t * self.tumble_speeds[1])
        rz = rotate_z(t * self.tumble_speeds[2])
        # Combine rotations (order matters, but random tumbling is fine)
        rot_matrix = rz @ ry @ rx

        # 2. Calculate Orbit Position (Center of shape)
        angle = self.orbit_phase + (t * self.orbit_speed)
        orbit_pos = np.array([
            np.cos(angle) * self.orbit_radius,
            np.sin(angle) * self.orbit_radius,
            self.orbit_height + np.sin(t + self.orbit_phase)*5.0 # Add subtle vertical bob
        ])

        # 3. Transform vertices to World Space
        # Rotate then Translate
        world_verts = (self.local_verts @ rot_matrix.T) + orbit_pos

        # 4. Densify Edges into Points
        points = []
        points_per_edge = 50 # High density for smooth lines

        for v_start_idx, v_end_idx in self.edges:
            p1 = world_verts[v_start_idx]
            p2 = world_verts[v_end_idx]

            # Linear interpolation between p1 and p2
            for i in range(points_per_edge):
                mix = i / (points_per_edge - 1)
                p = p1 * (1-mix) + p2 * mix
                points.append(p)

        return np.array(points)

def generate_clip():
    print(f"Generating Wireframes ({DURATION_SEC}s)...")

    # Create a few distinct objects
    objects = []
    num_shapes = 5
    for i in range(num_shapes):
        # Cycle through shape types
        objects.append(WireframeObject(i % len(SHAPE_TYPES)))

    clip_data = np.zeros((TOTAL_FRAMES, LED_COUNT, 3), dtype=np.uint8)

    # Pre-calc mapping data
    led_indices = np.arange(LED_COUNT)
    led_z_norm = led_indices / LED_COUNT
    led_theta = (led_indices % (LED_COUNT / PILLAR_WRAPS)) / (LED_COUNT / PILLAR_WRAPS)
    aspect = HEIGHT_INCHES / CIRCUMFERENCE_INCHES

    dt = 1.0 / FPS

    for f in range(TOTAL_FRAMES):
        if f % 30 == 0: print(f"Frame {f}/{TOTAL_FRAMES}")
        t = f * dt

        # Collect all points from all objects for this frame
        all_points = []
        all_colors = []
        for obj in objects:
            pts = obj.get_world_points(t)
            all_points.append(pts)
            # Repeat color for every point in this object
            all_colors.append(np.tile(obj.color, (len(pts), 1)))

        # Combine into one big list of render points
        render_pos = np.vstack(all_points)
        render_col = np.vstack(all_colors)

        # --- RENDERER ---
        # (Standard projection logic from previous scripts)

        p_x = render_pos[:, 0]
        p_y = render_pos[:, 1]
        p_z = render_pos[:, 2]

        p_r = np.sqrt(p_x**2 + p_y**2)
        p_theta = (np.arctan2(p_y, p_x) + np.pi) / (2 * np.pi)

        # Depth Fade (shapes further out are dimmer)
        dist_from_surface = p_r - RADIUS_INCHES
        brightness = 1.0 - (dist_from_surface / 15.0) # 15 inch fade range
        brightness = np.clip(brightness, 0.0, 1.0)

        # Filter
        valid_mask = (p_z >= 0) & (p_z <= HEIGHT_INCHES) & (dist_from_surface > 0)
        valid_indices = np.where(valid_mask)[0]

        # Optimized Nearest Neighbor Loop
        for idx in valid_indices:
            nz = p_z[idx] / HEIGHT_INCHES
            nt = p_theta[idx]

            d_z = np.abs(led_z_norm - nz) * aspect
            raw_dt = np.abs(led_theta - nt)
            d_t = np.minimum(raw_dt, 1.0 - raw_dt)

            dist_sq = d_z**2 + d_t**2
            nearest = np.argmin(dist_sq)

            # THE KEY TO LOW RES: Large hit radius
            if dist_sq[nearest] < WIRE_THICKNESS_RADIUS_SQ:
                b_val = brightness[idx]
                # Additive Color Mixing
                r = int(render_col[idx, 0] * 255 * b_val)
                g = int(render_col[idx, 1] * 255 * b_val)
                b = int(render_col[idx, 2] * 255 * b_val)

                # Explicit cast and clamp to avoid overflow
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