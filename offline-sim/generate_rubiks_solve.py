import numpy as np

LED_COUNT = 600
PILLAR_WRAPS = 19.4
HEIGHT_INCHES = 48.0
CIRCUMFERENCE_INCHES = 21.0
RADIUS_INCHES = CIRCUMFERENCE_INCHES / (2 * np.pi)

FPS = 60
MOVE_FRAMES = 18
PAUSE_FRAMES = 6
OUTPUT_FILE = "rubiks_solve_loop.npy"

CUBE_SIZE = 16.0
CUBIE_SIZE = CUBE_SIZE / 3.0
STICKER_MARGIN = 0.15

ORBIT_RADIUS_INCHES = RADIUS_INCHES + 9.0
ORBIT_Z_CENTER = HEIGHT_INCHES * 0.55
ORBIT_Z_AMP = HEIGHT_INCHES * 0.18

CUBE_CENTER_THETA_RAD = 0.0

CUBE_SCALE_XY = 0.70
CUBE_SCALE_Z = 1.55

CUBE_SPIN_Z_TURNS_PER_LOOP = 0.0

CUBE_CLEARANCE_INCHES = 1.25

AA_SAMPLES_THETA = 3
AA_SAMPLES_Z = 3

LED_FOV_THETA_MULT = 1.6
LED_FOV_Z_MULT = 1.6

GAMMA = 2.2

COLOR_U = np.array([255, 255, 255], dtype=np.uint8)
COLOR_D = np.array([255, 255, 0], dtype=np.uint8)
COLOR_L = np.array([255, 140, 0], dtype=np.uint8)
COLOR_R = np.array([255, 0, 0], dtype=np.uint8)
COLOR_F = np.array([0, 200, 0], dtype=np.uint8)
COLOR_B = np.array([0, 70, 255], dtype=np.uint8)
COLOR_NONE = None


def _rot(axis: str, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    raise ValueError(axis)


def _axis_and_layer(move: str):
    m = move[0]
    prime = move.endswith("'")
    if m == "U":
        return "z", 1, -1 if not prime else 1
    if m == "D":
        return "z", -1, 1 if not prime else -1
    if m == "R":
        return "x", 1, -1 if not prime else 1
    if m == "L":
        return "x", -1, 1 if not prime else -1
    if m == "F":
        return "y", 1, -1 if not prime else 1
    if m == "B":
        return "y", -1, 1 if not prime else -1
    raise ValueError(move)


def _rotate_pos_90(pos: np.ndarray, axis: str, direction: int) -> np.ndarray:
    angle = direction * (np.pi / 2.0)
    r = _rot(axis, angle)
    p = r @ pos.astype(float)
    return np.rint(p).astype(int)


def _outside_face_color(local_pos: np.ndarray, face: str):
    x, y, z = local_pos.tolist()
    if face == "+x":
        return COLOR_R if x == 1 else COLOR_NONE
    if face == "-x":
        return COLOR_L if x == -1 else COLOR_NONE
    if face == "+y":
        return COLOR_F if y == 1 else COLOR_NONE
    if face == "-y":
        return COLOR_B if y == -1 else COLOR_NONE
    if face == "+z":
        return COLOR_U if z == 1 else COLOR_NONE
    if face == "-z":
        return COLOR_D if z == -1 else COLOR_NONE
    raise ValueError(face)


_FACE_NORMALS_LOCAL = {
    "+x": np.array([1, 0, 0], dtype=int),
    "-x": np.array([-1, 0, 0], dtype=int),
    "+y": np.array([0, 1, 0], dtype=int),
    "-y": np.array([0, -1, 0], dtype=int),
    "+z": np.array([0, 0, 1], dtype=int),
    "-z": np.array([0, 0, -1], dtype=int),
}


_FACE_UV_LOCAL = {
    "+x": (np.array([0, 1, 0], dtype=float), np.array([0, 0, 1], dtype=float)),
    "-x": (np.array([0, 1, 0], dtype=float), np.array([0, 0, -1], dtype=float)),
    "+y": (np.array([1, 0, 0], dtype=float), np.array([0, 0, 1], dtype=float)),
    "-y": (np.array([1, 0, 0], dtype=float), np.array([0, 0, -1], dtype=float)),
    "+z": (np.array([1, 0, 0], dtype=float), np.array([0, 1, 0], dtype=float)),
    "-z": (np.array([1, 0, 0], dtype=float), np.array([0, -1, 0], dtype=float)),
}


class Cubie:
    def __init__(self, pos: np.ndarray):
        self.pos = pos.astype(int)
        self.orient = np.eye(3, dtype=float)
        self.stickers = {}
        for face in _FACE_NORMALS_LOCAL:
            self.stickers[face] = _outside_face_color(self.pos, face)


def _make_cube():
    cubies = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                if x == 0 and y == 0 and z == 0:
                    continue
                cubies.append(Cubie(np.array([x, y, z], dtype=int)))
    return cubies


def _select_slice(cubies, axis: str, layer: int):
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    return [c for c in cubies if c.pos[idx] == layer]


def _apply_move_discrete(cubies, move: str):
    axis, layer, direction = _axis_and_layer(move)
    slice_cubies = _select_slice(cubies, axis, layer)
    r = _rot(axis, direction * (np.pi / 2.0))
    for c in slice_cubies:
        c.pos = _rotate_pos_90(c.pos, axis, direction)
        c.orient = r @ c.orient


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    x = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    return x**GAMMA


def _linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    x = np.clip(lin, 0.0, 1.0)
    x = x ** (1.0 / GAMMA)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _ray_box_intersect(origin: np.ndarray, direction: np.ndarray, half_size: float):
    inv_d = 1.0 / np.where(np.abs(direction) < 1e-9, 1e-9, direction)
    t0 = (-half_size - origin) * inv_d
    t1 = (half_size - origin) * inv_d
    tmin = np.maximum.reduce(np.minimum(t0, t1))
    tmax = np.minimum.reduce(np.maximum(t0, t1))
    if tmax < 0.0 or tmin > tmax:
        return None
    t = tmin if tmin > 0.0 else tmax
    if t < 0.0:
        return None
    hit = origin + direction * t
    eps = 1e-4
    ax = np.abs(hit)
    m = np.max(ax)
    if np.abs(m - half_size) > 1e-2:
        m = half_size
    if ax[0] >= ax[1] and ax[0] >= ax[2] and np.abs(ax[0] - m) < eps:
        n = np.array([np.sign(hit[0]), 0.0, 0.0], dtype=np.float32)
    elif ax[1] >= ax[0] and ax[1] >= ax[2] and np.abs(ax[1] - m) < eps:
        n = np.array([0.0, np.sign(hit[1]), 0.0], dtype=np.float32)
    else:
        n = np.array([0.0, 0.0, np.sign(hit[2])], dtype=np.float32)
    return float(t), hit.astype(np.float32), n


def _ray_aabb_intersect(origin: np.ndarray, direction: np.ndarray, half_sizes: np.ndarray):
    inv_d = 1.0 / np.where(np.abs(direction) < 1e-9, 1e-9, direction)
    t0 = (-half_sizes - origin) * inv_d
    t1 = (half_sizes - origin) * inv_d
    tmin = np.maximum.reduce(np.minimum(t0, t1))
    tmax = np.minimum.reduce(np.maximum(t0, t1))
    if tmax < 0.0 or tmin > tmax:
        return None
    t = tmin if tmin > 0.0 else tmax
    if t < 0.0:
        return None
    hit = origin + direction * t
    eps = 2e-4
    ax = np.abs(hit)
    # Determine which slab we hit by comparing normalized distance to each half-size
    s = np.where(half_sizes < 1e-6, 1e-6, half_sizes)
    nx = ax[0] / s[0]
    ny = ax[1] / s[1]
    nz = ax[2] / s[2]
    m = max(nx, ny, nz)
    if m == nx and np.abs(ax[0] - half_sizes[0]) < eps:
        n = np.array([np.sign(hit[0]), 0.0, 0.0], dtype=np.float32)
    elif m == ny and np.abs(ax[1] - half_sizes[1]) < eps:
        n = np.array([0.0, np.sign(hit[1]), 0.0], dtype=np.float32)
    else:
        n = np.array([0.0, 0.0, np.sign(hit[2])], dtype=np.float32)
    return float(t), hit.astype(np.float32), n


def _anim_cubies_state(cubies, anim_axis=None, anim_layer=None, anim_angle=0.0):
    if anim_axis is None:
        return [(c.pos.astype(np.float32), c.orient.astype(np.float32), c.stickers) for c in cubies]

    anim_r = _rot(anim_axis, anim_angle).astype(np.float32)
    idx = {"x": 0, "y": 1, "z": 2}[anim_axis]
    out = []
    for c in cubies:
        p = c.pos.astype(np.float32)
        o = c.orient.astype(np.float32)
        if c.pos[idx] == anim_layer:
            p = anim_r @ p
            o = anim_r @ o
        out.append((p, o, c.stickers))
    return out


def _sticker_visibility_mask(uv_in_cell: np.ndarray) -> bool:
    inner = (1.0 - STICKER_MARGIN) * 0.5
    return (np.abs(uv_in_cell[0]) <= inner) and (np.abs(uv_in_cell[1]) <= inner)


def _sample_surrounding_cube_color(ray_o_world: np.ndarray, ray_d_world: np.ndarray, cube_half_sizes_world: np.ndarray, cube_center_world: np.ndarray, anim_state) -> np.ndarray:
    # Cube is axis-aligned in world space and surrounds the pillar.
    o = (ray_o_world - cube_center_world).astype(np.float32)
    d = ray_d_world.astype(np.float32)
    d_norm = float(np.linalg.norm(d))
    if d_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    d = d / d_norm

    hit = _ray_aabb_intersect(o, d, cube_half_sizes_world.astype(np.float32))
    if hit is None:
        return np.zeros(3, dtype=np.float32)

    t, p_hit, n_face = hit

    # "Inside looking out" lambert-ish term to avoid flat stickers.
    face_light = float(np.clip(np.dot(-n_face, d), 0.0, 1.0))
    dist_fade = 1.0 / (1.0 + 0.02 * t * t)
    shade = 0.20 + 0.80 * face_light
    shade *= dist_fade

    # Determine which global face we hit (+x,-x,+y,-y,+z,-z)
    if np.abs(n_face[0]) > 0.5:
        face_key = "+x" if n_face[0] > 0 else "-x"
        u_axis, v_axis = 1, 2
        face_normal = np.array([np.sign(n_face[0]), 0.0, 0.0], dtype=np.float32)
    elif np.abs(n_face[1]) > 0.5:
        face_key = "+y" if n_face[1] > 0 else "-y"
        u_axis, v_axis = 0, 2
        face_normal = np.array([0.0, np.sign(n_face[1]), 0.0], dtype=np.float32)
    else:
        face_key = "+z" if n_face[2] > 0 else "-z"
        u_axis, v_axis = 0, 1
        face_normal = np.array([0.0, 0.0, np.sign(n_face[2])], dtype=np.float32)

    # LED rays are horizontal (z=0), so top/bottom are effectively unseen.
    if face_key in ("+z", "-z"):
        return np.zeros(3, dtype=np.float32)

    # Normalize point into cube-local coordinates in [-1,1] per axis
    p_n = p_hit / cube_half_sizes_world.astype(np.float32)

    # Map hit point to cubie indices (in cube local coords -1,0,1)
    # For a face hit, one coordinate is fixed at ±1, the other two pick row/col.
    u = float(np.clip(p_n[u_axis], -0.999999, 0.999999))
    v = float(np.clip(p_n[v_axis], -0.999999, 0.999999))

    # 3 equal bands => index 0,1,2 then map to -1,0,1
    def _band(x: float) -> int:
        b = int(np.floor((x + 1.0) * 1.5))
        return int(np.clip(b, 0, 2))

    bu = _band(u)
    bv = _band(v)

    def _idx_to_coord(b: int) -> int:
        return -1 if b == 0 else (0 if b == 1 else 1)

    # Inside a cube, +x face has outward normal +x; the visible stickers for the pillar are on that face.
    if face_key in ("+x", "-x"):
        cubie_pos = np.array([1 if face_key == "+x" else -1, _idx_to_coord(bu), _idx_to_coord(bv)], dtype=np.float32)
    else:
        cubie_pos = np.array([_idx_to_coord(bu), 1 if face_key == "+y" else -1, _idx_to_coord(bv)], dtype=np.float32)

    # UV within the cubie cell centered at 0
    cell_u = (u - ((bu - 1) / 1.5)) * 1.5
    cell_v = (v - ((bv - 1) / 1.5)) * 1.5
    uv_in_cell = np.array([cell_u, cell_v], dtype=np.float32)

    if not _sticker_visibility_mask(uv_in_cell):
        return (np.array([8.0, 8.0, 8.0], dtype=np.float32) / 255.0) * shade

    # Find matching cubie in the animated state
    best_idx = -1
    best_d = 1e9
    for i, (cpos, _, _) in enumerate(anim_state):
        d2 = float(np.sum((cpos - cubie_pos) ** 2))
        if d2 < best_d:
            best_d = d2
            best_idx = i

    if best_idx < 0:
        return np.zeros(3, dtype=np.float32)

    _, corient, stickers = anim_state[best_idx]

    best_face = None
    best_dot = -1.0
    for face, col in stickers.items():
        if col is None:
            continue
        n_local = _FACE_NORMALS_LOCAL[face].astype(np.float32)
        n_world = corient @ n_local
        dd = float(np.dot(n_world, face_normal))
        if dd > best_dot:
            best_dot = dd
            best_face = face

    if best_face is None or best_dot < 0.6:
        return np.zeros(3, dtype=np.float32)

    lin = _srgb_to_linear(stickers[best_face])
    return lin * shade


def _render_frame(led_pos_world: np.ndarray, led_theta_step: float, led_z_step: float, cube_half_sizes_world: np.ndarray, cube_center_world: np.ndarray, anim_state) -> np.ndarray:
    frame_lin = np.zeros((LED_COUNT, 3), dtype=np.float32)

    fov_theta = float(led_theta_step * LED_FOV_THETA_MULT)
    fov_z = float(led_z_step * LED_FOV_Z_MULT)

    for i in range(LED_COUNT):
        acc = np.zeros(3, dtype=np.float32)
        base_theta = np.arctan2(led_pos_world[i, 1], led_pos_world[i, 0])
        base_z = led_pos_world[i, 2]

        for su in range(AA_SAMPLES_THETA):
            fu = (su + 0.5) / AA_SAMPLES_THETA
            dtheta = (fu - 0.5) * fov_theta
            theta = base_theta + dtheta
            dir_world = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)

            for sv in range(AA_SAMPLES_Z):
                fv = (sv + 0.5) / AA_SAMPLES_Z
                dz = (fv - 0.5) * fov_z
                o_world = np.array([RADIUS_INCHES * np.cos(theta), RADIUS_INCHES * np.sin(theta), base_z + dz], dtype=np.float32)
                if o_world[2] < 0.0 or o_world[2] > HEIGHT_INCHES:
                    continue
                acc += _sample_surrounding_cube_color(o_world, dir_world, cube_half_sizes_world, cube_center_world, anim_state)

        acc /= float(AA_SAMPLES_THETA * AA_SAMPLES_Z)
        frame_lin[i] = np.clip(acc, 0.0, 1.0)

    return _linear_to_srgb(frame_lin)


def generate_clip():
    scramble = ["R", "U", "R'", "U'", "F", "R", "U'", "F'"]
    solve = list(reversed(scramble))
    solve = [m[:-1] if m.endswith("'") else (m + "'") for m in solve]

    move_list = solve + scramble

    total_frames = (PAUSE_FRAMES + len(move_list) * MOVE_FRAMES + PAUSE_FRAMES)

    clip_data = np.zeros((total_frames, LED_COUNT, 3), dtype=np.uint8)

    led_indices = np.arange(LED_COUNT, dtype=np.float32)
    led_h_norm = led_indices / float(LED_COUNT)
    led_z_world = led_h_norm * HEIGHT_INCHES
    leds_per_wrap = LED_COUNT / PILLAR_WRAPS
    led_theta_norm = (led_indices % leds_per_wrap) / leds_per_wrap
    led_theta_rad = led_theta_norm * (2.0 * np.pi)

    led_x_world = RADIUS_INCHES * np.cos(led_theta_rad)
    led_y_world = RADIUS_INCHES * np.sin(led_theta_rad)
    led_pos_world = np.stack([led_x_world, led_y_world, led_z_world], axis=1).astype(np.float32)

    led_theta_step = float(2.0 * np.pi / leds_per_wrap)
    led_z_step = float(HEIGHT_INCHES / LED_COUNT)

    cubies = _make_cube()
    for m in scramble:
        _apply_move_discrete(cubies, m)

    # Surrounding cube centered on pillar axis.
    # The cube is bigger than the pillar radius so rays always hit a side face.
    cube_center_world = np.array([0.0, 0.0, ORBIT_Z_CENTER], dtype=np.float32)
    half_xy = (RADIUS_INCHES + CUBE_CLEARANCE_INCHES) / max(1e-6, float(CUBE_SCALE_XY))
    half_z = (HEIGHT_INCHES * 0.48) * float(CUBE_SCALE_Z)
    cube_half_sizes_world = np.array([half_xy, half_xy, half_z], dtype=np.float32)

    f = 0
    for _ in range(PAUSE_FRAMES):
        if f % 5 == 0 or f == total_frames - 1:
            print(f"Frame {f + 1}/{total_frames} (pause)")
        anim_state = _anim_cubies_state(cubies)
        clip_data[f] = _render_frame(led_pos_world, led_theta_step, led_z_step, cube_half_sizes_world, cube_center_world, anim_state)
        f += 1

    for mi, m in enumerate(move_list):
        axis, layer, direction = _axis_and_layer(m)
        for k in range(MOVE_FRAMES):
            t_norm = (k + 1) / MOVE_FRAMES
            angle = direction * (np.pi / 2.0) * t_norm
            if f % 5 == 0 or f == total_frames - 1:
                print(f"Frame {f + 1}/{total_frames} (move {mi + 1}/{len(move_list)} {m} step {k + 1}/{MOVE_FRAMES})")
            anim_state = _anim_cubies_state(cubies, anim_axis=axis, anim_layer=layer, anim_angle=angle)
            clip_data[f] = _render_frame(led_pos_world, led_theta_step, led_z_step, cube_half_sizes_world, cube_center_world, anim_state)
            f += 1

        _apply_move_discrete(cubies, m)

    for _ in range(PAUSE_FRAMES):
        if f % 5 == 0 or f == total_frames - 1:
            print(f"Frame {f + 1}/{total_frames} (pause)")
        anim_state = _anim_cubies_state(cubies)
        clip_data[f] = _render_frame(led_pos_world, led_theta_step, led_z_step, cube_half_sizes_world, cube_center_world, anim_state)
        f += 1

    print(f"Saving {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, clip_data)
    print("Done")


if __name__ == "__main__":
    generate_clip()
