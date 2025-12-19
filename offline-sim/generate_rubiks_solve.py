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
STICKER_POINT_GRID = 4

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


def _render_points(cubies, base_rot: np.ndarray, base_offset: np.ndarray, anim_axis=None, anim_layer=None, anim_angle=0.0):
    points = []
    colors = []

    if anim_axis is None:
        anim_r = np.eye(3, dtype=float)
    else:
        anim_r = _rot(anim_axis, anim_angle)
        idx = {"x": 0, "y": 1, "z": 2}[anim_axis]

    cubie_half = CUBIE_SIZE / 2.0
    sticker_half = cubie_half * (1.0 - STICKER_MARGIN)
    sticker_offset = cubie_half + 0.15

    for c in cubies:
        p = c.pos.astype(float)
        o = c.orient

        if anim_axis is not None and c.pos[idx] == anim_layer:
            p = anim_r @ p
            o = anim_r @ o

        p = (p * CUBIE_SIZE) @ base_rot.T
        o = base_rot @ o
        p = p + base_offset

        for face, col in c.stickers.items():
            if col is None:
                continue

            n_local = _FACE_NORMALS_LOCAL[face].astype(float)
            n = o @ n_local

            u_local, v_local = _FACE_UV_LOCAL[face]
            u = o @ u_local
            v = o @ v_local

            center = p + n * sticker_offset

            for iu in range(STICKER_POINT_GRID):
                fu = (iu / (STICKER_POINT_GRID - 1)) * 2.0 - 1.0
                for iv in range(STICKER_POINT_GRID):
                    fv = (iv / (STICKER_POINT_GRID - 1)) * 2.0 - 1.0
                    pt = center + u * (fu * sticker_half) + v * (fv * sticker_half)
                    points.append(pt)
                    colors.append(col)

    if len(points) == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)

    return np.asarray(points, dtype=float), np.asarray(colors, dtype=np.uint8)


def _project_to_leds(clip_frame: np.ndarray, pts: np.ndarray, cols: np.ndarray, led_h_norm: np.ndarray, led_theta_norm: np.ndarray, aspect: float):
    if pts.shape[0] == 0:
        return

    p_x = pts[:, 0]
    p_y = pts[:, 1]
    p_z = pts[:, 2]

    p_r = np.sqrt(p_x**2 + p_y**2)
    p_theta = (np.arctan2(p_y, p_x) + np.pi) / (2 * np.pi)

    dist_from_surface = p_r - RADIUS_INCHES
    brightness = 1.0 - (np.abs(dist_from_surface) / 4.5)
    brightness = np.clip(brightness, 0.0, 1.0)

    valid_mask = (p_z >= 0.0) & (p_z <= HEIGHT_INCHES) & (brightness > 0.02) & (dist_from_surface > -1.5) & (dist_from_surface < 5.0)
    idxs = np.where(valid_mask)[0]

    if len(idxs) == 0:
        return

    for i in idxs:
        nz = p_z[i] / HEIGHT_INCHES
        nt = p_theta[i]

        d_z = np.abs(led_h_norm - nz) * aspect
        raw_dt = np.abs(led_theta_norm - nt)
        d_t = np.minimum(raw_dt, 1.0 - raw_dt)

        dist_sq = d_z**2 + d_t**2
        nearest = np.argmin(dist_sq)

        if dist_sq[nearest] < 0.0035:
            b = brightness[i]
            add = (cols[i].astype(int) * b).astype(int)
            cur = clip_frame[nearest].astype(int)
            clip_frame[nearest] = np.minimum(cur + add, 255).astype(np.uint8)


def generate_clip():
    scramble = ["R", "U", "R'", "U'", "F", "R", "U'", "F'"]
    solve = list(reversed(scramble))
    solve = [m[:-1] if m.endswith("'") else (m + "'") for m in solve]

    move_list = solve + scramble

    total_frames = (PAUSE_FRAMES + len(move_list) * MOVE_FRAMES + PAUSE_FRAMES)

    clip_data = np.zeros((total_frames, LED_COUNT, 3), dtype=np.uint8)

    led_indices = np.arange(LED_COUNT)
    led_h_norm = led_indices / LED_COUNT
    leds_per_wrap = LED_COUNT / PILLAR_WRAPS
    led_theta_norm = (led_indices % leds_per_wrap) / leds_per_wrap
    aspect = HEIGHT_INCHES / CIRCUMFERENCE_INCHES

    base_rot = _rot("y", 0.25) @ _rot("z", 0.15)
    base_offset = np.array([RADIUS_INCHES + 2.0, 0.0, HEIGHT_INCHES * 0.52], dtype=float)

    cubies = _make_cube()
    for m in scramble:
        _apply_move_discrete(cubies, m)

    f = 0

    for _ in range(PAUSE_FRAMES):
        pts, cols = _render_points(cubies, base_rot, base_offset)
        _project_to_leds(clip_data[f], pts, cols, led_h_norm, led_theta_norm, aspect)
        f += 1

    for m in move_list:
        axis, layer, direction = _axis_and_layer(m)
        for k in range(MOVE_FRAMES):
            t = (k + 1) / MOVE_FRAMES
            angle = direction * (np.pi / 2.0) * t

            pts, cols = _render_points(cubies, base_rot, base_offset, anim_axis=axis, anim_layer=layer, anim_angle=angle)
            _project_to_leds(clip_data[f], pts, cols, led_h_norm, led_theta_norm, aspect)
            f += 1

        _apply_move_discrete(cubies, m)

    for _ in range(PAUSE_FRAMES):
        pts, cols = _render_points(cubies, base_rot, base_offset)
        _project_to_leds(clip_data[f], pts, cols, led_h_norm, led_theta_norm, aspect)
        f += 1

    np.save(OUTPUT_FILE, clip_data)


if __name__ == "__main__":
    generate_clip()
