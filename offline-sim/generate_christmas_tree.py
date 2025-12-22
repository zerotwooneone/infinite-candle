import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None


def _cupy_is_usable() -> bool:
    if cp is None:
        return False
    try:
        # Touch the runtime/compiler path; missing NVRTC DLLs will error here.
        _ = cp.asarray([0.0], dtype=cp.float32)
        _ = cp.sum(_)
        return True
    except Exception:
        return False

LED_COUNT = 600
PILLAR_WRAPS = 19.4
HEIGHT_INCHES = 48.0
CIRCUMFERENCE_INCHES = 21.0
RADIUS_INCHES = CIRCUMFERENCE_INCHES / (2 * np.pi)

FPS = 60
DURATION_SEC = 20.0
TOTAL_FRAMES = int(FPS * DURATION_SEC)
OUTPUT_FILE = "christmas_tree.npy"

TREE_Z_MIN = 2.0
TREE_Z_MAX = HEIGHT_INCHES * 0.92
TREE_CENTER_Z = (TREE_Z_MIN + TREE_Z_MAX) * 0.5

TREE_RADIUS_BASE = RADIUS_INCHES + 4.5
TREE_RADIUS_TOP = RADIUS_INCHES + 0.8
TREE_TIER_COUNT = 4
TREE_POINTS_PER_FACE = 380
TREE_COLOR = np.array([0.0, 0.55, 0.16], dtype=np.float32)
TREE_AMBIENT = 0.55

TREE_PYRAMID_ROTATIONS_RAD = [0.0, np.pi / 4.0, np.pi / 8.0]

TRUNK_Z_MIN = 0.0
TRUNK_Z_MAX = TREE_Z_MIN + 3.8
TRUNK_RADIUS = RADIUS_INCHES + 1.15
TRUNK_POINTS = 1500
TRUNK_COLOR = np.array([0.25, 0.14, 0.06], dtype=np.float32)
TRUNK_AMBIENT = 0.65

STAR_Z = TREE_Z_MAX + 0.4
STAR_RADIUS = RADIUS_INCHES + 0.15
STAR_COLOR = np.array([1.0, 0.9, 0.45], dtype=np.float32)
STAR_INTENSITY = 1.2

ORNAMENT_COUNT = 28
ORNAMENT_RADIUS = 1.05
ORNAMENT_AMBIENT = 0.55

ORNAMENT_SPHERE_SAMPLES = 28
ORNAMENT_SPECULAR_STRENGTH = 0.75
ORNAMENT_SPECULAR_SHININESS = 28.0

TWINKLE_LIGHT_COUNT = 45
TWINKLE_INTENSITY_MIN = 0.15
TWINKLE_INTENSITY_MAX = 1.0

ROT_LIGHT_RADIUS = TREE_RADIUS_BASE + 7.0
ROT_LIGHT_HEIGHT = TREE_CENTER_Z + 3.0
ROT_LIGHT_SPEED_TURNS_PER_LOOP = 0.22
ROT_LIGHT_A_COLOR = np.array([1.0, 0.82, 0.35], dtype=np.float32)
ROT_LIGHT_B_COLOR = np.array([0.92, 0.95, 1.0], dtype=np.float32)
ROT_LIGHT_INTENSITY = 1.15
ROT_LIGHT_ATTEN = 0.028

GLOBAL_EXPOSURE = 1.0

SPLAT_K = 18
SPLAT_SIGMA_THETA_MULT = 1.25
SPLAT_SIGMA_Z_MULT = 1.55

USE_CUPY = True
CUPY_CHUNK_SIZE = 2048

_CUPY_USABLE = False
_CUPY_FALLBACK_WARNED = False

if USE_CUPY and _cupy_is_usable():
    _CUPY_USABLE = True
elif USE_CUPY and cp is not None:
    print("CuPy detected but CUDA/NVRTC is not usable; falling back to NumPy.")

ENABLE_TREE = True
ENABLE_TRUNK = True
ENABLE_STAR = True
ENABLE_ORNAMENTS = True
ENABLE_TWINKLE_LIGHTS = True
ENABLE_ROTATING_LIGHTS = True

PRINT_DEBUG_STATS = True
DEBUG_EVERY_N_FRAMES = 60

TREE_OCCUPANCY_GAIN = 0.020
TRUNK_OCCUPANCY_GAIN = 0.030
STAR_OCCUPANCY_GAIN = 0.090

DEPTH_FADE_INCHES = 7.0


def _srgb8_to_linear01(rgb8: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    x = np.clip(rgb8.astype(np.float32) / 255.0, 0.0, 1.0)
    return x**gamma


def _linear01_to_srgb8(x: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    y = np.clip(x, 0.0, 1.0) ** (1.0 / gamma)
    return (y * 255.0 + 0.5).astype(np.uint8)


def _wrap01_dist(a: np.ndarray, b: float) -> np.ndarray:
    raw = np.abs(a - b)
    return np.minimum(raw, 1.0 - raw)


def _depth_fade(p_r: float) -> float:
    d = np.abs(p_r - RADIUS_INCHES)
    return float(np.clip(1.0 - (d / DEPTH_FADE_INCHES), 0.0, 1.0))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def _project_point_to_uv(point_world: np.ndarray):
    x, y, z = float(point_world[0]), float(point_world[1]), float(point_world[2])
    nz = z / HEIGHT_INCHES
    nt = (np.arctan2(y, x) + np.pi) / (2.0 * np.pi)
    return float(nt), float(nz)


def _precompute_splat(pts: np.ndarray, led_z_norm: np.ndarray, led_theta_norm: np.ndarray, aspect: float, sigma_theta: float, sigma_z: float, k: int):
    global _CUPY_USABLE, _CUPY_FALLBACK_WARNED
    if _CUPY_USABLE:
        try:
            return _precompute_splat_cupy(pts, led_z_norm, led_theta_norm, aspect, sigma_theta, sigma_z, k)
        except Exception as e:
            _CUPY_USABLE = False
            if not _CUPY_FALLBACK_WARNED:
                print(f"CuPy failed during splat precompute ({type(e).__name__}); falling back to NumPy.")
                _CUPY_FALLBACK_WARNED = True

    n_pts = int(pts.shape[0])
    idxs = np.zeros((n_pts, k), dtype=np.int32)
    wts = np.zeros((n_pts, k), dtype=np.float32)

    inv_2sig_t2 = 1.0 / max(1e-9, (2.0 * sigma_theta * sigma_theta))
    inv_2sig_z2 = 1.0 / max(1e-9, (2.0 * sigma_z * sigma_z))

    for i in range(n_pts):
        nt, nz = _project_point_to_uv(pts[i])
        d_t = _wrap01_dist(led_theta_norm, nt)
        d_z = (np.abs(led_z_norm - nz) * aspect)
        g = np.exp(-(d_t * d_t) * inv_2sig_t2 - (d_z * d_z) * inv_2sig_z2)

        top = np.argpartition(g, -k)[-k:]
        top = top[np.argsort(-g[top])]
        ww = g[top].astype(np.float32)
        s = float(np.sum(ww))
        if s < 1e-9:
            ww[:] = 0.0
        else:
            ww /= s

        idxs[i] = top.astype(np.int32)
        wts[i] = ww

    return idxs, wts


def _precompute_splat_cupy(pts: np.ndarray, led_z_norm: np.ndarray, led_theta_norm: np.ndarray, aspect: float, sigma_theta: float, sigma_z: float, k: int):
    # Chunked GPU path: compute gaussian weights to all LEDs and select top-k.
    pts = np.asarray(pts, dtype=np.float32)
    led_z_norm = np.asarray(led_z_norm, dtype=np.float32)
    led_theta_norm = np.asarray(led_theta_norm, dtype=np.float32)

    n_pts = int(pts.shape[0])
    idxs = np.zeros((n_pts, k), dtype=np.int32)
    wts = np.zeros((n_pts, k), dtype=np.float32)

    led_z_cp = cp.asarray(led_z_norm)
    led_t_cp = cp.asarray(led_theta_norm)

    inv_2sig_t2 = np.float32(1.0 / max(1e-9, (2.0 * sigma_theta * sigma_theta)))
    inv_2sig_z2 = np.float32(1.0 / max(1e-9, (2.0 * sigma_z * sigma_z)))
    aspect_cp = np.float32(aspect)

    for start in range(0, n_pts, int(CUPY_CHUNK_SIZE)):
        end = min(n_pts, start + int(CUPY_CHUNK_SIZE))
        chunk = pts[start:end]

        x = cp.asarray(chunk[:, 0])
        y = cp.asarray(chunk[:, 1])
        z = cp.asarray(chunk[:, 2])

        nz = z / np.float32(HEIGHT_INCHES)
        nt = (cp.arctan2(y, x) + np.float32(np.pi)) / np.float32(2.0 * np.pi)

        d_t = cp.abs(led_t_cp[None, :] - nt[:, None])
        d_t = cp.minimum(d_t, np.float32(1.0) - d_t)
        d_z = cp.abs(led_z_cp[None, :] - nz[:, None]) * aspect_cp

        g = cp.exp(-(d_t * d_t) * inv_2sig_t2 - (d_z * d_z) * inv_2sig_z2)

        # top-k along axis=1
        top = cp.argpartition(g, -k, axis=1)[:, -k:]
        top_g = cp.take_along_axis(g, top, axis=1)
        order = cp.argsort(-top_g, axis=1)
        top = cp.take_along_axis(top, order, axis=1)
        top_g = cp.take_along_axis(top_g, order, axis=1)

        s = cp.sum(top_g, axis=1, keepdims=True)
        top_w = cp.where(s > np.float32(1e-9), top_g / s, np.float32(0.0))

        idxs[start:end] = cp.asnumpy(top).astype(np.int32)
        wts[start:end] = cp.asnumpy(top_w).astype(np.float32)

    return idxs, wts


def _splat_add(frame_lin: np.ndarray, led_idxs: np.ndarray, led_wts: np.ndarray, rgb_lin: np.ndarray):
    if np.all(rgb_lin <= 0.0):
        return
    for j in range(int(led_idxs.shape[0])):
        frame_lin[int(led_idxs[j])] += rgb_lin * float(led_wts[j])


def _accum_weight_sum(led_count: int, led_idxs: np.ndarray, led_wts: np.ndarray, point_fade: np.ndarray) -> np.ndarray:
    global _CUPY_USABLE, _CUPY_FALLBACK_WARNED
    if _CUPY_USABLE:
        try:
            return _accum_weight_sum_cupy(led_count, led_idxs, led_wts, point_fade)
        except Exception as e:
            _CUPY_USABLE = False
            if not _CUPY_FALLBACK_WARNED:
                print(f"CuPy failed during weight accumulation ({type(e).__name__}); falling back to NumPy.")
                _CUPY_FALLBACK_WARNED = True

    # Returns per-LED weight sum: sum_i fade_i * w_ij
    s = np.zeros((led_count,), dtype=np.float32)
    for i in range(int(led_idxs.shape[0])):
        fi = float(point_fade[i])
        if fi <= 0.0:
            continue
        for j in range(int(led_idxs.shape[1])):
            s[int(led_idxs[i, j])] += fi * float(led_wts[i, j])
    return s


def _accum_weight_sum_cupy(led_count: int, led_idxs: np.ndarray, led_wts: np.ndarray, point_fade: np.ndarray) -> np.ndarray:
    led_idxs_cp = cp.asarray(np.asarray(led_idxs, dtype=np.int32))
    led_wts_cp = cp.asarray(np.asarray(led_wts, dtype=np.float32))
    fade_cp = cp.asarray(np.asarray(point_fade, dtype=np.float32))

    # Expand fade to match (n_pts, k) then scatter-add.
    contrib = led_wts_cp * fade_cp[:, None]
    out = cp.zeros((int(led_count),), dtype=cp.float32)
    cp.add.at(out, led_idxs_cp.ravel(), contrib.ravel())
    return cp.asnumpy(out).astype(np.float32)


def _occupancy_from_weight_sum(weight_sum: np.ndarray, gain: float) -> np.ndarray:
    # Turns additive sample density into a bounded [0,1) coverage term.
    return 1.0 - np.exp(-np.clip(weight_sum, 0.0, 1e9) * float(gain))


def _tree_radius_at_z(z: float) -> float:
    t = (z - TREE_Z_MIN) / max(1e-6, (TREE_Z_MAX - TREE_Z_MIN))
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * TREE_RADIUS_BASE + t * TREE_RADIUS_TOP


def _generate_pyramid_face_points(z0: float, z1: float, r0: float, r1: float, rng: np.random.Generator) -> np.ndarray:
    cx, cy = 0.0, 0.0
    points = []

    corners0 = np.array(
        [
            [cx + r0, cy + r0, z0],
            [cx - r0, cy + r0, z0],
            [cx - r0, cy - r0, z0],
            [cx + r0, cy - r0, z0],
        ],
        dtype=np.float32,
    )
    apex = np.array([cx, cy, z1], dtype=np.float32)

    for fi in range(4):
        a = corners0[fi]
        b = corners0[(fi + 1) % 4]
        for _ in range(TREE_POINTS_PER_FACE):
            u = rng.random()
            v = rng.random()
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            p = apex * (1.0 - u - v) + a * u + b * v
            points.append(p)

    return np.asarray(points, dtype=np.float32)


def _rotate_z(points: np.ndarray, ang: float) -> np.ndarray:
    c, s = float(np.cos(ang)), float(np.sin(ang))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = points[:, :2] @ rot.T
    out = points.copy()
    out[:, 0:2] = xy
    return out


def _generate_tree_points(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)

    tiers = []
    zs = np.linspace(TREE_Z_MIN, TREE_Z_MAX, TREE_TIER_COUNT + 1)
    for i in range(TREE_TIER_COUNT):
        z0 = float(zs[i])
        z1 = float(zs[i + 1])
        r0 = _tree_radius_at_z(z0)
        r1 = _tree_radius_at_z(z1)
        base = _generate_pyramid_face_points(z0, z1, r0, r1, rng)
        for ang in TREE_PYRAMID_ROTATIONS_RAD:
            tiers.append(_rotate_z(base, float(ang)))

    return np.concatenate(tiers, axis=0)


def _generate_trunk_points(seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, TRUNK_POINTS).astype(np.float32)
    z = rng.uniform(TRUNK_Z_MIN, TRUNK_Z_MAX, TRUNK_POINTS).astype(np.float32)
    r = (TRUNK_RADIUS * np.sqrt(rng.uniform(0.0, 1.0, TRUNK_POINTS))).astype(np.float32)
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _generate_star_points(seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 220
    pts = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        rr = float(rng.uniform(0.0, 1.0)) ** 0.5
        pts[i, 0] = float(np.cos(ang) * rr * STAR_RADIUS)
        pts[i, 1] = float(np.sin(ang) * rr * STAR_RADIUS)
        pts[i, 2] = float(STAR_Z + rng.uniform(-0.25, 0.25))
    return pts


def _generate_ornaments(seed: int = 2):
    rng = np.random.default_rng(seed)

    base_colors_srgb8 = np.array(
        [
            [255, 40, 40],
            [40, 120, 255],
            [255, 200, 40],
            [255, 70, 255],
        ],
        dtype=np.uint8,
    )
    base_colors = _srgb8_to_linear01(base_colors_srgb8)

    pos = []
    col = []
    for _ in range(ORNAMENT_COUNT):
        z = float(rng.uniform(TREE_Z_MIN + 2.0, TREE_Z_MAX - 3.0))
        r = _tree_radius_at_z(z) * float(rng.uniform(0.80, 0.98))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        pos.append([x, y, z])
        col.append(base_colors[int(rng.integers(0, len(base_colors)))])

    return np.asarray(pos, dtype=np.float32), np.asarray(col, dtype=np.float32)


def _generate_ornament_sphere_offsets(seed: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    offs = np.zeros((ORNAMENT_SPHERE_SAMPLES, 3), dtype=np.float32)
    for i in range(ORNAMENT_SPHERE_SAMPLES):
        u = float(rng.uniform(-1.0, 1.0))
        phi = float(rng.uniform(0.0, 2.0 * np.pi))
        s = float(np.sqrt(max(0.0, 1.0 - u * u)))
        n = np.array([np.cos(phi) * s, np.sin(phi) * s, u], dtype=np.float32)
        offs[i] = n * float(ORNAMENT_RADIUS)
    return offs


def _generate_tree_lights(seed: int = 3):
    rng = np.random.default_rng(seed)

    palette_srgb8 = np.array(
        [
            [255, 60, 60],
            [60, 255, 60],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    )
    palette = _srgb8_to_linear01(palette_srgb8)

    pos = []
    col = []
    phase = []
    rate = []

    for _ in range(TWINKLE_LIGHT_COUNT):
        z = float(rng.uniform(TREE_Z_MIN + 1.5, TREE_Z_MAX - 1.0))
        r = _tree_radius_at_z(z) * float(rng.uniform(0.85, 1.02))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        pos.append([x, y, z])
        col.append(palette[int(rng.integers(0, len(palette)))])
        phase.append(float(rng.uniform(0.0, 2.0 * np.pi)))
        rate.append(float(rng.uniform(0.6, 1.8)))

    return (
        np.asarray(pos, dtype=np.float32),
        np.asarray(col, dtype=np.float32),
        np.asarray(phase, dtype=np.float32),
        np.asarray(rate, dtype=np.float32),
    )


def _twinkle_intensity(t: float, phase: float, rate: float) -> float:
    base = 0.5 + 0.5 * np.sin((t * rate * 2.0 * np.pi) + phase)
    base = float(np.clip(base, 0.0, 1.0))
    return TWINKLE_INTENSITY_MIN + (TWINKLE_INTENSITY_MAX - TWINKLE_INTENSITY_MIN) * (base**2)


def _rotating_lights_world(t: float):
    ang = (t / DURATION_SEC) * (2.0 * np.pi) * float(ROT_LIGHT_SPEED_TURNS_PER_LOOP)
    a = np.array([np.cos(ang) * ROT_LIGHT_RADIUS, np.sin(ang) * ROT_LIGHT_RADIUS, ROT_LIGHT_HEIGHT], dtype=np.float32)
    b = np.array([np.cos(ang + np.pi) * ROT_LIGHT_RADIUS, np.sin(ang + np.pi) * ROT_LIGHT_RADIUS, ROT_LIGHT_HEIGHT], dtype=np.float32)
    return a, b


def _illum_from_light(p: np.ndarray, light_pos: np.ndarray) -> float:
    v = (light_pos - p).astype(np.float32)
    d2 = float(np.dot(v, v))
    return 1.0 / (1.0 + ROT_LIGHT_ATTEN * d2)


def _light_contrib(p: np.ndarray, n: np.ndarray, v_dir: np.ndarray, light_pos: np.ndarray, light_color: np.ndarray) -> np.ndarray:
    lvec = (light_pos - p).astype(np.float32)
    ldir = _unit(lvec)
    att = _illum_from_light(p, light_pos)

    diff = float(max(0.0, np.dot(n, ldir)))
    h = _unit(ldir + v_dir)
    spec = float(max(0.0, np.dot(n, h)) ** ORNAMENT_SPECULAR_SHININESS)
    return light_color * (ROT_LIGHT_INTENSITY * att) * (diff + ORNAMENT_SPECULAR_STRENGTH * spec)


def generate_clip():
    print(f"Generating Christmas Tree ({DURATION_SEC}s @ {FPS} FPS)...")

    tree_points = _generate_tree_points()
    trunk_points = _generate_trunk_points()
    star_points = _generate_star_points()
    ornaments_pos, ornaments_col = _generate_ornaments()
    ornament_offs = _generate_ornament_sphere_offsets()
    lights_pos, lights_col, lights_phase, lights_rate = _generate_tree_lights()

    clip_lin = np.zeros((TOTAL_FRAMES, LED_COUNT, 3), dtype=np.float32)

    led_idx = np.arange(LED_COUNT, dtype=np.float32)
    led_z_norm = led_idx / float(LED_COUNT)
    leds_per_wrap = LED_COUNT / PILLAR_WRAPS
    led_theta_norm = (led_idx % leds_per_wrap) / leds_per_wrap
    aspect = HEIGHT_INCHES / CIRCUMFERENCE_INCHES

    led_theta_step = float(1.0 / leds_per_wrap)
    led_z_step = float(1.0 / LED_COUNT)
    sigma_theta = led_theta_step * float(SPLAT_SIGMA_THETA_MULT)
    sigma_z = led_z_step * float(SPLAT_SIGMA_Z_MULT)

    tree_led_i, tree_led_w = _precompute_splat(tree_points, led_z_norm, led_theta_norm, aspect, sigma_theta, sigma_z, SPLAT_K)
    trunk_led_i, trunk_led_w = _precompute_splat(trunk_points, led_z_norm, led_theta_norm, aspect, sigma_theta, sigma_z, SPLAT_K)
    star_led_i, star_led_w = _precompute_splat(star_points, led_z_norm, led_theta_norm, aspect, sigma_theta, sigma_z, SPLAT_K)
    lights_led_i, lights_led_w = _precompute_splat(lights_pos, led_z_norm, led_theta_norm, aspect, sigma_theta, sigma_z, SPLAT_K)

    tree_fade = np.array([_depth_fade(float(np.sqrt(p[0] * p[0] + p[1] * p[1]))) for p in tree_points], dtype=np.float32)
    trunk_fade = np.array([_depth_fade(float(np.sqrt(p[0] * p[0] + p[1] * p[1]))) for p in trunk_points], dtype=np.float32)
    star_fade = np.array([_depth_fade(float(np.sqrt(p[0] * p[0] + p[1] * p[1]))) for p in star_points], dtype=np.float32)
    lights_fade = np.array([_depth_fade(float(np.sqrt(p[0] * p[0] + p[1] * p[1]))) for p in lights_pos], dtype=np.float32)

    orn_pts = np.zeros((ORNAMENT_COUNT, ORNAMENT_SPHERE_SAMPLES, 3), dtype=np.float32)
    orn_nrm = np.zeros((ORNAMENT_COUNT, ORNAMENT_SPHERE_SAMPLES, 3), dtype=np.float32)
    for i in range(ORNAMENT_COUNT):
        for s in range(ORNAMENT_SPHERE_SAMPLES):
            orn_pts[i, s] = ornaments_pos[i] + ornament_offs[s]
            orn_nrm[i, s] = _unit(ornament_offs[s])

    orn_flat = orn_pts.reshape((-1, 3))
    orn_led_i, orn_led_w = _precompute_splat(orn_flat, led_z_norm, led_theta_norm, aspect, sigma_theta, sigma_z, SPLAT_K)
    orn_fade = np.array([_depth_fade(float(np.sqrt(p[0] * p[0] + p[1] * p[1]))) for p in orn_flat], dtype=np.float32)

    static_lin = np.zeros((LED_COUNT, 3), dtype=np.float32)
    if ENABLE_TREE:
        tree_sum = _accum_weight_sum(LED_COUNT, tree_led_i, tree_led_w, tree_fade)
        tree_occ = _occupancy_from_weight_sum(tree_sum, TREE_OCCUPANCY_GAIN)
        static_lin += (TREE_COLOR * float(TREE_AMBIENT))[None, :] * tree_occ[:, None]

    if ENABLE_TRUNK:
        trunk_sum = _accum_weight_sum(LED_COUNT, trunk_led_i, trunk_led_w, trunk_fade)
        trunk_occ = _occupancy_from_weight_sum(trunk_sum, TRUNK_OCCUPANCY_GAIN)
        static_lin += (TRUNK_COLOR * float(TRUNK_AMBIENT))[None, :] * trunk_occ[:, None]

    if ENABLE_STAR:
        star_sum = _accum_weight_sum(LED_COUNT, star_led_i, star_led_w, star_fade)
        star_occ = _occupancy_from_weight_sum(star_sum, STAR_OCCUPANCY_GAIN)
        static_lin += (STAR_COLOR * float(STAR_INTENSITY))[None, :] * star_occ[:, None]

    static_lin = np.clip(static_lin, 0.0, 1.0)

    dt = 1.0 / FPS

    for f in range(TOTAL_FRAMES):
        if f % 30 == 0 or f == TOTAL_FRAMES - 1:
            print(f"Frame {f + 1}/{TOTAL_FRAMES}")

        t = f * dt
        frame_lin = static_lin.copy()

        a_pos, b_pos = _rotating_lights_world(t)

        if ENABLE_ORNAMENTS:
            for i in range(ORNAMENT_COUNT):
                base_col = ornaments_col[i]
                v_dir = _unit(np.array([ornaments_pos[i, 0], ornaments_pos[i, 1], 0.0], dtype=np.float32))
                if np.all(v_dir == 0.0):
                    v_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

                for s in range(ORNAMENT_SPHERE_SAMPLES):
                    p = orn_pts[i, s]
                    n = orn_nrm[i, s]
                    flat_idx = i * ORNAMENT_SPHERE_SAMPLES + s
                    fade = float(orn_fade[flat_idx])
                    if fade <= 0.01:
                        continue

                    col = base_col * float(ORNAMENT_AMBIENT)
                    if ENABLE_ROTATING_LIGHTS:
                        col += base_col * _light_contrib(p, n, v_dir, a_pos, ROT_LIGHT_A_COLOR)
                        col += base_col * _light_contrib(p, n, v_dir, b_pos, ROT_LIGHT_B_COLOR)
                    col *= fade

                    _splat_add(frame_lin, orn_led_i[flat_idx], orn_led_w[flat_idx], col)

        if ENABLE_TWINKLE_LIGHTS:
            for i in range(TWINKLE_LIGHT_COUNT):
                fade = float(lights_fade[i])
                if fade <= 0.01:
                    continue
                inten = _twinkle_intensity(t, float(lights_phase[i]), float(lights_rate[i]))
                col = lights_col[i] * float(inten) * fade
                _splat_add(frame_lin, lights_led_i[i], lights_led_w[i], col)

        out = np.clip(frame_lin * float(GLOBAL_EXPOSURE), 0.0, 1.0)
        if PRINT_DEBUG_STATS and (f % int(DEBUG_EVERY_N_FRAMES) == 0 or f == TOTAL_FRAMES - 1):
            mx = float(np.max(out))
            mean = float(np.mean(out))
            clipped = float(np.mean(out >= 0.999))
            print(f"  stats: mean={mean:.4f} max={mx:.4f} clipped={clipped * 100.0:.1f}%")
        clip_lin[f] = out

    clip = _linear01_to_srgb8(clip_lin)
    print(f"Saving {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, clip)
    print("Done")


if __name__ == "__main__":
    generate_clip()
