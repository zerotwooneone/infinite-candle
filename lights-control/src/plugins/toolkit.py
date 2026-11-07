# lights-control/src/plugins/toolkit.py
from __future__ import annotations
import math
import numpy as np


# =====================================================================
# 1. LOW-LEVEL MATH & SYMMETRY HELPERS
# =====================================================================

def wrap_delta(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Signed shortest angular delta on a 0.0..1.0 circle in range [-0.5, 0.5]."""
    return (np.asarray(a) - np.asarray(b) + 0.5) % 1.0 - 0.5


def wrap_dist(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Unsigned shortest distance on a 0.0..1.0 circle in range [0.0, 0.5]."""
    raw = np.abs(np.asarray(a) - np.asarray(b))
    return np.minimum(raw, 1.0 - raw)


def apply_symmetry(theta: np.ndarray, symmetry: int = 1, mirror: bool = False) -> np.ndarray:
    """
    Remap cylinder azimuth `theta` (0..1) into `symmetry` repeated sectors (0..1).
    - symmetry=1: full cylinder (1 copy)
    - symmetry=2: 2 opposing faces (front + back)
    - symmetry=4: 4 pillar faces
    If `mirror=True`, adjacent sectors mirror seamlessly.
    """
    n = max(1, int(symmetry))
    if n == 1 and not mirror:
        return theta
    scaled = (theta * n) % 1.0
    if mirror:
        sector_idx = np.floor(theta * n).astype(int)
        scaled = np.where(sector_idx % 2 == 1, 1.0 - scaled, scaled)
    return scaled


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Standard GLSL smoothstep(edge0, edge1, x) returning 0.0..1.0."""
    span = max(1e-6, float(edge1 - edge0))
    t = np.clip((x - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def soft_mask(distance: np.ndarray, radius: float, softness: float = 0.03) -> np.ndarray:
    """
    Convert a distance array into a 0.0..1.0 intensity mask:
    1.0 inside `radius - softness`, smoothly falling to 0.0 at `radius + softness`.
    """
    s = max(1e-5, float(softness))
    return 1.0 - smoothstep(radius - s, radius + s, distance)


# =====================================================================
# 2. 1D STRAND & RING HELPERS
# =====================================================================

def strand_chase(
    u: np.ndarray,
    head_pos: float,
    tail_length: float = 0.15,
    repeat: int = 1,
) -> np.ndarray:
    """
    1D comet along normalized coordinate `u` (0..1) with smooth tail decay.
    Returns (600,) float intensity in 0.0..1.0.
    """
    u_rep = (u * max(1, int(repeat))) % 1.0
    head = head_pos % 1.0
    dist_behind = (head - u_rep) % 1.0
    length = max(1e-4, float(tail_length))
    in_tail = dist_behind <= length
    intensity = np.where(in_tail, 1.0 - (dist_behind / length), 0.0)
    return intensity ** 1.5


# =====================================================================
# 3. 2D CYLINDER SURFACE PRIMITIVES (Broad, Anti-Aliased)
# =====================================================================

def band_y(
    y: np.ndarray,
    center_y: float,
    width: float = 0.12,
    softness: float = 0.04,
) -> np.ndarray:
    """Horizontal band around the cylinder at `center_y` with soft vertical AA."""
    dist = np.abs(y - center_y)
    return soft_mask(dist, width * 0.5, softness)


def wedge_theta(
    theta: np.ndarray,
    center_theta: float,
    width: float = 0.2,
    softness: float = 0.03,
    symmetry: int = 1,
) -> np.ndarray:
    """Vertical beam/wedge at `center_theta` (supports symmetry=1, 2, 4)."""
    n_sym = max(1, int(symmetry))
    t_sym = apply_symmetry(theta, n_sym)
    c_sym = (center_theta * n_sym) % 1.0
    dist = wrap_dist(t_sym, c_sym) / n_sym
    return soft_mask(dist, width * 0.5, softness)


def blobs_2d(
    theta: np.ndarray,
    y: np.ndarray,
    blob_x: np.ndarray,
    blob_y: np.ndarray,
    blob_radii: np.ndarray | float,
    aspect: float = 2.2857,
    symmetry: int = 1,
) -> np.ndarray:
    """
    Metaball influence field `sum(r^2 / d^2)` evaluated at all 600 LEDs.
    - `blob_x`, `blob_y`: shape `(N,)` in 0.0..1.0
    - Returns `(600,)` float field where values >= 1.0 are inside the merged blobs.
    """
    n_sym = max(1, int(symmetry))
    t_sym = apply_symmetry(theta, n_sym)[np.newaxis, :]  # (1, 600)
    led_y = y[np.newaxis, :]                             # (1, 600)

    bx = ((np.asarray(blob_x) * n_sym) % 1.0)[:, np.newaxis]  # (N, 1)
    by = np.asarray(blob_y)[:, np.newaxis]                    # (N, 1)
    br = np.asarray(blob_radii)
    if br.ndim == 0:
        br = np.full_like(bx, float(br))
    else:
        br = br[:, np.newaxis]

    dx = wrap_dist(t_sym, bx) / n_sym
    dy = (led_y - by) * aspect
    dist_sq = np.maximum(dx * dx + dy * dy, 1e-4)
    return np.sum((br * br) / dist_sq, axis=0)


def silhouette_cone(
    theta: np.ndarray,
    y: np.ndarray,
    center_theta: float,
    y_min: float,
    y_max: float,
    width_bottom: float,
    width_top: float,
    softness: float = 0.02,
    symmetry: int = 2,
) -> np.ndarray:
    """
    Tapered 2D shape on the cylinder surface (e.g. tree, flame, obelisk, hourglass).
    Defaults to `symmetry=2` (visible on 2 opposing sides of the candle).
    Returns `(600,)` float intensity 0.0..1.0.
    """
    h = max(1e-5, y_max - y_min)
    y_norm = np.clip((y - y_min) / h, 0.0, 1.0)
    half_w = 0.5 * (width_bottom * (1.0 - y_norm) + width_top * y_norm)

    n_sym = max(1, int(symmetry))
    t_sym = apply_symmetry(theta, n_sym)
    c_sym = (center_theta * n_sym) % 1.0
    dx = wrap_dist(t_sym, c_sym) / n_sym

    horiz = soft_mask(dx, half_w, softness)
    vert = soft_mask(np.abs(y - 0.5 * (y_min + y_max)), 0.5 * h, softness)
    return horiz * vert


# =====================================================================
# 4. 3D VOLUME & SIGNED DISTANCE FIELDS (Evaluated at the 600 LEDs)
# =====================================================================

def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """3x3 Euler rotation matrix (radians)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    my = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    mz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return mz @ my @ mx


def world_to_local(
    led_xyz: np.ndarray,
    center: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    angles_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Inverse-transform the 600 LED 3D positions into a moving/rotating object's local space.
    Returns `(600, 3)` local coordinates.
    """
    shifted = led_xyz - np.asarray(center, dtype=float)[np.newaxis, :]
    rot = rotation_matrix_xyz(*angles_xyz)
    # Inverse of rotation matrix is its transpose; for row vectors `shifted @ rot`
    return shifted @ rot


def sdf_box_3d(local_xyz: np.ndarray, half_extents: tuple[float, float, float] | float) -> np.ndarray:
    """
    Signed distance from each LED `(600, 3)` to an axis-aligned box of `half_extents`.
    Negative inside the box, 0 on surface, positive outside.
    """
    b = np.asarray(half_extents, dtype=float)
    q = np.abs(local_xyz) - b
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def sdf_box_wireframe_3d(
    local_xyz: np.ndarray,
    half_extents: tuple[float, float, float] | float,
) -> np.ndarray:
    """
    Distance from each LED `(600, 3)` to the 12 edges of a 3D box of `half_extents`.
    Analytic distance to thin box edges — no edge point sampling.
    """
    b = np.broadcast_to(np.asarray(half_extents, dtype=float), (3,))
    p = np.abs(local_xyz) - b[np.newaxis, :]  # (600, 3)

    # Distance to an edge: clamp along the edge's own axis (segment extent),
    # free in the two perpendicular axes.
    overhang = np.maximum(p, 0.0)
    # 3 candidate edge families (along X, Y, Z axes):
    d_edge_x = np.sqrt(overhang[:, 0] ** 2 + p[:, 1] ** 2 + p[:, 2] ** 2)
    d_edge_y = np.sqrt(p[:, 0] ** 2 + overhang[:, 1] ** 2 + p[:, 2] ** 2)
    d_edge_z = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2 + overhang[:, 2] ** 2)
    return np.minimum(np.minimum(d_edge_x, d_edge_y), d_edge_z)


def plasma_field_3d(
    xyz: np.ndarray,
    t: float,
    scale: float = 0.25,
) -> np.ndarray:
    """
    3D multi-sine interference field evaluated at the 600 LEDs.
    Returns `(600,)` values normalized to `0.0..1.0`.
    """
    x = xyz[:, 0] * scale
    y = xyz[:, 1] * scale
    z = xyz[:, 2] * (scale * 0.5)
    v = (
        np.sin(x + t)
        + np.sin(y + t * 0.8)
        + np.sin(z - t * 0.6)
        + 0.5 * np.sin((x + y + z) * 0.7 + t * 1.3)
    )
    # Range of v is [-3.5, 3.5] -> map to [0.0, 1.0]
    return np.clip((v + 3.5) / 7.0, 0.0, 1.0)


# =====================================================================
# 5. COLOR PALETTES & PERSISTENCE / TRAIL BUFFER
# =====================================================================

class ColorPalette:
    """Fast 256-entry RGB lookup table built from 2 or more RGB stops."""

    def __init__(self, stops: list[tuple[int, int, int] | list[int]]):
        arr = np.asarray(stops, dtype=float)
        if len(arr) < 2:
            arr = np.vstack([arr, arr])
        n_stops = len(arr)
        stop_positions = np.linspace(0.0, 1.0, n_stops)
        sample_positions = np.linspace(0.0, 1.0, 256)
        self.lut = np.stack(
            [np.interp(sample_positions, stop_positions, arr[:, c]) for c in range(3)],
            axis=1,
        )  # (256, 3) float

    def sample(self, t: np.ndarray | float) -> np.ndarray:
        """Map `t` in `0.0..1.0` (shape `(600,)`) to `(600, 3)` float RGB."""
        idx = np.clip(np.asarray(t) * 255.0, 0.0, 255.0).astype(int)
        return self.lut[idx]


class TrailBuffer:
    """Per-plugin persistence buffer for motion trails, afterglow, and sparks."""

    def __init__(self, led_count: int = 600):
        self.buffer = np.zeros((led_count, 3), dtype=float)

    def decay(self, dt: float, rate: float = 3.0) -> np.ndarray:
        """Exponential decay (`rate` higher = shorter trails)."""
        self.buffer *= math.exp(-max(0.0, float(rate)) * max(0.0, float(dt)))
        return self.buffer

    def add(self, colors: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Additive blend `colors` into the trail buffer and clamp to 0..255."""
        if mask is None:
            self.buffer = np.clip(self.buffer + colors, 0.0, 255.0)
        else:
            m = mask[:, np.newaxis] if mask.ndim == 1 else mask
            self.buffer = np.clip(self.buffer + colors * m, 0.0, 255.0)
        return self.buffer


# =====================================================================
# 6. COMPOSITING / BUFFER BLENDING HELPER
# =====================================================================

def blend_into_buffer(
    buffer: np.ndarray,
    colors: np.ndarray,
    *,
    alpha: np.ndarray | float = 1.0,
    mask: np.ndarray | None = None,
    mode: str = "alpha",
) -> None:
    """
    Blend `colors` `(600, 3)` into the live `buffer` `(600, 3)` uint8 array in-place.
    - `alpha`: scalar `0.0..1.0` or per-LED array `(600,)`
    - `mask`: optional boolean array `(600,)` (e.g. `(leds.y >= h_min) & (leds.y <= h_max)`)
    - `mode`: `'overwrite'`, `'alpha'`, `'add'`, or `'max'`
    """
    fg = np.clip(np.asarray(colors, dtype=float), 0.0, 255.0)
    if np.ndim(alpha) == 0:
        a = np.full((len(buffer), 1), float(np.clip(alpha, 0.0, 1.0)))
    else:
        a = np.clip(np.asarray(alpha, dtype=float), 0.0, 1.0)[:, np.newaxis]

    idx = slice(None) if mask is None else np.asarray(mask, dtype=bool)
    if mask is not None and not np.any(idx):
        return

    fg_sel = fg[idx]
    bg_sel = buffer[idx].astype(float)
    a_sel = a[idx]

    if mode == "overwrite":
        out = fg_sel
    elif mode == "add":
        out = np.clip(bg_sel + fg_sel * a_sel, 0.0, 255.0)
    elif mode == "max":
        out = np.maximum(bg_sel, fg_sel * a_sel)
    else:  # "alpha"
        out = bg_sel * (1.0 - a_sel) + fg_sel * a_sel

    buffer[idx] = out.astype(np.uint8)
