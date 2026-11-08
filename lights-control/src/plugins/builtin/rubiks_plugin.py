# lights-control/src/plugins/builtin/rubiks_plugin.py
from __future__ import annotations
import numpy as np
from src.plugins.base import (
    AnimationPlugin,
    BlendModeParam,
    BoolParam,
    ChoiceParam,
    FloatParam,
    OpacityParam,
    PluginContext,
)
from src.plugins.registry import register_plugin
from src.plugins import toolkit as tk


# Six-color palettes, index-aligned: U-white, D-yellow, R-red, L-orange, F-green, B-blue.
PALETTES = {
    "classic": np.array(
        [
            [255, 255, 255],
            [255, 220, 0],
            [220, 0, 0],
            [255, 110, 0],
            [0, 190, 60],
            [0, 60, 255],
        ],
        dtype=float,
    ),
    "neon": np.array(
        [
            [255, 0, 140],
            [0, 240, 255],
            [140, 255, 0],
            [150, 0, 255],
            [255, 120, 0],
            [220, 240, 255],
        ],
        dtype=float,
    ),
    "pastel": np.array(
        [
            [255, 200, 200],
            [200, 230, 255],
            [255, 240, 180],
            [200, 255, 210],
            [230, 200, 255],
            [255, 215, 235],
        ],
        dtype=float,
    ),
}


def _rotate_in_plane(pts: np.ndarray, axis: int, angle: float) -> np.ndarray:
    """Rotate (K, 3) points around `axis` by `angle` radians, holding the axis coord fixed."""
    c, s = np.cos(angle), np.sin(angle)
    a, b = (axis + 1) % 3, (axis + 2) % 3
    out = pts.copy()
    out[:, a] = pts[:, a] * c - pts[:, b] * s
    out[:, b] = pts[:, a] * s + pts[:, b] * c
    return out


@register_plugin
class RubiksBlocksPlugin(AnimationPlugin):
    plugin_id = "rubiks"
    display_name = "Rubik Blocks"
    description = (
        "Colorful cubelets tumbling in 3D that periodically twist whole slices, "
        "like a Rubik's cube mid-scramble."
    )
    params = [
        ChoiceParam("palette", "Palette", options=["classic", "neon", "pastel"], default="classic"),
        ChoiceParam("grid", "Cubelets Per Side", options=["2", "3", "4"], default="3", live=False),
        FloatParam("size", "Cube Size (in)", default=14.0, min=6.0, max=24.0, step=0.5),
        FloatParam("gap", "Block Gap", default=0.15, min=0.02, max=0.4, step=0.01),
        FloatParam("corner", "Corner Round", default=0.15, min=0.0, max=0.45, step=0.01),
        FloatParam("tumble_speed", "Tumble Speed", default=0.4, min=0.0, max=2.0, step=0.05),
        FloatParam("twist_every", "Twist Every (s)", default=2.0, min=0.0, max=8.0, step=0.1),
        FloatParam("twist_time", "Twist Duration (s)", default=0.9, min=0.25, max=2.5, step=0.05),
        BoolParam("transparent_bg", "Transparent", default=True),
        BlendModeParam(default="alpha"),
        OpacityParam(default=1.0),
    ]

    def on_init(self) -> None:
        n = int(self.values["grid"])
        # Integer half-cell coordinates (e.g. -2,0,2 for n=3) so 90-degree slice
        # rotations map exactly back onto the lattice.
        coords = ((np.arange(n) - (n - 1) / 2.0) * 2.0).astype(int)
        gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
        self.grid_pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        self._rng = np.random.default_rng()
        # Scrambled look: each cubelet carries one palette color that travels with it.
        self.colors = self._rng.integers(0, 6, size=len(self.grid_pos))
        self.twist: dict | None = None

    def _maybe_start_twist(self, ctx: PluginContext) -> None:
        every = float(self.values["twist_every"])
        if every <= 1e-3 or not ctx.clock.every(every, key="twist"):
            return
        axis = int(self._rng.integers(0, 3))
        layer_vals = np.unique(self.grid_pos[:, axis])
        layer = int(layer_vals[self._rng.integers(0, len(layer_vals))])
        self.twist = {
            "axis": axis,
            "layer": layer,
            "dir": int(self._rng.choice([-1, 1])),
            "p": 0.0,
        }

    def _advance_twist(self, dt: float, centers: np.ndarray) -> np.ndarray:
        """Apply the in-flight slice rotation to `centers`; commit the move on completion."""
        tw = self.twist
        duration = max(0.05, float(self.values["twist_time"]))
        tw["p"] += dt / duration
        mask = self.grid_pos[:, tw["axis"]] == tw["layer"]

        if tw["p"] >= 1.0:
            # Commit: rotate the slice's lattice positions a quarter-turn.
            angle = tw["dir"] * (np.pi / 2.0)
            moved = _rotate_in_plane(self.grid_pos[mask].astype(float), tw["axis"], angle)
            self.grid_pos[mask] = np.rint(moved).astype(int)
            self.twist = None
            return centers

        # Ease-in-out quarter turn for the animated cells.
        p = tw["p"]
        eased = p * p * (3.0 - 2.0 * p)
        angle = tw["dir"] * (np.pi / 2.0) * eased
        centers = centers.copy()
        centers[mask] = _rotate_in_plane(centers[mask], tw["axis"], angle)
        return centers

    def render(self, ctx: PluginContext):
        t = ctx.clock.t
        dt = ctx.clock.dt
        size = float(self.values["size"])
        n = int(self.values["grid"])
        spacing = size / n
        half = spacing * 0.5 * (1.0 - float(self.values["gap"]))
        radius = min(half * float(self.values["corner"]), half * 0.9)

        # 1. Cubelet centers on the lattice; an active twist rotates its slice.
        centers = self.grid_pos * (spacing * 0.5)
        if self.twist is None:
            self._maybe_start_twist(ctx)
        if self.twist is not None:
            centers = self._advance_twist(dt, centers)

        # 2. Whole-cube tumble + gentle vertical drift, then into cube-local space.
        speed = float(self.values["tumble_speed"])
        angles = (t * 0.55 * speed, t * 0.75 * speed, t * 0.4 * speed)
        center = (0.0, 0.0, np.sin(t * 0.25 * speed) * ctx.leds.height_inches * 0.15)
        local = tk.world_to_local(ctx.leds.xyz_centered, center=center, angles_xyz=angles)

        # 3. Rounded-box SDF to the nearest cubelet: (600, M)
        diff = local[:, np.newaxis, :] - centers[np.newaxis, :, :]
        q = np.abs(diff) - (half - radius)
        sdf = (
            np.linalg.norm(np.maximum(q, 0.0), axis=2)
            + np.minimum(np.max(q, axis=2), 0.0)
            - radius
        )
        nearest = np.argmin(sdf, axis=1)
        dmin = sdf[np.arange(local.shape[0]), nearest]

        # 4. Flat palette color per cubelet, soft anti-aliased block edges.
        intensity = tk.soft_mask(dmin, radius=0.0, softness=0.6)
        palette = PALETTES[str(self.values["palette"])]
        colors = palette[self.colors[nearest]] * intensity[:, np.newaxis]

        if self.values["transparent_bg"]:
            return colors, intensity
        return colors
