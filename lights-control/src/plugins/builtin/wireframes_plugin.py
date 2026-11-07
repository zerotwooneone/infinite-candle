# lights-control/src/plugins/builtin/wireframes_plugin.py
from __future__ import annotations
import numpy as np
from src.plugins.base import (
    AnimationPlugin,
    BlendModeParam,
    BoolParam,
    ChoiceParam,
    ColorParam,
    FloatParam,
    PluginContext,
)
from src.plugins.registry import register_plugin
from src.plugins import toolkit as tk


@register_plugin
class WireframesLivePlugin(AnimationPlugin):
    plugin_id = "wireframes_live"
    display_name = "3D Wireframes (Live)"
    description = "Real-time 3D rotating cube / compound lattice evaluated on the 600 LEDs."
    params = [
        ColorParam("edge_color", "Edge Color", default=[255, 160, 0]),
        ColorParam("surface_color", "Volume Fill Color", default=[0, 100, 180]),
        ChoiceParam("shape", "Geometry", options=["cube", "lattice", "octa_bars"], default="lattice"),
        FloatParam("cube_size", "Size (inches)", default=8.5, min=3.0, max=16.0, step=0.5),
        FloatParam("edge_thickness", "Edge Width (in)", default=1.1, min=0.4, max=3.0, step=0.1),
        FloatParam("rot_speed", "Rotation Speed", default=0.6, min=0.0, max=3.0, step=0.05),
        FloatParam("travel_amp", "Swing Amplitude", default=1.0, min=0.0, max=2.0, step=0.1),
        FloatParam("surface_opacity", "Volume Fill", default=0.75, min=0.0, max=1.0, step=0.05),
        BoolParam("transparent_bg", "Transparent Outside", default=True),
        BlendModeParam(default="alpha"),
    ]

    def render(self, ctx: PluginContext):
        t = ctx.clock.t * float(self.values["rot_speed"])
        size = float(self.values["cube_size"])
        amp_scale = float(self.values["travel_amp"])
        edge_w = float(self.values["edge_thickness"])

        # 1. Compute 3D center oscillation & tumble (matches generate_wireframes.py motion)
        swing_xy = (ctx.leds.radius_inches + size * 0.45) * amp_scale
        center = (
            np.sin(t) * swing_xy,
            np.cos(t * 2.0) * (swing_xy * 0.5),
            np.sin(t * 0.5) * (ctx.leds.height_inches * 0.32 * min(1.0, amp_scale + 0.3)),
        )
        angles = (t * 1.0, t * 0.7, t * 0.3)

        # 2. Inverse-transform the 600 LED positions into the cube's local 3D frame
        local_xyz = tk.world_to_local(ctx.leds.xyz_centered, center=center, angles_xyz=angles)

        shape = self.values["shape"]
        if shape == "lattice":
            # 3x3x3 compound lattice via modular folding in local space
            cell_spacing = size * 0.65
            clamped_cell = np.clip(np.round(local_xyz / cell_spacing), -1.0, 1.0)
            cell_local = local_xyz - clamped_cell * cell_spacing
            half_ext = (size * 0.30, size * 0.30, size * 0.30)
            vol_dist = tk.sdf_box_3d(cell_local, half_ext)
            edge_dist = tk.sdf_box_wireframe_3d(cell_local, half_ext)
        elif shape == "octa_bars":
            # Elongated intersecting prism bars
            half_ext = (size * 0.55, size * 0.55, size * 1.35)
            vol_dist = tk.sdf_box_3d(local_xyz, half_ext)
            edge_dist = tk.sdf_box_wireframe_3d(local_xyz, half_ext)
        else:
            # Single bold 3D cube
            half_ext = (size * 0.6, size * 0.6, size * 0.9)
            vol_dist = tk.sdf_box_3d(local_xyz, half_ext)
            edge_dist = tk.sdf_box_wireframe_3d(local_xyz, half_ext)

        # 3. Soft masks for volume interior and glowing wireframe edges
        surface_hit = tk.soft_mask(vol_dist, radius=0.35, softness=0.75) * float(self.values["surface_opacity"])
        edge_hit = tk.soft_mask(edge_dist, radius=edge_w * 0.5, softness=edge_w * 0.45)

        surf_col = np.asarray(self.values["surface_color"], dtype=float)
        edge_col = np.asarray(self.values["edge_color"], dtype=float)

        # Additive edge highlight on top of volume fill
        colors = np.clip(
            surf_col[np.newaxis, :] * surface_hit[:, np.newaxis]
            + edge_col[np.newaxis, :] * edge_hit[:, np.newaxis],
            0.0,
            255.0,
        )

        if self.values["transparent_bg"]:
            alpha = np.clip(np.maximum(surface_hit, edge_hit), 0.0, 1.0)
            return colors, alpha

        return colors
