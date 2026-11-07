# lights-control/src/plugins/builtin/lava_plugin.py
from __future__ import annotations
import numpy as np
from src.plugins.base import (
    AnimationPlugin,
    BoolParam,
    ChoiceParam,
    ColorParam,
    FloatParam,
    HeightRangeParams,
    IntParam,
    OpacityParam,
    PluginContext,
)
from src.plugins.registry import register_plugin
from src.plugins import toolkit as tk


@register_plugin
class LavaLampPlugin(AnimationPlugin):
    plugin_id = "lava_v2"
    display_name = "Lava Lamp (Plugin)"
    description = "Buoyant metaball wax blobs with adjustable softness, symmetry, and transparency."
    params = [
        ColorParam("color", "Blob Color", default=[255, 40, 0]),
        ColorParam("bg_color", "Fluid Color", default=[20, 0, 10]),
        IntParam("blob_count", "Blob Count", default=6, min=1, max=16, step=1, live=False),
        FloatParam("speed", "Flow Speed", default=0.6, min=0.05, max=3.0, step=0.05),
        FloatParam("blob_size", "Blob Size", default=0.15, min=0.06, max=0.35, step=0.01),
        FloatParam("edge_softness", "Edge Softness", default=0.5, min=0.1, max=1.2, step=0.05),
        ChoiceParam("symmetry", "Symmetry Faces", options=["1", "2", "4"], default="1"),
        BoolParam("transparent_bg", "Transparent Fluid", default=False),
        OpacityParam(default=1.0),
        *HeightRangeParams(0.0, 1.0),
    ]

    def on_init(self) -> None:
        count = int(self.values["blob_count"])
        rng = np.random.default_rng()
        self.y = rng.uniform(0.0, 1.0, count)
        self.x = rng.uniform(0.0, 1.0, count)
        self.temp = rng.uniform(0.0, 1.0, count)
        self.size_jitter = rng.uniform(0.75, 1.25, count)

    def render(self, ctx: PluginContext):
        dt = ctx.clock.dt
        speed = float(self.values["speed"])
        count = len(self.y)

        # 1. Buoyancy & thermal cycle physics
        heating = 0.5 * speed * dt
        cooling = 0.3 * speed * dt
        self.temp = np.where(self.y <= 0.12, self.temp + heating, self.temp)
        self.temp = np.where(self.y >= 0.88, self.temp - cooling, self.temp)
        np.clip(self.temp, 0.0, 1.0, out=self.temp)

        buoyancy = (self.temp - 0.5) * 0.4 * speed
        self.y = np.clip(self.y + buoyancy * dt, 0.0, 1.0)

        idx = np.arange(count, dtype=float)
        drift = np.sin(self.y * 10.0 + idx) * 0.05 * speed
        self.x = (self.x + drift * dt) % 1.0

        # 2. Evaluate 2D metaball field on cylinder
        radii = float(self.values["blob_size"]) * self.size_jitter
        symmetry = int(self.values["symmetry"])
        field = tk.blobs_2d(
            ctx.leds.theta,
            ctx.leds.y,
            self.x,
            self.y,
            radii,
            aspect=ctx.leds.aspect_ratio,
            symmetry=symmetry,
        )

        # 3. Soft threshold & color mix
        softness = float(self.values["edge_softness"])
        mix = tk.smoothstep(1.0 - softness * 0.5, 1.0 + softness * 0.5, field)

        blob_col = np.asarray(self.values["color"], dtype=float)
        bg_col = np.asarray(self.values["bg_color"], dtype=float)

        if self.values["transparent_bg"]:
            colors = np.broadcast_to(blob_col, (ctx.leds.count, 3))
            return colors, mix

        m = mix[:, np.newaxis]
        return bg_col * (1.0 - m) + blob_col * m
