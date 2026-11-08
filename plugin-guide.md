# Animation Plugin Guide

How to write a new animation for the Infinite Candle: **create one Python file** in `lights-control/src/plugins/builtin/`. The file is auto-discovered at startup, appears in the dashboard's *+ Add Layer* menu with its own controls, and is live-editable while running — no dashboard or engine changes required.

---

## 1. Hardware & Geometry Ground Truth

Every design decision is constrained by the physical candle (`lights-control/src/config.py`):

- **`LED_COUNT = 600`** addressable RGB LEDs wrapped around a 4-sided pillar.
- **`PILLAR_WRAPS = 19.4`** spiral turns from bottom (`y = 0.0`) to top (`y = 1.0`).
- **Physical dimensions:** `48.0"` height, `21.0"` circumference (`RADIUS_INCHES ≈ 3.342"`), `aspect_ratio ≈ 2.2857`.
- **Effective resolution:** ~30.9 LEDs per horizontal wrap (~7.7 per face) by 19.4 vertical wraps.

Because vertical spacing (~2.5") is ~3.6× coarser than horizontal spacing (~0.68"):

1. **Fine shapes, thin lines, and high-res grids alias or disappear.**
2. **What works visually:** broad anti-aliased bands, soft blobs/metaballs, 2-opposing-face (`symmetry=2`) or 4-face (`symmetry=4`) silhouettes, and 3D signed-distance fields (SDFs) evaluated **directly at the 600 LED coordinates**.
3. **Performance rule:** never iterate over thousands of sample points in Python loops. Keep everything vectorized at shape `(600,)` or `(600, K)` with `K <= 64`.

---

## 2. Core Principles

1. **Plugins own 100% of their UI controls.** A plugin layer does *not* inherit `opacity`, `h_min`, `h_max`, or `blend_mode`. The dashboard inspector renders **only** the `params` the plugin declares.
2. **Opt-in reusable presets.** Want an Opacity slider, a Blend Mode dropdown, or Height bounds? Include `OpacityParam()`, `BlendModeParam()`, or `HeightRangeParams()` in your `params` list.
3. **Direct frame access.** `render(ctx)` receives the LED mapper, the shared animation clock, and the live `(600, 3)` `uint8` buffer — or just return colors and let the adapter composite for you.

---

## 3. The Plugin Contract

```python
# lights-control/src/plugins/builtin/my_effect.py
import numpy as np
from src.plugins.base import AnimationPlugin, ColorParam, FloatParam, PluginContext
from src.plugins.registry import register_plugin
from src.plugins import toolkit as tk


@register_plugin
class MyEffect(AnimationPlugin):
    plugin_id = "my_effect"           # unique id used as the layer `type`
    display_name = "My Effect"        # shown in the dashboard
    description = "What it looks like."
    params = [ ... ]                  # the only UI controls this layer gets

    def on_init(self) -> None:
        """Optional: allocate state arrays from self.values (called on construction
        and again whenever a `live=False` param changes)."""
        pass

    def on_params_changed(self, changed: dict) -> None:
        """Optional: react to live parameter edits."""
        pass

    def render(self, ctx: PluginContext):
        """Compute one frame for all 600 LEDs. See return conventions below."""
        ...
```

### Discovery & lifecycle

- **Discovery:** every module in `src/plugins/builtin/` is imported at startup; `AnimationPlugin` subclasses with a non-empty `plugin_id` register automatically (the `@register_plugin` decorator is belt-and-suspenders — keep it).
- **Lifecycle:** construction → `on_init()` → `render(ctx)` every frame (~60 FPS). `PATCH /scene/layers/{id}` calls `update_params()`, which coerces values, then calls `on_params_changed(changed)` — or `on_init()` again if a `live=False` param changed.
- **State preservation:** layers whose `(id, type)` match an active layer keep their plugin instance across `POST /scene` pushes, so animation state (e.g. blob positions) survives config updates.

### `render(ctx)` return conventions

| Return | Behavior |
| :--- | :--- |
| `colors` `(600, 3)` | Composited into the buffer. Uses declared `opacity`, `blend_mode`, `h_min`/`h_max` if present — otherwise **overwrites**. |
| `(colors, alpha)` | `alpha` is `(600,)` or `float`; blended with `"alpha"` mode by default. |
| `None` | You mutated `ctx.buffer` yourself (e.g. `tk.blend_into_buffer(ctx.buffer, ...)`). |

The adapter automatically honors `opacity`, `blend_mode`, `h_min`, and `h_max` **only if** the plugin declared those params in `self.values` — return a `(colors, alpha)` tuple or a plain color array and it does the right thing.

---

## 4. Parameters (UI Controls)

Each `ParamSpec` in `params` maps 1:1 to a dashboard control:

| Factory | `kind` | Dashboard control | Notes |
| :--- | :--- | :--- | :--- |
| `FloatParam(name, label, default, min, max, step, live=True)` | `float` | slider + number box | |
| `IntParam(name, label, default, min, max, step, live=True)` | `int` | slider + number box | rounded on coerce |
| `ColorParam(name, label, default=[r,g,b], live=True)` | `color` | color picker | stored as `[r, g, b]` |
| `BoolParam(name, label, default, live=True)` | `bool` | checkbox | |
| `ChoiceParam(name, label, options, default, live=True)` | `choice` | dropdown | |
| `TextParam(name, label, default, max_len=64, live=True)` | `text` | single-line text input | truncated to `max_len` |

- **`live=False`** — changing this param re-runs `on_init()` (e.g. `blob_count`, which re-allocates state arrays). Live params update in place with no animation restart.
- Values arrive **coerced and clamped** in `self.values[name]` — always read params via `self.values`, never cache them.

### Opt-in shared presets

```python
from src.plugins.base import OpacityParam, BlendModeParam, HeightRangeParams

params = [
    ...,
    OpacityParam(default=1.0),            # "opacity" slider 0–1
    BlendModeParam(default="alpha"),      # "blend_mode" dropdown: alpha | add | max | overwrite
    *HeightRangeParams(0.0, 1.0),         # "h_min" + "h_max" sliders (height mask)
]
```

---

## 5. `PluginContext`: `leds`, `clock`, `buffer`

### `ctx.leds` — `PillarMapper` coordinate arrays

| Array | Shape | Meaning |
| :--- | :--- | :--- |
| `index`, `u` | `(600,)` | LED index / normalized position along the strip `0..1` |
| `wrap`, `wrap_u` | `(600,)` | spiral wrap index (`0..19`) / position within the wrap |
| `theta` (= `coords_x`), `theta_rad` | `(600,)` | azimuth `0..1` / `0..2π` |
| `y` (= `coords_y`) | `(600,)` | height `0.0` bottom → `1.0` top |
| `face`, `face_u` | `(600,)` | pillar face `0..3` / position within face |
| `xyz`, `xyz_centered` | `(600, 3)` | 3D inches; `xyz_centered` is Z-centered at 0 for rotations |
| `normals_xy` | `(600, 2)` | outward unit normals |
| Scalars | — | `count`, `wraps`, `faces`, `aspect_ratio`, `radius_inches`, `height_inches`, `circumference_inches` |

### `ctx.clock` — shared `AnimationClock`

| Member | Purpose |
| :--- | :--- |
| `t`, `dt`, `frame`, `bpm` | elapsed seconds / clamped frame delta / frame count / tempo |
| `phase(period_sec, offset)` | sawtooth ramp `0..1` per period |
| `wave(period_sec, offset)` | sine `0..1` per period |
| `triangle(period_sec, offset)` | triangle `0→1→0` per period |
| `beat(multiplier)` | sawtooth synced to `bpm` |
| `every(interval_sec, key)` | `True` once per interval — edge-triggered events |

`dt` is clamped to `0.25 s`, so `ctx.clock.dt` is always safe for physics integration.

### `ctx.buffer`

The live `(600, 3)` `uint8` frame buffer containing everything rendered by layers **below** this one. Mutate it directly (e.g. via `tk.blend_into_buffer`) only when you need compositing behavior the adapter's return conventions can't express.

---

## 6. Toolkit Reference (`src/plugins/toolkit.py`)

Vectorized NumPy helpers for the 600-LED cylinder — never write wrap-around math by hand:

- **Math & symmetry:** `wrap_delta`, `wrap_dist`, `apply_symmetry` (sectors `1`/`2`/`4`, `mirror=True` for seamless mirroring), `smoothstep`, `soft_mask`
- **1D strand:** `strand_chase` (comet + tail)
- **2D cylinder surface:** `band_y`, `wedge_theta`, `blobs_2d` (metaballs), `silhouette_cone` (trees, flames, obelisks)
- **3D volume / SDF:** `rotation_matrix_xyz`, `world_to_local`, `sdf_box_3d`, `sdf_box_wireframe_3d` (exact box-frame edges), `plasma_field_3d`
- **Color & trails:** `ColorPalette(stops).sample(t)`, `TrailBuffer().decay(dt, rate)` / `.add(colors, mask)`
- **Compositing:** `blend_into_buffer(buffer, colors, alpha=..., mask=..., mode="alpha")`

---

## 7. Layer JSON, Discovery & Live Updates

```jsonc
// POST /scene — a plugin layer is just {id, type, ...param values}
{ "layers": [ { "id": "L1", "type": "wireframes_live", "rot_speed": 1.2 } ] }
```

| Endpoint | Purpose |
| :--- | :--- |
| `GET /plugins` | metadata for every registered plugin (`id`, `name`, `description`, `params`, `defaults`) — the dashboard fetches this on load |
| `PATCH /scene/layers/{id}` | live-update a running layer's params without resetting animation state (dashboard sends these, throttled 50 ms) |

Keeping `id` stable across `POST /scene` pushes is what lets the engine preserve a running plugin's internal state.

---

## 8. Complete Minimal Example

```python
import numpy as np
from src.plugins.base import (
    AnimationPlugin, ChoiceParam, ColorParam, FloatParam, OpacityParam, PluginContext,
)
from src.plugins.registry import register_plugin
from src.plugins import toolkit as tk


@register_plugin
class LighthouseBeaconPlugin(AnimationPlugin):
    plugin_id = "beacon"
    display_name = "Rotating Beacon"
    description = "Dual opposing lighthouse beams with adjustable width and speed."
    params = [
        ColorParam("color", "Beam Color", default=[255, 220, 120]),
        FloatParam("speed", "Rotation Speed", default=0.5, min=0.1, max=3.0, step=0.1),
        FloatParam("width", "Beam Width", default=0.18, min=0.05, max=0.45, step=0.02),
        ChoiceParam("symmetry", "Beams", options=["1", "2", "4"], default="2"),
        OpacityParam(default=1.0),   # opt-in: only because this plugin wants an Opacity slider
    ]

    def render(self, ctx: PluginContext):
        angle = ctx.clock.phase(period_sec=1.0 / float(self.values["speed"]))
        intensity = tk.wedge_theta(
            ctx.leds.theta,
            center_theta=angle,
            width=float(self.values["width"]),
            softness=0.05,
            symmetry=int(self.values["symmetry"]),
        )
        color = np.asarray(self.values["color"], dtype=float)
        return color[np.newaxis, :] * intensity[:, np.newaxis], intensity
```

---

## 9. Verifying a Plugin

From `lights-control/`:

```powershell
# Confirm discovery + metadata:
uv run python -c "from src.plugins.registry import list_plugins_metadata; print([p['id'] for p in list_plugins_metadata()])"

# Smoke-test the full pipeline (simulation mode, no hardware):
uv run python -c "import src.config; src.config.IS_RASPBERRY_PI = False; from fastapi.testclient import TestClient; from src.main import app; c = TestClient(app); assert c.post('/scene', json={'layers': [{'id': 'L1', 'type': 'my_effect'}]}).status_code == 200; assert c.patch('/scene/layers/L1', json={'speed': 2.0}).status_code == 200; print('OK')"
```

---

## 10. Reference Implementations

Study these in `lights-control/src/plugins/builtin/`:

- **`lava_plugin.py` (`lava_v2`)** — 2D metaball simulation with `on_init` state allocation, `live=False` param (`blob_count`), and opt-in `OpacityParam` + `HeightRangeParams`.
- **`wireframes_plugin.py` (`wireframes_live`)** — real-time 3D SDF (`world_to_local` + `sdf_box_wireframe_3d`) with choice-driven geometry and `BlendModeParam`; deliberately *no* opacity/height controls.
