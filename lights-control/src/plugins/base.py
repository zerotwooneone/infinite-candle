# lights-control/src/plugins/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np

from src.effects.abstract import Effect
from src.engine.clock import AnimationClock
from src.engine.mapper import PillarMapper
from src.plugins import toolkit as tk


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    kind: Literal["float", "int", "color", "bool", "choice", "text"]
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | None = None
    live: bool = True  # If False, changing this param triggers `on_init()`
    max_len: int | None = None  # text params only: truncate to this many chars

    def coerce(self, value: Any) -> Any:
        if value is None:
            return self.default
        try:
            if self.kind == "float":
                v = float(value)
                if self.min is not None:
                    v = max(self.min, v)
                if self.max is not None:
                    v = min(self.max, v)
                return v
            if self.kind == "int":
                v = int(round(float(value)))
                if self.min is not None:
                    v = max(int(self.min), v)
                if self.max is not None:
                    v = min(int(self.max), v)
                return v
            if self.kind == "color":
                arr = list(value)
                if len(arr) != 3:
                    return list(self.default)
                return [max(0, min(255, int(c))) for c in arr]
            if self.kind == "bool":
                return bool(value)
            if self.kind == "choice":
                s = str(value)
                if self.options and s not in self.options:
                    return self.default
                return s
            if self.kind == "text":
                s = str(value)
                if self.max_len is not None:
                    s = s[: int(self.max_len)]
                return s
        except (TypeError, ValueError):
            return self.default
        return value

    def to_schema_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "options": self.options,
            "live": self.live,
            "max_len": self.max_len,
        }


def FloatParam(name: str, label: str, default: float, min: float = 0.0, max: float = 1.0, step: float = 0.05, live: bool = True) -> ParamSpec:
    return ParamSpec(name=name, label=label, kind="float", default=default, min=min, max=max, step=step, live=live)


def IntParam(name: str, label: str, default: int, min: int = 1, max: int = 100, step: int = 1, live: bool = True) -> ParamSpec:
    return ParamSpec(name=name, label=label, kind="int", default=default, min=min, max=max, step=step, live=live)


def ColorParam(name: str, label: str, default: list[int] | tuple[int, int, int] = (255, 255, 255), live: bool = True) -> ParamSpec:
    return ParamSpec(name=name, label=label, kind="color", default=list(default), live=live)


def BoolParam(name: str, label: str, default: bool = False, live: bool = True) -> ParamSpec:
    return ParamSpec(name=name, label=label, kind="bool", default=default, live=live)


def ChoiceParam(name: str, label: str, options: list[str], default: str, live: bool = True) -> ParamSpec:
    return ParamSpec(name=name, label=label, kind="choice", default=default, options=options, live=live)


def TextParam(name: str, label: str, default: str = "", max_len: int = 64, live: bool = True) -> ParamSpec:
    return ParamSpec(name=name, label=label, kind="text", default=default, live=live, max_len=max_len)


# --- Opt-in Reusable Parameter Presets ---
# Plugins include these in `params = [...]` ONLY when they want those UI controls shown.

def OpacityParam(default: float = 1.0, label: str = "Opacity") -> ParamSpec:
    return FloatParam("opacity", label, default=default, min=0.0, max=1.0, step=0.05)


def BlendModeParam(default: str = "alpha", label: str = "Blend Mode") -> ParamSpec:
    return ChoiceParam("blend_mode", label, options=["alpha", "add", "max", "overwrite"], default=default)


def HeightRangeParams(h_min: float = 0.0, h_max: float = 1.0) -> list[ParamSpec]:
    return [
        FloatParam("h_min", "Start Height", default=h_min, min=0.0, max=1.0, step=0.05),
        FloatParam("h_max", "End Height", default=h_max, min=0.0, max=1.0, step=0.05),
    ]


@dataclass
class PluginContext:
    leds: PillarMapper
    clock: AnimationClock
    buffer: np.ndarray  # Live (600, 3) uint8 frame buffer from lower layers


class AnimationPlugin(ABC):
    """
    Base class for all plugins.
    Each plugin defines ONLY the UI controls it wants in `params = [...]`.
    Nothing (`opacity`, `h_min`, `h_max`, `blend_mode`) is forced onto a plugin unless
    the plugin includes those `ParamSpec`s in `params`.
    """
    plugin_id: str = ""
    display_name: str = ""
    description: str = ""
    params: list[ParamSpec] = []

    def __init__(self, initial_params: dict[str, Any] | None = None):
        self._spec_map: dict[str, ParamSpec] = {p.name: p for p in self.params}
        self.values: dict[str, Any] = {p.name: p.default for p in self.params}
        if initial_params:
            for k, p in self._spec_map.items():
                if k in initial_params:
                    self.values[k] = p.coerce(initial_params[k])
        self.on_init()

    def on_init(self) -> None:
        """Optional hook to allocate state arrays based on `self.values`."""
        pass

    def update_params(self, incoming: dict[str, Any]) -> None:
        """
        Coerce and apply updated parameter values in-place.
        Calls `on_init()` if any `live=False` parameter changed; otherwise calls `on_params_changed(changed)`.
        """
        changed: dict[str, Any] = {}
        needs_rebuild = False
        for k, raw_val in incoming.items():
            spec = self._spec_map.get(k)
            if spec is None:
                continue
            new_val = spec.coerce(raw_val)
            if self.values.get(k) != new_val:
                self.values[k] = new_val
                changed[k] = new_val
                if not spec.live:
                    needs_rebuild = True

        if not changed:
            return
        if needs_rebuild:
            self.on_init()
        else:
            self.on_params_changed(changed)

    def on_params_changed(self, changed: dict[str, Any]) -> None:
        """Optional hook for custom reactions to live parameter updates."""
        pass

    @abstractmethod
    def render(self, ctx: PluginContext) -> np.ndarray | tuple[np.ndarray, np.ndarray | float] | None:
        """
        Compute one frame for the 600 LEDs.
        A plugin can either:
          1. Return `colors` `(600, 3)` -> blended into `ctx.buffer` using any opt-in `opacity` / `blend_mode` / `h_min` / `h_max` params declared in `self.values` (or overwritten if none declared).
          2. Return `(colors, alpha)` where `alpha` is `(600,)` or `float` -> blended with `alpha`.
          3. Mutate `ctx.buffer` directly (e.g. via `tk.blend_into_buffer(ctx.buffer, ...)`) and return `None`.
        """
        pass

    @classmethod
    def get_metadata(cls) -> dict[str, Any]:
        return {
            "id": cls.plugin_id,
            "name": cls.display_name or cls.plugin_id,
            "description": cls.description,
            "params": [p.to_schema_dict() for p in cls.params],
            "defaults": {p.name: p.default for p in cls.params},
        }


class PluginEffectAdapter(Effect):
    """Adapts an `AnimationPlugin` to the `Engine` layer interface."""

    def __init__(self, plugin: AnimationPlugin, config: Any, clock: AnimationClock):
        super().__init__(config)
        self.plugin = plugin
        self.clock = clock

    def apply_config_update(self, new_config: Any) -> None:
        self.config = new_config
        raw = new_config.model_dump() if hasattr(new_config, "model_dump") else dict(new_config.__dict__)
        self.plugin.update_params(raw)

    def update(self, dt: float) -> None:
        pass

    def render(self, buffer: np.ndarray, mapper: PillarMapper) -> None:
        ctx = PluginContext(leds=mapper, clock=self.clock, buffer=buffer)
        result = self.plugin.render(ctx)
        if result is None:
            return

        if isinstance(result, tuple):
            colors, pixel_alpha = result
        else:
            colors = result
            pixel_alpha = 1.0

        # Honor `opacity`, `blend_mode`, `h_min`, `h_max` ONLY if the plugin declared them in `self.values`
        vals = self.plugin.values
        opacity = float(vals.get("opacity", 1.0))
        blend_mode = str(vals.get("blend_mode", "alpha" if isinstance(result, tuple) or "opacity" in vals else "overwrite"))

        mask = None
        if "h_min" in vals or "h_max" in vals:
            h_min = float(vals.get("h_min", 0.0))
            h_max = float(vals.get("h_max", 1.0))
            mask = (mapper.coords_y >= h_min) & (mapper.coords_y <= h_max)

        combined_alpha = pixel_alpha * opacity if np.ndim(pixel_alpha) > 0 else float(pixel_alpha) * opacity
        tk.blend_into_buffer(buffer, colors, alpha=combined_alpha, mask=mask, mode=blend_mode)
