# lights-control/src/effects/factory.py
from src.effects.stripes import StripesEffect
from src.effects.solid import SolidEffect
from src.effects.snow import SnowEffect
from src.effects.fire import FireEffect
from src.effects.fireworks import FireworkEffect
from src.effects.gol import GameOfLifeEffect
from src.effects.lava import LavaLampEffect
from src.effects.breathing import BreathingEffect
from src.effects.player import ClipPlayerEffect
from src.effects.alien import AlienAbductionEffect
from src.effects.christmas_tree import ChristmasTreeEffect
from src.plugins.base import PluginEffectAdapter
from src.plugins.registry import get_plugin_class
from src.engine.clock import AnimationClock


def create_effect(config, clock: AnimationClock | None = None):
    t = config.type

    # 1. Check new plugin registry first
    plugin_cls = get_plugin_class(t)
    if plugin_cls is not None:
        raw = config.model_dump() if hasattr(config, "model_dump") else dict(config.__dict__)
        plugin = plugin_cls(initial_params=raw)
        return PluginEffectAdapter(plugin, config, clock or AnimationClock())

    # 2. Legacy built-in effects (unchanged)
    if t == "stripes": return StripesEffect(config)
    elif t == "solid": return SolidEffect(config)
    elif t == "snow": return SnowEffect(config)
    elif t == "fire": return FireEffect(config)
    elif t == "fireworks": return FireworkEffect(config)
    elif t == "gol": return GameOfLifeEffect(config)
    elif t == "lava": return LavaLampEffect(config)
    elif t == "breathing": return BreathingEffect(config)
    elif t == "clip": return ClipPlayerEffect(config)
    elif t == "alien": return AlienAbductionEffect(config)
    elif t == "christmas_tree": return ChristmasTreeEffect(config)

    return None