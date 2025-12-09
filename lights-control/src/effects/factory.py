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
def create_effect(config):
    t = config.type
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

    return None