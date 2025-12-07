# src/effects/factory.py
from src.effects.stripes import StripesEffect
from src.effects.solid import SolidEffect
from src.effects.snow import SnowEffect
from src.effects.fire import FireEffect  # <--- Import

def create_effect(config):
    t = config.type
    if t == "stripes": return StripesEffect(config)
    elif t == "solid": return SolidEffect(config)
    elif t == "snow": return SnowEffect(config)
    elif t == "fire": return FireEffect(config) # <--- Add

    return None