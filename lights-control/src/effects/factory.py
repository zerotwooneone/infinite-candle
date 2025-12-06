from src.effects.stripes import StripesEffect
from src.effects.solid import SolidEffect    # <--- Add this
from src.effects.snow import SnowEffect      # <--- Add this

def create_effect(config):
    if config.type == "stripes":
        return StripesEffect(config)
    elif config.type == "solid":
        return SolidEffect(config)
    elif config.type == "snow":
        return SnowEffect(config)

    return None