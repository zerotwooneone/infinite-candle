from typing import Literal, List, Union, Annotated
from pydantic import BaseModel, Field

# --- Base ---
class BaseLayer(BaseModel):
    opacity: float = 1.0
    # RESTORED: These now apply to ALL layers
    h_min: float = 0.0
    h_max: float = 1.0

# --- Effect Configs ---
class SolidLayer(BaseLayer):
    type: Literal["solid"]
    color: List[int]

class ChaseLayer(BaseLayer):
    type: Literal["chase"]
    color: List[int]
    speed: float = 20.0
    tail_length: int = 30

class StripesLayer(BaseLayer):
    type: Literal["stripes"]
    color_a: List[int] = [0, 0, 0]
    color_b: List[int] = [255, 255, 255]
    angle: float = 0.0
    width: float = 0.2
    speed: float = 1.0

class SnowLayer(BaseLayer):
    type: Literal["snow"]
    color: List[int] = [200, 200, 255]
    flake_count: int = 40
    gravity: float = 0.05

class FireLayer(BaseLayer):
    type: Literal["fire"]
    cooling: float = 0.15
    sparking: float = 0.3
    # NEW: Custom Color Gradient (Start -> End)
    color_start: List[int] = [255, 0, 0]    # Red
    color_end: List[int] = [255, 255, 0]    # Yellow

class FireworkLayer(BaseModel):
    type: Literal["fireworks"]
    opacity: float = 1.0
    launch_rate: float = 0.5    # Rockets per second
    burst_height: float = 0.8   # Average height of explosion (0.0 - 1.0)
    explosion_size: float = 0.15 # How wide the burst spreads

# --- The Union ---
EffectConfig = Annotated[
    Union[SolidLayer, ChaseLayer, StripesLayer, SnowLayer, FireLayer, FireworkLayer], # <--- Add FireworkLayer
    Field(discriminator="type")
]

class SceneRequest(BaseModel):
    layers: List[EffectConfig]