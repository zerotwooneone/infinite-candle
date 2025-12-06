from typing import Literal, List, Union, Annotated, Optional
from pydantic import BaseModel, Field

# --- Base ---
class BaseLayer(BaseModel):
    opacity: float = 1.0

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
    angle: float = 0.0      # 0 = Horizontal, 90 = Vertical
    width: float = 0.2      # Stripe thickness
    speed: float = 1.0      # Scroll speed

class SnowLayer(BaseLayer):
    type: Literal["snow"]
    color: List[int] = [200, 200, 255]
    flake_count: int = 40
    gravity: float = 0.5

# --- The Union ---
# The 'discriminator' tells Pydantic to look at the 'type' field 
# to decide which class to instantiate.
EffectConfig = Annotated[Union[SolidLayer, ChaseLayer, StripesLayer, SnowLayer], Field(discriminator="type")]

class SceneRequest(BaseModel):
    layers: List[EffectConfig]