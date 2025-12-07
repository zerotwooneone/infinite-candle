# src/api/schemas.py
from typing import Literal, List, Union, Annotated
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
    angle: float = 0.0
    width: float = 0.2
    speed: float = 1.0

class SnowLayer(BaseLayer):
    type: Literal["snow"]
    color: List[int] = [200, 200, 255]
    flake_count: int = 40
    gravity: float = 0.5

class FireLayer(BaseModel):
    type: Literal["fire"]
    opacity: float = 1.0
    wax_height: float = 0.15     # How much of the bottom is solid wax?
    cooling: float = 0.15        # How fast fire fades as it rises
    sparking: float = 0.3        # How vigorously it flickers at the base

# --- The Union ---
EffectConfig = Annotated[
    Union[SolidLayer, ChaseLayer, StripesLayer, SnowLayer, FireLayer], # <--- Add FireLayer
    Field(discriminator="type")
]

class SceneRequest(BaseModel):
    layers: List[EffectConfig]