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

class FireworkLayer(BaseLayer):
    # ---------------------------------------------------------
    type: Literal["fireworks"]
    launch_rate: float = 0.5
    burst_height: float = 0.8
    explosion_size: float = 0.15

class GameOfLifeLayer(BaseLayer):
    type: Literal["gol"]
    color: List[int] = [0, 255, 0]
    bg_color: List[int] = [0, 0, 0]
    speed: float = 10.0
    # NEW: Allow turning off transparency if they WANT the black box
    transparent: bool = True                # Generations per second

class LavaLampLayer(BaseLayer):
    type: Literal["lava"]
    color: List[int] = [255, 0, 0]      # The Blob Color (e.g., Red)
    bg_color: List[int] = [20, 0, 0]    # The Fluid Color (e.g., Dark Red)
    blob_count: int = 6                 # How many wax blobs?
    speed: float = 0.5                  # How fast they heat/cool

class BreathingLayer(BaseLayer):
    type: Literal["breathing"]
    color_a: List[int] = [50, 50, 50]   # Low state
    color_b: List[int] = [255, 255, 255] # High state
    speed: float = 0.5                  # Cycles per second (Breaths per second)

class ClipLayer(BaseLayer):
    type: Literal["clip"]
    filename: str

# Update Union
EffectConfig = Annotated[
    Union[SolidLayer, ChaseLayer, StripesLayer, SnowLayer, FireLayer,
    FireworkLayer, GameOfLifeLayer, LavaLampLayer, BreathingLayer,ClipLayer], # <--- Add
    Field(discriminator="type")
]

class SceneRequest(BaseModel):
    layers: List[EffectConfig]