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

    max_rockets: int = 6
    max_sparks: int = 1400
    spark_density: float = 1.0

    rocket_speed: float = 0.95
    rocket_wiggle: float = 0.08
    rocket_gravity: float = -0.35

    spark_gravity: float = -0.65
    spark_drag: float = 0.10

    trail_decay: float = 2.8
    brightness: float = 1.0

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
    transparent: bool = False

# src/api/schemas.py

class AlienLayer(BaseLayer):
    type: Literal["alien"]
    ship_color_1: List[int] = [50, 150, 255]  # Light Blue lines
    ship_color_2: List[int] = [0, 255, 50]    # Green Dashed line
    beam_color: List[int] = [100, 255, 255]   # Cyan Spotlight
    speed: float = 1.0
    transparent: bool = True                  # Usually want this overlaid

class ChristmasTreeLayer(BaseLayer):
    type: Literal["christmas_tree"]
    rotate_speed: float = 0.1
    brightness: float = 1.0
    thickness: float = 0.08
    ornament_count: int = 30
    ornament_size: float = 0.025
    tree_color: List[int] = [0, 120, 0]
    star_color: List[int] = [255, 220, 0]
    ornament_palette: List[List[int]] = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 120, 255],
        [255, 0, 255],
        [255, 140, 0],
        [0, 255, 255],
    ]
    star_height: float = 0.08

# Update Union
EffectConfig = Annotated[
    Union[SolidLayer, ChaseLayer, StripesLayer, SnowLayer, FireLayer,
    FireworkLayer, GameOfLifeLayer, LavaLampLayer, BreathingLayer,ClipLayer, AlienLayer, ChristmasTreeLayer], # <--- Add
    Field(discriminator="type")
]

class SceneRequest(BaseModel):
    layers: List[EffectConfig]