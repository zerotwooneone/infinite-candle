# src/engine/mapper.py
import numpy as np
from src.config import LED_COUNT, PILLAR_WRAPS, PILLAR_FACES

HEIGHT_INCHES = 48.0
CIRCUMFERENCE_INCHES = 21.0
RADIUS_INCHES = CIRCUMFERENCE_INCHES / (2.0 * np.pi)


class PillarMapper:
    def __init__(self):
        self.count = LED_COUNT
        self.wraps = PILLAR_WRAPS
        self.faces = PILLAR_FACES
        self.height_inches = HEIGHT_INCHES
        self.circumference_inches = CIRCUMFERENCE_INCHES
        self.radius_inches = RADIUS_INCHES

        # --- 1. 1D Strand Addressing ---
        self.index = np.arange(LED_COUNT, dtype=int)                  # (600,) 0..599
        self.u = self.index / float(LED_COUNT)                        # (600,) 0.0..1.0

        # --- 2. Wrap / Ring Addressing ---
        self.leds_per_wrap = LED_COUNT / PILLAR_WRAPS                 # ~30.9278
        self.wrap = np.floor(self.index / self.leds_per_wrap).astype(int)  # (600,) 0..19
        self.wrap_u = (self.index % self.leds_per_wrap) / self.leds_per_wrap  # (600,) 0.0..1.0

        # --- 3. 2D Cylinder Surface Addressing (Legacy + Alias Names) ---
        self.coords_y = self.u.copy()                                 # (600,) 0.0 (bottom) -> 1.0 (top)
        self.coords_x = self.wrap_u.copy()                            # (600,) 0.0 -> 1.0 azimuth
        self.y = self.coords_y
        self.theta = self.coords_x
        self.theta_rad = self.theta * (2.0 * np.pi)                   # (600,) 0..2pi
        self.aspect_ratio = HEIGHT_INCHES / CIRCUMFERENCE_INCHES      # ~2.2857

        # --- 4. Pillar Face Addressing (4 Square Faces: 0, 1, 2, 3) ---
        self.face = np.floor(self.theta * PILLAR_FACES).astype(int) % PILLAR_FACES  # (600,) 0..3
        self.face_u = (self.theta * PILLAR_FACES) % 1.0               # (600,) 0.0..1.0 within face

        # --- 5. 3D World Coordinates (Inches, Centered at X=0, Y=0, Z=0..48) ---
        self.world_x = RADIUS_INCHES * np.cos(self.theta_rad)         # (600,)
        self.world_y = RADIUS_INCHES * np.sin(self.theta_rad)         # (600,)
        self.world_z = self.y * HEIGHT_INCHES                         # (600,)
        self.xyz = np.stack([self.world_x, self.world_y, self.world_z], axis=1)  # (600, 3)

        # Centered 3D coordinates (Z centered at 0.0: -24"..+24") for 3D object rotations
        self.xyz_centered = np.stack(
            [self.world_x, self.world_y, self.world_z - (HEIGHT_INCHES * 0.5)],
            axis=1,
        )  # (600, 3)

        # Outward unit normal vectors on cylinder wall
        self.normals_xy = np.stack([np.cos(self.theta_rad), np.sin(self.theta_rad)], axis=1)  # (600, 2)