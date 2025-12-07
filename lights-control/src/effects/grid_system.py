import numpy as np
from src.effects.abstract import Effect
from src.config import LED_COUNT

class GridSystem(Effect):
    def __init__(self, config, width: int = 32, height: int = 72):
        super().__init__(config)

        # --- VIRTUAL DISPLAY RESOLUTION ---
        # 32 pixels around the circumference (~0.6 inches per pixel)
        # 72 pixels high (~0.6 inches per pixel) -> Roughly square pixels
        self.grid_w = width
        self.grid_h = height

        # The Framebuffer (Float 0.0-1.0 or Int 0-255 depending on subclass)
        # We use Float (N, M, 3) for full RGB support by default
        self.canvas = np.zeros((self.grid_h, self.grid_w, 3), dtype=float)

        # Optimization: Pre-calculated lookup tables
        self.led_map_x = np.zeros(LED_COUNT, dtype=int)
        self.led_map_y = np.zeros(LED_COUNT, dtype=int)
        self.needs_mapping = True

    def map_coords(self, mapper):
        """
        One-time setup: Figures out which Grid Cell each LED belongs to.
        """
        # Map LED 0.0-1.0 coordinates to Grid Integer coordinates
        # Y: 0 is Bottom, Grid_H-1 is Top
        self.led_map_y = (mapper.coords_y * (self.grid_h - 1)).astype(int)

        # X: 0 is Front, Grid_W-1 is Back
        self.led_map_x = (mapper.coords_x * (self.grid_w - 1)).astype(int)

        self.needs_mapping = False

    def render_grid(self, buffer, mapper):
        """
        Transfers the Virtual Canvas to the Physical LEDs
        """
        if self.needs_mapping:
            self.map_coords(mapper)

        # 1. Sample the Grid
        # "Advanced Indexing": We use the lookup arrays to pull colors 
        # for all 600 LEDs instantly from the 2D canvas.
        sampled_colors = self.canvas[self.led_map_y, self.led_map_x]

        # 2. Write to Buffer
        # We overwrite the buffer (Grid effects usually dominate the scene)
        buffer[:] = sampled_colors.astype(np.uint8)

    def draw_pixel(self, x, y, color):
        """Helper to draw safely with wrapping"""
        # Wrap X (Cylinder property)
        x = x % self.grid_w

        # Clamp Y (Floor/Ceiling property)
        if 0 <= y < self.grid_h:
            self.canvas[y, x] = color