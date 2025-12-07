import numpy as np
from src.effects.abstract import Effect
from src.config import LED_COUNT

class GridSystem(Effect):
    def __init__(self, config, width: int = 32, height: int = 72):
        super().__init__(config)

        # --- VIRTUAL DISPLAY RESOLUTION ---
        self.grid_w = width
        self.grid_h = height

        # The Framebuffer
        self.canvas = np.zeros((self.grid_h, self.grid_w, 3), dtype=float)

        # Optimization: Pre-calculated lookup tables
        self.led_map_x = np.zeros(LED_COUNT, dtype=int)
        self.led_map_y = np.zeros(LED_COUNT, dtype=int)
        self.needs_mapping = True

        # Transparency settings
        self.transparent_bg = getattr(config, 'transparent', True)
        self.bg_color = np.array(getattr(config, 'bg_color', [0,0,0]), dtype=int)
        self.opacity = getattr(config, 'opacity', 1.0)

    def map_coords(self, mapper):
        """One-time setup: Figures out which Grid Cell each LED belongs to."""
        self.led_map_y = (mapper.coords_y * (self.grid_h - 1)).astype(int)
        self.led_map_x = (mapper.coords_x * (self.grid_w - 1)).astype(int)
        self.needs_mapping = False

    def render_grid(self, buffer, mapper):
        """Transfers the Virtual Canvas to the Physical LEDs"""
        if self.needs_mapping:
            self.map_coords(mapper)

        # 1. Sample the Grid
        sampled_colors = self.canvas[self.led_map_y, self.led_map_x]

        # 2. Handle Transparency / Blending
        if not self.transparent_bg:
            if self.opacity >= 1.0:
                buffer[:] = sampled_colors.astype(np.uint8)
            else:
                current = buffer.astype(float)
                blended = (current * (1.0 - self.opacity)) + (sampled_colors * self.opacity)
                buffer[:] = blended.astype(np.uint8)
            return

        # Transparent Background (Overlay Mode)
        fg_mask = ~np.all(sampled_colors.astype(int) == self.bg_color, axis=1)

        if self.opacity >= 1.0:
            buffer[fg_mask] = sampled_colors[fg_mask].astype(np.uint8)
        else:
            current_fg = buffer[fg_mask].astype(float)
            incoming_fg = sampled_colors[fg_mask]
            blended = (current_fg * (1.0 - self.opacity)) + (incoming_fg * self.opacity)
            buffer[fg_mask] = blended.astype(np.uint8)

    def draw_pixel(self, x, y, color):
        """Helper to draw safely with wrapping"""
        x = x % self.grid_w
        if 0 <= y < self.grid_h:
            self.canvas[y, x] = color