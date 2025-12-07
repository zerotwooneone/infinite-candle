import numpy as np
from src.effects.abstract import Effect
from src.config import LED_COUNT

class GridSystem(Effect):
    def __init__(self, config, width: int = 32, height: int = 72):
        super().__init__(config)
        # ... existing setup ...

        # Check if the config wants transparency
        # (We default to True for most overlays)
        self.transparent_bg = getattr(config, 'transparent', True)
        self.bg_color = np.array(getattr(config, 'bg_color', [0,0,0]), dtype=int)
        self.opacity = config.opacity

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
        if self.needs_mapping:
            self.map_coords(mapper)
    
        # 1. Sample the Grid
        sampled_colors = self.canvas[self.led_map_y, self.led_map_x]
    
        # 2. Handle Transparency / Blending
    
        # Case A: No Transparency (Overwrite everything, like a TV screen)
        if not self.transparent_bg:
            if self.opacity >= 1.0:
                buffer[:] = sampled_colors.astype(np.uint8)
            else:
                current = buffer.astype(float)
                blended = (current * (1.0 - self.opacity)) + (sampled_colors * self.opacity)
                buffer[:] = blended.astype(np.uint8)
            return
    
        # Case B: Transparent Background (Overlay Mode)
        # We only draw pixels that are NOT the background color
    
        # Create a mask: (Color != BG_Color)
        # np.all(..., axis=1) checks if [R,G,B] matches BG [R,G,B]
        # We want pixels where they are DIFFERENT
        fg_mask = ~np.all(sampled_colors.astype(int) == self.bg_color, axis=1)
    
        # Only touch the foreground pixels
        if self.opacity >= 1.0:
            buffer[fg_mask] = sampled_colors[fg_mask].astype(np.uint8)
        else:
            # Blend only the foreground pixels
            current_fg = buffer[fg_mask].astype(float)
            incoming_fg = sampled_colors[fg_mask]
    
            blended = (current_fg * (1.0 - self.opacity)) + (incoming_fg * self.opacity)
            buffer[fg_mask] = blended.astype(np.uint8)

    def draw_pixel(self, x, y, color):
        """Helper to draw safely with wrapping"""
        # Wrap X (Cylinder property)
        x = x % self.grid_w

        # Clamp Y (Floor/Ceiling property)
        if 0 <= y < self.grid_h:
            self.canvas[y, x] = color