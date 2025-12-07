# src/effects/fire.py
import numpy as np
import random
from src.effects.abstract import Effect
from src.config import LED_COUNT, PILLAR_WRAPS

class FireEffect(Effect):
    def __init__(self, config):
        super().__init__(config)

        # --- Configuration ---
        self.wax_height_pct = config.wax_height
        # Cooling & Sparking must be scaled down for a 60fps loop
        # We divide by a rough factor to make the API inputs feel intuitive (0.0-1.0)
        self.cooling = config.cooling / 10.0
        self.sparking = config.sparking / 5.0

        # Colors
        self.color_wax = np.array([50, 45, 40], dtype=np.uint8) # Dull warm white/beige
        # Fire Palette (Heat 0.0 -> 1.0)
        self.color_cold = np.array([0, 0, 0], dtype=float)    # Black
        self.color_hot = np.array([255, 60, 0], dtype=float)  # Red-Orange
        self.color_bright = np.array([255, 200, 50], dtype=float) # Bright Yellow

        # --- 2D HEAT MAP STATE ---
        # We define a grid to run the physics simulation on.
        # Height = ~100 rows (enough resolution for smooth rising)
        # Width = Number of wraps (approx 19 columns for your pillar)
        self.grid_h = 100
        self.grid_w = int(round(PILLAR_WRAPS))
        # A 2D array storing heat values from 0.0 (cold) to 1.0 (hottest)
        self.heat_map = np.zeros((self.grid_h, self.grid_w), dtype=float)

        # Pre-calculate which heat map cell each LED belongs to
        # This saves massive CPU during the render loop
        # led_map_y will be an array of 600 integers (e.g., [0, 0, 1, 1, 2...])
        self.led_map_y = np.zeros(LED_COUNT, dtype=int)
        self.led_map_x = np.zeros(LED_COUNT, dtype=int)
        self.needs_mapping = True

    def update(self, dt: float):
        """Run the Fire Physics Simulation"""
        # 1. Cool Down
        # Every cell loses a little heat. Randomness prevents static patterns.
        cool_map = np.random.uniform(0.0, self.cooling, self.heat_map.shape)
        self.heat_map -= cool_map
        # Clamp to prevent negative heat
        np.clip(self.heat_map, 0.0, 1.0, out=self.heat_map)

        # 2. Heat Rises (Convection)
        # Roll the array upwards by 1 row.
        # The top row wraps to bottom, but we immediately overwrite it in step 3.
        self.heat_map = np.roll(self.heat_map, -1, axis=0)

        # 3. Ignite Base
        # The bottom row (index -1 after the roll) is the source of the fire.
        # Fill it with new sparks.

        # Determine the "ignite zone" - just above the wax
        wax_rows = int(self.grid_h * self.wax_height_pct)
        ignite_row_idx = self.grid_h - wax_rows - 1

        # Create sparks: A baseline heat + random intense flicker
        base_heat = 0.3
        flicker = np.random.uniform(0.0, self.sparking, self.grid_w)
        new_sparks = base_heat + flicker

        # Apply sparks to the ignition row
        self.heat_map[ignite_row_idx, :] = new_sparks
        # Ensure rows below ignition (the wax area) stay absolutely cold in the sim
        self.heat_map[ignite_row_idx+1:, :] = 0.0

        np.clip(self.heat_map, 0.0, 1.0, out=self.heat_map)

    def render(self, buffer: np.ndarray, mapper):
        """Map the 2D heat simulation to the 1D LEDs"""
        # One-time setup to map LEDs to grid cells based on geometry
        if self.needs_mapping:
            # mapper.coords_y is 0.0 to 1.0. Map to 0 to grid_h-1
            # We flip Y because heat grid 0 is top, but height 0.0 is bottom
            self.led_map_y = ((1.0 - mapper.coords_y) * (self.grid_h - 1)).astype(int)
            self.led_map_x = (mapper.coords_x * (self.grid_w - 1)).astype(int)
            self.needs_mapping = False

        # 1. Sample Heat for every LED
        # Use integer array indexing to pull values fast
        led_heat = self.heat_map[self.led_map_y, self.led_map_x]

        # 2. Color Mapping (Heat -> Color)
        # Create empty color array for all LEDs
        led_colors = np.zeros((LED_COUNT, 3), dtype=float)

        # Define masks for different heat zones
        is_wax = mapper.coords_y < self.wax_height_pct
        is_fire = ~is_wax

        # --- Render Wax ---
        led_colors[is_wax] = self.color_wax

        # --- Render Fire ---
        # Get heat values only for fire LEDs
        fire_heat = led_heat[is_fire]

        # Interpolate colors based on heat
        # Heat 0.0 - 0.5: Interpolate Black -> Red
        # Heat 0.5 - 1.0: Interpolate Red -> Yellow

        # Mask for lower half of heat range
        low_heat_mask = fire_heat < 0.5
        high_heat_mask = ~low_heat_mask

        # Normalize heat for interpolation (0.0-1.0 range for each half)
        heat_low_norm = fire_heat[low_heat_mask] * 2.0
        heat_high_norm = (fire_heat[high_heat_mask] - 0.5) * 2.0

        # Calculate colors (Vectorized linear interpolation)
        # Low: (1-t)*Black + t*Red
        colors_low = (1.0 - heat_low_norm[:, np.newaxis]) * self.color_cold + \
                     heat_low_norm[:, np.newaxis] * self.color_hot

        # High: (1-t)*Red + t*Yellow
        colors_high = (1.0 - heat_high_norm[:, np.newaxis]) * self.color_hot + \
                      heat_high_norm[:, np.newaxis] * self.color_bright

        # Assign back to the main color array
        # We need to map the subset (fire LEDs) back to the full set
        fire_colors_subset = np.zeros((np.sum(is_fire), 3), dtype=float)
        fire_colors_subset[low_heat_mask] = colors_low
        fire_colors_subset[high_heat_mask] = colors_high

        led_colors[is_fire] = fire_colors_subset

        # 3. Write to Final Buffer
        # Apply opacity if needed (config.opacity * led_colors)
        buffer[:] = led_colors.astype(np.uint8)