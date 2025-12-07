import numpy as np
from src.effects.abstract import Effect
from src.config import LED_COUNT, PILLAR_WRAPS

class FireEffect(Effect):
    def __init__(self, config):
        super().__init__(config)

        self.cooling = config.cooling / 10.0
        self.sparking = config.sparking / 5.0
        self.h_min = config.h_min
        self.h_max = config.h_max

        # --- Generic Color Palette ---
        self.color_bg = np.array([0, 0, 0], dtype=float)
        self.color_start = np.array(config.color_start, dtype=float)
        self.color_end = np.array(config.color_end, dtype=float)

        # --- 2D HEAT MAP STATE ---
        self.grid_h = 100
        self.grid_w = int(round(PILLAR_WRAPS))
        self.heat_map = np.zeros((self.grid_h, self.grid_w), dtype=float)

        # Pre-calculation flags
        self.led_map_y = np.zeros(LED_COUNT, dtype=int)
        self.led_map_x = np.zeros(LED_COUNT, dtype=int)
        self.needs_mapping = True

    def update(self, dt: float):
        # 1. Cool Down
        cool_map = np.random.uniform(0.0, self.cooling, self.heat_map.shape)
        self.heat_map -= cool_map
        np.clip(self.heat_map, 0.0, 1.0, out=self.heat_map)

        # 2. Heat Rises
        self.heat_map = np.roll(self.heat_map, -1, axis=0)

        # 3. Ignite Base (At h_min)
        # We calculate which row in the grid corresponds to h_min
        ignite_row = int((1.0 - self.h_min) * (self.grid_h - 1))

        # Keep bounds safe
        ignite_row = np.clip(ignite_row, 0, self.grid_h - 1)

        # Create sparks
        base_heat = 0.3
        flicker = np.random.uniform(0.0, self.sparking, self.grid_w)

        self.heat_map[ignite_row, :] = base_heat + flicker

        # IMPORTANT: Force everything "below" the fire source to be cold
        # (Since index 0 is Top, "below" means index > ignite_row)
        if ignite_row < self.grid_h - 1:
            self.heat_map[ignite_row+1:, :] = 0.0

    def render(self, buffer: np.ndarray, mapper):
        if self.needs_mapping:
            # Map 0.0-1.0 height to grid coordinates
            self.led_map_y = ((1.0 - mapper.coords_y) * (self.grid_h - 1)).astype(int)
            self.led_map_x = (mapper.coords_x * (self.grid_w - 1)).astype(int)
            self.needs_mapping = False

        # 1. Height Masking (Optimization)
        # We only care about LEDs within the active fire zone
        valid_mask = (mapper.coords_y >= self.h_min) & (mapper.coords_y <= self.h_max)

        # 2. Get Heat
        # Extract heat only for valid LEDs
        relevant_leds_y = self.led_map_y[valid_mask]
        relevant_leds_x = self.led_map_x[valid_mask]
        heat_values = self.heat_map[relevant_leds_y, relevant_leds_x]

        # 3. Color Interpolation
        # Heat 0.0 -> 0.5: Black -> StartColor
        # Heat 0.5 -> 1.0: StartColor -> EndColor

        colors = np.zeros((len(heat_values), 3), dtype=float)

        # Split into Low/High heat groups
        mask_low = heat_values < 0.5
        mask_high = ~mask_low

        # Normalize 0.0-0.5 range to 0.0-1.0
        heat_low = heat_values[mask_low] * 2.0
        # Normalize 0.5-1.0 range to 0.0-1.0
        heat_high = (heat_values[mask_high] - 0.5) * 2.0

        # Low Heat: Interpolate Black -> Start
        colors[mask_low] = (1.0 - heat_low[:, None]) * self.color_bg + \
                           heat_low[:, None] * self.color_start

        # High Heat: Interpolate Start -> End
        colors[mask_high] = (1.0 - heat_high[:, None]) * self.color_start + \
                            heat_high[:, None] * self.color_end

        # 4. Write to Buffer
        buffer[valid_mask] = colors.astype(np.uint8)