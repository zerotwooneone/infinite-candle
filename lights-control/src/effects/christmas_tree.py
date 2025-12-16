import math

import numpy as np

from src.effects.abstract import Effect


def _wrap_distance(a: np.ndarray, b: float) -> np.ndarray:
    raw = np.abs(a - b)
    return np.minimum(raw, 1.0 - raw)


class ChristmasTreeEffect(Effect):
    def __init__(self, config):
        super().__init__(config)
        self.t = 0.0

        self.rotate_speed = float(getattr(config, "rotate_speed", 0.1))
        self.brightness = float(getattr(config, "brightness", 1.0))

        self.ornament_count = int(getattr(config, "ornament_count", 30))
        self.ornament_size = float(getattr(config, "ornament_size", 0.025))

        self.tree_color = np.array(getattr(config, "tree_color", [0, 120, 0]), dtype=float)
        self.star_color = np.array(getattr(config, "star_color", [255, 220, 0]), dtype=float)

        palette = getattr(
            config,
            "ornament_palette",
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 120, 255],
                [255, 0, 255],
                [255, 140, 0],
                [0, 255, 255],
            ],
        )
        self.ornament_palette = np.array(palette, dtype=float)

        rng = np.random.default_rng()
        self.ornament_x = rng.uniform(0.0, 1.0, self.ornament_count)
        self.ornament_y = rng.uniform(self.config.h_min, self.config.h_max, self.ornament_count)
        self.ornament_color_idx = rng.integers(0, len(self.ornament_palette), size=self.ornament_count)

    def update(self, dt: float):
        self.t += dt

    def render(self, buffer: np.ndarray, mapper):
        x = mapper.coords_x
        y = mapper.coords_y

        h_min = float(getattr(self.config, "h_min", 0.0))
        h_max = float(getattr(self.config, "h_max", 1.0))

        height = max(1e-6, h_max - h_min)
        y_norm = np.clip((y - h_min) / height, 0.0, 1.0)

        rot = (self.t * self.rotate_speed) % 1.0
        x_rot = (x + rot) % 1.0

        layer = np.zeros(buffer.shape, dtype=float)

        # --- Tree body (cone silhouette)
        # Make a cone "wedge" centered on x=0.5. Width shrinks toward the top.
        tree_mask_y = (y >= h_min) & (y <= h_max)
        half_width = 0.48 * (1.0 - y_norm) + 0.03
        dx_center = _wrap_distance(x_rot, 0.5)
        tree_mask = tree_mask_y & (dx_center <= half_width)

        if np.any(tree_mask):
            # Subtle shading so it feels dimensional
            shade = 0.65 + 0.35 * (1.0 - (dx_center / np.maximum(half_width, 1e-6)))
            shade = shade[:, np.newaxis]
            layer[tree_mask] = (self.tree_color * shade[tree_mask]).astype(float)

        # --- Ornaments (colored spots)
        if self.ornament_count > 0:
            led_y = y[np.newaxis, :]
            led_x = x_rot[np.newaxis, :]

            p_y = self.ornament_y[:, np.newaxis]
            p_x = (self.ornament_x + rot) % 1.0
            p_x = p_x[:, np.newaxis]

            dy = np.abs(led_y - p_y) * mapper.aspect_ratio
            dx = np.minimum(np.abs(led_x - p_x), 1.0 - np.abs(led_x - p_x))

            dist_sq = (dx * dx) + (dy * dy)
            sigma = max(1e-6, self.ornament_size)
            w = np.exp(-dist_sq / (2.0 * sigma * sigma))

            # Only show ornaments on the tree body
            w *= tree_mask[np.newaxis, :].astype(float)

            # Accumulate with a soft-max style blend to avoid blowing out
            ornament_colors = self.ornament_palette[self.ornament_color_idx]
            for i in range(self.ornament_count):
                wi = w[i][:, np.newaxis]
                layer = np.maximum(layer, ornament_colors[i] * wi)

        # --- Star (top region, 5-point modulation)
        star_height = float(getattr(self.config, "star_height", 0.08))
        star_y0 = h_max - (star_height * height)
        star_mask_y = (y >= star_y0) & (y <= h_max)

        if np.any(star_mask_y):
            theta = (x_rot - 0.5) * (2.0 * math.pi)
            # 5-point "spikiness" around the circumference
            spikes = 0.5 + 0.5 * np.cos(5.0 * theta)
            # Larger spikes near the very top
            tip = np.clip((y - star_y0) / max(1e-6, (h_max - star_y0)), 0.0, 1.0)
            spike_threshold = 0.35 + 0.55 * (1.0 - tip)
            star_mask = star_mask_y & (spikes >= spike_threshold)

            layer[star_mask] = self.star_color

        # Apply brightness and opacity blend
        layer *= np.clip(self.brightness, 0.0, 1.0)
        alpha = float(getattr(self.config, "opacity", 1.0))

        if alpha >= 1.0:
            buffer[:] = layer.astype(np.uint8)
        else:
            bg = buffer.astype(float)
            buffer[:] = ((bg * (1.0 - alpha)) + (layer * alpha)).astype(np.uint8)
