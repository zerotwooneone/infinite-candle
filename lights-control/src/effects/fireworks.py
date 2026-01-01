# src/effects/fireworks.py
import math
from typing import Optional

import numpy as np

from src.effects.abstract import Effect


def _wrap_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 0.5) % 1.0 - 0.5


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = (h % 1.0) * 6.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    out = np.zeros((h.shape[0], 3), dtype=np.float32)
    m0 = i == 0
    m1 = i == 1
    m2 = i == 2
    m3 = i == 3
    m4 = i == 4
    m5 = i >= 5

    out[m0] = np.stack([v[m0], t[m0], p[m0]], axis=1)
    out[m1] = np.stack([q[m1], v[m1], p[m1]], axis=1)
    out[m2] = np.stack([p[m2], v[m2], t[m2]], axis=1)
    out[m3] = np.stack([p[m3], q[m3], v[m3]], axis=1)
    out[m4] = np.stack([t[m4], p[m4], v[m4]], axis=1)
    out[m5] = np.stack([v[m5], p[m5], q[m5]], axis=1)
    return out


class FireworkEffect(Effect):
    def __init__(self, config):
        super().__init__(config)
        self.t = 0.0

        self.launch_rate = float(getattr(config, "launch_rate", 0.5))
        self.burst_height = float(getattr(config, "burst_height", 0.8))
        self.explosion_size = float(getattr(config, "explosion_size", 0.55))

        self.max_rockets = int(getattr(config, "max_rockets", 6))
        self.max_sparks = int(getattr(config, "max_sparks", 1400))
        self.spark_density = float(getattr(config, "spark_density", 1.0))

        self.rocket_speed = float(getattr(config, "rocket_speed", 0.95))
        self.rocket_wiggle = float(getattr(config, "rocket_wiggle", 0.08))
        self.rocket_gravity = float(getattr(config, "rocket_gravity", -0.35))

        self.spark_gravity = float(getattr(config, "spark_gravity", -0.65))
        self.spark_drag = float(getattr(config, "spark_drag", 0.10))

        self.trail_decay = float(getattr(config, "trail_decay", 2.8))
        self.brightness = float(getattr(config, "brightness", 1.0))
        self.opacity = float(getattr(config, "opacity", 1.0))

        self._rng = np.random.default_rng()
        self._spawn_accum = 0.0

        self._trail = None

        self._rocket_alive = np.zeros((self.max_rockets,), dtype=bool)
        self._rocket_x = np.zeros((self.max_rockets,), dtype=np.float32)
        self._rocket_y = np.zeros((self.max_rockets,), dtype=np.float32)
        self._rocket_vx = np.zeros((self.max_rockets,), dtype=np.float32)
        self._rocket_vy = np.zeros((self.max_rockets,), dtype=np.float32)
        self._rocket_burst_y = np.zeros((self.max_rockets,), dtype=np.float32)
        self._rocket_hue = np.zeros((self.max_rockets,), dtype=np.float32)
        self._rocket_seed = np.zeros((self.max_rockets,), dtype=np.float32)

        self._spark_alive = np.zeros((self.max_sparks,), dtype=bool)
        self._spark_kind = np.zeros((self.max_sparks,), dtype=np.int8)
        self._spark_x = np.zeros((self.max_sparks,), dtype=np.float32)
        self._spark_y = np.zeros((self.max_sparks,), dtype=np.float32)
        self._spark_vx = np.zeros((self.max_sparks,), dtype=np.float32)
        self._spark_vy = np.zeros((self.max_sparks,), dtype=np.float32)
        self._spark_age = np.zeros((self.max_sparks,), dtype=np.float32)
        self._spark_life = np.ones((self.max_sparks,), dtype=np.float32)
        self._spark_rgb = np.zeros((self.max_sparks, 3), dtype=np.float32)
        self._spark_seed = np.zeros((self.max_sparks,), dtype=np.float32)

    def update(self, dt: float):
        self.t += float(dt)

        if self._trail is not None:
            self._trail *= float(math.exp(-self.trail_decay * dt))

        self._spawn_accum += dt * max(0.0, self.launch_rate)
        if self._spawn_accum >= 1.0:
            shells = int(self._spawn_accum)
            self._spawn_accum -= shells
            for _ in range(shells):
                self._spawn_rocket()

        self._step_rockets(dt)
        self._step_sparks(dt)

    def _spawn_rocket(self) -> None:
        free = np.where(~self._rocket_alive)[0]
        if free.size == 0:
            return

        i = int(free[0])
        h_min = float(getattr(self.config, "h_min", 0.0))
        h_max = float(getattr(self.config, "h_max", 1.0))
        y0 = h_min + 0.01

        target = float(np.clip(self.burst_height, 0.0, 1.0))
        burst_y = h_min + (h_max - h_min) * target
        burst_y += float(self._rng.uniform(-0.05, 0.04)) * (h_max - h_min)
        burst_y = float(np.clip(burst_y, h_min + 0.08, h_max - 0.05))

        self._rocket_alive[i] = True
        self._rocket_x[i] = float(self._rng.uniform(0.0, 1.0))
        self._rocket_y[i] = float(y0)
        self._rocket_vx[i] = float(self._rng.uniform(-0.02, 0.02))
        self._rocket_vy[i] = float(self.rocket_speed) * float(self._rng.uniform(0.85, 1.12))
        self._rocket_burst_y[i] = float(burst_y)
        self._rocket_hue[i] = float(self._rng.uniform(0.0, 1.0))
        self._rocket_seed[i] = float(self._rng.uniform(0.0, 1.0))

    def _step_rockets(self, dt: float) -> None:
        alive = self._rocket_alive
        if not np.any(alive):
            return

        idx = np.where(alive)[0]
        tt = float(self.t)
        phase = (tt * 4.0 + self._rocket_seed[idx] * 10.0) * (2.0 * math.pi)
        wiggle = np.sin(phase).astype(np.float32) * float(self.rocket_wiggle)

        self._rocket_vx[idx] = (self._rocket_vx[idx] * 0.92) + (wiggle * 0.02)
        self._rocket_vy[idx] += float(self.rocket_gravity) * dt

        self._rocket_x[idx] = (self._rocket_x[idx] + self._rocket_vx[idx] * dt) % 1.0
        self._rocket_y[idx] = self._rocket_y[idx] + self._rocket_vy[idx] * dt

        detonate = (self._rocket_y[idx] >= self._rocket_burst_y[idx]) | (self._rocket_vy[idx] <= 0.0)
        if np.any(detonate):
            for ridx in idx[detonate]:
                self._explode(int(ridx))

        out_of_bounds = self._rocket_y[idx] > 1.2
        if np.any(out_of_bounds):
            self._rocket_alive[idx[out_of_bounds]] = False

    def _step_sparks(self, dt: float) -> None:
        alive = self._spark_alive
        if not np.any(alive):
            return

        idx = np.where(alive)[0]
        self._spark_vy[idx] += float(self.spark_gravity) * dt
        drag = float(max(0.0, min(0.95, self.spark_drag)))
        self._spark_vx[idx] *= (1.0 - drag * dt)
        self._spark_vy[idx] *= (1.0 - drag * dt)

        self._spark_x[idx] = (self._spark_x[idx] + self._spark_vx[idx] * dt) % 1.0
        self._spark_y[idx] = self._spark_y[idx] + self._spark_vy[idx] * dt

        self._spark_age[idx] += dt
        self._spark_life[idx] -= dt

        dead = (self._spark_life[idx] <= 0.0) | (self._spark_y[idx] < -0.2) | (self._spark_y[idx] > 1.2)
        if np.any(dead):
            self._spark_alive[idx[dead]] = False

    def _explode(self, rocket_idx: int) -> None:
        x0 = float(self._rocket_x[rocket_idx])
        y0 = float(self._rocket_y[rocket_idx])
        hue0 = float(self._rocket_hue[rocket_idx])
        seed0 = float(self._rocket_seed[rocket_idx])
        self._rocket_alive[rocket_idx] = False

        free = np.where(~self._spark_alive)[0]
        if free.size == 0:
            return

        kinds = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        weights = np.array([0.28, 0.22, 0.20, 0.16, 0.14], dtype=np.float32)
        kind = int(self._rng.choice(kinds, p=(weights / np.sum(weights))))

        base = 140
        extra = int(220 * max(0.2, self.spark_density))
        count = int(base + self._rng.integers(0, extra))
        count = int(min(count, free.size))
        if count <= 0:
            return

        k = free[:count]
        self._spark_alive[k] = True
        self._spark_kind[k] = kind
        self._spark_x[k] = x0
        self._spark_y[k] = y0
        self._spark_age[k] = 0.0
        self._spark_seed[k] = float(seed0)

        life = float(self._rng.uniform(0.9, 1.45))
        if kind == 3:
            life *= 1.55
        self._spark_life[k] = life

        sat = float(self._rng.uniform(0.80, 1.0))
        val = float(self._rng.uniform(0.85, 1.0))
        hue = (hue0 + self._rng.normal(0.0, 0.06, count).astype(np.float32)) % 1.0
        rgb = _hsv_to_rgb(hue.astype(np.float32), np.full((count,), sat, dtype=np.float32), np.full((count,), val, dtype=np.float32))
        self._spark_rgb[k] = rgb * 255.0

        speed = float(self._rng.uniform(0.35, 0.65))
        if kind == 0:
            speed *= float(self._rng.uniform(0.95, 1.25))
        elif kind == 1:
            speed *= float(self._rng.uniform(0.75, 1.05))
        elif kind == 2:
            speed *= float(self._rng.uniform(0.70, 0.95))
        elif kind == 3:
            speed *= float(self._rng.uniform(0.55, 0.85))
        else:
            speed *= float(self._rng.uniform(0.95, 1.45))

        ang = self._rng.uniform(0.0, 2.0 * math.pi, count).astype(np.float32)
        ca = np.cos(ang)
        sa = np.sin(ang)

        if kind == 1:
            sa *= 0.28
        elif kind == 2:
            sa = np.sign(sa) * (np.abs(sa) ** 0.55)
        elif kind == 3:
            sa = np.abs(sa) ** 0.35

        vx = ca * speed
        vy = sa * speed

        if kind == 2:
            vy += float(self._rng.uniform(-0.05, 0.02))

        self._spark_vx[k] = vx
        self._spark_vy[k] = vy

        if kind == 4:
            # "crackle" is handled in render via stochastic flicker.
            pass

    def render(self, buffer: np.ndarray, mapper):
        if self._trail is None or self._trail.shape != buffer.shape:
            self._trail = np.zeros(buffer.shape, dtype=np.float32)

        if np.any(self._rocket_alive):
            self._render_points(
                mapper,
                self._rocket_x[self._rocket_alive],
                self._rocket_y[self._rocket_alive],
                np.full((int(np.sum(self._rocket_alive)), 3), 255.0, dtype=np.float32),
                intensity=0.18,
                sigma=max(0.004, self.explosion_size * 0.06),
                cutoff=0.020,
                flicker=None,
            )

        if np.any(self._spark_alive):
            idx = np.where(self._spark_alive)[0]
            life = np.clip(self._spark_life[idx], 0.0, 10.0)
            fade = (np.clip(life, 0.0, 1.0) ** 1.9).astype(np.float32)

            rgb = self._spark_rgb[idx] * fade[:, None]
            flicker = np.ones((idx.shape[0],), dtype=np.float32)

            crackle = self._spark_kind[idx] == 4
            if np.any(crackle):
                r = self._rng.random(np.sum(crackle)).astype(np.float32)
                flicker[crackle] = (r > 0.55).astype(np.float32) * (0.65 + 0.35 * self._rng.random(np.sum(crackle)).astype(np.float32))

            self._render_points(
                mapper,
                self._spark_x[idx],
                self._spark_y[idx],
                rgb,
                intensity=0.62,
                sigma=max(0.0035, self.explosion_size * 0.055),
                cutoff=0.030,
                flicker=flicker,
            )

        layer = np.clip(self._trail * float(np.clip(self.brightness, 0.0, 2.0)), 0.0, 255.0)
        alpha = float(np.clip(self.opacity, 0.0, 1.0))
        if alpha <= 0.0:
            return
        if alpha >= 1.0:
            cur = buffer.astype(np.float32)
            out = cur + layer
            np.clip(out, 0.0, 255.0, out=out)
            buffer[:] = out.astype(np.uint8)
        else:
            cur = buffer.astype(np.float32)
            add = layer * alpha
            out = cur + add
            np.clip(out, 0.0, 255.0, out=out)
            buffer[:] = out.astype(np.uint8)

    def _render_points(
        self,
        mapper,
        x: np.ndarray,
        y: np.ndarray,
        rgb: np.ndarray,
        intensity: float,
        sigma: float,
        cutoff: float,
        flicker: Optional[np.ndarray],
    ) -> None:
        if x.size == 0:
            return

        led_x = mapper.coords_x[None, :].astype(np.float32)
        led_y = mapper.coords_y[None, :].astype(np.float32)

        p_x = x[:, None].astype(np.float32)
        p_y = y[:, None].astype(np.float32)

        dx = _wrap_delta(led_x, p_x)
        dy = (led_y - p_y) * np.float32(mapper.aspect_ratio)
        dist_sq = (dx * dx) + (dy * dy)

        k = 7
        k = int(min(k, dist_sq.shape[1]))
        nn = np.argpartition(dist_sq, kth=(k - 1), axis=1)[:, :k]
        rows = np.arange(dist_sq.shape[0])[:, None]
        d = dist_sq[rows, nn]

        sig2 = max(1e-8, 2.0 * sigma * sigma)
        w = np.exp(-d / sig2).astype(np.float32)
        w *= (d <= (cutoff * cutoff)).astype(np.float32)

        if flicker is not None:
            w *= flicker.astype(np.float32)[:, None]

        w *= np.float32(intensity)

        idx_flat = nn.reshape(-1)
        w_flat = w.reshape(-1)

        c = rgb.astype(np.float32)
        c_flat = (np.repeat(c, k, axis=0) * w_flat[:, None]).astype(np.float32)

        np.add.at(self._trail[:, 0], idx_flat, c_flat[:, 0])
        np.add.at(self._trail[:, 1], idx_flat, c_flat[:, 1])
        np.add.at(self._trail[:, 2], idx_flat, c_flat[:, 2])