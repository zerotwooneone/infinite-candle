# src/engine/clock.py
import math
from typing import Hashable


class AnimationClock:
    """
    Shared timebase ticked once per frame by Engine before updating/rendering layers.
    Provides synchronized periodic waveforms and edge-triggered interval events.
    """

    def __init__(self, bpm: float = 120.0):
        self.t: float = 0.0
        self.dt: float = 0.0
        self.frame: int = 0
        self.bpm: float = bpm
        self._last_triggers: dict[Hashable, int] = {}

    def tick(self, dt: float) -> None:
        # Clamp dt to avoid giant physics jumps if the process stalls
        clamped_dt = max(0.0, min(float(dt), 0.25))
        self.dt = clamped_dt
        self.t += clamped_dt
        self.frame += 1

    def phase(self, period_sec: float = 1.0, offset: float = 0.0) -> float:
        """Sawtooth ramp from 0.0 to 1.0 repeating every `period_sec` seconds."""
        if period_sec <= 1e-6:
            return 0.0
        return ((self.t / period_sec) + offset) % 1.0

    def wave(self, period_sec: float = 1.0, offset: float = 0.0) -> float:
        """Smooth sine wave normalized to 0.0..1.0 repeating every `period_sec` seconds."""
        p = self.phase(period_sec, offset)
        return 0.5 + 0.5 * math.sin(p * 2.0 * math.pi)

    def triangle(self, period_sec: float = 1.0, offset: float = 0.0) -> float:
        """Linear triangle wave 0.0 -> 1.0 -> 0.0 repeating every `period_sec` seconds."""
        p = self.phase(period_sec, offset)
        return 1.0 - abs(2.0 * p - 1.0)

    def beat(self, multiplier: float = 1.0) -> float:
        """0.0..1.0 sawtooth synced to `self.bpm`."""
        period = (60.0 / max(1.0, self.bpm)) / max(1e-6, multiplier)
        return self.phase(period)

    def every(self, interval_sec: float, key: Hashable = "default") -> bool:
        """Returns True once each time `interval_sec` elapses for a given `key`."""
        if interval_sec <= 1e-6:
            return True
        bucket = int(self.t // interval_sec)
        prev = self._last_triggers.get(key)
        if prev != bucket:
            self._last_triggers[key] = bucket
            return True
        return False
