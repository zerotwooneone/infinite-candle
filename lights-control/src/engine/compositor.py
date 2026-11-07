import threading
import time
import numpy as np
from src.drivers.led_interface import LEDInterface
from src.engine.mapper import PillarMapper
from src.engine.clock import AnimationClock
from src.config import LED_COUNT
from src.effects.factory import create_effect
from src.plugins.base import PluginEffectAdapter

class Engine:
    def __init__(self):
        self.driver = LEDInterface()
        self.mapper = PillarMapper()
        self.clock = AnimationClock()
        self.active_effects = []
        self.buffer = np.zeros((LED_COUNT, 3), dtype=np.uint8)
        self.running = False
        # Guards active_effects swaps / in-place param patches vs the render loop.
        self._state_lock = threading.RLock()
        # Snapshot of the last composited frame for WebSocket streaming.
        # Guarded by _stream_lock so readers never see a torn frame.
        self._stream_lock = threading.Lock()
        self._stream_frame = np.zeros((LED_COUNT, 3), dtype=np.uint8)

    def update_layers(self, layer_configs):
        """
        Swaps or updates the layer stack. Preserves running plugin instances
        whose `id` and `type` match an existing active layer.
        """
        with self._state_lock:
            existing_by_id = {}
            for eff in self.active_effects:
                lid = getattr(getattr(eff, "config", None), "id", None)
                ltype = getattr(getattr(eff, "config", None), "type", None)
                if lid and ltype:
                    existing_by_id[(lid, ltype)] = eff

            new_stack = []
            for conf in layer_configs:
                lid = getattr(conf, "id", None)
                ltype = getattr(conf, "type", None)
                reused = existing_by_id.get((lid, ltype)) if lid else None

                if isinstance(reused, PluginEffectAdapter):
                    reused.apply_config_update(conf)
                    new_stack.append(reused)
                else:
                    effect = create_effect(conf, clock=self.clock)
                    if effect:
                        new_stack.append(effect)

            self.active_effects = new_stack

    def patch_layer(self, layer_ref: str, updates: dict) -> bool:
        """
        Live-update a single layer by `id` (or integer index string "0", "1", ...).
        For PluginEffectAdapter layers, updates parameters in-place without resetting state.
        """
        with self._state_lock:
            target_idx = None
            for i, eff in enumerate(self.active_effects):
                lid = getattr(getattr(eff, "config", None), "id", None)
                if lid is not None and str(lid) == str(layer_ref):
                    target_idx = i
                    break
            if target_idx is None and str(layer_ref).isdigit():
                idx = int(layer_ref)
                if 0 <= idx < len(self.active_effects):
                    target_idx = idx

            if target_idx is None:
                return False

            eff = self.active_effects[target_idx]
            conf = eff.config
            raw = conf.model_dump() if hasattr(conf, "model_dump") else dict(conf.__dict__)
            raw.update(updates)
            new_conf = conf.__class__(**raw)

            if isinstance(eff, PluginEffectAdapter):
                eff.apply_config_update(new_conf)
            else:
                rebuilt = create_effect(new_conf, clock=self.clock)
                if rebuilt:
                    self.active_effects[target_idx] = rebuilt
            return True

    def start_loop(self):
        self.running = True
        last_time = time.time()

        while self.running:
            now = time.time()
            dt = now - last_time
            last_time = now

            with self._state_lock:
                self.clock.tick(dt)
                self.buffer[:] = 0 # Clear frame

                for effect in self.active_effects:
                    effect.update(dt)
                    effect.render(self.buffer, self.mapper)

            with self._stream_lock:
                np.copyto(self._stream_frame, self.buffer)

            self.driver.show(self.buffer)
            time.sleep(0.001)

    def get_stream_frame(self) -> bytes:
        """Return the latest composited frame as raw RGB bytes (600*3 = 1800)."""
        with self._stream_lock:
            return self._stream_frame.tobytes()

    def stop_loop(self):
        self.running = False