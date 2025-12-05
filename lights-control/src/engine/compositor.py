# lights-control/src/engine/compositor.py
import time
import threading
import numpy as np
from src.drivers.led_interface import LEDInterface
from src.engine.mapper import PillarMapper
from src.config import LED_COUNT

class Engine:
    def __init__(self):
        self.driver = LEDInterface()
        self.mapper = PillarMapper()
        self.running = False
        self.buffer = np.zeros((LED_COUNT, 3), dtype=np.uint8)

        # The Scene Graph (Just a list of dicts for V1)
        # Default: A faint red background (Cthulhu style)
        self.layers = [
            {
                "type": "solid",
                "color": [10, 0, 0],
                "faces": [0,1,2,3],
                "h_min": 0.0,
                "h_max": 1.0
            }
        ]

    def update_layers(self, new_layers):
        """Thread-safe update of the scene"""
        self.layers = new_layers

    def render(self):
        """Calculates one frame"""
        # Clear buffer to black
        self.buffer[:] = 0

        for layer in self.layers:
            # 1. Get the mask for where this layer applies
            mask = self.mapper.get_indices_for_region(
                layer.get("faces", [0,1,2,3]),
                layer.get("h_min", 0.0),
                layer.get("h_max", 1.0)
            )

            # 2. Calculate Color (Simple Solid Color logic for now)
            if layer["type"] == "solid":
                color = np.array(layer["color"], dtype=np.uint8)
                # Apply color to masked area
                self.buffer[mask] = color

            # TODO: Add 'pulse', 'fire', etc here

        # Push to hardware
        self.driver.show(self.buffer)

    def start_loop(self):
        self.running = True
        while self.running:
            start_time = time.time()
            self.render()

            # Cap at ~60 FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0/60.0) - elapsed)
            time.sleep(sleep_time)

    def stop_loop(self):
        self.running = False