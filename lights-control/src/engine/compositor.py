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

        # Default Layer: Faint red background
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

        current_time = time.time()

        for layer in self.layers:

            # --- TYPE: SOLID ---
            if layer["type"] == "solid":
                mask = self.mapper.get_indices_for_region(
                    layer.get("faces", [0,1,2,3]),
                    layer.get("h_min", 0.0),
                    layer.get("h_max", 1.0)
                )
                color = np.array(layer["color"], dtype=np.uint8)
                self.buffer[mask] = color

            # --- TYPE: CHASE ---
            elif layer["type"] == "chase":
                # 1. Get Parameters
                base_color = np.array(layer["color"], dtype=float)
                speed = layer.get("speed", 20.0)    # Pixels per second
                tail_length = layer.get("tail", 30) # How long is the glow?

                # 2. Calculate Head Position
                head_pos = (current_time * speed) % LED_COUNT

                # 3. Draw the Tail
                for i in range(tail_length):
                    distance_from_head = i
                    pixel_index = int(head_pos - distance_from_head) % LED_COUNT

                    # Exponential Fade (Squared for smoother look)
                    fade = 1.0 - (distance_from_head / tail_length)
                    fade = fade ** 2

                    dimmed_color = (base_color * fade).astype(np.uint8)

                    # Additive Blending (prevent overflow)
                    existing = self.buffer[pixel_index].astype(float)
                    new_val = existing + dimmed_color
                    np.clip(new_val, 0, 255, out=new_val)

                    self.buffer[pixel_index] = new_val.astype(np.uint8)

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