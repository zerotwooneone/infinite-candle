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

    current_time = time.time()

    for layer in self.layers:

        # --- TYPE: SOLID (Existing) ---
        if layer["type"] == "solid":
            mask = self.mapper.get_indices_for_region(
                layer.get("faces", [0,1,2,3]),
                layer.get("h_min", 0.0),
                layer.get("h_max", 1.0)
            )
            color = np.array(layer["color"], dtype=np.uint8)
            self.buffer[mask] = color

        # --- TYPE: CHASE (New!) ---
        elif layer["type"] == "chase":
            # 1. Get Parameters
            base_color = np.array(layer["color"], dtype=float)
            speed = layer.get("speed", 20.0)    # Pixels per second
            tail_length = layer.get("tail", 30) # How long is the glow?

            # 2. Calculate Head Position (Float allows smooth sub-pixel feel later)
            # We use modulo (%) so it wraps around the pillar forever
            head_pos = (current_time * speed) % LED_COUNT

            # 3. Draw the Tail
            for i in range(tail_length):
                # How far back is this pixel?
                distance_from_head = i

                # Calculate Index (Wrap around backwards)
                pixel_index = int(head_pos - distance_from_head) % LED_COUNT

                # Calculate Brightness (Linear Fade)
                # 1.0 at head, 0.0 at end of tail
                fade = 1.0 - (distance_from_head / tail_length)

                # Make it "Subtle" (Exponential curve looks more natural than linear)
                # Squaring the fade makes the tail drop off faster but linger gently
                fade = fade ** 2

                # Apply fade to color
                # We add to the existing buffer so chases can cross each other!
                dimmed_color = (base_color * fade).astype(np.uint8)

                # Simple Additive Blending (Optional: Cap at 255)
                # This prevents integer overflow wrapping (250 + 10 = 4, not 255)
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