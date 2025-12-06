import time
import numpy as np
from src.drivers.led_interface import LEDInterface
from src.engine.mapper import PillarMapper
from src.config import LED_COUNT
from src.effects.factory import create_effect

class Engine:
    def __init__(self):
        self.driver = LEDInterface()
        self.mapper = PillarMapper()
        self.active_effects = []
        self.buffer = np.zeros((LED_COUNT, 3), dtype=np.uint8)
        self.running = False

    def update_layers(self, layer_configs):
        """
        Receives List[EffectConfig] from API
        """
        new_stack = []
        for conf in layer_configs:
            effect = create_effect(conf)
            if effect:
                new_stack.append(effect)
        self.active_effects = new_stack

    def start_loop(self):
        self.running = True
        last_time = time.time()

        while self.running:
            now = time.time()
            dt = now - last_time
            last_time = now

            self.buffer[:] = 0 # Clear frame

            for effect in self.active_effects:
                effect.update(dt)
                effect.render(self.buffer, self.mapper)

            self.driver.show(self.buffer)
            time.sleep(0.001)

    def stop_loop(self):
        self.running = False