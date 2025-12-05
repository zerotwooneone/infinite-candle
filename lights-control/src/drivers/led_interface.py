# lights-control/src/drivers/led_interface.py
import numpy as np
from src.config import *

class LEDInterface:
    def __init__(self):
        self.count = LED_COUNT
        self.strip = None

        if IS_RASPBERRY_PI:
            from rpi_ws281x import PixelStrip, Color
            self.strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
            self.strip.begin()
            print(f"Hardware Driver Initialized: {LED_COUNT} LEDs on Pin {LED_PIN}")
        else:
            print("⚠️ Simulation Mode: No hardware detected. Using Mock Driver.")

    def show(self, pixels: np.ndarray):
        """
        Input: numpy array of shape (N, 3) representing [R, G, B]
        """
        if self.strip:
            for i in range(self.count):
                # WS281x expects a 24-bit integer, not an [R,G,B] list
                # Also, GRB ordering is common in 12V strips, might need swapping later
                r, g, b = pixels[i]
                color = (int(r) << 16) | (int(g) << 8) | int(b)
                self.strip.setPixelColor(i, color)
            self.strip.show()