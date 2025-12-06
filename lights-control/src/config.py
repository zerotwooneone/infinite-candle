# lights-control/src/config.py
import os

# Hardware Settings
LED_COUNT = 600          # 2 Reels of WS2815
LED_PIN = 18             # GPIO 18 (PWM)
LED_FREQ_HZ = 800000     # 800khz
LED_DMA = 10             # DMA Channel
LED_BRIGHTNESS = 255     # 0-255 (We limit current in WLED, but here we do it via max brightness)
LED_INVERT = False
LED_CHANNEL = 0

# Geometry Settings
PILLAR_WRAPS = 19.3        # How many times the strip circles the pillar
PILLAR_FACES = 4         # Square pillar

# Simulation Mode
# If we are not on a Pi, fallback to printing to console
IS_RASPBERRY_PI = False
try:
    import rpi_ws281x
    IS_RASPBERRY_PI = True
except ImportError:
    pass