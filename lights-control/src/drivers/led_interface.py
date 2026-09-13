# lights-control/src/drivers/led_interface.py
import ctypes
import numpy as np
from src.config import *


class LEDInterface:
    def __init__(self):
        self.count = LED_COUNT
        self.strip = None
        # numpy uint32 view aliasing rpi_ws281x's internal DMA buffer (if mapped)
        self._dma_view = None

        if IS_RASPBERRY_PI:
            from rpi_ws281x import PixelStrip
            self.strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
            self.strip.begin()
            if LED_BULK_WRITE:
                self._dma_view = self._map_dma_buffer()
            mode = "bulk DMA writes" if self._dma_view is not None else "per-pixel fallback"
            print(f"Hardware Driver Initialized: {LED_COUNT} LEDs on Pin {LED_PIN} ({mode})")
        else:
            print("⚠️ Simulation Mode: No hardware detected. Using Mock Driver.")

    def _map_dma_buffer(self):
        """
        Attempt to obtain a numpy uint32 array that aliases rpi_ws281x's
        internal channel->leds buffer. Writing into this view and calling
        strip.show() pushes the whole frame with zero per-pixel Python calls.

        Returns None if the driver's internals can't be resolved — callers
        must fall back to setPixelColor().
        """
        try:
            channel = getattr(self.strip, "_channel", None)
            if channel is None:
                return None

            leds = getattr(channel, "leds", None)
            if leds is None:
                return None

            addr = self._pointer_to_addr(leds)
            if addr is None:
                return None

            buf_t = ctypes.c_uint32 * self.count
            return np.ctypeslib.as_array(buf_t.from_address(addr))
        except Exception:
            return None

    @staticmethod
    def _pointer_to_addr(leds):
        """Best-effort extraction of a raw address from a SWIG pointer."""
        # Strategy 1: already an int / int-able
        try:
            return int(leds)
        except (TypeError, ValueError):
            pass
        # Strategy 2: SWIG proxy exposes .this
        inner = getattr(leds, "this", None)
        if inner is not None:
            try:
                return int(inner)
            except (TypeError, ValueError):
                pass
        # Strategy 3: parse "<Swig Object of type 'ws2811_led_t *' at 0x...>"
        try:
            rep = repr(leds)
            if " at 0x" in rep:
                return int(rep.rsplit(" at 0x", 1)[1].rstrip(">"), 16)
        except Exception:
            pass
        return None

    def show(self, pixels: np.ndarray):
        """
        Input: numpy array of shape (N, 3) representing [R, G, B]
        """
        if not self.strip:
            return

        # Vectorized pack: WS281x expects a 24-bit integer per pixel (R<<16|G<<8|B)
        packed = (
            (pixels[:, 0].astype(np.uint32) << 16)
            | (pixels[:, 1].astype(np.uint32) << 8)
            | pixels[:, 2].astype(np.uint32)
        )

        if self._dma_view is not None:
            # Single bulk copy into the DMA buffer, then trigger the render.
            np.copyto(self._dma_view, packed)
            self.strip.show()
        else:
            for i in range(self.count):
                self.strip.setPixelColor(i, int(packed[i]))
            self.strip.show()