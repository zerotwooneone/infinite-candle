import numpy as np
import os
from src.effects.abstract import Effect

class ClipPlayerEffect(Effect):
    def __init__(self, config):
        super().__init__(config)

        # Load the file
        # In a real app, use config.filename
        file_path = f"/home/pi/infinite-candle/clips/{config.filename}"

        if not os.path.exists(file_path):
            print(f"Error: Clip {file_path} not found.")
            self.clip = np.zeros((1, 600, 3), dtype=np.uint8)
        else:
            # Memory Map the file!
            # This allows playing huge files without loading them fully into RAM
            self.clip = np.load(file_path, mmap_mode='r')

        self.total_frames = self.clip.shape[0]
        self.current_frame = 0
        self.fps_accum = 0.0
        self.playback_speed = 60.0 # Match the generation FPS

    def update(self, dt: float):
        # Frame Pacing
        self.fps_accum += dt * self.playback_speed
        if self.fps_accum >= 1.0:
            frames_to_advance = int(self.fps_accum)
            self.current_frame = (self.current_frame + frames_to_advance) % self.total_frames
            self.fps_accum -= frames_to_advance

    def render(self, buffer, mapper):
        frame_data = self.clip[self.current_frame]

        # Check for transparency config
        transparent = getattr(self.config, 'transparent', False)

        if transparent:
            # Create a mask of pixels that are NOT black
            # axis=1 means check R,G,B. If sum > 0, it has color.
            has_color = np.sum(frame_data, axis=1) > 10 # Threshold 10 for noise

            # Only write colored pixels, preserving the background buffer
            if hasattr(self.config, 'opacity') and self.config.opacity < 1.0:
                # Blend
                fg = frame_data[has_color].astype(float)
                bg = buffer[has_color].astype(float)
                blended = (bg * (1.0 - self.config.opacity)) + (fg * self.config.opacity)
                buffer[has_color] = blended.astype(np.uint8)
            else:
                # Overwrite
                buffer[has_color] = frame_data[has_color]

        else:
            # Old Behavior (Overwrite everything)
            if hasattr(self.config, 'opacity') and self.config.opacity < 1.0:
                buffer[:] = (frame_data * self.config.opacity).astype(np.uint8)
            else:
                buffer[:] = frame_data