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
        # Direct Memory Copy
        # Fast and efficient
        frame_data = self.clip[self.current_frame]

        # Handle opacity if needed
        if hasattr(self.config, 'opacity') and self.config.opacity < 1.0:
            buffer[:] = (frame_data * self.config.opacity).astype(np.uint8)
        else:
            buffer[:] = frame_data