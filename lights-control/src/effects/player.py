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