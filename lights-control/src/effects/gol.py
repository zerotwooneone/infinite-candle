import numpy as np
from src.effects.grid_system import GridSystem

class GameOfLifeEffect(GridSystem):
    def __init__(self, config):
        # 30x70 resolution gives nice chunky pixels that are visible from a distance
        super().__init__(config, width=30, height=70)

        self.color_alive = np.array(config.color, dtype=float)
        self.color_dead = np.array(config.bg_color, dtype=float)

        self.update_interval = 1.0 / config.speed
        self.timer = 0.0

        # The Game State (0 or 1)
        self.cells = np.zeros((self.grid_h, self.grid_w), dtype=int)

        # History for stagnation detection
        self.prev_cells = np.zeros_like(self.cells)
        self.stagnant_frames = 0

        self.reset_grid()

    def reset_grid(self):
        """Randomly seed the board"""
        # 20% chance of life
        self.cells = (np.random.random((self.grid_h, self.grid_w)) < 0.2).astype(int)

    def update(self, dt: float):
        self.timer += dt
        if self.timer < self.update_interval:
            return # Wait for next tick

        self.timer = 0.0

        # --- THE GAME OF LIFE LOGIC (Vectorized) ---

        # 1. Calculate Neighbors using "Roll" (Shift)
        # We roll the array in 8 directions to count overlaps
        # Axis 0 = Y (Up/Down), Axis 1 = X (Left/Right)

        # Neighbors Count
        n = np.zeros_like(self.cells)

        # Cardinal
        n += np.roll(self.cells,  1, axis=0) # North (Note: this wraps Y, which acts like a Torus)
        n += np.roll(self.cells, -1, axis=0) # South
        n += np.roll(self.cells,  1, axis=1) # East (Wraps X - Cylinder!)
        n += np.roll(self.cells, -1, axis=1) # West

        # Diagonals
        n += np.roll(np.roll(self.cells,  1, axis=0),  1, axis=1) # NE
        n += np.roll(np.roll(self.cells,  1, axis=0), -1, axis=1) # NW
        n += np.roll(np.roll(self.cells, -1, axis=0),  1, axis=1) # SE
        n += np.roll(np.roll(self.cells, -1, axis=0), -1, axis=1) # SW

        # 2. Apply Rules
        # Rule 1: Alive and 2 or 3 neighbors -> Stay Alive
        # Rule 2: Dead and exactly 3 neighbors -> Born
        # All else -> Die

        # Boolean masks
        born = (self.cells == 0) & (n == 3)
        survive = (self.cells == 1) & ((n == 2) | (n == 3))

        new_cells = np.zeros_like(self.cells)
        new_cells[born | survive] = 1

        # 3. Check for Stagnation (Did the frame change?)
        if np.array_equal(new_cells, self.cells) or np.array_equal(new_cells, self.prev_cells):
            self.stagnant_frames += 1
        else:
            self.stagnant_frames = 0

        if self.stagnant_frames > 10 or np.sum(new_cells) == 0:
            self.reset_grid() # Reboot if boring
        else:
            self.prev_cells = self.cells.copy()
            self.cells = new_cells

    def render(self, buffer, mapper):
        # Translate Cells (0/1) to RGB Canvas
        # We use broadcasting to set the generic self.canvas

        # Create mask for Alive cells
        mask = self.cells == 1

        # Fill Canvas
        # Note: We need to broadcast (H, W, 3)
        self.canvas[mask] = self.color_alive
        self.canvas[~mask] = self.color_dead

        # Delegate to base class to push canvas to LEDs
        self.render_grid(buffer, mapper)