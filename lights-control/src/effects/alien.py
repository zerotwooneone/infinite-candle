import numpy as np
import random
from src.effects.grid_system import GridSystem

# --- STATE DEFINITIONS ---
S0_DARK_START   = 0
S1_PERSON_WAIT  = 1
S2_SHIP_DESCEND = 2
S3_PRE_BEAM     = 3
S4_SPOTLIGHT_IN = 4
S5_PRE_RINGS    = 5
S6_RINGS_FALL   = 6
S7_SCAN_DOTS    = 7
S8_DOTS_RISE    = 8
S9_PERSON_RISE  = 9
S10_ABDUCTED    = 10
S11_SHIP_DEPART = 11
S12_DARK_END    = 12

GRID_W = 32
GRID_H = 72
HOVER_H = 55.0
GROUND_H = 0.0

class AlienAbductionEffect(GridSystem):
    def __init__(self, config):
        # Fixed grid size for predictable quadrant math
        super().__init__(config, width=GRID_W, height=GRID_H)

        # Colors
        self.c_ship_blue = np.array(config.ship_color_1, dtype=float)
        self.c_ship_green = np.array(config.ship_color_2, dtype=float)
        self.c_beam = np.array(config.beam_color, dtype=float)
        self.c_person = np.array([200, 150, 100], dtype=float)
        self.c_ring = np.array([0, 50, 255], dtype=float) # Deep blue rings
        self.scan_colors = [
            np.array([255, 0, 0]), np.array([255, 255, 0]),
            np.array([0, 255, 255]), np.array([255, 0, 255])
        ]

        self.speed_mult = config.speed
        self.reset_scene()

    def reset_scene(self):
        self.state = S0_DARK_START
        self.state_timer = 0.0
        self.global_time = 0.0

        # Actors
        self.ship_y = GRID_H + 5.0
        self.person_y = GROUND_H
        self.rings_y = [] # List of active rings
        self.dots_y = GROUND_H

        # Visibility Flags
        self.show_person = False
        self.show_ship = False
        self.show_beam = False
        self.beam_opacity = 0.0
        self.show_dots = False
        self.person_abducted = False

    def draw_rect(self, x, y, w, h, color, opacity=1.0):
        """Helper to draw rectangles on the grid with wrapping"""
        ix, iy, iw, ih = int(x), int(y), int(w), int(h)
        for r in range(iy, iy + ih):
            for c in range(ix, ix + iw):
                if 0 <= r < self.grid_h:
                    col_idx = c % self.grid_w
                    if opacity >= 1.0:
                        self.canvas[r, col_idx] = color
                    else:
                        bg = self.canvas[r, col_idx]
                        self.canvas[r, col_idx] = (bg * (1.0 - opacity)) + (color * opacity)

    def draw_at_quadrants(self, draw_func, base_x, *args, **kwargs):
        """Helper to execute a draw command 4 times around the cylinder"""
        quad_width = GRID_W // 4
        for i in range(4):
            offset_x = i * quad_width
            draw_func(base_x + offset_x, *args, **kwargs)

    # --- ACTOR DRAWING FUNCTIONS ---

    def draw_ship_actor(self, x, y):
        # Ship is 5 pixels tall
        # Top Blue
        self.draw_rect(x, y+4, 10, 1, self.c_ship_blue)
        # Bottom Blue
        self.draw_rect(x, y, 10, 1, self.c_ship_blue)

        # Middle Green Dashed Scrolling Line
        iy = int(y) + 2
        if 0 <= iy < GRID_H:
            # Scroll speed
            scroll_offset = int(self.global_time * 10)
            for c in range(int(x), int(x) + 10):
                # Dash pattern: 2 on, 2 off
                if (c + scroll_offset) % 4 < 2:
                    col_idx = c % self.grid_w
                    self.canvas[iy, col_idx] = self.c_ship_green

    def draw_person_actor(self, x, y):
        if self.person_abducted: return
        # 2x4 rectangle
        self.draw_rect(x-1, y, 2, 4, self.c_person)

    def draw_spotlight(self):
        if not self.show_beam: return

        # Draw from ground up to ship bottom
        ship_bottom = int(self.ship_y)

        for r in range(ship_bottom):
            # Opacity fades towards the ship (brightest at bottom)
            # Use S4 timer for fade-in effect
            dist_factor = 1.0 - (r / ship_bottom) # 1.0 at bottom, 0.0 at ship
            dist_factor = dist_factor ** 2 # Non-linear fade looks better

            final_opacity = dist_factor * self.beam_opacity * 0.6 # Max 60% opacity

            # Draw beam under all 4 quadrants
            self.draw_at_quadrants(self.draw_rect, 3, r, 4, 1, self.c_beam, opacity=final_opacity)

    def draw_rings_actor(self, x, y_list):
        for ry in y_list:
            # Double blue ring with gap
            self.draw_rect(x-2, ry, 6, 1, self.c_ring) # Bottom
            self.draw_rect(x-2, ry+2, 6, 1, self.c_ring) # Top (gap of 1 pixel)

    def draw_dots_actor(self, x, y):
        # 4 colorful dots around the center point
        offsets = [(-2, 0), (2, 0), (0, 2), (0, 4)]
        for i, (ox, oy) in enumerate(offsets):
            self.draw_rect(x+ox, y+oy, 1, 1, self.scan_colors[i])


    def update(self, dt: float):
        dt *= self.speed_mult
        self.state_timer += dt
        self.global_time += dt
        self.canvas[:] = 0 # Clear frame

        # --- THE STATE MACHINE DIRECTOR ---

        if self.state == S0_DARK_START:
            if self.state_timer > 3.0:
                self.state = S1_PERSON_WAIT
                self.state_timer = 0
                self.show_person = True

        elif self.state == S1_PERSON_WAIT:
            if self.state_timer > 3.0:
                self.state = S2_SHIP_DESCEND
                self.state_timer = 0
                self.show_ship = True

        elif self.state == S2_SHIP_DESCEND:
            self.ship_y -= 15.0 * dt # Descent speed
            if self.ship_y <= HOVER_H:
                self.ship_y = HOVER_H
                self.state = S3_PRE_BEAM
                self.state_timer = 0

        elif self.state == S3_PRE_BEAM:
            if self.state_timer > 1.0:
                self.state = S4_SPOTLIGHT_IN
                self.state_timer = 0
                self.show_beam = True

        elif self.state == S4_SPOTLIGHT_IN:
            # Fade in beam over 2 seconds
            self.beam_opacity = min(1.0, self.state_timer / 2.0)
            if self.state_timer > 2.0:
                self.state = S5_PRE_RINGS
                self.state_timer = 0

        elif self.state == S5_PRE_RINGS:
            if self.state_timer > 1.0:
                self.state = S6_RINGS_FALL
                self.state_timer = 0
                # Spawn the double ring at ship
                self.rings_y.append(self.ship_y - 2)

        elif self.state == S6_RINGS_FALL:
            # Move rings down
            finished = False
            new_rings = []
            for ry in self.rings_y:
                ry -= 20.0 * dt
                if ry > GROUND_H:
                    new_rings.append(ry)
                else:
                    finished = True # Ring hit ground
            self.rings_y = new_rings

            if finished:
                self.state = S7_SCAN_DOTS
                self.state_timer = 0
                self.show_dots = True

        elif self.state == S7_SCAN_DOTS:
            if self.state_timer > 0.5:
                self.state = S8_DOTS_RISE
                self.state_timer = 0

        elif self.state == S8_DOTS_RISE:
            self.dots_y += 15.0 * dt
            if self.dots_y >= self.ship_y:
                self.show_dots = False
                self.state = S9_PERSON_RISE
                self.state_timer = 0

        elif self.state == S9_PERSON_RISE:
            # Jitter while rising
            self.person_y += 10.0 * dt
            if self.person_y >= self.ship_y - 2:
                self.person_abducted = True
                self.state = S10_ABDUCTED
                self.state_timer = 0
                self.show_beam = False # Spotlight off instantly

        elif self.state == S10_ABDUCTED:
            if self.state_timer > 1.0:
                self.state = S11_SHIP_DEPART
                self.state_timer = 0

        elif self.state == S11_SHIP_DEPART:
            self.ship_y += 40.0 * dt # Fast ascent
            if self.ship_y > GRID_H + 10:
                self.state = S12_DARK_END
                self.state_timer = 0
                self.show_ship = False

        elif self.state == S12_DARK_END:
            if self.state_timer > 3.0:
                self.reset_scene() # Loop

        # --- RENDER CALLS ---
        # Center X for actors (middle of first quadrant)
        cx = 5

        self.draw_spotlight()

        if self.show_person:
            # Add jitter if rising
            jy = 0 if self.state != S9_PERSON_RISE else np.random.uniform(-0.5, 0.5)
            self.draw_at_quadrants(self.draw_person_actor, cx, self.person_y + jy)

        if self.show_dots:
            self.draw_at_quadrants(self.draw_dots_actor, cx, self.dots_y)

        if self.rings_y:
            self.draw_at_quadrants(self.draw_rings_actor, cx, self.rings_y)

        if self.show_ship:
            self.draw_at_quadrants(self.draw_ship_actor, cx-4, self.ship_y) # Ship is wider, offset x

    def render(self, buffer, mapper):
        self.render_grid(buffer, mapper)