# lights-control/src/plugins/builtin/marquee_plugin.py
from __future__ import annotations
import numpy as np
from src.plugins.base import (
    AnimationPlugin,
    BlendModeParam,
    BoolParam,
    ChoiceParam,
    ColorParam,
    FloatParam,
    IntParam,
    OpacityParam,
    PluginContext,
    TextParam,
)
from src.plugins.registry import register_plugin
from src.plugins import toolkit as tk


# 3x5 bitmap font, written as 5 rows of 3 chars ('#' = lit pixel).
# Bold low-res glyphs read far better than thin 5x7 strokes on 19 sparse rows.
_FONT_ROWS = {
    " ": ("...", "...", "...", "...", "..."),
    "A": ("###", "#.#", "###", "#.#", "#.#"),
    "B": ("##.", "#.#", "##.", "#.#", "##."),
    "C": (".##", "#..", "#..", "#..", ".##"),
    "D": ("##.", "#.#", "#.#", "#.#", "##."),
    "E": ("###", "#..", "##.", "#..", "###"),
    "F": ("###", "#..", "##.", "#..", "#.."),
    "G": ("###", "#..", "#.#", "#.#", "###"),
    "H": ("#.#", "#.#", "###", "#.#", "#.#"),
    "I": ("###", ".#.", ".#.", ".#.", "###"),
    "J": ("..#", "..#", "..#", "#.#", ".#."),
    "K": ("#.#", "#.#", "##.", "#.#", "#.#"),
    "L": ("#..", "#..", "#..", "#..", "###"),
    "M": ("#.#", "###", "#.#", "#.#", "#.#"),
    "N": ("#.#", "##.", "#.#", ".##", "#.#"),
    "O": ("###", "#.#", "#.#", "#.#", "###"),
    "P": ("###", "#.#", "###", "#..", "#.."),
    "Q": ("###", "#.#", "#.#", "###", "..#"),
    "R": ("###", "#.#", "##.", "#.#", "#.#"),
    "S": ("###", "#..", "###", "..#", "###"),
    "T": ("###", ".#.", ".#.", ".#.", ".#."),
    "U": ("#.#", "#.#", "#.#", "#.#", "###"),
    "V": ("#.#", "#.#", "#.#", "#.#", ".#."),
    "W": ("#.#", "#.#", "#.#", "###", "#.#"),
    "X": ("#.#", "#.#", ".#.", "#.#", "#.#"),
    "Y": ("#.#", "#.#", ".#.", ".#.", ".#."),
    "Z": ("###", "..#", ".#.", "#..", "###"),
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("###", "..#", "###", "#..", "###"),
    "3": ("###", "..#", ".##", "..#", "###"),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "###", "..#", "###"),
    "6": ("###", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", "..#", ".#.", ".#."),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "###"),
    "!": (".#.", ".#.", ".#.", "...", ".#."),
    "?": ("###", "..#", ".##", "...", ".#."),
    ".": ("...", "...", "...", "...", ".#."),
    ",": ("...", "...", "...", ".#.", "#.."),
    "'": (".#.", ".#.", "...", "...", "..."),
    "-": ("...", "...", "###", "...", "..."),
    "_": ("...", "...", "...", "...", "###"),
    ":": ("...", ".#.", "...", ".#.", "..."),
    "+": ("...", ".#.", "###", ".#.", "..."),
    "*": ("#.#", ".#.", "###", ".#.", "#.#"),
    "<": ("..#", ".#.", "#..", ".#.", "..#"),
    ">": ("#..", ".#.", "..#", ".#.", "#.."),
    "=": ("...", "###", "...", "###", "..."),
    "/": ("..#", "..#", ".#.", "#..", "#.."),
    "♥": ("#.#", "###", "###", ".#.", "..."),
}

# Glyph bitmaps indexed [column, row] so horizontal strips concatenate along axis 0.
FONT = {
    ch: np.array([[c == "#" for c in row] for row in rows], dtype=np.float32).T
    for ch, rows in _FONT_ROWS.items()
}

_RAINBOW = tk.ColorPalette(
    [(255, 0, 0), (255, 150, 0), (255, 240, 0), (0, 200, 80), (0, 120, 255), (150, 0, 255)]
)


@register_plugin
class MarqueeTextPlugin(AnimationPlugin):
    plugin_id = "marquee"
    display_name = "Marquee Text"
    description = (
        "Scrolling text that wraps the pillar, stepping forward one letter at a time."
    )
    params = [
        TextParam("text", "Text", default="HELLO WORLD ", max_len=48),
        ColorParam("color", "Letter Color", default=[255, 220, 40]),
        BoolParam("rainbow", "Rainbow Letters", default=False),
        FloatParam("speed", "Speed (letters/s)", default=0.8, min=0.05, max=5.0, step=0.05),
        ChoiceParam("motion", "Motion", options=["step", "smooth"], default="step"),
        IntParam("letter_gap", "Letter Gap (px)", default=1, min=0, max=4, step=1),
        FloatParam("letters_around", "Letters Around", default=5.5, min=2.0, max=12.0, step=0.5),
        FloatParam("v_pos", "Position", default=0.5, min=0.0, max=1.0, step=0.02),
        FloatParam("v_size", "Text Height", default=0.45, min=0.1, max=1.0, step=0.02),
        BoolParam("reverse", "Reverse Direction", default=False),
        BoolParam("transparent_bg", "Transparent Background", default=True),
        BlendModeParam(default="alpha"),
        OpacityParam(default=1.0),
    ]

    def on_init(self) -> None:
        self._steps = 0.0   # whole letter-cells advanced (step mode)
        self._acc = 0.0     # dwell accumulator (step mode)
        self._pos_f = 0.0   # continuous position in letter-cells (smooth mode)
        self._rebuild_strip()

    def on_params_changed(self, changed: dict) -> None:
        self._rebuild_strip()

    def _rebuild_strip(self) -> None:
        """Rasterize the text into a (W, 5) bitmap strip of glyph + gap columns."""
        text = str(self.values.get("text", "")).upper()
        gap = max(0, int(self.values["letter_gap"]))
        self._n_chars = max(1, len(text))
        cols = []
        blank = np.zeros((gap, 5), dtype=np.float32)
        for ch in text:
            cols.append(FONT.get(ch, FONT["?"]))
            if gap > 0:
                cols.append(blank)
        if not cols:
            cols = [np.zeros((3, 5), dtype=np.float32)]
        self._strip = np.concatenate(cols, axis=0)
        self._cell_px = 3 + gap
        self._w = int(self._strip.shape[0])

    def _scroll_cells(self, ctx: PluginContext) -> float:
        """Current scroll position in letter cells (fractional during a step ease)."""
        speed = max(0.02, float(self.values["speed"]))
        dt = ctx.clock.dt
        if str(self.values["motion"]) == "step":
            dwell = 1.0 / speed
            self._acc += dt
            if self._acc >= dwell:
                self._steps += int(self._acc / dwell)
                self._acc %= dwell
                self._steps %= self._n_chars
            # Ease-in-out snap across the last-style segment of each dwell.
            move = min((self._acc / dwell) / 0.4, 1.0)
            eased = move * move * (3.0 - 2.0 * move)
            return self._steps + eased
        self._pos_f = (self._pos_f + dt * speed) % self._n_chars
        return self._pos_f

    def render(self, ctx: PluginContext):
        t = ctx.clock.t
        cell = float(self._cell_px)
        width = float(self._w)
        px_per_wrap = float(self.values["letters_around"]) * cell
        direction = -1.0 if self.values["reverse"] else 1.0
        scroll_px = self._scroll_cells(ctx) * cell * direction

        # 1. Horizontal: map theta onto strip columns, blending adjacent columns
        #    so stepped/smooth motion glides instead of flickering.
        xw = (ctx.leds.theta * px_per_wrap - scroll_px) % width
        x0 = np.floor(xw).astype(int)
        frac = xw - x0
        x1 = (x0 + 1) % int(width)

        # 2. Vertical: 5 glyph rows inside a band centered at v_pos.
        v_pos = float(self.values["v_pos"])
        v_size = max(0.02, float(self.values["v_size"]))
        frac_y = (ctx.leds.y - (v_pos - v_size * 0.5)) / v_size
        row = np.floor((1.0 - frac_y) * 5.0).astype(int)
        in_band = (row >= 0) & (row <= 4)
        row_c = np.clip(row, 0, 4)

        lit = self._strip[x0, row_c] * (1.0 - frac) + self._strip[x1, row_c] * frac
        lit = np.where(in_band, lit, 0.0)

        # 3. Color: solid letter color, or a slowly cycling per-letter rainbow.
        if self.values["rainbow"]:
            char_idx = x0 // int(cell)
            hue = (char_idx.astype(float) * 0.11 + t * 0.05) % 1.0
            base = _RAINBOW.sample(hue)
        else:
            base = np.asarray(self.values["color"], dtype=float)[np.newaxis, :]
        colors = base * lit[:, np.newaxis]

        if self.values["transparent_bg"]:
            return colors, lit
        return colors
