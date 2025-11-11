from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pynput.mouse import Button, Controller

from aircursor.utils.screen_utils import ScreenInfo, normalized_to_screen


@dataclass
class MouseEngineConfig:
    smooth_factor: float = 0.2
    dead_zone: float = 0.05
    scroll_speed_scale: float = 2000.0


class MouseEngine:
    def __init__(self, cfg: MouseEngineConfig, screen: ScreenInfo) -> None:
        self.cfg = cfg
        self.screen = screen
        self.mouse = Controller()
        self._cursor_x: Optional[float] = None
        self._cursor_y: Optional[float] = None
        self._left_pressed = False

    def move_cursor(self, nx: float, ny: float) -> None:
        if nx < self.cfg.dead_zone and ny < self.cfg.dead_zone:
            return
        if self._cursor_x is None or self._cursor_y is None:
            self._cursor_x, self._cursor_y = nx, ny
        else:
            self._cursor_x += (nx - self._cursor_x) * self.cfg.smooth_factor
            self._cursor_y += (ny - self._cursor_y) * self.cfg.smooth_factor

        px, py = normalized_to_screen(self._cursor_x, self._cursor_y, self.screen)
        self.mouse.position = (px, py)

    def left_click(self, pressed: bool) -> None:
        if pressed and not self._left_pressed:
            self.mouse.press(Button.left)
            self._left_pressed = True
        elif not pressed and self._left_pressed:
            self.mouse.release(Button.left)
            self._left_pressed = False

    def right_click(self) -> None:
        self.mouse.click(Button.right, 1)

    def scroll(self, dy: float) -> None:
        self.mouse.scroll(0, int(dy * self.cfg.scroll_speed_scale))

