from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ScreenInfo:
    width: int
    height: int


def get_primary_screen_size() -> ScreenInfo:
    """使用 Quartz (macOS) / tkinter (其他平台) 获取主屏幕大小。"""

    try:
        import AppKit  # type: ignore

        screen = AppKit.NSScreen.mainScreen()
        if screen is None:
            raise RuntimeError("无法获取主屏幕")
        frame = screen.frame()
        return ScreenInfo(width=int(frame.size.width), height=int(frame.size.height))
    except Exception:
        # 备用：tkinter
        import tkinter  # type: ignore

        root = tkinter.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return ScreenInfo(width=int(width), height=int(height))


def normalized_to_screen(x: float, y: float, screen: ScreenInfo) -> Tuple[int, int]:
    """将 [0,1] 范围的归一化坐标映射到屏幕像素。"""

    x_clamped = np.clip(x, 0.0, 1.0)
    y_clamped = np.clip(y, 0.0, 1.0)
    px = int(x_clamped * screen.width)
    py = int(y_clamped * screen.height)
    return px, py

