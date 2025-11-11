from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np


@dataclass
class TrajectoryFrame:
    """存放单帧的手部中心位置与时间戳。"""

    x: float
    y: float
    timestamp: float


class TrajectoryRecorder:
    """维护固定长度的轨迹缓存，供动态手势识别使用。"""

    def __init__(self, maxlen: int = 48) -> None:
        self._buffer: Deque[TrajectoryFrame] = deque(maxlen=maxlen)

    def append(self, x: float, y: float, timestamp: float) -> None:
        self._buffer.append(TrajectoryFrame(x=x, y=y, timestamp=timestamp))

    def clear(self) -> None:
        self._buffer.clear()

    def to_numpy(self) -> np.ndarray:
        if not self._buffer:
            return np.zeros((0, 3), dtype=np.float32)
        data = np.array([[f.x, f.y, f.timestamp] for f in self._buffer], dtype=np.float32)
        return data

    def recent_displacement(self, seconds: float = 0.4) -> Tuple[float, float]:
        """计算最近一段时间的位移向量，用于滚动等操作。"""

        if len(self._buffer) < 2:
            return 0.0, 0.0
        latest = self._buffer[-1]
        for frame in reversed(self._buffer):
            if latest.timestamp - frame.timestamp >= seconds:
                dx = latest.x - frame.x
                dy = latest.y - frame.y
                return float(dx), float(dy)
        first = self._buffer[0]
        return float(latest.x - first.x), float(latest.y - first.y)

    def is_stationary(self, threshold: float = 0.01) -> bool:
        """判断轨迹在阈值内是否基本静止。"""

        if len(self._buffer) < 2:
            return True
        xs = np.array([f.x for f in self._buffer], dtype=np.float32)
        ys = np.array([f.y for f in self._buffer], dtype=np.float32)
        return float(xs.max() - xs.min()) < threshold and float(ys.max() - ys.min()) < threshold

