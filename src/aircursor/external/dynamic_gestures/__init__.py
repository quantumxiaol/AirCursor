"""动态手势识别库

基于 ONNX 模型和 OC-SORT 跟踪算法的实时动态手势识别系统。

主要组件：
- DynamicGestureController: 主控制器，负责手部检测、分类和跟踪
- Event: 动态手势事件枚举
- Drawer: 调试绘制工具
"""

from .controller import DynamicGestureController
from .utils import Drawer, Event, HandPosition, targets

__all__ = [
    "DynamicGestureController",
    "Drawer",
    "Event",
    "HandPosition",
    "targets",
]

