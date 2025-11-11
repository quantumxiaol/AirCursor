from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from aircursor.external.dynamic_gestures import DynamicGestureController, Drawer, Event, targets


SCROLL_UP_EVENTS = {
    Event.SWIPE_UP,
    Event.SWIPE_UP2,
    Event.SWIPE_UP3,
    Event.FAST_SWIPE_UP,
}
SCROLL_DOWN_EVENTS = {
    Event.SWIPE_DOWN,
    Event.SWIPE_DOWN2,
    Event.SWIPE_DOWN3,
    Event.FAST_SWIPE_DOWN,
}
SWIPE_LEFT_EVENTS = {Event.SWIPE_LEFT, Event.SWIPE_LEFT2, Event.SWIPE_LEFT3}
SWIPE_RIGHT_EVENTS = {Event.SWIPE_RIGHT, Event.SWIPE_RIGHT2, Event.SWIPE_RIGHT3}
DRAG_START_EVENTS = {Event.DRAG, Event.DRAG2, Event.DRAG3}
DRAG_END_EVENTS = {Event.DROP, Event.DROP2, Event.DROP3}
CLICK_EVENTS = {Event.TAP, Event.DOUBLE_TAP}


@dataclass
class DynamicAdapterConfig:
    detector_path: Path
    classifier_path: Path
    debug: bool = False


class HagridDynamicAdapter:
    """
    将 HaGRID 动态手势流水线封装为可选适配器。
    """

    def __init__(self, cfg: DynamicAdapterConfig) -> None:
        self.cfg = cfg
        if not cfg.detector_path.exists():
            raise FileNotFoundError(f"未找到手部检测模型：{cfg.detector_path}")
        if not cfg.classifier_path.exists():
            raise FileNotFoundError(f"未找到手势分类模型：{cfg.classifier_path}")

        self.controller = DynamicGestureController(str(cfg.detector_path), str(cfg.classifier_path))
        self.drawer = Drawer() if cfg.debug else None
        self.drag_active = False

    def process(self, frame_bgr: np.ndarray) -> List[Event]:
        """
        运行检测 + 动作识别，返回发生的事件列表。
        保留内部轨迹状态以便检测持续事件（拖拽、缩放等）。
        """

        bboxes, ids, labels = self.controller(frame_bgr)
        events: List[Event] = []

        if (
            self.cfg.debug
            and bboxes is not None
            and ids is not None
            and labels is not None
            and len(bboxes)
        ):
            self._draw_debug(frame_bgr, bboxes.astype(np.int32), ids, labels)

        for track in self.controller.tracks:
            if track["tracker"].time_since_update < 1 and track["hands"].action is not None:
                events.append(track["hands"].action)
                track["hands"].action = None
        return events

    def apply_events(self, events: Sequence[Event], mouse_engine) -> None:
        """
        将事件映射为鼠标操作。保留拖拽状态，避免重复触发。
        """

        for event in events:
            if event in CLICK_EVENTS:
                mouse_engine.left_click(True)
                mouse_engine.left_click(False)
            elif event in DRAG_START_EVENTS:
                if not self.drag_active:
                    mouse_engine.left_click(True)
                    self.drag_active = True
            elif event in DRAG_END_EVENTS:
                if self.drag_active:
                    mouse_engine.left_click(False)
                    self.drag_active = False
            elif event in SCROLL_UP_EVENTS:
                mouse_engine.scroll(-2.0)
            elif event in SCROLL_DOWN_EVENTS:
                mouse_engine.scroll(2.0)
            elif event == Event.ZOOM_IN:
                mouse_engine.scroll(-4.0)
            elif event == Event.ZOOM_OUT:
                mouse_engine.scroll(4.0)
            elif event in SWIPE_LEFT_EVENTS:
                dx = int(-mouse_engine.cfg.scroll_speed_scale)
                mouse_engine.mouse.scroll(dx, 0)
            elif event in SWIPE_RIGHT_EVENTS:
                dx = int(mouse_engine.cfg.scroll_speed_scale)
                mouse_engine.mouse.scroll(dx, 0)

    def _draw_debug(self, frame: np.ndarray, bboxes: np.ndarray, ids, labels) -> None:
        if ids is None or labels is None:
            return
        for i in range(bboxes.shape[0]):
            box = bboxes[i]
            gesture_idx = labels[i] if labels[i] is not None else None
            gesture_name = targets[gesture_idx] if gesture_idx is not None and gesture_idx < len(targets) else "None"
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {ids[i]}: {gesture_name}",
                (box[0], max(box[1] - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

