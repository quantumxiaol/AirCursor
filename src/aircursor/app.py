from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml

from aircursor.core.gesture_fusion import GestureFusionConfig, GestureFusionEngine
from aircursor.core.hagrid_dynamic_adapter import DynamicAdapterConfig, HagridDynamicAdapter
from aircursor.core.hand_tracker import HandTracker, HandTrackerConfig
from aircursor.core.mouse_engine import MouseEngine, MouseEngineConfig
from aircursor.models.dynamic_lstm import DynamicGestureConfig
from aircursor.models.static_mlp import StaticGestureConfig
from aircursor.utils.landmark_preprocess import compute_hand_center
from aircursor.utils.screen_utils import get_primary_screen_size


def load_config(project_root: Path) -> dict:
    """从项目根目录加载配置文件
    
    Args:
        project_root: 项目根目录路径
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 如果配置文件不存在
    """
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"未找到配置文件：{config_path}\n"
            "请确保项目根目录存在 config.yaml 文件。\n"
            "你可以从 config.yaml.example 复制一份并修改。"
        )
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    torch.set_grad_enabled(False)

    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent.parent  # src/aircursor/ -> src/ -> project_root/
    config = load_config(project_root)

    cam_cfg = config.get("camera", {})
    cursor_cfg = config.get("cursor", {})
    gesture_cfg = config.get("gesture", {})
    scroll_cfg = config.get("scroll", {})
    hagrid_cfg = config.get("dynamic_hagrid", {})

    capture = cv2.VideoCapture(cam_cfg.get("index", 0))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 960))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 540))

    screen = get_primary_screen_size()
    mouse_engine = MouseEngine(
        MouseEngineConfig(
            smooth_factor=float(cursor_cfg.get("smooth_factor", 0.2)),
            dead_zone=float(cursor_cfg.get("dead_zone", 0.05)),
            scroll_speed_scale=float(scroll_cfg.get("speed_scale", 5.0)),
        ),
        screen=screen,
    )

    def resolve_path(raw: Optional[str], default: Optional[str] = None) -> Optional[Path]:
        target = raw or default
        if not target:
            return None
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        return candidate

    hand_model_path = resolve_path(gesture_cfg.get("hand_landmarker_path"), "weights/hand_landmarker.task")
    tracker = HandTracker(
        HandTrackerConfig(
            model_path=hand_model_path,
        )
    )

    fusion_engine = GestureFusionEngine(
        GestureFusionConfig(
            static=StaticGestureConfig(
                model_path=resolve_path(gesture_cfg.get("static_model_path")),
                confidence=float(gesture_cfg.get("static_confidence", 0.6)),
            ),
            dynamic=DynamicGestureConfig(
                model_path=resolve_path(gesture_cfg.get("dynamic_model_path")),
                confidence=float(gesture_cfg.get("dynamic_confidence", 0.6)),
            ),
            max_idle_frames=int(gesture_cfg.get("max_idle_frames", 15)),
        )
    )

    hagrid_adapter: Optional[HagridDynamicAdapter] = None
    if hagrid_cfg.get("enabled", False):
        detector_path = resolve_path(hagrid_cfg.get("detector_path"), "weights/hand_detector.onnx")
        classifier_path = resolve_path(hagrid_cfg.get("classifier_path"), "weights/crops_classifier.onnx")
        try:
            hagrid_adapter = HagridDynamicAdapter(
                DynamicAdapterConfig(
                    detector_path=detector_path,
                    classifier_path=classifier_path,
                    debug=bool(hagrid_cfg.get("debug", False)),
                )
            )
            print(f"[HaGRID] 已加载动态手势模型: {detector_path.name}, {classifier_path.name}")
        except Exception as exc:
            print(f"[HaGRID] 动态手势适配器初始化失败：{exc}")
            hagrid_adapter = None

    print("AirCursor 已启动，按 'q' 退出。")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("无法从摄像头读取数据。")
                break

            packet = tracker.process(frame)
            if packet is None:
                fusion_engine.on_no_hand()
                cv2.imshow("AirCursor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            timestamp = time.time()
            actions = fusion_engine.update(packet, timestamp)

            if actions.get("move", 0.0) > 0.5:
                cx, cy = compute_hand_center(packet.landmarks)
                mouse_engine.move_cursor(cx, cy)

            click_prob = actions.get("click", 0.0)
            mouse_engine.left_click(click_prob > 0.6)

            if actions.get("right_click", 0.0) > 0.6:
                mouse_engine.right_click()

            scroll_up = actions.get("scroll_up", 0.0)
            scroll_down = actions.get("scroll_down", 0.0)
            scroll_mode = actions.get("scroll", 0.0) > 0.6
            if scroll_mode or scroll_up > 0.5 or scroll_down > 0.5:
                if scroll_mode:
                    _, disp_y = fusion_engine.recorder.recent_displacement(seconds=0.25)
                    dy = float(np.clip(-disp_y * 50.0, -2.0, 2.0))
                else:
                    dy = -1.0 if scroll_up > scroll_down else 1.0
                mouse_engine.scroll(dy)

            if hagrid_adapter:
                dynamic_events = hagrid_adapter.process(frame)
                if dynamic_events:
                    print("动态手势:", ", ".join(event.name for event in dynamic_events))
                    hagrid_adapter.apply_events(dynamic_events, mouse_engine)

            cv2.imshow("AirCursor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.release()
        tracker.close()
        cv2.destroyAllWindows()

