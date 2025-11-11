from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from aircursor.utils.landmark_preprocess import LandmarkPacket


@dataclass
class HandTrackerConfig:
    model_path: Optional[Path] = None
    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class HandTracker:
    def __init__(self, cfg: HandTrackerConfig) -> None:
        self.cfg = cfg
        self._timestamp_ms = 0

        model_path = cfg.model_path
        project_root = Path(__file__).resolve().parents[3]
        if model_path is None:
            model_path = project_root / "weights" / "hand_landmarker.task"
        if not model_path.exists():
            raise FileNotFoundError(
                f"未找到手部关键点模型：{model_path}\n"
                "请从 https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/hand_landmarker.task 下载 "
                "并放置于 weights/ 目录。"
            )

        base_options = BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.max_num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr) -> Optional[LandmarkPacket]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.hand_landmarks:
            return None

        idx = 0
        landmark_list = result.hand_landmarks[idx]
        landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32)

        world_np = None
        if result.hand_world_landmarks:
            world_list = result.hand_world_landmarks[idx]
            world_np = np.array([[lm.x, lm.y, lm.z] for lm in world_list], dtype=np.float32)

        handedness = "Right"
        if result.handedness and len(result.handedness) > idx:
            hand_info = result.handedness[idx]
            # MediaPipe 0.10.21+ 返回的是 Classifications 对象
            if hasattr(hand_info, 'categories') and hand_info.categories:
                handedness = hand_info.categories[0].category_name
            elif isinstance(hand_info, list) and len(hand_info) > 0:
                handedness = hand_info[0].category_name

        return LandmarkPacket(handedness=handedness, landmarks=landmarks_np, world_landmarks=world_np)

    def close(self) -> None:
        if hasattr(self._landmarker, "close"):
            self._landmarker.close()

