from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from aircursor.models.dynamic_lstm import DynamicGestureClassifier, DynamicGestureConfig
from aircursor.models.static_mlp import StaticGestureClassifier, StaticGestureConfig
from aircursor.utils.landmark_preprocess import LandmarkPacket, build_static_feature
from aircursor.utils.trajectory_recorder import TrajectoryRecorder

STATIC_ACTION_MAP = {
    "open": "move",
    "closed": "click",
    "peace": "scroll",
    "three": "right_click",
}


@dataclass
class GestureFusionConfig:
    static: StaticGestureConfig
    dynamic: DynamicGestureConfig
    max_idle_frames: int = 15


class GestureFusionEngine:
    def __init__(self, cfg: GestureFusionConfig) -> None:
        self.cfg = cfg
        self.static_classifier = StaticGestureClassifier(cfg.static)
        self.dynamic_classifier = DynamicGestureClassifier(cfg.dynamic)
        self.recorder = TrajectoryRecorder()
        self.idle_counter = 0
        self.current_mode: Optional[str] = None

    def update(self, packet: LandmarkPacket, timestamp: float) -> Dict[str, float]:
        feature_np = build_static_feature(packet)
        feature_tensor = torch.from_numpy(feature_np).unsqueeze(0).to(torch.float32)

        static_probs = self.static_classifier.predict(feature_tensor, packet.landmarks)

        self.recorder.append(
            float(packet.landmarks[:, 0].mean()),
            float(packet.landmarks[:, 1].mean()),
            timestamp,
        )
        dynamic_probs = self.dynamic_classifier.predict(self.recorder)

        action_score = {
            STATIC_ACTION_MAP[k]: v for k, v in static_probs.items() if k in STATIC_ACTION_MAP
        }

        if dynamic_probs.get("idle", 1.0) < 0.5:
            if dynamic_probs.get("swipe_up", 0.0) > dynamic_probs.get("swipe_down", 0.0):
                action_score["scroll_up"] = dynamic_probs["swipe_up"]
            else:
                action_score["scroll_down"] = dynamic_probs.get("swipe_down", 0.0)

        return action_score

    def on_no_hand(self) -> None:
        self.recorder.clear()
        self.idle_counter += 1
        if self.idle_counter > self.cfg.max_idle_frames:
            self.current_mode = None

