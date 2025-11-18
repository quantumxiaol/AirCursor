from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from aircursor.utils.trajectory_recorder import TrajectoryRecorder

DYNAMIC_CLASSES = ["idle", "swipe_up", "swipe_down", "circle_cw"]


class DynamicGestureLSTM(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, len(DYNAMIC_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1]
        return self.fc(last_hidden)


@dataclass
class DynamicGestureConfig:
    model_path: Optional[Path] = None
    confidence: float = 0.6


class DynamicGestureClassifier:
    def __init__(self, cfg: DynamicGestureConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.model = DynamicGestureLSTM()
        self.model_loaded = False
        if cfg.model_path and cfg.model_path.exists():
            self._load_model(cfg.model_path)

    def _load_model(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()
        self.model_loaded = True

    def predict(self, recorder: TrajectoryRecorder) -> Dict[str, float]:
        if not recorder.to_numpy().size or len(recorder._buffer) < 4:
            return {cls: (1.0 if cls == "idle" else 0.0) for cls in DYNAMIC_CLASSES}

        sequence = recorder.to_numpy()
        sequence[:, :2] = sequence[:, :2] - sequence[0, :2]
        input_tensor = torch.from_numpy(sequence).unsqueeze(0).to(torch.float32)

        if self.model_loaded:
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                return {cls: float(prob) for cls, prob in zip(DYNAMIC_CLASSES, probs)}

        # 简单启发式：根据整体位移判断上下挥手
        dx, dy = recorder.recent_displacement()
        if abs(dy) > abs(dx) * 1.2 and abs(dy) > 0.2:
            if dy < 0:
                return {"swipe_up": 1.0, "swipe_down": 0.0, "circle_cw": 0.0, "idle": 0.0}
            return {"swipe_up": 0.0, "swipe_down": 1.0, "circle_cw": 0.0, "idle": 0.0}

        return {"idle": 1.0, "swipe_up": 0.0, "swipe_down": 0.0, "circle_cw": 0.0}



