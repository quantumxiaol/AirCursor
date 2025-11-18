from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aircursor.utils.landmark_preprocess import infer_static_label_from_geometry

STATIC_CLASSES = ["open", "closed", "peace"]


class StaticGestureMLP(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, len(STATIC_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class StaticGestureConfig:
    model_path: Optional[Path] = None
    confidence: float = 0.6


class StaticGestureClassifier:
    """包装好的静态手势分类器，如果未加载模型则退化为启发式。"""

    def __init__(self, cfg: StaticGestureConfig) -> None:
        self.cfg = cfg
        self.model = StaticGestureMLP()
        self.device = torch.device("cpu")
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

    def predict(self, feature: torch.Tensor, landmarks_np) -> Dict[str, float]:
        if self.model_loaded:
            with torch.no_grad():
                logits = self.model(feature.to(self.device))
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
        else:
            heur_label = infer_static_label_from_geometry(landmarks_np)
            probs = [0.0] * len(STATIC_CLASSES)
            if heur_label and heur_label in STATIC_CLASSES:
                probs[STATIC_CLASSES.index(heur_label)] = 1.0
            else:
                probs = [1.0 / len(STATIC_CLASSES)] * len(STATIC_CLASSES)
        return {cls: float(prob) for cls, prob in zip(STATIC_CLASSES, probs)}

