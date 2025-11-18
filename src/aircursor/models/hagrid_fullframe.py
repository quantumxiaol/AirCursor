from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from aircursor.external.hagrid import targets as HAGRID_TARGETS


HAGRID_TARGET_LIST: List[str] = [HAGRID_TARGETS[i] for i in sorted(HAGRID_TARGETS)]

HAGRID_TO_STATIC = {
    "closed": {
        "fist",  # 握拳，四指收起
    },
    "peace": {
        "peace",
        "peace_inverted",
        "two_up",
        "two_up_inverted",
    },
    "open": {
        "palm",
        "stop",
        "stop_inverted",
    },
}


@dataclass
class HaGRIDFullFrameConfig:
    model_path: Path
    device: str = "cpu"
    architecture: str = "resnet18"


class HaGRIDFullFrameClassifier:
    """
    轻量封装 HaGRID 预训练全帧分类模型（默认 ResNet18 架构）。
    将 18 种 HaGRID 手势映射到本项目所需的三个静态手势（石头、剪刀、布）。
    """

    def __init__(self, cfg: HaGRIDFullFrameConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = self._build_model(cfg.architecture)
        self._load_weights(cfg.model_path)
        self.model.eval().to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.label_to_index: Dict[str, int] = {label: idx for idx, label in enumerate(HAGRID_TARGET_LIST)}
        self.static_mapping: Dict[str, List[int]] = {
            target: [
                self.label_to_index[label]
                for label in labels
                if label in self.label_to_index
            ]
            for target, labels in HAGRID_TO_STATIC.items()
        }

    def _build_model(self, architecture: str) -> torch.nn.Module:
        if not hasattr(models, architecture):
            raise ValueError(f"无法在 torchvision 中找到架构：{architecture}")

        model_fn = getattr(models, architecture)
        model = model_fn(weights=None)
        num_classes = len(HAGRID_TARGET_LIST)

        if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Module):
            in_features = model.fc.in_features  # type: ignore[attr-defined]
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, "classifier"):
            classifier = getattr(model, "classifier")
            if isinstance(classifier, torch.nn.Sequential):
                last_idx = -1
                in_features = classifier[last_idx].in_features  # type: ignore[index]
                classifier[last_idx] = torch.nn.Linear(in_features, num_classes)
                model.classifier = classifier
            else:
                raise ValueError(f"暂不支持该模型的 classifier 类型：{type(classifier)}")
        else:
            raise ValueError(f"模型 {architecture} 缺少可替换的最终分类层。")
        return model

    def _load_weights(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"未找到 HaGRID 权重文件：{path}\n"
                "请从 https://github.com/hukenovs/hagrid/releases 下载如 resnet18_hagridv2.pth"
            )
        state = torch.load(path, map_location=self.device)
        # 支持多种权重格式
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "MODEL_STATE" in state:
                # HaGRID 训练检查点格式
                state = state["MODEL_STATE"]
            elif "state_dict" in state:
                state = state["state_dict"]
        
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"加载权重时发生问题 missing={missing}, unexpected={unexpected}")

    @torch.no_grad()
    def predict(self, frame_bgr) -> Dict[str, float]:
        """
        输入 BGR 图像（可为整帧或手部裁剪），输出四个静态手势的置信度。
        """

        if frame_bgr is None:
            return {}

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        aggregated: Dict[str, float] = {}
        for target, indices in self.static_mapping.items():
            if not indices:
                aggregated[target] = 0.0
            else:
                aggregated[target] = float(probs[indices].sum())
        return aggregated

    @torch.no_grad()
    def predict_raw(self, frame_bgr) -> Dict[str, float]:
        """
        输入 BGR 图像，输出所有 HaGRID 原始类别的置信度。
        """
        if frame_bgr is None:
            return {}

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        # 返回所有原始类别的概率
        raw_probs: Dict[str, float] = {}
        for idx, label in enumerate(HAGRID_TARGET_LIST):
            raw_probs[label] = float(probs[idx])
        return raw_probs

    @staticmethod
    def top_label(probs: Dict[str, float]) -> str:
        if not probs:
            return "unknown"
        label, score = max(probs.items(), key=lambda kv: kv[1])
        return f"{label}({score:.2f})"
    
    @staticmethod
    def get_mapping_for_label(raw_label: str) -> Optional[str]:
        """
        获取原始 HaGRID 标签映射到的静态手势类别。
        """
        for static_label, hagrid_labels in HAGRID_TO_STATIC.items():
            if raw_label in hagrid_labels:
                return static_label
        return None

