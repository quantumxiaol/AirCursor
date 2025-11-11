from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 7, 11, 15, 19]
FINGER_MCP = [2, 5, 9, 13, 17]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


@dataclass
class LandmarkPacket:
    """打包 MediaPipe 返回的手部关键点数组。"""

    handedness: str
    landmarks: np.ndarray  # (21, 3) 归一化坐标
    world_landmarks: Optional[np.ndarray] = None  # (21, 3)


def mp_landmarks_to_np(mp_landmarks) -> np.ndarray:
    """将 MediaPipe Landmarks 转换为 numpy 数组。"""

    if mp_landmarks is None:
        return np.zeros((21, 3), dtype=np.float32)

    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in mp_landmarks.landmark], dtype=np.float32
    )
    return coords


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """以手腕为原点进行平移，再缩放至单位范围，提升泛化能力。"""

    if landmarks.shape != (21, 3):
        raise ValueError("landmarks 形状必须为 (21, 3)")

    origin = landmarks[0:1]
    translated = landmarks - origin
    max_range = np.linalg.norm(translated, axis=1).max()
    if max_range < 1e-6:
        return translated
    return translated / max_range


def vectorize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """将 21 个坐标展平成长度 63 的特征向量。"""

    return landmarks.reshape(-1)


def build_static_feature(packet: LandmarkPacket) -> np.ndarray:
    """静态手势模型输入：归一化 + 向量化后拼接左右手信息。"""

    normalized = normalize_landmarks(packet.landmarks)
    feature = vectorize_landmarks(normalized)
    handedness_flag = np.array([1.0 if packet.handedness == "Right" else 0.0], dtype=np.float32)
    return np.concatenate([feature, handedness_flag], axis=0)


def infer_static_label_from_geometry(landmarks: np.ndarray) -> Optional[str]:
    """
    当没有训练好的模型时，使用基于关键点的启发式分类。
    返回 open/closed/peace 之一，若无法判断返回 None。
    """

    normalized = normalize_landmarks(landmarks)
    palm_indices = [0, 1, 5, 9, 13, 17]
    palm_center = normalized[palm_indices].mean(axis=0)

    finger_states = {}
    tip_coords = {}

    for name, tip_idx, pip_idx, mcp_idx in zip(
        FINGER_NAMES, FINGER_TIPS, FINGER_PIPS, FINGER_MCP
    ):
        tip = normalized[tip_idx]
        pip = normalized[pip_idx]
        mcp = normalized[mcp_idx]

        v1 = pip - mcp
        v2 = tip - pip

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_angle = -1.0
        if norm_v1 > 1e-5 and norm_v2 > 1e-5:
            cos_angle = float(np.dot(v1, v2) / (norm_v1 * norm_v2))

        tip_dist = float(np.linalg.norm(tip - palm_center))
        pip_dist = float(np.linalg.norm(pip - palm_center))
        radial_gain = tip_dist - pip_dist

        if name == "thumb":
            finger_states[name] = (
                cos_angle > 0.4 and radial_gain > 0.01 and tip_dist > 0.25
            )
        else:
            finger_states[name] = (
                cos_angle > 0.6 and radial_gain > 0.015 and tip_dist > 0.32
            )

        tip_coords[name] = tip

    index_open = finger_states["index"]
    middle_open = finger_states["middle"]
    ring_open = finger_states["ring"]
    pinky_open = finger_states["pinky"]
    thumb_open = finger_states["thumb"]

    # open_count 只统计四指（食指、中指、无名指、小指），不包括大拇指
    open_count = sum([index_open, middle_open, ring_open, pinky_open])

    # 石头（closed）：四指都收起，大拇指朝内或朝外都可以
    if open_count == 0:
        return "closed"

    thumb_index_dist = np.linalg.norm(tip_coords["thumb"] - tip_coords["index"])
    thumb_middle_dist = np.linalg.norm(tip_coords["thumb"] - tip_coords["middle"])
    index_middle_dist = np.linalg.norm(tip_coords["index"] - tip_coords["middle"])
    middle_ring_dist = np.linalg.norm(tip_coords["middle"] - tip_coords["ring"])
    ring_pinky_dist = np.linalg.norm(tip_coords["ring"] - tip_coords["pinky"])

    # 剪刀手势：食指和中指伸出且分开，无名指和小指收起
    if (
        index_open
        and middle_open
        and not ring_open
        and not pinky_open
    ):
        if index_middle_dist > 0.18:
            return "peace"

    # 布（open）：至少3个四指伸出，且手指展开
    # 要求大拇指也伸出，或者四指全部伸出（避免握拳+大拇指伸出被误判为布）
    spread = np.linalg.norm(tip_coords["index"] - tip_coords["pinky"])
    if open_count >= 3:
        # 条件1：手指展开且大拇指也伸出
        # 条件2：四指全部伸出（此时即使大拇指收起也是布）
        if (spread > 0.40 and thumb_open) or open_count == 4:
            return "open"

    return None


def compute_hand_center(landmarks: np.ndarray) -> Tuple[float, float]:
    """返回二维屏幕坐标映射的参考手部中心。"""

    center = landmarks.mean(axis=0)
    return float(center[0]), float(center[1])

