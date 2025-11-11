from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np
import torch
from PyQt6 import QtCore, QtGui, QtWidgets

from aircursor.core.hand_tracker import HandTracker, HandTrackerConfig
from aircursor.models.hagrid_fullframe import HaGRIDFullFrameClassifier, HaGRIDFullFrameConfig
from aircursor.models.static_mlp import StaticGestureClassifier, StaticGestureConfig
from aircursor.utils.landmark_preprocess import LandmarkPacket, build_static_feature

HAND_CONNECTIONS: Sequence[tuple[int, int]] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass
class PreviewConfig:
    camera_index: int
    mirror: bool
    model_path: Optional[Path]
    min_confidence: float
    hagrid_model_path: Optional[Path]
    hagrid_arch: str = "resnet18"
    hand_model_path: Optional[Path] = None


class GesturePreviewWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: PreviewConfig) -> None:
        super().__init__()
        self.setWindowTitle("AirCursor 静态手势预览")

        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {cfg.camera_index}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540

        self.tracker = HandTracker(
            HandTrackerConfig(
                model_path=cfg.hand_model_path,
            )
        )

        classifier_cfg = StaticGestureConfig(
            model_path=cfg.model_path,
            confidence=cfg.min_confidence,
        )
        self.classifier = StaticGestureClassifier(classifier_cfg)
        self.hagrid_classifier: Optional[HaGRIDFullFrameClassifier] = None
        if cfg.hagrid_model_path:
            try:
                self.hagrid_classifier = HaGRIDFullFrameClassifier(
                    HaGRIDFullFrameConfig(
                        model_path=cfg.hagrid_model_path,
                        architecture=cfg.hagrid_arch,
                    )
                )
                print(f"✓ 已加载 HaGRID 模型：{cfg.hagrid_model_path.name}")
            except Exception as exc:  # pragma: no cover - 运行期提示
                print(f"⚠ HaGRID 模型加载失败：{exc}")
                self.hagrid_classifier = None

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(30)

    def _update_frame(self) -> None:
        success, frame = self.cap.read()
        if not success:
            self.status_bar.showMessage("摄像头读取失败，正在重试…")
            return

        if self.cfg.mirror:
            frame = cv2.flip(frame, 1)

        packet = self.tracker.process(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display_frame = frame_rgb.copy()
        overlay_text = "未检测到手势"
        hagrid_text = None

        if packet is not None:
            handedness = packet.handedness
            landmarks_np = packet.landmarks
            self._draw_landmarks(display_frame, landmarks_np)

            feature_np = build_static_feature(packet)
            feature_tensor = torch.from_numpy(feature_np).unsqueeze(0).to(torch.float32)

            probs = self.classifier.predict(feature_tensor, landmarks_np)
            top_label, top_score = self._select_top(probs)

            overlay_text = (
                f"MLP/Heuristic: {top_label} ({top_score:.2f}) | "
                f"{handedness} hand"
            )

            if self.hagrid_classifier is not None:
                crop = self._extract_hand_crop(frame, landmarks_np)
                
                # 获取原始预测
                raw_probs = self.hagrid_classifier.predict_raw(crop)
                if raw_probs:
                    # 获取 top 3 预测
                    sorted_probs = sorted(raw_probs.items(), key=lambda kv: kv[1], reverse=True)
                    raw_label, raw_score = sorted_probs[0]
                    
                    # 获取映射后的类别
                    mapped_label = self.hagrid_classifier.get_mapping_for_label(raw_label)
                    if mapped_label:
                        hagrid_text = f"HaGRID: {raw_label}({raw_score:.2f}) -> {mapped_label}"
                    else:
                        # 尝试找第一个有映射的预测
                        mapped_found = None
                        for label, score in sorted_probs[:5]:  # 检查 top 5
                            mapped = self.hagrid_classifier.get_mapping_for_label(label)
                            if mapped:
                                mapped_found = (label, score, mapped)
                                break
                        
                        if mapped_found:
                            hagrid_text = (
                                f"HaGRID: {raw_label}({raw_score:.2f}) unmapped | "
                                f"Best mapped: {mapped_found[0]}({mapped_found[1]:.2f})->{mapped_found[2]}"
                            )
                        else:
                            hagrid_text = f"HaGRID: {raw_label}({raw_score:.2f}) -> (no mapping in top 5)"

        display_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        text_lines = [overlay_text]
        if hagrid_text:
            text_lines.append(hagrid_text)

        # 主要预测信息（顶部，大字体，绿色）
        for idx, text in enumerate(text_lines):
            cv2.putText(
                display_bgr,
                text,
                (20, 40 + idx * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        
        # 标签映射说明（底部，小字体，黄色）
        if self.hagrid_classifier is not None:
            mapping_info = [
                "HaGRID Mapping: closed={fist} peace={peace,peace_inverted,two_up,two_up_inverted}",
                "                open={palm,stop,stop_inverted}"
            ]
            h, w = display_bgr.shape[:2]
            for idx, info_text in enumerate(mapping_info):
                cv2.putText(
                    display_bgr,
                    info_text,
                    (10, h - 40 + idx * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 200, 255),  # 黄色
                    1,
                    cv2.LINE_AA,
                )

        h, w, ch = display_bgr.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(
            display_bgr.data,
            w,
            h,
            bytes_per_line,
            QtGui.QImage.Format.Format_BGR888,
        )
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)
        status_msg = " | ".join(text_lines)
        self.status_bar.showMessage(status_msg)

    def _extract_hand_crop(self, frame_bgr, landmarks_np) -> Optional[np.ndarray]:
        if frame_bgr is None or landmarks_np.size == 0:
            return None
        h, w, _ = frame_bgr.shape
        min_xy = landmarks_np[:, :2].min(axis=0)
        max_xy = landmarks_np[:, :2].max(axis=0)
        min_x = int(max((min_xy[0] - 0.1) * w, 0))
        min_y = int(max((min_xy[1] - 0.1) * h, 0))
        max_x = int(min((max_xy[0] + 0.1) * w, w))
        max_y = int(min((max_xy[1] + 0.1) * h, h))
        if max_x <= min_x or max_y <= min_y:
            return None
        return frame_bgr[min_y:max_y, min_x:max_x]

    def _draw_landmarks(self, frame_rgb: np.ndarray, landmarks_np: np.ndarray) -> None:
        if landmarks_np.size == 0:
            return
        h, w, _ = frame_rgb.shape
        points: list[tuple[int, int]] = []
        for x, y, _ in landmarks_np:
            px = int(np.clip(x, 0.0, 1.0) * w)
            py = int(np.clip(y, 0.0, 1.0) * h)
            points.append((px, py))

        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame_rgb, points[start], points[end], (0, 255, 255), 2)

        for px, py in points:
            cv2.circle(frame_rgb, (px, py), 4, (0, 255, 0), -1)

    @staticmethod
    def _select_top(probs: Dict[str, float]) -> tuple[str, float]:
        if not probs:
            return "unknown", 0.0
        label, score = max(probs.items(), key=lambda kv: kv[1])
        return label, score

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.timer.stop()
        self.tracker.close()
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)


def parse_args() -> PreviewConfig:
    parser = argparse.ArgumentParser(description="实时预览静态手势识别效果")
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引 (默认: 0)")
    parser.add_argument("--mirror", action="store_true", help="启用水平镜像")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="静态手势模型权重路径（可选，未提供则使用启发式）",
    )
    parser.add_argument(
        "--hagrid-model-path",
        type=Path,
        default=None,
        help="HaGRID 全帧分类模型权重路径（如 resnet18_hagridv2.pth）",
    )
    parser.add_argument(
        "--hagrid-arch",
        type=str,
        default="resnet18",
        help="HaGRID 模型架构（示例：resnet18、resnet152）",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="预测置信度阈值 (仅对加载模型时生效)",
    )
    parser.add_argument(
        "--hand-model-path",
        type=Path,
        default=None,
        help="MediaPipe Hand Landmarker 模型路径（默认 weights/hand_landmarker.task）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]

    def resolve_path(path: Optional[Path], default: Optional[str] = None) -> Optional[Path]:
        if path is None:
            if default is None:
                return None
            candidate = project_root / default
        else:
            candidate = path if path.is_absolute() else (project_root / path)
        return candidate

    return PreviewConfig(
        camera_index=args.camera_index,
        mirror=args.mirror,
        model_path=resolve_path(args.model_path),
        min_confidence=args.min_confidence,
        hagrid_model_path=resolve_path(args.hagrid_model_path),
        hagrid_arch=args.hagrid_arch,
        hand_model_path=resolve_path(args.hand_model_path, "weights/hand_landmarker.task"),
    )


def main() -> None:
    cfg = parse_args()
    app = QtWidgets.QApplication([])
    window = GesturePreviewWindow(cfg)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

