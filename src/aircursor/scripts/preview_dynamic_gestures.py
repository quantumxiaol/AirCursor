from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from aircursor.external.dynamic_gestures import DynamicGestureController, Drawer, Event, targets


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    # 从脚本位置向上 3 层到达项目根目录
    # __file__ -> preview_dynamic_gestures.py
    # parents[0] -> scripts/
    # parents[1] -> aircursor/
    # parents[2] -> src/
    # parents[3] -> 项目根目录
    project_root = Path(__file__).resolve().parents[3]
    return (project_root / path).resolve()


@dataclass
class DynamicPreviewConfig:
    camera_index: int
    mirror: bool
    detector_path: Path
    classifier_path: Path
    debug: bool


class DynamicGesturePreviewWindow(QtWidgets.QMainWindow):
    """动态手势预览窗口（PyQt6 GUI）"""
    
    def __init__(self, cfg: DynamicPreviewConfig) -> None:
        super().__init__()
        self.setWindowTitle("AirCursor 动态手势预览")
        
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {cfg.camera_index}")
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
        
        # 设置摄像头分辨率（更高的分辨率有助于手势识别）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 初始化动态手势控制器（优化参数以提高响应速度和准确性）
        self.controller = DynamicGestureController(
            str(cfg.detector_path),
            str(cfg.classifier_path),
            max_age=28,        # 轨迹最大存活帧数（默认30，降低以减少延迟）
            min_hits=2,        # 确认轨迹的最小检测次数（默认3，降低以快速响应）⭐
            iou_threshold=0.3, # IOU阈值（保持默认）
            maxlen=35,         # 轨迹历史最大长度（默认30，略微增加）
            min_frames=18      # 确认手势的最小帧数（默认20，降低以快速响应）⭐
        )
        self.drawer = Drawer() if cfg.debug else None
        
        # 事件历史记录（保留最近 5 个事件）
        self.event_history = deque(maxlen=5)
        
        # 设置中央窗口部件和布局
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        
        layout = QtWidgets.QVBoxLayout(central)
        
        # 视频显示标签
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)
        
        # 状态栏
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("准备就绪 | 等待检测动态手势...")
        
        # 定时器（30ms ≈ 33 FPS）
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(30)
        
        self.last_time = time.time()
    
    def _update_frame(self) -> None:
        """更新帧并处理动态手势"""
        success, frame = self.cap.read()
        if not success:
            self.status_bar.showMessage("摄像头读取失败，正在重试…")
            return
        
        if self.cfg.mirror:
            frame = cv2.flip(frame, 1)
        
        # 处理动态手势（使用 MainController，就像 run_demo.py）
        start = time.time()
        bboxes, ids, labels = self.controller(frame)
        
        # 收集事件（参考 run_demo.py 的实现）
        events = []
        for trk in self.controller.tracks:
            if trk["tracker"].time_since_update < 1:
                if trk["hands"].action is not None:
                    # 记录事件
                    events.append(trk["hands"].action)
                    
                    # 添加到历史记录（去重：如果和最后一个相同则不添加）
                    if not self.event_history or self.event_history[-1] != trk["hands"].action:
                        self.event_history.append(trk["hands"].action)
                    
                    # 关键：消费事件，避免重复识别
                    # DRAG 类事件需要持续保持，直到 DROP
                    if trk["hands"].action not in [Event.DRAG, Event.DRAG2, Event.DRAG3]:
                        trk["hands"].action = None
        
        fps = 1.0 / (time.time() - start + 1e-6)
        
        # 绘制调试信息（如果启用）
        if self.cfg.debug and bboxes is not None:
            bboxes = bboxes.astype(np.int32)
            for i in range(bboxes.shape[0]):
                box = bboxes[i, :]
                gesture = targets[labels[i]] if labels[i] is not None else "None"
                
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
                cv2.putText(
                    frame,
                    f"ID {ids[i]}: {gesture}",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        
        # 构建文字信息
        text_lines = []
        text_lines.append(f"Dynamic Gesture | FPS: {fps:.1f}")
        
        # 显示事件历史：最新 <- 次新 <- ... <- waiting
        if self.event_history:
            # 从最新到最旧排列，格式：E3 <- E2 <- E1 <- waiting
            history_text = " <- ".join(event.name for event in reversed(list(self.event_history)))
            if not events:  # 如果当前没有事件，添加 waiting
                history_text = f"waiting <- {history_text}"
            text_lines.append(f"Events: {history_text}")
            
            # 状态栏显示最新的事件
            latest_event = list(self.event_history)[-1].name
            status_text = f"最新事件: {latest_event} | 历史: {len(self.event_history)} 个事件"
        else:
            text_lines.append("Events: waiting...")
            status_text = "等待检测动态手势..."
        
        # 使用原始 BGR 帧
        display_bgr = frame.copy()
        
        # 显示主要信息（顶部，大字体，绿色）
        for idx, text in enumerate(text_lines):
            cv2.putText(
                display_bgr,
                text,
                (20, 40 + idx * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),  # 绿色
                2,
                cv2.LINE_AA,
            )
        
        # 显示底部说明（小字体，黄色）- 手势到鼠标操作的映射
        h, w = display_bgr.shape[:2]
        help_info = [
            "Mouse Mapping: TAP->Left Click | ZOOM->Right Click | DRAG/DROP->Drag",
            "               SWIPE_UP/DOWN->Scroll Vertical | SWIPE_LEFT/RIGHT->Scroll Horizontal"
        ]
        for idx, info_text in enumerate(help_info):
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
        
        # 转换为 QImage 并显示
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
        
        # 更新状态栏
        self.status_bar.showMessage(f"{status_text} | FPS: {fps:.1f}")
    
    def closeEvent(self, event) -> None:
        """窗口关闭时释放资源"""
        self.timer.stop()
        self.cap.release()
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description="HaGRID 动态手势实时预览（PyQt6 GUI）")
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引（默认 0）")
    parser.add_argument("--mirror", action="store_true", help="是否镜像画面（适合自拍镜头）")
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("weights/hand_detector.onnx"),
        help="手部检测 ONNX 模型路径",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("weights/crops_classifier.onnx"),
        help="手势分类 ONNX 模型路径",
    )
    parser.add_argument("--debug", action="store_true", help="绘制检测框与标签")
    args = parser.parse_args()

    detector_path = resolve_path(args.detector)
    classifier_path = resolve_path(args.classifier)
    
    print(f"✓ 手部检测模型: {detector_path}")
    print(f"✓ 手势分类模型: {classifier_path}")
    print(f"✓ 调试模式: {'开启' if args.debug else '关闭'}")
    print("启动 PyQt6 GUI 窗口...")

    cfg = DynamicPreviewConfig(
        camera_index=args.camera_index,
        mirror=args.mirror,
        detector_path=detector_path,
        classifier_path=classifier_path,
        debug=args.debug,
    )

    app = QtWidgets.QApplication([])
    window = DynamicGesturePreviewWindow(cfg)
    window.resize(960, 720)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

