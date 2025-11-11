from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


@dataclass
class ROIBox:
    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height

    def is_valid(self, min_size: int = 10) -> bool:
        return self.width >= min_size and self.height >= min_size


class ROIOverlayLabel(QtWidgets.QLabel):
    roiChanged = QtCore.pyqtSignal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._scaled_pixmap: Optional[QtGui.QPixmap] = None
        self._drawing = False
        self._origin = QtCore.QPoint()
        self._current_rect: Optional[QtCore.QRect] = None
        self._scale_x = 1.0
        self._scale_y = 1.0

    def setFramePixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = pixmap
        self._update_scaled_pixmap()
        self.update()

    def clearRoi(self) -> None:
        self._current_rect = None
        self.update()
        self.roiChanged.emit(None)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()
        self.update()

    def _update_scaled_pixmap(self) -> None:
        if not self._pixmap:
            self._scaled_pixmap = None
            self._scale_x = self._scale_y = 1.0
            return
        label_size = self.size()
        pixmap_size = self._pixmap.size()
        scaled = self._pixmap.scaled(
            label_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._scaled_pixmap = scaled
        if pixmap_size.width() == 0 or pixmap_size.height() == 0:
            self._scale_x = self._scale_y = 1.0
        else:
            self._scale_x = pixmap_size.width() / scaled.width()
            self._scale_y = pixmap_size.height() / scaled.height()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0))
        pixmap_rect = self._current_pixmap_rect()
        if self._scaled_pixmap:
            painter.drawPixmap(pixmap_rect.topLeft(), self._scaled_pixmap)
        if self._current_rect:
            painter.translate(pixmap_rect.topLeft())
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(self._current_rect)
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drawing = True
            pixmap_rect = self._current_pixmap_rect()
            pos = event.position().toPoint()
            if not pixmap_rect.contains(pos):
                self._drawing = False
                return
            self._origin = pos - pixmap_rect.topLeft()
            self._current_rect = QtCore.QRect(self._origin, self._origin)
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drawing:
            pixmap_rect = self._current_pixmap_rect()
            pos = event.position().toPoint() - pixmap_rect.topLeft()
            pos.setX(max(0, min(pos.x(), pixmap_rect.width())))
            pos.setY(max(0, min(pos.y(), pixmap_rect.height())))
            rect = QtCore.QRect(self._origin, pos).normalized()
            self._current_rect = rect
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            if self._current_rect and (
                self._current_rect.width() < 10 or self._current_rect.height() < 10
            ):
                self._current_rect = None
            self.update()
            self.roiChanged.emit(self.get_roi_box())

    def _current_pixmap_rect(self) -> QtCore.QRect:
        if self._scaled_pixmap:
            rect = QtCore.QRect(
                0,
                0,
                self._scaled_pixmap.width(),
                self._scaled_pixmap.height(),
            )
            rect.moveCenter(self.rect().center())
            return rect
        return QtCore.QRect(0, 0, self.width(), self.height())

    def get_roi_box(self) -> Optional[ROIBox]:
        if not self._current_rect:
            return None
        rect = self._current_rect
        pixmap_rect = self._current_pixmap_rect()
        offset_x = pixmap_rect.x()
        offset_y = pixmap_rect.y()
        # 将相对于缩放后画布的坐标映射回原始帧坐标
        x = int((rect.x() - offset_x) * self._scale_x)
        y = int((rect.y() - offset_y) * self._scale_y)
        w = int(rect.width() * self._scale_x)
        h = int(rect.height() * self._scale_y)
        return ROIBox(x, y, w, h)


class LabelingWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        camera_index: int,
        labels: List[str],
        output_root: Path,
        category: str,
        mirror: bool,
    ) -> None:
        super().__init__()
        self.setWindowTitle("AirCursor 手势标注工具")

        self.labels = labels
        self.output_root = output_root
        self.category = category
        self.mirror = mirror

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_index}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.frame_width == 0 or self.frame_height == 0:
            self.frame_width = 960
            self.frame_height = 540

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)

        self.video_label = ROIOverlayLabel()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.video_label.roiChanged.connect(self._on_roi_changed)
        layout.addWidget(self.video_label, stretch=1)

        button_layout = QtWidgets.QHBoxLayout()
        for label in labels:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda _, l=label: self._save_frame(l))
            button_layout.addWidget(button)

        clear_button = QtWidgets.QPushButton("清除 ROI")
        clear_button.clicked.connect(self.video_label.clearRoi)
        button_layout.addWidget(clear_button)

        exit_button = QtWidgets.QPushButton("退出")
        exit_button.clicked.connect(self.close)
        button_layout.addWidget(exit_button)

        layout.addLayout(button_layout)

        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(30)

        self._current_frame: Optional[np.ndarray] = None
        self._current_roi: Optional[ROIBox] = None

    def _on_roi_changed(self, roi: Optional[ROIBox]) -> None:
        self._current_roi = roi
        if roi is None:
            self.status_bar.showMessage("已清除 ROI")
        else:
            self.status_bar.showMessage(
                f"ROI: x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}"
            )

    def _ensure_dir(self, label: str) -> Path:
        target = self.output_root / self.category / label
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _on_timer(self) -> None:
        success, frame = self.cap.read()
        if not success:
            self.status_bar.showMessage("摄像头读取失败，正在重试...")
            return

        if self.mirror:
            frame = cv2.flip(frame, 1)

        self._current_frame = frame.copy()

        display_frame = frame.copy()
        if self._current_roi:
            x0, y0, x1, y1 = self._current_roi.as_tuple()
            cv2.rectangle(display_frame, (x0, y0), (x1, y1), (0, 0, 255), 2)

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.video_label.setFramePixmap(pixmap)

    def _save_frame(self, label: str) -> None:
        if self._current_frame is None:
            self.status_bar.showMessage("未捕获到画面。")
            return

        frame = self._current_frame
        if self._current_roi and self._current_roi.is_valid():
            x0, y0, x1, y1 = self._current_roi.as_tuple()
            x0 = max(0, min(self.frame_width - 1, x0))
            y0 = max(0, min(self.frame_height - 1, y0))
            x1 = max(0, min(self.frame_width, x1))
            y1 = max(0, min(self.frame_height, y1))
            if x1 > x0 and y1 > y0:
                frame = frame[y0:y1, x0:x1]

        timestamp = int(time.time() * 1000)
        target_dir = self._ensure_dir(label)
        filename = target_dir / f"{label}_{timestamp}.jpg"
        success = cv2.imwrite(str(filename), frame)
        if success:
            self.status_bar.showMessage(f"已保存 {filename}")
        else:
            self.status_bar.showMessage("保存失败，请检查目录权限。")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AirCursor 手势图像标注工具 (PyQt)")
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引 (默认: 0)")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["open", "closed", "peace"],
        help="手势标签列表（石头剪刀布）",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="输出根目录 (默认: data)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="static",
        help="标签类别子目录 (默认: static)",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="启用水平镜像，适合自拍摄像头",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    app = QtWidgets.QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = LabelingWindow(
        camera_index=args.camera_index,
        labels=args.labels,
        output_root=args.output_root,
        category=args.category,
        mirror=args.mirror,
    )
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

