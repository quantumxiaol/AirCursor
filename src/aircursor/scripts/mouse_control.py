#!/usr/bin/env python3
"""手势控制鼠标脚本

使用动态手势识别控制鼠标操作：
- TAP: 左键点击
- ZOOM（单手）: 右键点击
- DRAG/DROP: 拖拽操作
- SWIPE UP/DOWN: 垂直滚动
- SWIPE LEFT/RIGHT: 水平滚动
- 手部移动: 光标移动
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyautogui
from PyQt6 import QtCore, QtGui, QtWidgets

from aircursor.external.dynamic_gestures import DynamicGestureController, Event

# 配置 pyautogui
pyautogui.FAILSAFE = True  # 鼠标移到屏幕角落可以中止
pyautogui.PAUSE = 0.01  # 操作间隔


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[3]
    return (project_root / path).resolve()


@dataclass
class MouseControlConfig:
    """鼠标控制配置"""
    camera_index: int = 0
    detector_path: Path = Path("weights/hand_detector.onnx")
    classifier_path: Path = Path("weights/crops_classifier.onnx")
    mirror: bool = True
    debug_coords: bool = False  # 是否打印坐标调试信息
    show_display: bool = True  # 是否显示窗口
    
    # 鼠标控制参数
    cursor_smooth: float = 0.3  # 光标平滑系数 (0-1)
    scroll_speed: int = 20  # 滚动速度
    click_cooldown: float = 0.3  # 点击冷却时间（秒），降低以支持更快点击
    click_freeze_duration: float = 0.15  # 点击时光标冻结时间（秒），防止手动时光标偏移
    dead_zone: float = 0.15  # 死区比例 (0-0.5)，视频边缘不映射的区域
    
    # 手势识别参数（降低阈值以提高灵敏度）
    max_age: int = 25  # 轨迹最大存活帧数（降低以减少延迟）
    min_hits: int = 2   # 确认轨迹的最小检测次数（保持较低以快速响应）
    iou_threshold: float = 0.3  # IOU阈值（保持默认）
    maxlen: int = 30    # 轨迹历史最大长度（略微减少）
    min_frames: int = 12  # 确认手势的最小帧数（从18降到12，提高灵敏度）⭐


class MouseController:
    """手势到鼠标操作的控制器"""
    
    def __init__(self, cfg: MouseControlConfig):
        self.cfg = cfg
        
        # 获取屏幕尺寸
        self.screen_width, self.screen_height = pyautogui.size()
        
        # 初始化光标位置平滑
        self.cursor_history = deque(maxlen=5)
        
        # 状态跟踪
        self.last_click_time = 0.0
        self.last_action_time = 0.0  # 上次动作时间（用于点击冻结）
        self.is_dragging = False
        self.drag_start_pos = None
        self.initialized = False  # 是否已初始化光标位置
        self.cursor_frozen = False  # 光标是否冻结（点击时短暂冻结）
        
        # 事件历史（用于去重）
        self.event_history = deque(maxlen=3)
        
        print(f"🖥️  屏幕尺寸: {self.screen_width}x{self.screen_height}")
        print(f"💡 提示: 光标需要几帧初始化，请稍等...")
    
    def hand_to_screen(self, hand_x: float, hand_y: float, frame_width: int, frame_height: int) -> tuple[int, int]:
        """将手部坐标转换为屏幕坐标（带死区处理）
        
        死区（Dead Zone）：视频边缘一定比例的区域不映射到屏幕，
        这样手在视频边缘时也能操作到屏幕边缘。
        
        例如：dead_zone=0.15 时，视频边缘 15% 的区域被裁剪，
        只有中间 70% 的区域映射到整个屏幕。
        
        Args:
            hand_x: 手部中心 X 坐标（像素）
            hand_y: 手部中心 Y 坐标（像素）
            frame_width: 帧宽度
            frame_height: 帧高度
            
        Returns:
            屏幕坐标 (x, y)
        """
        # 归一化到 0-1
        norm_x = hand_x / frame_width
        norm_y = hand_y / frame_height
        
        # 应用死区：将 [dead_zone, 1-dead_zone] 重映射到 [0, 1]
        dz = self.cfg.dead_zone
        if dz > 0:
            # 重映射公式：(x - dz) / (1 - 2*dz)
            norm_x = (norm_x - dz) / (1 - 2 * dz)
            norm_y = (norm_y - dz) / (1 - 2 * dz)
        
        # 限制范围到 [0, 1]
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        
        # 映射到屏幕坐标
        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)
        
        # 确保不超出屏幕边界
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        # 平滑处理
        self.cursor_history.append((screen_x, screen_y))
        if len(self.cursor_history) >= 3:
            avg_x = int(sum(x for x, y in self.cursor_history) / len(self.cursor_history))
            avg_y = int(sum(y for x, y in self.cursor_history) / len(self.cursor_history))
            # 再次确保平滑后的坐标在范围内
            avg_x = max(0, min(self.screen_width - 1, avg_x))
            avg_y = max(0, min(self.screen_height - 1, avg_y))
            return avg_x, avg_y
        
        return screen_x, screen_y
    
    def move_cursor(self, hand_center: tuple[float, float], frame_width: int, frame_height: int, debug: bool = False):
        """移动光标"""
        # 检查是否在点击冻结时间内
        current_time = time.time()
        if current_time - self.last_action_time < self.cfg.click_freeze_duration:
            # 在冻结时间内，不移动光标
            if debug:
                print(f"❄️  光标冻结中... (剩余 {self.cfg.click_freeze_duration - (current_time - self.last_action_time):.2f}s)")
            return
        
        hand_x, hand_y = hand_center
        screen_x, screen_y = self.hand_to_screen(hand_x, hand_y, frame_width, frame_height)
        
        # 获取当前光标位置
        current_x, current_y = pyautogui.position()
        
        # 如果还未初始化，使用更强的平滑来避免突然跳转
        if not self.initialized:
            # 前几帧使用极低的平滑系数，让光标逐渐靠近目标
            smooth_factor = 0.05
            if len(self.cursor_history) >= 5:
                self.initialized = True
                print("✅ 光标初始化完成")
        else:
            smooth_factor = self.cfg.cursor_smooth
        
        # 平滑移动（插值）
        target_x = int(current_x + (screen_x - current_x) * smooth_factor)
        target_y = int(current_y + (screen_y - current_y) * smooth_factor)
        
        # 确保目标位置在屏幕范围内
        target_x = max(0, min(self.screen_width - 1, target_x))
        target_y = max(0, min(self.screen_height - 1, target_y))
        
        # 调试信息
        if debug:
            print(f"📍 手部: ({hand_x:.0f}, {hand_y:.0f}) → "
                  f"归一化: ({hand_x/frame_width:.2f}, {hand_y/frame_height:.2f}) → "
                  f"屏幕目标: ({screen_x}, {screen_y}) → "
                  f"平滑后: ({target_x}, {target_y}) | "
                  f"当前: ({current_x}, {current_y})")
        
        # 直接移动到目标位置（不使用 pyautogui 的动画）
        pyautogui.moveTo(target_x, target_y, _pause=False)
    
    def handle_event(self, event: Event) -> bool:
        """处理手势事件
        
        Args:
            event: 识别到的手势事件
            
        Returns:
            是否成功处理
        """
        current_time = time.time()
        
        # 去重：避免同一事件重复触发
        if self.event_history and self.event_history[-1] == event:
            return False
        
        self.event_history.append(event)
        
        try:
            # TAP -> 左键点击
            if event == Event.TAP:
                if current_time - self.last_click_time > self.cfg.click_cooldown:
                    pyautogui.click()
                    self.last_click_time = current_time
                    self.last_action_time = current_time  # 冻结光标
                    print("🖱️  左键点击")
                    return True
            
            # ZOOM_IN/OUT（单手）-> 右键点击
            elif event in [Event.ZOOM_IN, Event.ZOOM_OUT]:
                if current_time - self.last_click_time > self.cfg.click_cooldown:
                    pyautogui.rightClick()
                    self.last_click_time = current_time
                    self.last_action_time = current_time  # 冻结光标
                    print("🖱️  右键点击")
                    return True
            
            # DOUBLE_TAP -> 双击
            elif event == Event.DOUBLE_TAP:
                if current_time - self.last_click_time > self.cfg.click_cooldown:
                    pyautogui.doubleClick()
                    self.last_click_time = current_time
                    self.last_action_time = current_time  # 冻结光标
                    print("🖱️  双击")
                    return True
            
            # DRAG -> 开始拖拽
            elif event in [Event.DRAG, Event.DRAG2, Event.DRAG3]:
                if not self.is_dragging:
                    self.is_dragging = True
                    self.drag_start_pos = pyautogui.position()
                    pyautogui.mouseDown()
                    print("🖱️  开始拖拽")
                    return True
            
            # DROP -> 结束拖拽
            elif event in [Event.DROP, Event.DROP2, Event.DROP3]:
                if self.is_dragging:
                    pyautogui.mouseUp()
                    self.is_dragging = False
                    print("🖱️  结束拖拽")
                    return True
            
            # SWIPE UP/DOWN -> 垂直滚动
            elif event in [Event.SWIPE_UP, Event.SWIPE_UP2, Event.SWIPE_UP3, Event.FAST_SWIPE_UP]:
                pyautogui.scroll(self.cfg.scroll_speed)
                print("🖱️  向上滚动")
                return True
            
            elif event in [Event.SWIPE_DOWN, Event.SWIPE_DOWN2, Event.SWIPE_DOWN3, Event.FAST_SWIPE_DOWN]:
                pyautogui.scroll(-self.cfg.scroll_speed)
                print("🖱️  向下滚动")
                return True
            
            # SWIPE LEFT/RIGHT -> 水平滚动
            elif event in [Event.SWIPE_LEFT, Event.SWIPE_LEFT2, Event.SWIPE_LEFT3]:
                pyautogui.hscroll(-self.cfg.scroll_speed)
                print("🖱️  向左滚动")
                return True
            
            elif event in [Event.SWIPE_RIGHT, Event.SWIPE_RIGHT2, Event.SWIPE_RIGHT3]:
                pyautogui.hscroll(self.cfg.scroll_speed)
                print("🖱️  向右滚动")
                return True
            
        except Exception as e:
            print(f"❌ 鼠标操作错误: {e}")
            return False
        
        return False


class MouseControlWindow(QtWidgets.QMainWindow):
    """手势鼠标控制的 PyQt6 窗口"""
    
    def __init__(self, cfg: MouseControlConfig):
        super().__init__()
        self.cfg = cfg
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(cfg.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 无法打开摄像头 {cfg.camera_index}")
        
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 初始化控制器
        self.gesture_controller = DynamicGestureController(
            str(cfg.detector_path),
            str(cfg.classifier_path),
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_threshold,
            maxlen=cfg.maxlen,
            min_frames=cfg.min_frames,
        )
        
        self.mouse_controller = MouseController(cfg)
        
        # 事件历史（用于显示）
        self.event_history = deque(maxlen=5)
        
        # FPS 计算
        self.last_time = time.time()
        self.fps = 0
        
        # 设置 UI
        self._setup_ui()
        
        # 定时器
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(30)  # ~30 FPS
    
    def _setup_ui(self):
        """设置 UI"""
        self.setWindowTitle("AirCursor - Mouse Control")
        
        # 中央窗口
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # 视频显示
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)
        
        # 状态栏
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing...")
    
    def _update_frame(self):
        """更新帧"""
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Failed to read frame")
            return
        
        if self.cfg.mirror:
            frame = cv2.flip(frame, 1)
        
        # 计算 FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
        self.last_time = current_time
        
        # 手势识别
        bboxes, ids, labels = self.gesture_controller(frame)
        
        # 处理结果
        if bboxes is not None and len(bboxes) > 0:
            # 获取主手（第一个检测到的手）
            main_hand_bbox = bboxes[0]
            hand_center_x = (main_hand_bbox[0] + main_hand_bbox[2]) / 2
            hand_center_y = (main_hand_bbox[1] + main_hand_bbox[3]) / 2
            
            # 移动光标
            self.mouse_controller.move_cursor(
                (hand_center_x, hand_center_y),
                self.frame_width,
                self.frame_height,
                debug=self.cfg.debug_coords
            )
            
            # 处理手势事件
            for trk in self.gesture_controller.tracks:
                if trk["tracker"].time_since_update < 1:
                    if trk["hands"].action is not None:
                        action = trk["hands"].action
                        
                        # 添加到历史
                        if not self.event_history or self.event_history[-1] != action:
                            self.event_history.append(action)
                        
                        # 处理事件
                        self.mouse_controller.handle_event(action)
                        
                        # 清除事件（除了 DRAG）
                        if action not in [Event.DRAG, Event.DRAG2, Event.DRAG3]:
                            trk["hands"].action = None
            
            # 绘制检测框和死区
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 显示 ID
                cv2.putText(
                    frame,
                    f"Hand {i+1}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            
            # 绘制手部中心
            cx, cy = int(hand_center_x), int(hand_center_y)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        
        # 绘制死区边界
        if self.cfg.dead_zone > 0:
            dz = self.cfg.dead_zone
            dz_x = int(self.frame_width * dz)
            dz_y = int(self.frame_height * dz)
            
            # 外边界（红色虚线）
            cv2.rectangle(
                frame,
                (dz_x, dz_y),
                (self.frame_width - dz_x, self.frame_height - dz_y),
                (0, 0, 255),
                2,
            )
            
            # 死区标注
            cv2.putText(
                frame,
                f"Dead Zone: {int(dz*100)}%",
                (10, self.frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        
        # 顶部信息
        info_lines = [
            f"FPS: {self.fps:.1f}",
            f"Status: {'Controlling' if bboxes is not None and len(bboxes) > 0 else 'Waiting'}",
        ]
        
        if self.event_history:
            history_text = " <- ".join(event.name for event in reversed(list(self.event_history)))
            info_lines.append(f"Events: {history_text}")
        else:
            info_lines.append("Events: waiting...")
        
        # 绘制顶部信息
        y_offset = 30
        for line in info_lines:
            cv2.putText(
                frame,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y_offset += 30
        
        # 底部帮助信息
        help_lines = [
            "TAP:L-Click | ZOOM:R-Click | DRAG:Drag | SWIPE:Scroll",
            "Press 'Q' or ESC to quit",
        ]
        
        y_offset = self.frame_height - 50
        for line in help_lines:
            cv2.putText(
                frame,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 20
        
        # 转换为 QPixmap 并显示
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        
        self.video_label.setPixmap(pixmap)
        
        # 更新状态栏
        if bboxes is not None and len(bboxes) > 0:
            if self.event_history:
                latest_event = self.event_history[-1].name
                self.status_bar.showMessage(f"Latest Event: {latest_event} | Hands: {len(bboxes)} | FPS: {self.fps:.1f}")
            else:
                self.status_bar.showMessage(f"Controlling | Hands: {len(bboxes)} | FPS: {self.fps:.1f}")
        else:
            self.status_bar.showMessage(f"Waiting for gesture... | FPS: {self.fps:.1f}")
    
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """键盘事件"""
        if event.key() in [QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape]:
            self.close()
    
    def closeEvent(self, event):
        """关闭事件"""
        self.timer.stop()
        self.cap.release()
        print("\n👋 程序已退出")
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="手势控制鼠标 | Gesture Mouse Control")
    
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引 | Camera index")
    parser.add_argument("--detector", type=Path, required=True, help="手部检测模型路径 | Hand detector model path")
    parser.add_argument("--classifier", type=Path, required=True, help="手势分类模型路径 | Gesture classifier model path")
    parser.add_argument("--mirror", action="store_true", help="镜像翻转画面 | Mirror flip the frame")
    parser.add_argument("--no-display", action="store_true", help="不显示窗口 | No display window")
    parser.add_argument("--cursor-smooth", type=float, default=0.3, help="光标平滑系数 (0-1) | Cursor smoothing factor")
    parser.add_argument("--scroll-speed", type=int, default=20, help="滚动速度 | Scroll speed")
    parser.add_argument("--dead-zone", type=float, default=0.15, help="死区比例 (0-0.5) | Dead zone ratio for edge mapping")
    parser.add_argument("--click-freeze", type=float, default=0.15, help="点击时光标冻结时间 (秒) | Click freeze duration (seconds)")
    parser.add_argument("--min-frames", type=int, default=12, help="手势确认最小帧数 (降低以提高灵敏度) | Min frames for gesture (lower for faster)")
    parser.add_argument("--debug-coords", action="store_true", help="打印坐标映射调试信息 | Print coordinate mapping debug info")
    
    args = parser.parse_args()
    
    # 配置
    cfg = MouseControlConfig(
        camera_index=args.camera,
        detector_path=resolve_path(args.detector),
        classifier_path=resolve_path(args.classifier),
        mirror=args.mirror,
        debug_coords=args.debug_coords,
        show_display=not args.no_display,
        cursor_smooth=args.cursor_smooth,
        scroll_speed=args.scroll_speed,
        dead_zone=args.dead_zone,
        click_freeze_duration=args.click_freeze,
        min_frames=args.min_frames,
    )
    
    # 检查模型文件
    if not cfg.detector_path.exists():
        print(f"❌ 找不到检测模型: {cfg.detector_path}")
        print("💡 请运行: python download_models.py")
        return 1
    
    if not cfg.classifier_path.exists():
        print(f"❌ 找不到分类模型: {cfg.classifier_path}")
        print("💡 请运行: python download_models.py")
        return 1
    
    # 打印启动信息
    print("🚀 启动手势鼠标控制... | Starting Gesture Mouse Control...")
    print(f"📹 摄像头 | Camera: {cfg.camera_index}")
    print(f"🤖 检测模型 | Detector: {cfg.detector_path.name}")
    print(f"🤖 分类模型 | Classifier: {cfg.classifier_path.name}")
    print()
    print("⚙️  优化参数 | Optimized Parameters:")
    print(f"  🎯 死区 | Dead Zone: {int(cfg.dead_zone*100)}%")
    print(f"  ✨ 平滑系数 | Smoothing: {cfg.cursor_smooth}")
    print(f"  ❄️  点击冻结 | Click Freeze: {cfg.click_freeze_duration}s")
    print(f"  ⚡ 手势帧数 | Min Frames: {cfg.min_frames} (降低以提高灵敏度 | Lower for faster)")
    print()
    print("📋 手势映射 | Gesture Mapping:")
    print("  🤏 TAP           → 左键点击 | Left Click")
    print("  👌 ZOOM          → 右键点击 | Right Click")
    print("  ✊ DRAG/DROP     → 拖拽 | Drag")
    print("  👆 SWIPE UP/DOWN → 垂直滚动 | Vertical Scroll")
    print("  👉 SWIPE L/R     → 水平滚动 | Horizontal Scroll")
    print("  🖐️  手部移动 | Hand Move → 光标移动 | Cursor Move")
    print()
    print("⚠️  提示 | Tips:")
    print("  • 移动鼠标到屏幕角落可以紧急停止 | Move mouse to corner to emergency stop")
    print("  • 按 'Q' 或 ESC 键退出 | Press 'Q' or ESC to quit")
    print("  • 红色矩形框显示死区边界 | Red rectangle shows dead zone boundary")
    print("  • 点击时光标会短暂冻结，防止偏移 | Cursor freezes briefly during clicks")
    print()
    
    try:
        # 创建 PyQt6 应用
        app = QtWidgets.QApplication([])
        
        # 创建窗口
        if cfg.show_display:
            window = MouseControlWindow(cfg)
            window.show()
            print("✅ 初始化完成 | Initialization complete")
            print("🎮 窗口已打开，开始控制... | Window opened, control started...")
            print()
            
            # 运行应用
            return app.exec()
        else:
            # 无显示模式（暂不支持，因为需要显著重构）
            print("⚠️  无显示模式暂不支持 | No-display mode not yet supported")
            print("💡 请移除 --no-display 参数 | Please remove --no-display flag")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  程序被用户中断 | Program interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ 错误 | Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

