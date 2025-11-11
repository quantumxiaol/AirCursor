#!/usr/bin/env python3
"""静态手势控制鼠标脚本

使用静态手势（石头剪刀布）控制鼠标操作：
- ✊ 拳头（石头）移动：光标移动
- ✋ → ✌️ 布变剪刀：左键点击
- ✊ → ✋ 石头变布：右键点击
- ✋ 布移动：按下左键拖拽
- ✌️ 剪刀移动：按下左键拖拽
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

from aircursor.core.hand_tracker import HandTracker, HandTrackerConfig
from aircursor.models.static_mlp import StaticGestureClassifier

# 配置 pyautogui
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[3]
    return (project_root / path).resolve()


@dataclass
class StaticMouseConfig:
    """静态手势鼠标控制配置"""
    camera_index: int = 0
    model_path: Optional[Path] = None
    landmarker_path: Path = Path("weights/hand_landmarker.task")
    mirror: bool = True
    debug: bool = False
    
    # 鼠标控制参数
    cursor_smooth: float = 0.3
    dead_zone: float = 0.15
    click_freeze_duration: float = 0.15
    
    # 手势识别参数
    confidence_threshold: float = 0.7


class StaticMouseController:
    """静态手势到鼠标操作的控制器"""
    
    def __init__(self, cfg: StaticMouseConfig):
        self.cfg = cfg
        
        # 获取屏幕尺寸
        self.screen_width, self.screen_height = pyautogui.size()
        
        # 手势状态
        self.current_gesture: Optional[str] = None
        self.previous_gesture: Optional[str] = None
        self.gesture_start_time = 0.0
        
        # 鼠标状态
        self.cursor_history = deque(maxlen=5)
        self.last_action_time = 0.0
        self.is_button_down = False  # 是否处于按下状态（布或剪刀移动时）
        self.initialized = False
        
        # 时序信息：手势历史记录（用于稳定识别）
        from collections import deque
        self.gesture_history = deque(maxlen=5)  # 保存最近5帧的手势
        self.stable_gesture = None  # 稳定的手势（经过时序过滤）
        
        print(f"🖥️  屏幕尺寸: {self.screen_width}x{self.screen_height}")
        print(f"📋 手势映射:")
        print(f"  ✊ 拳头移动 → 光标移动")
        print(f"  ✊ → ✋ 石头变布 → 左键点击")
        print(f"  ✊ → ✌️  石头变剪刀 → 右键点击")
        print(f"  ✋ 布移动 → 拖拽（按下左键）")
        print(f"  ✌️  剪刀移动 → 拖拽（按下左键）")
        print(f"  ⏱️  使用时序信息稳定手势识别（5帧投票）")
        print()
    
    def hand_to_screen(self, hand_x: float, hand_y: float, frame_width: int, frame_height: int) -> tuple[int, int]:
        """将手部坐标转换为屏幕坐标（带死区处理）"""
        # 归一化到 0-1
        norm_x = hand_x / frame_width
        norm_y = hand_y / frame_height
        
        # 应用死区
        dz = self.cfg.dead_zone
        if dz > 0:
            norm_x = (norm_x - dz) / (1 - 2 * dz)
            norm_y = (norm_y - dz) / (1 - 2 * dz)
        
        # 限制范围
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        
        # 映射到屏幕
        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)
        
        # 边界检查
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        # 平滑处理
        self.cursor_history.append((screen_x, screen_y))
        if len(self.cursor_history) >= 3:
            avg_x = int(sum(x for x, y in self.cursor_history) / len(self.cursor_history))
            avg_y = int(sum(y for x, y in self.cursor_history) / len(self.cursor_history))
            avg_x = max(0, min(self.screen_width - 1, avg_x))
            avg_y = max(0, min(self.screen_height - 1, avg_y))
            return avg_x, avg_y
        
        return screen_x, screen_y
    
    def move_cursor(self, hand_center: tuple[float, float], frame_width: int, frame_height: int, gesture: str):
        """移动光标（仅在拳头状态或初始化时）"""
        # 检查是否在点击冻结时间内
        current_time = time.time()
        if current_time - self.last_action_time < self.cfg.click_freeze_duration:
            return
        
        hand_x, hand_y = hand_center
        screen_x, screen_y = self.hand_to_screen(hand_x, hand_y, frame_width, frame_height)
        
        # 获取当前光标位置
        current_x, current_y = pyautogui.position()
        
        # 初始化阶段使用低平滑系数
        if not self.initialized:
            smooth_factor = 0.05
            if len(self.cursor_history) >= 5:
                self.initialized = True
                print("✅ 光标初始化完成")
        else:
            smooth_factor = self.cfg.cursor_smooth
        
        # 平滑移动
        target_x = int(current_x + (screen_x - current_x) * smooth_factor)
        target_y = int(current_y + (screen_y - current_y) * smooth_factor)
        
        # 边界检查
        target_x = max(0, min(self.screen_width - 1, target_x))
        target_y = max(0, min(self.screen_height - 1, target_y))
        
        pyautogui.moveTo(target_x, target_y, _pause=False)
    
    def _get_stable_gesture(self, gesture: Optional[str]) -> Optional[str]:
        """使用时序信息获取稳定的手势（投票机制）
        
        Args:
            gesture: 当前帧识别的手势
            
        Returns:
            稳定的手势（需要在历史中占多数）
        """
        if gesture is None:
            self.gesture_history.clear()
            return None
        
        # 添加当前手势到历史
        self.gesture_history.append(gesture)
        
        # 如果历史记录不足3帧，使用之前的稳定手势（避免过早切换）
        if len(self.gesture_history) < 3:
            return self.stable_gesture
        
        # 投票机制：统计最近5帧中每种手势的出现次数
        gesture_counts = {}
        for g in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        # 找出出现次数最多的手势
        max_count = max(gesture_counts.values())
        
        # 需要至少出现3次才能被认为是稳定的（超过半数）
        if max_count >= 3:
            most_common = [g for g, count in gesture_counts.items() if count == max_count]
            # 如果有多个手势出现次数相同，优先返回当前手势（如果在其中）
            if gesture in most_common:
                return gesture
            else:
                return most_common[0]
        
        # 如果没有手势出现3次以上，保持之前的稳定手势
        return self.stable_gesture
    
    def update_gesture(self, gesture: Optional[str], hand_center: tuple[float, float], frame_width: int, frame_height: int):
        """更新手势状态并执行相应操作"""
        if gesture is None:
            # 没有检测到手势，释放按键
            if self.is_button_down:
                pyautogui.mouseUp()
                self.is_button_down = False
                print("🖱️  释放鼠标")
            self.current_gesture = None
            self.previous_gesture = None
            self.stable_gesture = None
            self.gesture_history.clear()
            return
        
        # 使用时序信息稳定手势
        stable_gesture = self._get_stable_gesture(gesture)
        
        if stable_gesture is None:
            # 还在收集历史信息，暂不处理
            return
        
        # 更新稳定手势
        if stable_gesture != self.stable_gesture:
            self.stable_gesture = stable_gesture
        
        # 使用稳定后的手势
        gesture = stable_gesture
        
        current_time = time.time()
        
        # 检测手势切换
        if self.current_gesture != gesture:
            # 手势发生变化
            self.previous_gesture = self.current_gesture
            self.current_gesture = gesture
            self.gesture_start_time = current_time
            
            # 处理手势切换触发的点击
            if self.previous_gesture and self.current_gesture:
                # ✊ → ✋ 石头变布 → 左键点击
                if self.previous_gesture == "closed" and self.current_gesture == "open":
                    pyautogui.click()
                    self.last_action_time = current_time
                    print("🖱️  左键点击（石头→布）")
                
                # ✊ → ✌️ 石头变剪刀 → 右键点击
                elif self.previous_gesture == "closed" and self.current_gesture == "peace":
                    pyautogui.rightClick()
                    self.last_action_time = current_time
                    print("🖱️  右键点击（石头→剪刀）")
            
            # 检查是否需要按下鼠标（布或剪刀）
            if self.current_gesture in ["open", "peace"]:
                if not self.is_button_down:
                    pyautogui.mouseDown()
                    self.is_button_down = True
                    print(f"🖱️  按下鼠标（{self.current_gesture}）")
            else:
                # 拳头状态，释放鼠标
                if self.is_button_down:
                    pyautogui.mouseUp()
                    self.is_button_down = False
                    print("🖱️  释放鼠标")
        
        # 根据当前手势移动光标
        if self.current_gesture == "closed":
            # 拳头：正常光标移动
            self.move_cursor(hand_center, frame_width, frame_height, gesture)
        elif self.current_gesture in ["open", "peace"]:
            # 布或剪刀：拖拽移动
            self.move_cursor(hand_center, frame_width, frame_height, gesture)


class StaticMouseWindow(QtWidgets.QMainWindow):
    """静态手势鼠标控制的 PyQt6 窗口"""
    
    def __init__(self, cfg: StaticMouseConfig):
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
        
        # 初始化手部跟踪器
        tracker_cfg = HandTrackerConfig(
            model_path=cfg.landmarker_path,
            max_num_hands=1,
        )
        self.hand_tracker = HandTracker(tracker_cfg)
        
        # 初始化手势分类器
        if cfg.model_path and cfg.model_path.exists():
            self.classifier = StaticGestureClassifier(str(cfg.model_path))
            print(f"✅ 加载模型: {cfg.model_path}")
        else:
            self.classifier = None
            print("⚠️  未加载模型，使用启发式规则")
        
        # 初始化鼠标控制器
        self.mouse_controller = StaticMouseController(cfg)
        
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
        self.setWindowTitle("AirCursor - Static Gesture Mouse Control")
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)
        
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
        
        # 手部跟踪
        hand_packet = self.hand_tracker.process(frame)
        
        gesture = None
        hand_center = None
        
        if hand_packet:
            # landmarks 是 numpy 数组 (21, 3)
            landmarks_np = hand_packet.landmarks
            
            # 手势分类
            if self.classifier:
                gesture = self.classifier.predict(landmarks_np)
            else:
                # 使用启发式规则
                gesture = self._heuristic_classify(landmarks_np)
            
            # 计算手部中心（手腕，第0个关键点）
            wrist = landmarks_np[0]  # [x, y, z]
            hand_center = (wrist[0] * self.frame_width, wrist[1] * self.frame_height)
            
            # 绘制手部关键点
            for lm in landmarks_np:
                x, y = int(lm[0] * self.frame_width), int(lm[1] * self.frame_height)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # 绘制手部中心
            cx, cy = int(hand_center[0]), int(hand_center[1])
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        
        # 更新鼠标控制
        if hand_center:
            self.mouse_controller.update_gesture(gesture, hand_center, self.frame_width, self.frame_height)
        else:
            self.mouse_controller.update_gesture(None, (0, 0), self.frame_width, self.frame_height)
        
        # 绘制死区边界
        if self.cfg.dead_zone > 0:
            dz = self.cfg.dead_zone
            dz_x = int(self.frame_width * dz)
            dz_y = int(self.frame_height * dz)
            cv2.rectangle(
                frame,
                (dz_x, dz_y),
                (self.frame_width - dz_x, self.frame_height - dz_y),
                (0, 0, 255),
                2,
            )
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
            f"Gesture: {gesture if gesture else 'None'}",
            f"Status: {'Dragging' if self.mouse_controller.is_button_down else 'Moving'}",
        ]
        
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
            "Fist:Move | Closed->Open:L-Click | Closed->Peace:R-Click",
            "Open/Peace Move:Drag | Temporal Smoothing(5 frames) | Press 'Q' or ESC to quit",
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
        if gesture:
            status_text = f"Current: {gesture} | "
            if self.mouse_controller.is_button_down:
                status_text += "Dragging"
            else:
                status_text += "Moving"
            status_text += f" | FPS: {self.fps:.1f}"
            self.status_bar.showMessage(status_text)
        else:
            self.status_bar.showMessage(f"Waiting for hand... | FPS: {self.fps:.1f}")
    
    def _heuristic_classify(self, landmarks_np: np.ndarray) -> str:
        """启发式手势分类（简单规则）
        
        Args:
            landmarks_np: numpy 数组 (21, 3)，每行是 [x, y, z]
        """
        # 计算手指伸展度
        def finger_extended(tip_idx: int, pip_idx: int) -> bool:
            tip = landmarks_np[tip_idx]  # [x, y, z]
            pip = landmarks_np[pip_idx]
            palm = landmarks_np[0]  # 手腕
            # 简单判断：指尖离手腕的距离 > PIP离手腕的距离
            tip_dist = ((tip[0] - palm[0])**2 + (tip[1] - palm[1])**2)**0.5
            pip_dist = ((pip[0] - palm[0])**2 + (pip[1] - palm[1])**2)**0.5
            return tip_dist > pip_dist * 1.2
        
        # 检查每根手指
        thumb_extended = finger_extended(4, 3)
        index_extended = finger_extended(8, 6)
        middle_extended = finger_extended(12, 10)
        ring_extended = finger_extended(16, 14)
        pinky_extended = finger_extended(20, 18)
        
        extended_count = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
        
        # 分类规则
        if extended_count >= 4:
            return "open"  # 布
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "peace"  # 剪刀
        else:
            return "closed"  # 石头
    
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """键盘事件"""
        if event.key() in [QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape]:
            self.close()
    
    def closeEvent(self, event):
        """关闭事件"""
        # 确保释放鼠标
        if self.mouse_controller.is_button_down:
            pyautogui.mouseUp()
        
        self.timer.stop()
        self.cap.release()
        self.hand_tracker.close()
        print("\n👋 程序已退出")
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="静态手势控制鼠标 | Static Gesture Mouse Control")
    
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--model", type=Path, help="静态手势模型路径（可选）")
    parser.add_argument("--landmarker", type=Path, default=Path("weights/hand_landmarker.task"), help="MediaPipe 模型路径")
    parser.add_argument("--mirror", action="store_true", help="镜像翻转画面")
    parser.add_argument("--cursor-smooth", type=float, default=0.3, help="光标平滑系数")
    parser.add_argument("--dead-zone", type=float, default=0.15, help="死区比例")
    parser.add_argument("--click-freeze", type=float, default=0.15, help="点击冻结时间")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    # 配置
    cfg = StaticMouseConfig(
        camera_index=args.camera,
        model_path=resolve_path(args.model) if args.model else None,
        landmarker_path=resolve_path(args.landmarker),
        mirror=args.mirror,
        debug=args.debug,
        cursor_smooth=args.cursor_smooth,
        dead_zone=args.dead_zone,
        click_freeze_duration=args.click_freeze,
    )
    
    # 检查 landmarker
    if not cfg.landmarker_path.exists():
        print(f"❌ 找不到 MediaPipe 模型: {cfg.landmarker_path}")
        print("💡 请运行: python download_models.py")
        return 1
    
    # 打印启动信息
    print("🚀 启动静态手势鼠标控制... | Starting Static Gesture Mouse Control...")
    print(f"📹 摄像头: {cfg.camera_index}")
    print(f"🤖 MediaPipe 模型: {cfg.landmarker_path.name}")
    if cfg.model_path:
        print(f"🤖 手势模型: {cfg.model_path.name}")
    print()
    print("📋 手势映射 | Gesture Mapping:")
    print("  ✊ 拳头移动 | Fist Move → 光标移动 | Cursor Move")
    print("  ✊ → ✋ 石头变布 | Closed->Open → 左键点击 | Left Click")
    print("  ✊ → ✌️  石头变剪刀 | Closed->Peace → 右键点击 | Right Click")
    print("  ✋ 布移动 | Open Move → 拖拽 | Drag")
    print("  ✌️  剪刀移动 | Peace Move → 拖拽 | Drag")
    print("  ⏱️  时序稳定 | Temporal Smoothing → 5帧投票机制")
    print()
    
    try:
        # 创建 PyQt6 应用
        app = QtWidgets.QApplication([])
        
        window = StaticMouseWindow(cfg)
        window.show()
        
        print("✅ 初始化完成 | Initialization complete")
        print("🎮 窗口已打开，开始控制... | Window opened, control started...")
        print()
        
        return app.exec()
    
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

