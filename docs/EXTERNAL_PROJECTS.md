# 外部项目说明 | External Projects

本项目整合了两个优秀的开源手势识别项目，并在此基础上实现了鼠标控制功能。

---

## 1. HaGRID - 手势识别数据集和模型

### 项目地址
- GitHub: https://github.com/hukenovs/hagrid
- 论文: https://arxiv.org/abs/2206.08219

### 项目介绍
HaGRID (HAnd Gesture Recognition Image Dataset) 是一个大规模手势识别数据集，包含：
- **554,800+ 张图像**
- **18 种手势类别**：call, dislike, fist, four, like, mute, ok, one, palm, peace, peace_inverted, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted, no_gesture
- **ResNet 系列模型**：ResNet18, ResNet152 等

### 在本项目中的使用
我们使用 HaGRID 的：
1. **预训练模型**（ResNet18）用于静态手势分类
2. **标签映射**：将 HaGRID 手势映射到我们的 3 种基础手势（石头、剪刀、布）
   - `fist` → **closed（石头/拳头）**
   - `peace`, `peace_inverted`, `two_up`, `two_up_inverted` → **peace（剪刀）**
   - `palm`, `stop`, `stop_inverted` → **open（布/手掌）**

### 识别原理
- **输入**：摄像头图像
- **处理**：MediaPipe 提取手部区域 → 裁剪手部图像
- **分类**：ResNet 模型预测手势类别
- **输出**：手势标签和置信度

---

## 2. Dynamic Gestures - 动态手势识别

### 项目地址
- GitHub: https://github.com/ai-forever/dynamic_gestures
- 作者: Sber AI

### 项目介绍
Dynamic Gestures 是一个实时动态手势识别系统，使用 ONNX 模型识别连续的手部动作：
- **手部检测**：基于 YOLO 的手部检测器
- **手势分类**：基于序列的手势识别
- **轨迹跟踪**：OC-SORT 算法跟踪手部运动

### 支持的手势类别
**24 种动态手势事件**：
- **点击类**：TAP, DOUBLE_TAP
- **缩放类**：ZOOM_IN, ZOOM_OUT
- **拖拽类**：DRAG, DROP (支持多手)
- **滑动类**：
  - 上下：SWIPE_UP, SWIPE_DOWN (1-3 手)
  - 左右：SWIPE_LEFT, SWIPE_RIGHT (1-3 手)
  - 快速：FAST_SWIPE_UP, FAST_SWIPE_DOWN

完整列表见 [DYNAMIC_GESTURES.md](DYNAMIC_GESTURES.md)

### 识别原理
1. **检测**：ONNX 手部检测器定位手部位置
2. **跟踪**：OC-SORT 算法跟踪每只手的运动轨迹
3. **序列分析**：分析连续帧的手势轨迹
4. **事件触发**：识别完整手势后触发事件

### 关键参数
- `min_frames`: 确认手势的最小帧数（默认 12）
- `max_age`: 轨迹最大存活时间（默认 25）
- `min_hits`: 确认轨迹的最小检测次数（默认 2）

---

## 本项目的手势映射

### 🎮 动态手势鼠标控制

基于 Dynamic Gestures 的鼠标控制方案（**推荐**）：

| 动态手势 | 鼠标操作 | 说明 |
|---------|---------|------|
| **TAP** (捏合) | 左键点击 | 快速捏合食指和拇指 |
| **ZOOM** (单手捏合) | 右键点击 | 单手做捏合放大手势 |
| **DRAG/DROP** | 拖拽 | 保持捏合并移动，松开释放 |
| **SWIPE UP/DOWN** | 垂直滚动 | 手掌上下快速挥动 |
| **SWIPE LEFT/RIGHT** | 水平滚动 | 手掌左右快速挥动 |
| **手部移动** | 光标移动 | 检测到手后光标跟随手部中心 |

**特点**：
- ⚡ 识别速度：约 0.4 秒
- 🎯 死区映射：边缘 15% 不映射，易于到达屏幕边缘
- ❄️ 点击冻结：点击时光标短暂冻结，防止偏移

---

### ✋ 静态手势鼠标控制

基于 HaGRID 的静态手势控制方案（**备选**）：

| 静态手势 | 鼠标操作 | 说明 |
|---------|---------|------|
| **✊ 拳头（石头）移动** | 光标移动 | 握拳状态下移动手控制光标 |
| **✊ → ✋ 石头变布** | 左键点击 | 从拳头变为手掌（常用操作） |
| **✊ → ✌️ 石头变剪刀** | 右键点击 | 从拳头变为剪刀手（菜单操作） |
| **✋ 布移动** | 拖拽（按下） | 保持手掌状态移动 = 按下左键拖动 |
| **✌️ 剪刀移动** | 拖拽（按下） | 保持剪刀手移动 = 按下左键拖动 |

**特点**：
- 🎯 状态清晰：手势形状直观对应操作
- 🔄 手势切换：从拳头切换到其他手势触发点击
- ⏱️ 时序稳定：5帧投票机制，避免误识别
- ✋✌️✊ 仅需 3 种手势：石头、剪刀、布
- 👌 符合直觉：布（展开）=左键（常用），剪刀（特殊）=右键（少用）

---

## 技术栈

### 手部检测
- **MediaPipe Hand Landmarker**（静态手势）
  - 21 个手部关键点
  - 实时跟踪
  - 支持多手检测

- **ONNX Hand Detector**（动态手势）
  - 基于 YOLO
  - 高速检测
  - 轨迹跟踪

### 手势分类
- **静态**：ResNet18/152 (HaGRID) + MLP (自训练)
- **动态**：序列分析 + ONNX 分类器

### 鼠标控制
- **pyautogui**：跨平台鼠标控制
- **坐标映射**：死区处理 + 平滑插值
- **事件管理**：冷却时间 + 冻结机制

---

## 参考文献

1. HaGRID Dataset:
   ```
   @article{kapitanov2022hagrid,
     title={HaGRID--HAnd Gesture Recognition Image Dataset},
     author={Kapitanov, Alexander and Kvanchiani, Karina and Nagaev, Alexander and Kraynov, Roman and Makhlyarchuk, Andrei},
     journal={arXiv preprint arXiv:2206.08219},
     year={2022}
   }
   ```

2. Dynamic Gestures: https://github.com/ai-forever/dynamic_gestures

3. MediaPipe: https://developers.google.com/mediapipe

---

**更新时间**：2024-11-11

