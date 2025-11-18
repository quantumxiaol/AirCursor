# AirCursor – 基于 PyTorch 的视觉手势控制鼠标系统

非接触式空中鼠标：仅需普通笔记本前置摄像头，通过手势实现光标移动、点击、拖拽、滚动等操作。

## 特性

### 🎮 两种控制方案

**方案一：动态手势控制**（推荐）
- 🤏 捏合（TAP） → 左键点击
- 👌 缩放（ZOOM） → 右键点击
- ✊ 拖放（DRAG/DROP） → 拖拽
- 👆 滑动上下 → 垂直滚动
- 👉 滑动左右 → 水平滚动
- 🖐️ 手部移动 → 光标移动

**方案二：静态手势控制**（备选）
- ✊ 拳头移动 → 光标移动
- ✊ → ✋ 石头变布 → 左键点击
- ✊ → ✌️ 石头变剪刀 → 右键点击
- ✋ 布移动 → 拖拽
- ✌️ 剪刀移动 → 拖拽
- ⏱️ 时序稳定 → 5帧投票机制

技术栈：
- 手部检测：MediaPipe Hand Landmarker + ONNX 检测器
- 手势分类：ResNet（HaGRID）+ ONNX 分类器 / MLP（自训练）
- 鼠标控制：PyAutoGUI + 坐标映射 + 平滑算法

基于 MediaPipe + PyTorch + PyAutoGUI

## ✨ 手势控制鼠标

### 🎮 方案一：动态手势控制（推荐）

使用连续动作手势直接控制鼠标，响应快、准确度高：

- 🤏 **TAP**（捏合）: 左键点击
- 👌 **ZOOM**（单手缩放）: 右键点击
- ✊ **DRAG/DROP**（拖放）: 拖拽操作
- 👆 **SWIPE UP/DOWN**（上下滑动）: 垂直滚动
- 👉 **SWIPE LEFT/RIGHT**（左右滑动）: 水平滚动
- 🖐️ **手部移动**: 光标跟随

**✨ 特性**:
- 🎯 **死区映射**：视频边缘 15% 区域不映射，让手在边缘也能到达屏幕边界
- 🪟 **PyQt6 窗口**：现代化 GUI，支持更好的文本渲染和中文显示
- 🎨 **实时可视化**：红色矩形框标记死区边界，绿色框显示手部检测
- 📊 **事件历史**：显示最近 5 个手势事件，便于调试
- ⚡ **灵敏度优化**：手势识别速度提升 30-40%（min_frames: 18→12）
- ❄️ **点击冻结**：点击时光标短暂冻结 0.15s，防止手动导致的偏移

### ✋ 方案二：静态手势控制（备选）

使用石头剪刀布手势控制鼠标，状态清晰、直观易用：

- ✊ **拳头移动**: 光标移动（正常移动模式）
- ✊ → ✋ **石头变布**: 左键点击（常用操作）
- ✊ → ✌️ **石头变剪刀**: 右键点击（菜单操作）
- ✋ **布移动**: 拖拽（按下左键移动）
- ✌️ **剪刀移动**: 拖拽（按下左键移动）

**✨ 特性**:
- 🎯 **状态清晰**：手势形状直观对应操作状态
- 🔄 **手势切换触发**：从拳头切换到其他手势触发点击
- ⏱️ **时序稳定**：5帧投票机制，避免误识别导致的误操作
- ✋✌️✊ **仅需 3 种手势**：石头、剪刀、布，简单易学
- 👌 **符合直觉**：布（展开）=左键（常用），剪刀（特殊）=右键（少用）

**动态手势控制使用方法**:

```bash
# 基础使用（默认死区 15%）
python -m aircursor.scripts.mouse_control \
  --detector weights/hand_detector.onnx \
  --classifier weights/crops_classifier.onnx \
  --mirror

# 自定义死区（推荐 MacBook 用户使用 20%）| Custom dead zone (20% recommended for MacBook)
python -m aircursor.scripts.mouse_control \
  --detector weights/hand_detector.onnx \
  --classifier weights/crops_classifier.onnx \
  --mirror \
  --dead-zone 0.20

# 调试模式（查看坐标映射）| Debug mode (view coordinate mapping)
python -m aircursor.scripts.mouse_control \
  --detector weights/hand_detector.onnx \
  --classifier weights/crops_classifier.onnx \
  --mirror \
  --dead-zone 0.15 \
  --debug-coords

# 调整平滑度（更快响应）| Adjust smoothing (faster response)
python -m aircursor.scripts.mouse_control \
  --detector weights/hand_detector.onnx \
  --classifier weights/crops_classifier.onnx \
  --mirror \
  --cursor-smooth 0.2  # 默认 0.3，越小越快但越不稳定

# 提高灵敏度（更快识别手势）| Higher sensitivity (faster gesture recognition)
python -m aircursor.scripts.mouse_control \
  --detector weights/hand_detector.onnx \
  --classifier weights/crops_classifier.onnx \
  --mirror \
  --min-frames 10  # 默认 12，越小越灵敏但可能误触

# 调整点击冻结（防止点击时光标偏移）| Adjust click freeze (prevent cursor drift on click)
python -m aircursor.scripts.mouse_control \
  --detector weights/hand_detector.onnx \
  --classifier weights/crops_classifier.onnx \
  --mirror \
  --click-freeze 0.2  # 默认 0.15，越大越稳定但可能影响连续操作

# 调整滚动速度 | Adjust scroll speed
python -m aircursor.scripts.mouse_control \
  --detector weights/hand_detector.onnx \
  --classifier weights/crops_classifier.onnx \
  --mirror \
  --scroll-speed 30  # 默认 20

# 组合优化（最灵敏配置）| Combined optimization (most sensitive)
python -m aircursor.scripts.mouse_control \
  --detector weights/hand_detector.onnx \
  --classifier weights/crops_classifier.onnx \
  --mirror \
  --min-frames 10 \
  --click-freeze 0.2 \
  --cursor-smooth 0.25 \
  --dead-zone 0.18
```

**静态手势控制使用方法**:

```bash
# 基础使用（仅需 MediaPipe 模型）
python -m aircursor.scripts.static_mouse_control \
  --landmarker weights/hand_landmarker.task \
  --mirror

# 使用自训练模型（可选）
python -m aircursor.scripts.static_mouse_control \
  --landmarker weights/hand_landmarker.task \
  --model modelsweights/static_mlp.pth \
  --mirror

# 自定义参数
python -m aircursor.scripts.static_mouse_control \
  --landmarker weights/hand_landmarker.task \
  --mirror \
  --cursor-smooth 0.4 \
  --dead-zone 0.18 \
  --click-freeze 0.2
```

### 📦 模型自动下载

下载所有必需的模型：

```bash
# 下载所有模型（~80MB）
python download_models.py

# 查看可用模型
python download_models.py --list

# 只下载特定模型
python download_models.py --models hand_landmarker.task ResNet18.pth
```

支持的模型：
- `hand_landmarker.task` (~26MB) - MediaPipe 手部关键点检测
- `hand_detector.onnx` (~9MB) - 动态手势检测
- `crops_classifier.onnx` (~1.5MB) - 动态手势分类
- `ResNet18.pth` (~43MB) - HaGRID 静态手势分类

## 快速开始

### 1. 配置文件初始化

首次使用需要创建个人配置文件：

```bash
# 从模板复制配置文件
cp config.yaml.example config.yaml

# 根据需要调整参数（可选）
# vim config.yaml
```

**个性化参数说明**：
- `camera.index` - 摄像头索引（多摄像头时需调整）
- `cursor.smooth_factor` - 光标平滑度（0.1-0.5，越小越灵敏）
- `cursor.dead_zone` - 抖动抑制范围（0.05-0.15）
- `scroll.speed_scale` - 滚动速度（1-10）

> 💡 `config.yaml` 已加入 `.gitignore`，可以自由调整而不影响版本控制。

### 2. 安装依赖

```bash
uv venv
source .venv/bin/activate
uv lock
uv sync
```

> macOS 用户如遇摄像头权限问题，可先执行 `brew install opencv` 并在"隐私与安全性"中允许终端使用摄像头；Apple Silicon 机器建议使用 python.org 官方发行版。

### 3. 运行主程序

```bash
python -m aircursor
```

**石头剪刀布手势**（仅用于旧版主程序，推荐使用下方的鼠标控制脚本）：
- 👋 `open`（布）：移动光标  
- ✊ `closed`（石头）：按住左键（松手自动释放），用于点击/拖拽  
- ✌️ `peace`（剪刀）：滚动模式，上下挥手滚屏

默认使用启发式规则保证开箱即用；若提供训练好的模型权重（详见下文），系统会自动加载并切换为神经网络推理。

> **💡 提示**：推荐使用下方的"手势控制鼠标"功能，体验更好！

### 4. 手势数据快速标注

提供基于 PyQt6 的可视化打标工具，可在窗口内圈定 ROI 并通过按钮一键保存样本：

```bash
# 石头剪刀布三种手势（用于静态手势控制）
python -m aircursor.scripts.label_tool --labels open closed peace --category static
```

- 左键拖拽即可绘制或调整 ROI（红框）；点击「清除 ROI」恢复全帧。
- 点击任意手势按钮后，工具会截取当前帧（若设置 ROI 则裁剪）并保存到 `data/<category>/<label>/` 目录。
- **手势标签说明**：
  - `open`（布）：五指展开的手掌
  - `closed`（石头）：握拳
  - `peace`（剪刀）：食指和中指伸出
- 支持 `--mirror` 选项用于水平镜像自拍摄像头，以及 `--camera-index`、`--output-root` 自定义来源与存储路径。


### 5. 实时预览与调试

想快速验证启发式或训练后的模型效果，可运行：

```bash
# 静态手势预览（基于模式识别）
python -m aircursor.scripts.preview_static_gestures --mirror

# 静态手势预览 + HaGRID 全帧模型（ResNet18）
python -m aircursor.scripts.preview_static_gestures \
  --mirror \
  --hagrid-model-path weights/ResNet18.pth \
  --hagrid-arch resnet18

# 动态手势预览（ONNX 模型）
python -m aircursor.scripts.preview_dynamic_gestures \
  --mirror \
  --detector weights/hand_detector.onnx \
  --classifier weights/crops_classifier.onnx

# 动态手势预览（调试模式，显示检测框）
python -m aircursor.scripts.preview_dynamic_gestures \
  --mirror \
  --detector weights/hand_detector.onnx \
  --classifier weights/crops_classifier.onnx \
  --debug
```

- `preview_static_gestures` 窗口会绘制 MediaPipe 关键点，并同时显示：
  - **第一行**：MLP/Heuristic 模型的预测结果（您自己训练的模型或内置启发式规则）
  - **第二行**：HaGRID 原始标签（如 `palm`, `fist`）→ 映射后的标签（如 `open`, `closed`）
  - **底部**：完整的标签映射关系说明
- `preview_dynamic_gestures` 识别动态手势并映射到鼠标操作：
  - **TAP / CLICK** → 左键点击
  - **ZOOM（单手）** → 右键点击
  - **DRAG / DROP** → 拖拽操作
  - **SWIPE_UP / DOWN** → 垂直滚动
  - **SWIPE_LEFT / RIGHT** → 水平滚动
  - **手部移动** → 光标移动
  - 完整事件列表（24 种）见 [docs/EXTERNAL_PROJECTS.md](docs/EXTERNAL_PROJECTS.md)
- 两个预览界面现已统一设计风格（颜色、布局、字体），使用 PyQt6 实现
- **标签映射**：HaGRID 的 18 个手势类别会自动映射到 AirCursor 的 3 个基础手势（open/closed/peace），详见 [docs/EXTERNAL_PROJECTS.md](docs/EXTERNAL_PROJECTS.md)


### 6. 动态手势与鼠标操作映射

项目内置动态手势流水线（源自 [HaGRID Dynamic Gestures](https://github.com/ai-forever/dynamic_gestures)），可识别多种动态手势并映射到鼠标操作。

#### 手势到鼠标操作的映射

| 动态手势 | 鼠标操作 | 说明 |
|---------|---------|------|
| **TAP / CLICK** | 左键点击 | 单指向前，模拟鼠标左键 |
| **ZOOM（单手）** | 右键点击 | 单手捏合放大手势，模拟右键菜单 |
| **DRAG / DROP** | 拖拽操作 | 保持捏合并移动，松开后释放 |
| **SWIPE_UP / DOWN** | 垂直滚动 | 上下滑动手势，控制页面上下滚动 |
| **SWIPE_LEFT / RIGHT** | 水平滚动 | 左右滑动手势，控制页面左右滚动 |
| **手部移动** | 光标移动 | 手掌/食指移动，实时跟踪光标位置 |

#### 启用动态手势

默认关闭，若要启用，在 `config.yaml` 中设置：

```yaml
dynamic_hagrid:
  enabled: true
  detector_path: "weights/hand_detector.onnx"
  classifier_path: "weights/crops_classifier.onnx"
  debug: false
```

- 可直接使用仓库自带的 ONNX 模型，或替换为官方最新权重（推荐复制到 `weights/` 后更新路径）
- 启用后系统将实时识别动态手势，并映射为对应的鼠标操作
- 在控制台会打印识别到的事件以便调试
- 使用 `preview_dynamic_gestures` 可预览手势识别和映射关系

### 7. 调整配置

项目根目录的 `config.yaml` 提供了常用参数：

- `camera`：摄像头索引、分辨率
- `cursor`：光标平滑系数、死区（抑制抖动）
- `gesture`：静态/动态模型路径及判定阈值
- `gesture.hand_landmarker_path`：MediaPipe `.task` 模型路径（默认 `weights/hand_landmarker.task`）
- `scroll`：滚动速度倍率

> 💡 如果根目录没有 `config.yaml`，可以从 `config.yaml.example` 复制一份。

## 项目结构

```
AirCursor/
├── pyproject.toml
├── README.md
├── LICENSE
├── config.yaml              # 配置文件（从 config.yaml.example 复制，已加入 .gitignore）
├── config.yaml.example      # 配置文件模板
├── download_models.py       # 模型下载脚本
├── docs/                    # 文档
│   ├── README.md
│   └── EXTERNAL_PROJECTS.md # 外部项目说明（HaGRID、动态手势）
├── weights/                 # 模型权重文件
│   ├── hand_landmarker.task
│   ├── hand_detector.onnx
│   ├── crops_classifier.onnx
│   └── ResNet18.pth
├── models/                  # 自训练模型
│   └── static_mlp.pth
├── data/                    # 数据采集目录
│   └── static/
└── src/
    └── aircursor/
        ├── __init__.py
        ├── __main__.py          # 允许 python -m aircursor 启动
        ├── app.py               # 主程序逻辑
        ├── core/                # 核心组件
        │   ├── hand_tracker.py
        │   ├── gesture_fusion.py
        │   ├── mouse_engine.py
        │   └── hagrid_dynamic_adapter.py
        ├── models/              # PyTorch 模型定义
        │   ├── static_mlp.py
        │   ├── dynamic_lstm.py
        │   └── hagrid_fullframe.py
        ├── utils/               # 工具函数
        │   ├── landmark_preprocess.py
        │   ├── trajectory_recorder.py
        │   └── screen_utils.py
        ├── scripts/             # 数据采集与工具
        │   ├── collect_static.py
        │   ├── collect_dynamic.py
        │   ├── label_tool.py
        │   ├── preview_static_gestures.py
        │   ├── preview_dynamic_gestures.py
        │   ├── mouse_control.py
        │   ├── static_mouse_control.py
        │   ├── train_static.py
        │   ├── convert_static_images.py
        │   └── hagrid_import.py
        └── external/            # 整合的外部项目代码
            ├── dynamic_gestures/  # 动态手势识别（源自 ai-forever/dynamic_gestures）
            │   ├── controller.py
            │   ├── onnx_models.py
            │   ├── ocsort/          # OC-SORT 跟踪算法
            │   └── utils/
            └── hagrid/            # HaGRID 手势数据集工具（源自 ai-forever/hagrid）
                ├── constants.py
                ├── custom_utils/
                ├── dataset/
                └── models/
```

数据默认保存在 `data/` 目录，若不存在运行脚本时会自动创建。

## 自行训练模型

1. 使用上述采集脚本收集 CSV（静态）和 NPY（动态）数据；
2. 可借助 `python -m aircursor.scripts.label_tool` 快速扩充静态样本；
3. 可直接运行 `python -m aircursor.scripts.train_static --data-root data/static --output modelsweights/static_mlp.pth` 训练静态手势分类器；脚本会自动划分训练/验证集并保存权重。
4. 如已采集的是原始图像，可先运行 `python -m aircursor.scripts.convert_static_images --input-root data/static` 将其转换为 21×3 的关键点 CSV，再启动训练。
5. 训练完的模型可通过 `preview_static_gestures` 实时验证；若表现不佳，可继续采集数据或调优阈值。
6. 需要更大规模样本时，可借助 `hagrid_import` 脚本从 HaGRID v2 注释中筛选 `open/closed/peace` 等类别并自动生成训练特征；也可结合动态手势模块获取高级交互能力。
7. 依据 `src/aircursor/models/static_mlp.py` 与 `src/aircursor/models/dynamic_lstm.py` 结构自定义训练策略，或扩展动态手势训练脚本。
8. 将权重文件路径写入根目录 `config.yaml` 中的 `gesture.static_model_path` / `gesture.dynamic_model_path`，重启程序即可启用（相对路径相对于项目根目录解析）。

## 常见问题

### 使用问题
- **摄像头无法打开**：确认系统权限已允许终端或 IDE 访问摄像头。
- **光标抖动**：增大 `cursor.smooth_factor` 或 `cursor.dead_zone`。
- **滚动过快/过慢**：调整 `scroll.speed_scale`。
- **想切换手**：覆盖数据采集流程或自行训练左右手兼容模型。

### 预览功能问题
- **protobuf 兼容性错误**：确保已运行 `uv sync` 更新依赖，参考 [docs/CHANGELOG_FIX.md](docs/CHANGELOG_FIX.md)
- **HaGRID 模型加载失败**：现已支持训练检查点格式，详见文档
- **handedness 属性错误**：已修复 MediaPipe 0.10.21 API 兼容性问题

**快速验证**：运行 `./verify_preview.sh` 检查所有功能是否就绪

### 脚本使用
- 数据采集：`python -m aircursor.scripts.collect_static` / `collect_dynamic`
- 可视化标注：`python -m aircursor.scripts.label_tool --labels open closed peace --category static`
- 静态预览：`python -m aircursor.scripts.preview_static_gestures --mirror`
- 动态预览：`python -m aircursor.scripts.preview_dynamic_gestures --mirror --detector weights/hand_detector.onnx --classifier weights/crops_classifier.onnx`
- 鼠标控制：
  - 动态手势：`python -m aircursor.scripts.mouse_control --detector weights/hand_detector.onnx --classifier weights/crops_classifier.onnx --mirror`
  - 静态手势：`python -m aircursor.scripts.static_mouse_control --landmarker weights/hand_landmarker.task --mirror`

## 📚 文档说明

### 外部项目集成

本项目整合了两个优秀的开源手势识别项目：

1. **HaGRID**（静态手势）
   - 项目地址：https://github.com/hukenovs/hagrid
   - 用途：18 种手势分类，映射为石头✊、剪刀✌️、布✋
   - 模型：ResNet18/152
   
2. **Dynamic Gestures**（动态手势）
   - 项目地址：https://github.com/ai-forever/dynamic_gestures
   - 用途：24 种动态手势事件（TAP、ZOOM、SWIPE等）
   - 模型：ONNX 手部检测器 + 序列分类器

详见 [外部项目说明](docs/EXTERNAL_PROJECTS.md)

### 参数调优

**动态手势控制参数**:
- `--min-frames`: 手势确认最小帧数（8-20），越小越灵敏但可能误触
- `--click-freeze`: 点击时光标冻结时间（0.1-0.3秒），防止点击时手动导致的偏移
- `--scroll-speed`: 滚动速度（10-50），根据使用场景调整
- `--cursor-smooth`: 光标平滑系数（0.2-0.5），平衡速度和稳定性
- `--dead-zone`: 死区比例（0.08-0.25），根据屏幕大小调整

**静态手势控制参数**:
- `--cursor-smooth`: 光标平滑系数（0.2-0.5）
- `--dead-zone`: 死区比例（0.08-0.25）
- `--click-freeze`: 点击冻结时间（0.1-0.3秒）
- `--model`: 可选的自训练模型路径