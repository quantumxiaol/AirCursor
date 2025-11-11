# 📚 AirCursor 文档

AirCursor 项目的文档说明。

---

## 📖 文档列表

### [EXTERNAL_PROJECTS.md](EXTERNAL_PROJECTS.md)

**外部项目集成说明** - 核心文档

包含以下内容：

#### 1. 外部项目介绍
- **HaGRID** - 手势识别数据集和模型
  - 项目地址和论文链接
  - 18 种原始手势类别
  - ResNet 模型介绍
  - 识别原理

- **Dynamic Gestures** - 动态手势识别系统
  - 项目地址
  - 24 种动态手势事件
  - ONNX 模型和 OC-SORT 跟踪
  - 识别原理和关键参数

#### 2. 手势映射表

**HaGRID 原始手势 → AirCursor 基础手势**
- 18 种 HaGRID 手势 → 3 种基础手势（石头、剪刀、布）
- 映射规则和原理

**动态手势 → 鼠标操作**（推荐）
- TAP → 左键点击
- ZOOM → 右键点击
- DRAG/DROP → 拖拽
- SWIPE → 滚动
- 手部移动 → 光标移动

**静态手势 → 鼠标操作**（备选）
- 拳头移动 → 光标移动
- 石头变布 → 左键点击
- 石头变剪刀 → 右键点击
- 布/剪刀移动 → 拖拽
- 时序稳定机制（5帧投票）

#### 3. 技术栈
- 手部检测（MediaPipe / ONNX）
- 手势分类（ResNet / ONNX）
- 鼠标控制（PyAutoGUI）

---

## 🔗 快速导航

- **主 README**: [../README.md](../README.md)
- **外部项目说明**: [EXTERNAL_PROJECTS.md](EXTERNAL_PROJECTS.md)
- **HaGRID GitHub**: https://github.com/hukenovs/hagrid
- **Dynamic Gestures GitHub**: https://github.com/ai-forever/dynamic_gestures

---


