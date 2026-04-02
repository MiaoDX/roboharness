# Robot-Harness: Context Document

> **文档版本**: v0.1-draft | **日期**: 2026-04-02
> **用途**: 本文件是 robot-harness 项目的完整上下文文档（Context Document），供 Claude Code、Codex 等 AI Agent 在进行代码 Review、架构设计和功能开发时作为背景参考。

## 第一部分：项目概述与动机

### 1.1 什么是 Robot-Harness

Robot-Harness 是一个 **为 AI Coding Agent 设计的机器人仿真视觉测试框架**。它的核心目标是让 Claude Code、OpenAI Codex 等编程 Agent 能够：

1. **控制仿真步进**（step-by-step execution）— 在关键时刻暂停仿真
2. **采集多视角截图** — 在同一仿真时刻从不同相机位置获取 RGB/深度图像
3. **自主判断任务结果** — Agent 直接观察截图，判断运动是否合理、抓取是否成功
4. **迭代优化算法** — 基于视觉判断结果，Agent 自主修改控制代码并重跑

**与传统方案的根本区别**：我们不需要单独的 VLM 模型来做视觉评估。Claude Code 和 Codex 本身就是多模态 Agent，它们既能写代码、又能看图片、又能做决策。Robot-Harness 的职责是 **把仿真的视觉信息以 Agent 能直接消费的格式高效地呈现出来**。

### 1.2 核心使用场景

以抓取任务为例，完整的 Agent-in-the-loop 流程是：

1. Agent 编写/修改抓取控制代码
2. Robot-Harness 运行仿真，在预定义的检查点（plan 开始、plan 结束、接触点、抬升完成）自动暂停
3. 在每个检查点，Harness 从多个视角截图并保存为文件
4. Agent 查看截图 + 结构化状态数据，判断当前阶段是否正常
5. 如果发现问题，Agent 修改代码并从适当的检查点重跑
6. 循环迭代直到任务成功

## 第二部分：当前实践（已验证的工作流）

### 2.1 三工具协同架构

当前实践基于三个可视化工具的分工协作：

#### MuJoCo — 纯物理仿真引擎

- 仅负责物理模拟（碰撞、接触力、重力）
- 手部（末端执行器）在 MuJoCo 中不做运动学计算，保持仿真简单
- 通过 `mj_step()` 控制仿真步进
- 状态通过 `mj_getState`/`mj_setState` 进行快照和恢复

#### Meshcat — 轨迹可视化与交互调试

- 浏览器端 3D 可视化，基于 three.js/WebGL
- 负责展示手部运动轨迹、抓取过程的完整动画
- Agent（Codex）已实现自主控制 Meshcat 的步进：
  - 在抓取各阶段自动暂停（plan 开始、plan 结束、抓取过程中的固定位置）
  - 暂停时切换视角进行多角度截图
  - 基于截图自主判断抓取效果并决定下一步
- 添加物体和轨迹比较便捷（相比 MuJoCo 原生渲染）

#### Rerun — 多模态数据验证

- 支持深度数据的可视化
- 与机器人 URDF 良好配合，适合手眼标定验证
- 时间线回放功能支持任意时刻的数据检查
- 多模态数据（RGB、深度、点云、关节状态）的同步展示

### 2.2 已验证的 Agent 闭环流程

Codex 在当前项目中已经实现了以下闭环：

```
Codex 编写控制代码
    ↓
运行仿真 → Meshcat 渲染
    ↓
Codex 控制 Meshcat step → 在关键位置暂停
    ↓
Codex 切换视角 → 截图（多视角）
    ↓
Codex 分析截图 → 判断抓取效果
    ↓
如果效果不好 → Codex 修改代码 → 重跑
如果效果好 → 完成
```

这个流程已经跑通并产生了实际效果，但目前是针对某一个具体项目的定制实现。

### 2.3 当前实践的局限性

- **项目耦合**：视觉测试逻辑嵌入在具体项目代码中，无法复用
- **后端固定**：硬编码了 MuJoCo + Meshcat + Rerun 的组合，无法扩展到 Isaac Lab 等其他仿真器
- **接口非标准**：Agent 与 Harness 的交互没有统一的 API 合约
- **数据格式临时**：截图和状态数据的存储没有规范化的格式

## 第三部分：社区调研 — 现有工具和实践

### 3.1 仿真器原生的录制与多视角渲染能力

#### MuJoCo

- `mujoco.Renderer`：通过 `update_scene(data, camera="cam_name")` 从任意命名相机渲染 RGB/深度/分割图到 numpy 数组
- `mujoco.viewer.launch_passive`：支持暂停（SPACE）和单步（RIGHT），但无内建视频录制
- MuJoCo USD Exporter：导出完整轨迹到 USD 格式，可在 Blender/Omniverse 中做多相机光追渲染
- MJX-Warp（v3.3.5+）：GPU 加速批量渲染
- `mujoco-python-viewer`：第三方查看器，支持 `read_pixels(camid=N)` 离屏渲染
- `mujoco_tools`：支持多相机视角、4K 视频录制、轨迹渲染、CLI 驱动截取 — 最接近单一 MuJoCo Visual Harness 的工具
- MuJoCo Playground：参考 arXiv:2502.08844

#### Robosuite

- 多视角基础设施最完整：`render_camera` 参数接受多相机名称列表，同时渲染 RGB/深度/分割
- 内建 `demo_video_recording.py`、`CameraMover` 类、`DemoPlaybackCameraMover`
- MuJoCo 状态存储为 `.npz` 文件，支持确定性轨迹回放 — "stop the world" 的天然基础
- 渲染后端可插拔：MuJoCo 原生（~60fps）、NVISII 光追（~0.5fps）、iGibson PBR（~1500fps）

#### ManiSkill v3

- `RecordEpisode` wrapper：同时录制视频和 HDF5 轨迹数据，GPU 并行环境下达 30,000+ FPS（RTX 4090）
- `CameraConfig` 对象定义独立相机位姿和分辨率

#### NVIDIA Isaac Lab

- `TiledCamera`：将数千个相机批量拼接到单次 GPU 渲染
- Synthetic Data Recorder GUI + Camera Placement 优化工具
- Renderer ≠ Visualizer 分离（3.0）：传感器数据生成与交互调试解耦
- ~57,000 FPS 离屏渲染（RTX 3090）

#### PyBullet

- `getCameraImage()` 支持任意视角编程式截图
- `startStateLogging(STATE_LOGGING_VIDEO_MP4)` 录制 GUI 视口
- `pybullet-blender-recorder`：录制 link 位姿为 pickle，导入 Blender 做高质量离线渲染

#### Gymnasium

- `RecordVideo` wrapper：从任何 `render_mode="rgb_array"` 环境录制 MP4
- 单相机，无多视角或步级截取能力

### 3.2 可视化后端：Meshcat 与 Rerun

#### Meshcat

- 轻量级浏览器 3D 查看器（three.js + WebGL），无需 GPU，支持 Jupyter
- 动画模块：关键帧录制
- Drake 集成最完善：`StartRecording()`/`StopRecording()`/`PublishRecording()` 捕获所有变换
- `StaticHtml()` 导出独立 HTML 文件
- `RoboMeshCat`：上下文管理器风格的视频录制
- Pinocchio `MeshcatVisualizer`：提供 `captureImage(w, h)` 编程式截图
- **局限**：无原生多视角截取、无深度/分割渲染、无时间同步多模态日志

#### Rerun.io

- GitHub 6,200+ stars，Rust 实现，Python/C++/Rust SDK
- 时间感知 Entity Component System：图像、点云、3D 变换、网格、时间序列、深度图、分割掩码、视频帧 — 全部跨多时间线同步
- 完整时间回放：拖拽、步进、多时间线
- 可编程 Blueprint（`rerun.blueprint`）：定义标准化诊断布局
- 同时录制 + 可视化：多 sink 支持（实时 viewer + `.rrd` 文件）
- Dataframe API：从录制中程序化提取标量指标 — 可用于自动化 pass/fail 断言
- URDF 支持（v0.24+）：通过日志变换动态驱动关节动画
- Isaac Lab 3.0 官方集成 as Visualizer 后端
- MCAP/ROS 2 支持
- LeRobot 集成：HuggingFace LeRobot 使用 Rerun 作为主要可视化后端
- `.rrd` 文件可归档、共享、在 web viewer 中重新打开

### 3.3 VLM/LLM 评估机器人行为的研究现状

> **注意**：我们的方案中不需要单独的 VLM 做判断，Claude Code/Codex 自身即为评估者。但这些研究的评估方法论（如何提问、如何构造视觉输入、评估维度）仍有参考价值。

- **SuccessVQA**（Du et al., 2023）：将机器人成功检测建模为视觉问答任务
- **AHA**（2024-2025）：微调 LLaVA 检测操作失败并解释原因，超越 GPT-4 10.3%
- **GVL**（Generative Value Learning）：打乱视频帧让 VLM 重排，产生逐帧进度评分 — 零样本、无需微调
- **VLM-RMs**（ICLR 2024）：CLIP 余弦相似度作为零样本奖励信号
- **RL-VLM-F**（ICML 2024）：VLM 对比图像对学习奖励函数
- **RoboCLIP**（NeurIPS 2023）：视频-语言模型计算轨迹相似度
- **StepEval**（2025）：子目标分解 + VLM 逐阶段评估 — 最接近 robot-harness 需要的评估粒度
- **Robo2VLM**（2025）：从 176K 真实轨迹生成 684,710 个 VQA 问题

### 3.4 AI Agent 驱动的仿真迭代框架

#### Eureka（NVIDIA, ICLR 2024）

- GPT-4 生成奖励函数代码 → GPU RL 训练 → 文本统计反馈 → 迭代改进
- 29 个任务中 83% 超越人类专家奖励，平均提升 52%
- 不使用视觉反馈，纯文本统计驱动
- DrEureka 扩展至 sim-to-real

#### AOR（Act-Observe-Rewrite, 2025年3月）

- 最接近我们实践的学术工作
- 多模态 LLM（Claude）接收关键帧 RGB 图像 + 结构化诊断信号
- 每次迭代输出完整 Python 控制器类代码，动态编译加载
- robosuite Lift/PickPlaceCan 达到 100% 成功率，无梯度更新、无演示、无奖励工程
- 参考 arXiv:2603.04466
