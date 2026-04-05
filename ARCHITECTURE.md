# Roboharness Architecture

Roboharness 是一个面向 **AI Coding Agent** 的机器人仿真视觉测试框架。它的核心假设是：AI Agent 在编写机器人控制代码时，需要在关键时刻「暂停 → 观察 → 判断 → 迭代」—— 就像人类工程师盯着仿真画面调试一样。

## 设计理念

### 为什么需要 Roboharness？

传统的机器人仿真测试依赖数值断言（关节角度是否在范围内、末端执行器误差是否小于阈值）。但很多问题——错误的坐标变换、翻转的轴向、不自然的运动轨迹——数值上可能"通过"，视觉上一眼就能看出问题。

Roboharness 让 AI Agent 在仿真的关键节点自动截取多视角截图，结合数值状态一起判断，形成 **"视觉 + 数值"双通道验证**。

### 三个核心原则

1. **Protocol-Driven（协议驱动）**：所有外部依赖（仿真器、控制器、可视化器）通过 Protocol（结构化类型）接入，不需要继承基类。
2. **Checkpoint-Oriented（检查点导向）**：仿真不是连续跑完的，而是在语义关键点暂停、采集、检查。
3. **Agent-First（Agent 优先）**：API 设计围绕 AI Agent 的工作流——加载任务协议、执行动作序列、获取视觉反馈、决定下一步。

## 整体架构

```
                    ┌──────────────────────────────┐
                    │         AI Coding Agent       │
                    │   (编写/修改控制代码的 LLM)     │
                    └──────────┬───────────────────┘
                               │ command
                               ▼
                    ┌──────────────────────┐
                    │     Controller       │  高级命令 → 低级动作
                    │     (Protocol)       │  e.g. 目标位姿 → 关节角
                    └──────────┬───────────┘
                               │ action
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Harness                                 │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ TaskProtocol │  │ Checkpoint[] │  │ CheckpointStore        │  │
│  │ (语义阶段)   │→ │ (采集点列表)  │  │ (状态快照 save/restore)│  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
│                                                                  │
│  step() ──→ run_to_next_checkpoint() ──→ capture() ──→ save()   │
│                                                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  SimulatorBackend   │  仿真器适配层
              │     (Protocol)      │  step / get_state / capture_camera / ...
              └─────────────────────┘
              Implementations:
                • MuJoCoMeshcatBackend
                • (Isaac Lab, ManiSkill, ...)
```

## 模块详解

### `core/` — 框架核心

| 文件 | 职责 |
|------|------|
| `harness.py` | **Harness** 类 + **SimulatorBackend** Protocol。Harness 管理仿真循环、检查点调度、多视角采集 |
| `protocol.py` | **TaskProtocol** + **TaskPhase**。语义任务协议，定义任务的自然阶段（见下文详述） |
| `checkpoint.py` | **Checkpoint** 数据类 + **CheckpointStore** 状态快照管理 |
| `capture.py` | **CameraView**（单相机帧）+ **CaptureResult**（一个检查点的完整采集结果） |
| `controller.py` | **Controller** Protocol。高级命令到低级动作的转换接口 |
| `lifecycle.py` | **ComponentLifecycle** 元数据系统。标记组件的存在假设和过期时间，用于定期"harness 瘦身"评审 |
| `rerun_logger.py` | Rerun 可视化日志集成（可选） |

### `core/protocol.py` — 语义任务协议

这是我们最近引入的核心概念。传统方式是按仿真步数采集（"每 100 步截一张图"），这样做简单但丢失了语义信息。TaskProtocol 让采集点对应任务的自然阶段：

```python
# 一个抓取任务的语义协议
GRASP_PROTOCOL = TaskProtocol(
    name="grasp",
    phases=[
        TaskPhase("plan",      "规划抓取轨迹，可视化目标路径"),
        TaskPhase("pre_grasp", "移动到预抓取位姿"),
        TaskPhase("approach",  "沿规划路径接近物体"),
        TaskPhase("grasp",     "闭合夹爪抓取物体"),
        TaskPhase("lift",      "提起物体"),
        TaskPhase("place",     "放置物体到目标位置"),
        TaskPhase("home",      "回到初始位姿"),
    ],
)

# 使用时一行加载
harness.load_protocol(GRASP_PROTOCOL)
# 或只选需要的阶段
harness.load_protocol(GRASP_PROTOCOL, phases=["pre_grasp", "grasp", "lift"])
```

**内置四种协议：**

| 协议 | 适用场景 | 阶段 |
|------|---------|------|
| `GRASP_PROTOCOL` | 抓取/放置 | plan → pre_grasp → approach → grasp → lift → place → home |
| `LOCOMOTION_PROTOCOL` | 行走 | initial → accelerate → steady → decelerate → terminal |
| `LOCO_MANIPULATION_PROTOCOL` | 移动抓取 | navigate → pre_grasp → grasp → transport → place → retreat |
| `DANCE_PROTOCOL` | 舞蹈/节律动作 | ready → sequence → finale |

**自定义协议非常简单：**

```python
my_protocol = TaskProtocol(
    name="assembly",
    phases=[
        TaskPhase("pick", "拿起零件", cameras=["front", "wrist"]),
        TaskPhase("align", "对准装配位", cameras=["top", "wrist"]),
        TaskPhase("insert", "插入到位", cameras=["front", "side"]),
    ],
)
```

`BUILTIN_PROTOCOLS` 字典提供所有内置协议的注册表，方便发现和遍历。

### `backends/` — 仿真器适配层

**SimulatorBackend** 是一个 `@runtime_checkable` Protocol，定义了 7 个方法：

```python
class SimulatorBackend(Protocol):
    def step(self, action) -> dict[str, Any]: ...       # 推进一步
    def get_state(self) -> dict[str, Any]: ...           # 读取当前状态
    def save_state(self) -> dict[str, Any]: ...          # 保存完整状态（用于回滚）
    def restore_state(self, state) -> None: ...          # 恢复到某个状态
    def capture_camera(self, camera_name) -> CameraView: # 截取相机画面
    def get_sim_time(self) -> float: ...                 # 仿真时间
    def reset(self) -> dict[str, Any]: ...               # 重置
```

新仿真器只需实现这 7 个方法即可接入，**不需要继承任何基类**。当前实现：

- **MuJoCoMeshcatBackend** — MuJoCo 仿真 + Meshcat 3D 可视化导出
- **MeshcatVisualizer** — 独立的 Meshcat 交互式场景导出器

### `evaluate/` — 评估引擎

自动化的约束检查和评估系统：

```
report.json ──→ MetricAssertion[] ──→ AssertionEngine ──→ EvaluationResult
                                                              │
                                                              ├── Verdict: PASS / DEGRADED / FAIL
                                                              └── AssertionResult[]
```

- **MetricAssertion** — 单条约束（`grip_error < 5.0mm`, `lift_height > 0.02m`）
- **AssertionEngine** — 对一份 report 运行所有约束，输出 Verdict
- **Severity** — CRITICAL（一条失败 → FAIL）、MAJOR（失败 → DEGRADED）、MINOR、INFO
- **Operator** — lt, le, eq, gt, ge, in_range
- **Constraints** — 约束可以从 JSON/YAML 文件加载，实现配置与代码分离

**批量评估** (`evaluate/batch.py`)：跨多个 trial 的聚合分析——成功率、失败阶段分布、变体对比。

### `runner.py` — 并行试验执行

```python
runner = ParallelTrialRunner(
    backend_factory=lambda: MyBackend(),   # 每个 trial 独立的仿真器实例
    store=my_store,                         # 输出存储
    max_workers=4,                          # 并发数
)
batch = runner.run(specs, trial_fn=my_trial)
print(batch.success_rate)
```

- **TrialSpec** — 单次试验的规格（variant_name, trial_id, metadata）
- **ParallelTrialRunner** — 基于 ThreadPoolExecutor 的并发执行器，每个 trial 获得独立的 backend 和输出目录
- **BatchResult** — 聚合结果：成功率、per-variant 统计、failure_phase_distribution

### `storage/` — 存储系统

分层的文件组织：

```
harness_output/
└── pick_and_place/                    # 任务名
    ├── task_config.json
    ├── grasp_position_001/            # 变体（如不同抓取位置）
    │   ├── position.json
    │   ├── trial_001/                 # 第 1 次尝试
    │   │   ├── pre_grasp/             # 检查点
    │   │   │   ├── front_rgb.png      # 多视角截图
    │   │   │   ├── side_rgb.png
    │   │   │   ├── state.json         # 仿真状态
    │   │   │   └── metadata.json
    │   │   ├── grasp/
    │   │   ├── lift/
    │   │   └── result.json            # 试验结果
    │   └── summary.json               # 变体汇总
    └── report.json                    # 全局报告
```

- **TaskStore** — 通用的 task → variant → trial → checkpoint 存储
- **GraspTaskStore** — 抓取任务特化的存储，预定义检查点 `["plan_start", "pre_grasp", "contact", "lift"]`
- **EvaluationHistory** — 追加式 JSONL 日志，记录每次评估的成功率、指标。支持趋势检测（regression/improvement/stable）

### `wrappers/` — Gymnasium 集成

**RobotHarnessWrapper** 让任何 Gymnasium 环境零修改接入 Roboharness：

```python
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RobotHarnessWrapper(env,
    checkpoints=[{"name": "early", "step": 10}, {"name": "late", "step": 100}],
    output_dir="./output",
)

obs, info = env.reset()
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(action)
    if "checkpoint" in info:
        print(f"Checkpoint: {info['checkpoint']['name']}")
```

自动检测环境的多相机能力（`render_camera()` 方法、Isaac Lab TiledCamera、或 fallback 到 `env.render()`）。

### `robots/` — 机器人特化代码

目前支持 **Unitree G1** 人形机器人：

- **GrootLocomotionController** — 基于 ONNX 推理的行走控制器（15 自由度下肢）
- **HolosomaLocomotionController** — 29 自由度全身控制器
- **SonicLocomotionController** — 支持多模式（walk/dance/track）的高级控制器，10Hz 重规划 + 插值

### `reporting.py` — HTML 报告生成

自动生成自包含的 HTML 报告，包含每个检查点的多视角截图、状态元数据、可选的 Meshcat 3D 交互场景嵌入。

### `cli.py` — 命令行工具

```bash
roboharness inspect ./harness_output    # 检查输出目录内容
roboharness report ./harness_output     # 生成 HTML 报告 + JSON 摘要
roboharness evaluate report.json        # 运行约束评估
roboharness evaluate-batch ./output/    # 批量评估
roboharness trend ./output/             # 趋势检测（回归/改善）
```

### `core/lifecycle.py` — 组件生命周期

一个独特的元数据系统：每个框架组件可以标注"它为什么存在"的假设和预计过期时间。随着 AI 模型能力提升，某些辅助组件可能不再需要。

```python
ComponentLifecycle(
    name="intermediate_checkpoints",
    horizon=ExpirationHorizon.LONG_TERM,
    assumptions=[
        ComponentAssumption(
            description="模型无法从最终结果反推中间过程的问题",
            removal_condition="模型能从最终截图准确诊断中间步骤的错误",
        ),
    ],
)
```

## 数据流

一个典型的 Agent 工作流：

```python
from roboharness import Harness, GRASP_PROTOCOL

# 1. 创建后端和 Harness
backend = MuJoCoMeshcatBackend(xml_string=model_xml, cameras=["front", "side", "top"])
harness = Harness(backend, output_dir="./output", task_name="pick_cube")

# 2. 加载语义协议（自动注册检查点）
harness.load_protocol(GRASP_PROTOCOL, phases=["pre_grasp", "grasp", "lift"])

# 3. 重置仿真
harness.reset()

# 4. 逐阶段执行
for phase_name, actions in my_action_sequences.items():
    result = harness.run_to_next_checkpoint(actions)
    # result.views — 多视角截图
    # result.state — 仿真状态（关节角、接触力等）
    # result.sim_time — 仿真时间

    # Agent 检查截图，决定是否需要调整
    if not looks_good(result):
        harness.restore_checkpoint("pre_grasp")  # 回滚到之前的检查点
        # 重新尝试...
```

## 依赖关系

**核心（零额外依赖）：** numpy

**可选依赖组：**

| 组 | 用途 | 关键包 |
|---|------|-------|
| `[mujoco]` | MuJoCo 仿真 | mujoco ≥ 3.0 |
| `[meshcat]` | 3D 交互可视化 | meshcat ≥ 0.3 |
| `[rerun]` | Rerun 时序可视化 | rerun-sdk ≥ 0.18 |
| `[wbc]` | 全身控制（IK） | pinocchio, pink, qpsolvers |
| `[lerobot]` | LeRobot 策略推理 | onnxruntime, huggingface_hub |
| `[dev]` | 开发工具 | pytest, ruff, mypy |

## 扩展指南

### 添加新的仿真器后端

实现 `SimulatorBackend` 的 7 个方法即可，不需要继承：

```python
class MySimBackend:
    def step(self, action): ...
    def get_state(self): ...
    def save_state(self): ...
    def restore_state(self, state): ...
    def capture_camera(self, camera_name): ...
    def get_sim_time(self): ...
    def reset(self): ...
```

### 添加新的任务协议

```python
MY_PROTOCOL = TaskProtocol(
    name="my_task",
    description="描述这个任务类型",
    phases=[
        TaskPhase("phase_1", "第一阶段做什么", cameras=["front"]),
        TaskPhase("phase_2", "第二阶段做什么", cameras=["front", "top"]),
    ],
)
```

### 添加新的控制器

实现 `Controller` Protocol 的 `compute()` 方法：

```python
class MyController:
    def compute(self, command: dict, state: dict) -> Any:
        # command → action 的转换逻辑
        return joint_positions
```
