# Roboharness：面向 AI Coding Agent 的机器人仿真可视化验证框架

> **文档定位：正式飞书文档。** 可直接转飞书发布。报名操作见 `submission`，详细素材库见 `feishu-full`。

让 AI agent 真正”看见机器人”，而不只是”看见日志”。

## 团队信息

- 缪东旭 / `miaodongxu` / `miaodongxu@xiaomi.com`
- 丁松 / `dingsong1` / `dingsong1@xiaomi.com`

## 项目概述

Roboharness 是一个为机器人仿真场景设计的 **agent-first visual harness**。

它不是只记录日志，也不是只做截图，而是把抓取、行走、全身控制等机器人任务拆成语义化 checkpoint，在关键阶段自动采集多视角图像、状态 JSON、HTML 报告和可恢复快照，让 Claude Code、Codex 等 AI Coding Agent 能基于真实可视结果完成”写代码 → 跑仿真 → 判断成败 → 修改代码 → 再验证”的闭环。

### 项目一览

| 指标 | 数据 |
|------|------|
| 源码规模 | 34 个 Python 模块，~4,000 行有效代码（`src/`） |
| 测试规模 | ~3,500 行测试代码（`tests/`），覆盖率阈值 90% |
| CI 工作流 | 4 条（ci / docker / pages / release），支持 Python 3.10–3.13 |
| 示例场景 | 8 个可运行 example（MuJoCo grasp、G1 reach、LeRobot native 等） |
| 提交统计 | 50 次提交，其中 Claude 完成 48 次（96%） |
| Issue 驱动 | 13+ 个 issue 驱动开发，覆盖从 MuJoCo grasp 到 SONIC locomotion |
| 在线 Demo | 5 个公开 HTML visual report（GitHub Pages 自动部署） |

我们希望解决的问题很明确：在机器人研发里，很多任务的成败根本无法仅靠日志判断。抓取是否成功、姿态是否稳定、轨迹是否偏移、接触是否合理，都必须靠看画面、看阶段状态、看行为结果。对 AI Coding Agent 来说也是一样——如果只给它日志，它其实看不见机器人行为本身，自动迭代就容易退化成”改一版、跑一下、继续猜”。

因此，Roboharness 的核心不是替 agent 写控制算法，而是给 agent 一个真正可用的工作环境：

- 在哪里停下来观察
- 该看哪些视角
- 如何保存状态
- 如何从中间阶段恢复
- 如何把结果沉淀为报告和评估依据

## 核心方案

Roboharness 做的是一层 `harness / validation layer`，主要能力包括：

1. **语义化任务分阶段**
   - 例如抓取任务会被拆成：`plan → pre_grasp → approach → grasp → lift → place → home`

2. **关键阶段自动抓取多视角结果**
   - 保存 RGB 图像
   - 保存状态 JSON
   - 保存元数据
   - 保存可恢复仿真快照

3. **生成 agent 可消费的结果包**
   - PNG + JSON + HTML 报告
   - 人可以直接看，agent 也可以继续处理

4. **支持恢复与继续迭代**
   - 失败后从 checkpoint 继续调试
   - 不必每次从头重跑完整任务

5. **支持后续评估与趋势分析**
   - report
   - evaluate
   - trend

一句话概括就是：

> Roboharness 负责 `pause → capture → restore → report → evaluate`，控制逻辑由你自己或 AI agent 编写。

## 为什么它不是普通 demo

很多项目可以做出一个机器人演示，但很难沉淀成一套可以长期复用的基础设施。Roboharness 的独特价值在于，它不仅能展示机器人行为，还能把这个过程沉淀成：

- 可复用 API
- 可扩展 backend
- 可 checkpoint 恢复
- 可自动报告
- 可接入 agent 工作流
- 可进入 CI / PR 验证流程

它不是在做”单次秀肌肉”，而是在做机器人 Agent 开发的公共地基。

从创意性来看，它把 harness engineering 从网页/软件工程迁移到了机器人仿真场景；从落地性来看，它直接服务于机器人算法开发、仿真回归验证、PR 验证和失败定位，非常适合作为团队研发效能基础设施。

### 与现有工具的对比

| 维度 | Rerun | Meshcat | pytest / CI | **Roboharness** |
|------|-------|---------|-------------|-----------------|
| Agent 可消费的结构化输出 | ✗ | ✗ | 部分（文本） | ✓ PNG + JSON + HTML |
| 语义化 checkpoint 分阶段 | ✗ | ✗ | ✗ | ✓ plan → grasp → lift … |
| Checkpoint 恢复 & 重试 | ✗ | ✗ | ✗ | ✓ 从中间阶段继续 |
| 多仿真后端统一接口 | ✗ | ✗ | N/A | ✓ MuJoCo / Gym / LeRobot |
| 约束评估 & 趋势分析 | ✗ | ✗ | 需自建 | ✓ YAML 规则 + CLI |
| 面向 AI Agent 设计 | ✗ | ✗ | ✗ | ✓ agent-first |

Rerun 和 Meshcat 是优秀的可视化工具，但它们的定位是「给人看」；pytest/CI 擅长自动化但不理解机器人行为语义。Roboharness 把两者连起来：在仿真关键阶段自动采集、以 agent 可消费的格式输出、并支持约束评估和 checkpoint 恢复。

## 支持不同机器人、平台与任务

Roboharness 不是只服务于单一 demo，而是在形成一层可复用的机器人验证基础设施。当前已经覆盖或明确支持的对象包括：

### 不同机器人形态

- 2 指夹爪抓取任务
- Unitree G1 humanoid
- G1 native LeRobot 场景

### 不同仿真 / 接入平台

- MuJoCo
- LeRobot 原生 `make_env()`
- Gymnasium wrapper
- Isaac Lab compatibility 路线

### 不同可视化 / 验证后端

- Meshcat 3D
- HTML report
- PNG + JSON 结构化输出
- Rerun logging 路线

### 不同任务类型

- grasp
- whole-body reach
- locomotion
- motion tracking

这说明 Roboharness 不是只对单个机器人写死的脚本，而是一层正在成型的统一验证层。

## 现有 Demo 与 HTML Reports

当前仓库已经有一整套公开可访问的 HTML demo / visual reports：

- 总入口：https://miaodx.com/roboharness/
- MuJoCo Grasp：https://miaodx.com/roboharness/grasp/
- G1 WBC Reach：https://miaodx.com/roboharness/g1-reach/
- G1 Locomotion：https://miaodx.com/roboharness/g1-loco/
- G1 Native LeRobot：https://miaodx.com/roboharness/g1-native/
- SONIC Motion Tracking：https://miaodx.com/roboharness/sonic/

这些页面不是静态摆拍，而是由 GitHub Pages 工作流自动构建：每次推送到 `main`，CI 会跑示例、生成 HTML report、整理 checkpoint 资源，并把结果部署成可直接浏览的 demo 站点。

这件事对评委的感知价值非常高，因为它把“项目能跑”变成了“项目能被直接看到、直接点开、直接验证”。

推荐展示素材包括：

- `assets/architecture.svg`
- `assets/X32_Y28_Z13_front_view.gif`
- `assets/X26_Y22_Z13_topdown_view.gif`
- `assets/example_mujoco_grasp/pre_grasp_front.png`
- `assets/example_mujoco_grasp/grasp_front.png`
- `assets/example_mujoco_grasp/lift_front.png`

## 真实案例：Agent 闭环迭代是怎么发生的

这不是假设场景——以下是仓库里真实发生的一次 agent 闭环迭代：

**背景：** MuJoCo 抓取示例在 CI 中运行，checkpoint 截图看起来一切正常（夹爪确实夹住了方块）。但当我们给 CI 加上 `--assert-success` 物理约束验证后，**断言立刻失败了**：cube z 高度 ≈ 0，说明方块根本没被检测为”抬起”。

**根因：** Agent 生成的代码假设 MuJoCo `qpos` 数组中 slide joint 排在前面，因此用 `qpos[5]` 读取 cube z 坐标。但 MuJoCo 实际上把 free joint（cube 的 6-DOF）排在前面，正确索引是 `qpos[2]`。

**闭环过程：**

| 阶段 | Before | After |
|------|--------|-------|
| CI 门控 | 只检查”example 有没有崩溃” | `--assert-success`：检查 cube z > 桌面 + 5mm、夹爪接触存在、qvel 稳定 |
| qpos 索引 | `qpos[5]`（错误） | `qpos[2]`（正确） |
| 失败时产物 | 只在成功时上传 | `if: always()` 确保失败时也能看到 checkpoint 截图 |
| 约束定义 | 硬编码在 Python 中 | 外部化到 `constraints/grasp_default.yaml`，CLI 可评估 |

**这说明什么：** 视觉截图说”看起来 OK”，但结构化约束验证说”物理上不对”。两者结合才是完整的 agent 反馈闭环。这正是 Roboharness 要解决的问题。

> 相关提交：`102a593` — `fix: correct qpos index for cube z-position in grasp assertion`

## 开发方式：这本身就是一次 Agentic Development 实验

这个项目最值得强调的一点是：

> **功能代码 100% 由 Claude Code / Codex 生成，人类只负责 issue 定义、方向决策和最终验收。**

我们对这个仓库采用了一个极端约束：

- 人类不直接手写功能代码（项目配置、issue 描述、验收标准由人类编写）
- 人类负责定义目标、拆 issue、给出验收口径、做最终取舍
- 代码实现交给 agent 在云端完成

这使它不是”做了一个 AI 相关项目”，而是”用 AI agent 真正把项目做出来”。

### Claude 负责项目规划与 steering

Claude 在这套开发方式里承担的是高层规划与 steering 的角色：

- 帮助定义项目方向和阶段目标
- 帮助把想法整理成 issue / ticket
- 帮助进行方案比较、路线收敛和任务拆解
- 帮助在开发过程中持续 steer：当前应该先做什么、如何裁剪范围、哪里该补文档、哪里该补验证

一句话概括：

> Claude 不是写几段代码的助手，而是在承担项目规划和技术 steering 的工作。

### Claude Code 与 Codex 在云端执行编码

实际编码执行层面，主要由 Claude Code 与 Codex 在云端完成：

- 新功能实现
- 文档生成与重写
- CI 修复
- 示例补充
- 代码重构
- PR 产出与后续修正

这意味着整个项目不是本地 IDE 的人肉 coding，而是真正的 **cloud agent workflow**。

## Issues-driven 开发与云端验证体系

这个仓库的开发闭环非常清晰：

1. 人类提出问题与方向
2. Claude 帮助规划、拆解、steer
3. 形成 GitHub Issues / tickets
4. Claude Code 与 Codex 云端接手实现
5. PR 进入仓库
6. GitHub 云端 CI 验证
7. HTML demo / report 输出 proof-of-work
8. 再回到下一个 issue

### GitHub 云端 CI

当前项目已经把验证工作尽量放到云端：

- lint
- type check
- pytest
- 多 Python 版本验证
- MuJoCo 示例验证
- GitHub Pages 部署

### GPU CI 路线：Cirun + AWS

对更重型的仿真 / GPU 评测场景，项目已经有明确的 GPU CI 路线设计：

- 使用 Cirun 做云端 GPU runner 编排
- 使用 AWS 作为 GPU 资源承载平台之一
- 面向未来更复杂的机器人 benchmark 与验证流程

一句话概括就是：

> 开发在云端 agent 上完成，验证在云端 CI 上完成，结果通过 HTML demo 被直接发布出来。

## GitHub 路线演进

这个仓库不是一次性写完，而是在持续 issue-driven 演进。比较适合引用的代表 issue 包括：

- `#83` Native LeRobot env creation support
- `#86` SONIC locomotion Phase 1
- `#92` SONIC locomotion Phase 2
- `#81` Unitree / real-robot communication 路线
- `#34` G1 WBC reach 与多视角能力
- `#4` Isaac Lab compatibility
- `#2` MuJoCo grasp 端到端示例

这些 issue 说明项目不是凭灵感一把梭的 hack，也不是只为了 demo 的一次性产物，而是有路线、有阶段演进、有长期方向、有持续沉淀的工程项目。

## 为什么它有真实落地意义

Roboharness 对应的是机器人研发里一条非常真实的链路：

- 控制器开发
- 仿真调试
- 回归验证
- 行为问题定位
- 失败样本复盘
- PR 验证与持续集成

它的目标用户也很明确：

- 机器人算法工程师
- 仿真平台工程师
- 机器人测试工程师
- 使用 Claude Code / Codex / Cursor 的 AI 编程团队

如果落到公司内部，它可以成为：

- 机器人仿真任务的验证层
- agent 自动迭代的观测层
- PR 的行为回归层
- 失败案例的复盘层
- demo / benchmark / evaluation 的统一结果层

它未必是一个面向所有人的大众产品，但对目标团队来说，它是一层高价值基础设施。

## 当前局限与下一步

我们主动说明项目当前的不足，因为我们认为这也是它真实和可信的一部分：

1. **Agent 端到端自动闭环的验证还有限。** 当前的闭环案例（如 qpos 修复）主要依赖 CI 约束门控触发，agent 主动"看图 → 判断 → 改代码"的全自动链路还需要更多场景打磨。
2. **更多仿真后端的适配还在进行中。** 目前最成熟的是 MuJoCo，Isaac Lab 和更多内部仿真环境的 backend 还在路线上。
3. **约束评估（evaluate）目前还比较初级。** 当前主要是基于阈值的 pass/fail 判定，更复杂的行为质量评分、跨试次趋势分析还需要继续迭代。
4. **单人 + Agent 的开发模式有天花板。** 如果要真正进入团队日常工作流，还需要更多人参与场景适配和需求反馈。

这些不足恰恰说明项目不是 PPT 概念，而是一个已经在跑、正在面对真实工程问题的系统。

## 评委最应该记住的 4 句话

1. **它不是另一个机器人 demo，而是一层机器人研发验证基础设施。**
2. **它让 AI agent 不再只看日志，而是真正看见机器人在做什么。**
3. **我们不是在替 agent 写控制代码，而是在给 agent 建一个可观察、可判断、可回放、可验证的工作环境。**
4. **这个仓库本身就是一次 agentic development 实验：GitHub Issues 驱动方向，Claude 负责规划与 steering，Claude Code / Codex 云端完成绝大多数实现，最终由 GitHub CI、GPU CI 路线和 HTML demo 共同验收。**

## 3 分钟答辩稿

### `[0:00–0:30]` 问题

大家好，我们这次带来的项目叫 **Roboharness**。

机器人任务是否成功，很多时候根本不能只看日志。抓取有没有抓住、步态稳不稳、轨迹偏没偏，这些都必须看画面。对 Claude Code、Codex 这样的 AI Coding Agent 来说也是一样——如果只给它日志，它其实看不见机器人行为，自动迭代很容易退化成”改一版、跑一下、继续猜”。

### `[0:30–1:15]` 方案

Roboharness 做的事情，就是把机器人任务拆成语义化 checkpoint，在关键阶段自动抓取多视角图像、保存状态 JSON、生成 HTML report，并支持 checkpoint 恢复和约束评估。这样 agent 就可以基于真实视觉结果完成”写代码 → 跑仿真 → 判断成败 → 修改代码 → 再验证”的闭环。

*（此处展示 MuJoCo 抓取 GIF 或 HTML report）*

### `[1:15–1:45]` 真实案例

举一个真实的例子：我们的 agent 写的抓取代码，checkpoint 截图看起来没问题，但加上物理约束验证后发现 qpos 索引写错了——方块根本没被”检测到”抬起来。视觉说”OK”，约束说”不对”，两者结合才是完整的反馈。这就是 Roboharness 要做的闭环。

### `[1:45–2:15]` 证明它是真的

这个项目不是 PPT：34 个 Python 模块、4000 行源码、测试覆盖 90%+、4 条 CI 工作流、8 个可运行示例、5 个公开的在线 visual report。已经覆盖 MuJoCo grasp、G1 reach、LeRobot native、SONIC 等多个方向。

### `[2:15–2:45]` 开发方式

更特别的是，这个仓库本身就是一次 agentic development 实验。功能代码 100% 由 Claude Code / Codex 生成，人类只负责 issue 定义和最终验收。50 次提交中 48 次由 Claude 完成。开发在云端 agent 上完成，验证在云端 CI 上完成，结果通过 HTML demo 直接发布。

### `[2:45–3:00]` 收尾

所以 Roboharness 的意义不只是做一个机器人 demo，而是在给机器人研发团队提供一层真正 agent-first 的验证基础设施。谢谢。

## 结语

Roboharness 的核心价值，不是替代工程师，也不是替代控制算法，而是为 AI Coding Agent 提供机器人研发所缺失的那层“可见、可判、可回放、可验证”的工作环境。

它既是一个机器人验证框架，也是一个由 **GitHub Issues 驱动方向、Claude 负责规划与 steering、Claude Code / Codex 云端完成绝大多数实现、GitHub CI / GPU CI / HTML demo 联合验收** 的真实 agentic engineering 样板。
