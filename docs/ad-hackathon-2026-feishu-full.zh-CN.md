# Roboharness：面向 AI Coding Agent 的机器人仿真可视化验证框架

> 本文档为详细参考版（草稿素材库）。正式飞书文档请以 `ad-hackathon-2026-feishu-final.zh-CN.md` 为准。

> 让 Claude Code、Codex 等 AI Coding Agent **看见机器人在做什么**，**判断任务有没有真的成功**，并在真实的仿真反馈里持续迭代。

---

## 1. 痛点与判断

在机器人研发里，最痛苦的不是"代码写不出来"，而是：

- 机器人任务的成败**无法从日志中直接判断**——抓取是否成功、姿态是否稳定、接触是否合理，必须靠看图、看阶段状态。
- 对 AI Coding Agent 来说更严重：只给日志，agent **看不见机器人行为本身**，自动迭代退化成"改一版、跑一下、继续猜"。

> **What the agent can't see doesn't exist.**

因此，Roboharness 的核心不是替 agent 写控制代码，而是给 agent 一个"能看、能判、能回放、能验证"的工作环境。

---

## 2. 方案

Roboharness 是一层 **harness / validation layer**，核心能力：

1. **语义化任务分阶段** — `plan → pre_grasp → approach → grasp → lift → place → home`
2. **关键阶段自动采集** — RGB 图像 + 状态 JSON + 元数据 + 可恢复仿真快照
3. **Agent 可消费的结果包** — PNG + JSON + HTML 报告
4. **Checkpoint 恢复** — 失败后从中间阶段继续，不必重头跑
5. **约束评估与趋势分析** — YAML 规则 + CLI，支持 pass/degraded/fail 判定

> Roboharness 负责 `pause → capture → restore → report → evaluate`，控制逻辑由你自己或 AI agent 编写。

### 与现有工具的对比

| 维度 | Rerun | Meshcat | pytest / CI | **Roboharness** |
|------|-------|---------|-------------|-----------------|
| Agent 可消费的结构化输出 | ✗ | ✗ | 部分（文本） | ✓ PNG + JSON + HTML |
| 语义化 checkpoint 分阶段 | ✗ | ✗ | ✗ | ✓ |
| Checkpoint 恢复 & 重试 | ✗ | ✗ | ✗ | ✓ |
| 多仿真后端统一接口 | ✗ | ✗ | N/A | ✓ MuJoCo / Gym / LeRobot |
| 约束评估 & 趋势分析 | ✗ | ✗ | 需自建 | ✓ YAML 规则 + CLI |
| 面向 AI Agent 设计 | ✗ | ✗ | ✗ | ✓ agent-first |

---

## 3. 项目一览（量化数据）

| 指标 | 数据 |
|------|------|
| 源码规模 | 34 个 Python 模块，~4,000 行有效代码（`src/`） |
| 测试规模 | ~3,500 行测试代码（`tests/`），覆盖率阈值 90% |
| CI 工作流 | 4 条（ci / docker / pages / release），支持 Python 3.10–3.13 |
| 示例场景 | 8 个可运行 example |
| 提交统计 | 50 次提交，Claude 完成 48 次（96%） |
| Issue 驱动 | 13+ 个 issue 驱动开发 |
| 在线 Demo | 5 个公开 HTML visual report（GitHub Pages 自动部署） |

---

## 4. 当前能力

### 覆盖范围

- **机器人形态**：2 指夹爪、Unitree G1 humanoid、G1 native LeRobot
- **仿真平台**：MuJoCo、LeRobot `make_env()`、Gymnasium wrapper、Isaac Lab compatibility 路线
- **可视化后端**：Meshcat 3D、HTML report、PNG + JSON、Rerun logging 路线
- **任务类型**：grasp、whole-body reach、locomotion、motion tracking

### 在线 Demo

- 总入口：`https://miaodx.com/roboharness/`
- MuJoCo Grasp / G1 WBC Reach / G1 Locomotion / G1 Native LeRobot / SONIC Motion Tracking

由 GitHub Pages 工作流自动构建：每次推送到 `main`，CI 跑示例、生成报告、部署。

---

## 5. 真实案例：Agent 闭环迭代

以下是仓库里真实发生的一次闭环：

**背景：** MuJoCo 抓取 CI 中，checkpoint 截图显示夹爪抓住了方块。加上 `--assert-success` 物理约束后断言失败：cube z ≈ 0。

**根因：** Agent 用 `qpos[5]` 读 cube z，但 MuJoCo 实际上 free joint 排在前面，正确索引是 `qpos[2]`。

| | Before | After |
|---|--------|-------|
| CI 门控 | 只检查崩溃 | 物理约束（cube z、接触、qvel） |
| qpos 索引 | `qpos[5]`（错） | `qpos[2]`（对） |
| 失败时产物 | 不上传 | `if: always()` 确保可用 |

> 视觉说"OK"，约束说"不对"。两者结合才是完整闭环。

---

## 6. 开发方式：Agentic Development 实验

> **功能代码 100% 由 Claude Code / Codex 生成，人类只负责 issue 定义、方向决策和最终验收。**

### 角色分工

| 角色 | 职责 |
|------|------|
| 人类 | 定义目标、拆 issue、验收标准、最终取舍 |
| Claude | 项目规划、方案比较、路线收敛、持续 steering |
| Claude Code / Codex | 云端编码：功能实现、文档、CI、示例、重构 |
| GitHub CI | lint、pytest、多版本测试、MuJoCo 验证、Pages 部署 |
| Cirun + AWS | GPU CI 路线（重型仿真/评测） |

### 开发闭环

人类提出方向 → Claude 规划拆解 → Issue → Claude Code/Codex 云端实现 → PR → CI 验证 → HTML demo 输出 → 下一个 issue

### 代表性 Issue

- `#2` MuJoCo grasp 端到端示例
- `#4` Isaac Lab compatibility
- `#34` G1 WBC reach
- `#83` Native LeRobot env support
- `#86` SONIC locomotion Phase 1
- `#92` SONIC locomotion Phase 2

---

## 7. 落地价值

面向的真实研发链路：控制器开发 → 仿真调试 → 回归验证 → 行为定位 → 失败复盘 → PR 验证

目标用户：机器人算法工程师、仿真平台工程师、测试工程师、AI Coding Agent 团队

公司内部可以成为：仿真验证层 / agent 观测层 / PR 回归层 / 失败复盘层 / 统一评估层

---

## 8. 当前局限

1. **Agent 端到端全自动闭环还有限** — 目前主要依赖 CI 约束门控触发
2. **更多仿真后端适配在进行中** — MuJoCo 最成熟，Isaac Lab 等在路线上
3. **约束评估还比较初级** — 主要是阈值 pass/fail，复杂行为评分待迭代
4. **单人 + Agent 模式有天花板** — 需要更多人参与场景适配

---

## 9. 创意性 vs 落地性

**创意性**：把 harness engineering 从网页/软件工程迁移到机器人仿真；直接把多模态 coding agent 变成机器人任务的观察者。

**落地性**：直接服务于机器人算法开发、仿真回归测试、PR 验证、失败定位。

**趋势**：随着 AI Coding Agent 越来越强，"给 agent 提供好的工作环境"比"写更长的 prompt"更重要。在软件工程里这叫 harness engineering，在机器人仿真里才刚刚开始。

---

## 10. 金句备用

- **标题句**：Roboharness：面向 AI Coding Agent 的机器人仿真可视化验证框架
- **核心价值句**：它让 AI agent 不再只看日志，而是真正看见机器人在做什么。
- **方法论句**：我们不是在替 agent 写控制代码，而是在给 agent 建一个可观察、可判断、可回放、可验证的工作环境。
- **开发方式句**：这个仓库本身就是一次 agentic development 实验。
- **落地句**：它不是另一个机器人 demo，而是一层可以进入真实研发流程的验证基础设施。

---

## 11. 团队信息

- 缪东旭 / `miaodongxu` / `miaodongxu@xiaomi.com`
- 丁松 / `dingsong1` / `dingsong1@xiaomi.com`

推荐赛道：创意赛、应用落地赛
