# Roboharness 参赛材料草案（AD Hackathon 2026）

> **文档定位：报名操作指南。** 包含报名字段、CLI 命令、提交流程。正式飞书文档见 `feishu-final`，详细素材库见 `feishu-full`。

本文档用于把当前 `roboharness` 项目包装成一套可直接报名、补充材料、现场演示的参赛内容。

## 1. 建议报名策略

- 推荐赛道：`创意赛,应用落地赛`
- 推荐主标题：`Roboharness：面向 AI Coding Agent 的机器人仿真可视化验证框架`
- 备选标题 1：`让 AI 看见机器人：机器人任务的可视化验证与回归闭环`
- 备选标题 2：`Roboharness：给 Claude Code / Codex 用的机器人仿真 Harness`

推荐使用“双赛道”报名的原因：

1. 它有明确的技术创新点：不是传统测试框架，而是把 AI Coding Agent 直接接入机器人仿真的“观察—判断—迭代”闭环。
2. 它也有较强的落地属性：适合机器人算法、仿真平台、测试与研发效能团队做日常回归验证与问题定位。

## 2. 一句话版本

Roboharness 是一个面向 AI Coding Agent 的机器人仿真可视化验证框架：它把机器人任务拆成语义化 checkpoint，在关键阶段自动抓取多视角画面和状态数据，让 Claude Code、Codex 等 agent 不再只看日志，而是能真正“看见”机器人在做什么、判断任务是否成功，并基于结果持续迭代控制代码。

## 3. 报名字段草稿

以下内容可以直接用于 `agentic-hackathon project-create`。

### 3.1 标题

`Roboharness：面向 AI Coding Agent 的机器人仿真可视化验证框架`

### 3.2 描述

Roboharness 是一个面向机器人仿真场景的 agent-first 验证框架。它把抓取、行走、全身控制等任务拆成 `plan / pre_grasp / approach / grasp / lift` 这类语义化 checkpoint，在关键阶段自动暂停、抓取多视角图像、保存结构化状态，并生成 agent 可直接消费的 PNG、JSON 与 HTML 报告。

相比传统”只看日志”的调试方式，Roboharness 让 Claude Code、Codex 等 AI Coding Agent 能直接观察机器人行为、判断任务是否成功，并围绕 checkpoint 回放、结果对比和约束评估持续迭代控制代码。项目目前已在 MuJoCo 抓取、G1 全身控制、LeRobot 原生环境等场景完成验证，并具备继续扩展到更多仿真后端和内部验证流程的能力。

当前规模：34 个 Python 模块、~4,000 行源码、测试覆盖 90%+、4 条 CI 工作流、8 个示例、5 个公开在线 visual report。功能代码 100% 由 Claude Code / Codex 生成。

### 3.3 预期用户

机器人算法工程师、仿真平台研发工程师、测试工程师、机器人控制工程师，以及正在用 Claude Code / Codex / Cursor 等 AI Coding Agent 开发机器人能力的团队。

### 3.4 使用场景

1. 机器人抓取、行走、操作等任务的阶段性回归验证。
2. AI Agent 自动修改控制代码后，对行为结果进行可视化验收，而不是只看文本日志。
3. 仿真任务的 PR 验证、失败定位、checkpoint 回放和多视角对比。
4. 不同控制策略、不同模型、不同仿真后端之间的统一评估与报告生成。

### 3.5 标签

`机器人仿真,AI Agent,可视化验证,自动评测,MuJoCo,研发效能`

### 3.6 队员

当前建议按以下团队信息报名：

- 提交人：缪东旭 / `miaodongxu` / `miaodongxu@xiaomi.com`
- 队员：丁松 / `dingsong1` / `dingsong1@xiaomi.com`

说明：

- CLI 的 `--team-members` 参数使用用户名，不使用中文名或邮箱。
- 由于当前登录账号是 `miaodongxu`，项目提交人会自动是你本人。
- 因此 `--team-members` 更稳妥的写法是只填写额外队员：`dingsong1`。

### 3.7 痛点

当前机器人任务调试高度依赖人工盯画面、翻日志、重跑仿真，问题定位成本高、复现链路长、回归验证效率低。对 AI Coding Agent 来说，这个问题更严重：日志无法充分表达“机器人到底做对了没有”，agent 也很难从纯文本中判断抓取是否成功、姿态是否异常、轨迹是否偏移，导致自动迭代经常停留在“改代码—跑一下—看不懂—继续猜”的低效循环里。

### 3.8 行业参考

OpenAI Harness Engineering、Cursor Self-Driving Codebases、GitHub Actions 可视化回归、Rerun / Meshcat 机器人可视化调试工具链。

### 3.9 开发方式亮点

建议在报名材料或飞书文档里补上下面这段亮点总结：

- 功能代码 100% 由 Claude Code / Codex 生成，人类只负责 issue 定义、方向决策和最终验收
- Claude 负责项目规划、路线拆解与 steering
- Claude Code 与 Codex 在云端完成代码实现与大部分 PR
- GitHub 云端 CI 持续验证，GPU 测试路线采用 Cirun + AWS
- 当前已支持或明确覆盖多种机器人与平台场景：MuJoCo grasp、G1 WBC reach、G1 locomotion、Native LeRobot、Isaac Lab compatibility 路线
- 仓库已有 5 个可直接展示的 HTML demo / live reports：`grasp`、`g1-reach`、`g1-loco`、`g1-native`、`sonic`

## 4. 可直接执行的最终报名命令

下面是建议直接复制执行的最终版本：

```bash
agentic-hackathon project-create \
  --tracks 创意赛,应用落地赛 \
  --title "Roboharness：面向 AI Coding Agent 的机器人仿真可视化验证框架" \
  --description "Roboharness 是一个面向机器人仿真场景的 agent-first 验证框架。它把抓取、行走、全身控制等任务拆成语义化 checkpoint，在关键阶段自动暂停、抓取多视角图像、保存结构化状态，并生成 agent 可直接消费的 PNG、JSON 与 HTML 报告。相比传统只看日志的调试方式，它让 Claude Code、Codex 等 AI Coding Agent 能直接观察机器人行为、判断任务是否成功，并围绕 checkpoint 回放、结果对比和约束评估持续迭代控制代码。" \
  --expected-users "机器人算法工程师、仿真平台研发工程师、测试工程师、机器人控制工程师，以及正在使用 Claude Code / Codex / Cursor 等 AI Coding Agent 的团队" \
  --usage-scenarios "机器人抓取、行走、操作任务的阶段性回归验证；AI Agent 修改控制代码后的可视化验收；仿真任务的 PR 验证、失败定位、checkpoint 回放和多视角对比；不同控制策略和仿真后端的统一评估与报告生成" \
  --tags 机器人仿真,AI-Agent,可视化验证,自动评测,MuJoCo,研发效能 \
  --team-members dingsong1 \
  --pain-points "当前机器人任务调试高度依赖人工盯画面、翻日志、重跑仿真，问题定位成本高、复现链路长、回归验证效率低。对于 AI Coding Agent，纯日志无法充分表达机器人行为是否正确，导致自动迭代长期停留在低效试错阶段。" \
  --industry-reference "OpenAI Harness Engineering, Cursor Self-Driving Codebases, GitHub Actions visual regression, Rerun, Meshcat"
```

如果你希望在项目说明或飞书文档首页明确写出完整团队信息，建议使用下面的展示方式：

- 缪东旭（`miaodongxu` / `miaodongxu@xiaomi.com`）
- 丁松（`dingsong1` / `dingsong1@xiaomi.com`）

## 5. 作品资料建议

项目创建成功后，建议补充两类作品资料：

1. 飞书文档：用于讲清背景、方案、演示和落地计划。
2. Git 仓库：直接提交当前仓库地址。

对应命令模板：

```bash
agentic-hackathon submit <projectId> \
  --doc <你的飞书文档链接> \
  --git <当前仓库地址>
```

如果仓库需要只读访问 Token，再补：

```bash
agentic-hackathon submit <projectId> \
  --doc <你的飞书文档链接> \
  --git <当前仓库地址> \
  --git-access-token <readonly-token>
```

## 6. 推荐飞书文档结构

建议把飞书文档控制在 5 到 8 页以内，重点突出“痛点真实、方案可复现、演示可感知、落地可推进”。

### 6.1 背景与问题

- 机器人仿真调试强依赖视觉观察。
- 纯日志难以表达抓取、接触、姿态、步态等真实行为。
- AI Coding Agent 想闭环迭代，必须有“可看、可判、可回放”的验证环境。

### 6.2 方案概述

- 语义化任务协议：把任务拆成 `plan / pre_grasp / approach / grasp / lift` 等阶段。
- 多视角采集：在关键 checkpoint 记录图片与状态。
- checkpoint 回放：失败后从中间阶段继续，而不是每次重头跑。
- 报告与评估：生成 HTML 报告，并可进一步接约束评估与趋势分析。

### 6.3 系统架构

建议直接放仓库中的架构图：

- `assets/architecture.svg`

同时补一句解释：

> Harness 负责 pause → capture → resume，agent 负责写控制代码和根据观测继续迭代。

### 6.4 演示素材

建议优先使用仓库现成素材：

- `assets/X32_Y28_Z13_front_view.gif`
- `assets/X26_Y22_Z13_topdown_view.gif`
- `assets/example_mujoco_grasp/pre_grasp_front.png`
- `assets/example_mujoco_grasp/grasp_front.png`
- `assets/example_mujoco_grasp/lift_front.png`

如果要录制视频，优先演示 MuJoCo Grasp，因为最直观、最容易说明“为什么 agent 需要看画面”。

### 6.5 当前进展

- 已有核心库结构与对外 API。
- 已支持 checkpoint、capture、report、evaluate、trend 等能力。
- 已有 MuJoCo 抓取、G1 reach、LeRobot native 等示例。
- 已配置 CI、pytest、ruff、mypy 等基础工程质量保障。

### 6.6 下一步落地方向

1. 对接内部更多仿真环境与 benchmark。
2. 增加任务级约束和自动验收规则，把“看图判断”进一步沉淀成稳定评估标准。
3. 接入团队日常 PR 验证和回归流程，形成真正的 agent-in-the-loop 研发闭环。

## 7. 差异化说法

如果评委问“这和普通自动评测/仿真工具有什么区别”，建议这样回答：

### 7.1 相比通用 Agent 自动评测系统

我们不是在评测普通网页应用或通用代码任务，而是在评测机器人行为。机器人任务是否成功，很多时候无法从日志判断，必须看多视角画面、看阶段状态、看动作是否连续可解释。Roboharness 解决的是“让 agent 看见机器人”的问题。

### 7.2 相比传统仿真平台工具

我们不是只做仿真平台运维、文件统计或批量作业，而是面向 AI Coding Agent 的验证闭环。重点不是把仿真跑起来，而是让 agent 能在仿真结果上持续判断、对比、回放和改进。

### 7.3 相比截图工具

我们不是简单截图，而是把截图组织成语义化 checkpoint，并绑定状态、报告、恢复、评估流程，形成可以嵌入研发闭环的 harness。

## 8. 3 分钟演示脚本

### 8.1 开场（30 秒）

“机器人调试最痛的一点是：日志告诉不了你机器人到底有没有真的抓住、站稳、走对。对 AI Coding Agent 更是这样，光看日志它很难判断自己写的控制代码到底对不对。”

### 8.2 展示核心能力（60 秒）

- 展示 `README` 里的 MuJoCo 抓取示意图或 GIF。
- 说明 Roboharness 会在 `plan / pre_grasp / approach / grasp / lift` 等阶段自动抓图。
- 展示生成的 HTML 报告，说明每个 checkpoint 都能看图、看状态、看阶段结果。

### 8.3 展示为什么它适合 Agent（45 秒）

- 强调输出是 PNG + JSON，agent 可以直接消费。
- 强调 checkpoint restore，失败后不必从头重跑。
- 强调这不是人类专用调试工具，而是给 Claude Code / Codex 这种 agent 用的“工作环境”。

### 8.4 展示扩展性（30 秒）

- MuJoCo 只是当前最成熟的演示。
- 架构上通过 backend protocol 可以扩展到更多仿真后端。
- G1、LeRobot 场景说明它不是单点 demo，而是可以继续成长为统一验证层。

### 8.5 收尾（15 秒）

“我们做的不是另一个机器人 demo，而是在给 AI Coding Agent 修一条真正可用的机器人研发闭环。”

## 9. 评委常见问题回答

### 9.1 这个项目真正落地在哪里？

落地在机器人算法开发、仿真验证、回归测试和 agent 自动迭代这几个场景。尤其适合内部机器人团队把“写控制代码—跑仿真—看结果—改代码”的流程工程化。

### 9.2 为什么不是直接让 agent 看视频？

因为仅有视频不够稳定，难以结构化消费。我们提供的是语义阶段、多视角图片、状态 JSON、checkpoint 恢复和报告生成，这比单一视频更适合 agent 自动使用。

### 9.3 为什么你们不是做一个更通用的评测平台？

因为机器人是最需要可视化验证的场景之一。先把这个高价值、高复杂度场景打透，比一开始做泛化平台更有落地性。

### 9.4 当前最大的不足是什么？

目前最佳验证路径主要集中在 MuJoCo 与已有示例，更多内部仿真环境和任务约束还需要继续补齐。但这也正说明项目不是 PPT，而是已经有工作原型、正在进入可落地阶段。

## 10. 建议你现场强调的三个点

1. **不是日志工具，而是让 agent 看见机器人。**
2. **不是单点 demo，而是可复用的 harness 层。**
3. **不是纯创意，而是能接入真实研发闭环的工程基础设施。**

