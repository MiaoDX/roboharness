# Roboharness 15-20 min 分享 · 分镜稿

> **定版口径**：18-20 min, 19 页，目标时长 20 min，可压缩到 18 min
> **主线**：让 AI agent 能在机器人开发里自己判断自己的工作，从而真正自主跑长程任务
> **副线**：trust-boundary 工程同时用在产品层（for agents）和开发层（by agents）

---

## 📌 全局占位资产清单

| 编号 | 内容 | 状态 | 用在 |
|---|---|---|---|
| 图 #1 | 两个月前那张旧 vision 图（"AI 完全接手测试"） | **待补** | 页 3 |
| 图 #2 | 现在的三阶段工作流图（Plan / Execute / Review） | **待补** | 页 6 |
| 图 #3 | 拇指朝下抓 bottle 的对比图（正常抓 vs 拇指朝下） | **待补** | 页 9 |
| Demo 链接 | https://miaodx.com/roboharness/grasp/ | 已就绪 | 页 12 |
| Demo 截图 1 | 顶部 banner: "PASS — 10/10 constraints satisfied" + "No surfaced cases" | **待补**（备用） | 页 12 |
| Demo 截图 2 | Phase Timeline 一格（如 grasp/contact 阶段）含三视角图 | **待补**（备用） | 页 12 |
| Demo 截图 3 | Constraint Evaluation 10 个 metric 全 PASS 的表格 | **待补**（备用） | 页 12 |
| QR 码 | 公众号二维码 | **待补** | 页 19 |

---

# 页 1 · 标题页 [0:00-0:30]

**屏幕呈现**

```
Roboharness
  ——给 AI Agent 装上"自己判断自己工作"的能力

副标题：这个项目是给 AI Agent 用的，也是用 AI Agent 做出来的
```

底部：你的名字 / 团队 / 日期

**讲稿要点**

- "今天分享的项目叫 roboharness。一句话定位——给 AI Agent 装上自己判断自己工作的能力，这是我们的目标。"
- 双线提一下："这个项目是给 AI Agent 用的，也是用 AI Agent 做出来的。这两件事背后是同一个工程思路，待会儿会看到。"

**备注** 标题页节奏快，不超过 30 秒。

---

# 页 2 · 一句话定义 [0:30-2:00]

**屏幕呈现**

```
机器人开发里，AI Agent 能不能长时间自主跑
关键不是模型多聪明
是它能不能 ——
       自己判断「这一步成了没」
```

可以在右侧放一个小示意：agent 在调代码 → 跑仿真 → 一个问号（无法判断）。

**讲稿要点**

- 整个 talk 围绕这个判断展开
- "agent 能写代码、能调参、能跑仿真——这些早就不是瓶颈。卡住的是它不知道自己做的事到底成没成。"
- "今天我会用一个我们刚做完的真实案例证明：只要补上这层判断能力，agent 能完成相当复杂的、原本需要人盯几天的工程任务。"

---

# 页 3 · 两个月前那张图 [2:00-3:00]

**屏幕呈现**

```
[图 #1：两个月前内部分享的最后一张图，"AI 完全接手测试自动化"愿景图]

[ 占位区域 ——
  含义：当时这是一个 vision，今天回来汇报：做到了
]
```

**讲稿要点**

- "两个月前我在内部分享过一次，讲的就是我们的机器人抓取项目。最后一张图（指着）——上面写的是：**下一步，让 AI 完全接手这个测试自动化**。"
- "那张图当时是个愿景。今天来汇报：这件事我们做到了。"
- "怎么做到的，是接下来 5 分钟要讲的。"

**备注** 这一页就是钩子。讲快，往下走。

---

# 页 4 · 当时为什么做不到 [3:00-4:30]

**屏幕呈现**

```
当时的卡点

  Unitree G1 + GR00T Decoupled WBC
  下肢 RL + 上肢 IK 的混合架构 · 2025-11

  AI Agent 能改代码、调参 ✓
                          ──── 这早就不是瓶颈

  卡住的是 判断：
    · 夹爪夹住没？
    · 方块滑了没？
    · 姿态合理吗？
  → 日志看不出来，必须看画面
```

**讲稿要点**

- "项目背景：宇树 G1 机器人 + GR00T Decoupled WBC——就是 2025 年 11 月那版，下肢用 RL、上肢用 IK 的混合架构，做抓取。"
- "AI agent 改控制代码、调参数没问题——这些不是瓶颈。"
- "**真正卡住的是判断**：夹爪有没有夹住？物体有没有滑？姿态合理吗？这些事光看日志看不出来，必须看画面。"

---

# 页 5 · 关键洞察 [4:30-5:00]

**屏幕呈现**

```
这不是算力问题
是 注意力问题

  工程师只能盯着：
  agent 跑一轮 → 人看一眼 → 再跑一轮 → 再看一眼

  瓶颈不在 GPU
  在工程师的注意力
```

**讲稿要点**

- "工程师只能盯着——agent 跑一轮人看一眼，再跑一轮再看一眼。"
- "这不是算力问题，**是注意力问题**。"
- "当时我们已经看清：agent 要真自主，唯一的路是让它自己有判断依据。"
- 过渡："过去两个月我们做的就是这件事。下面看一个真实案例。"

---

# 页 6 · 兑现：SONIC 升级 [5:00-7:00]

**屏幕呈现**

```
[图 #2：现在的三阶段工作流图]

  Plan          →    Execute        →   Review
  (人主导)           (Agent 完全接管)    (人回来)
                     工程师不在场

  详尽计划       harness 每改一步自跑    先看 metric
  每步成/不成    metric+视觉证据自评     metric OK 再看视觉
  判据明确       Codex /goal 锚定长程    最后人工 E2E

──────────────────────────────────────────────────
真实例子：今年农历年 (2026-02-19) NVIDIA 发布 GEAR-SONIC

   Decoupled WBC            →   GEAR-SONIC
   (RL + IK 混合)                (单一 Transformer 基础模型)

   "把控制栈底层从模块化拼接 → 端到端基础模型"

   整条控制栈范式级重写 → agent 自主连续跑数小时完成
```

**讲稿要点**

- "过去 2 个月我们补的是 **metric + Visual Harness 双轨证据**：每个关键 phase 自动采 metric + 多视角截图，agent 自己读、自己判断。"
- "给一个真实例子。今年农历年期间（2026 年 2 月 19 日），NVIDIA 发了 GEAR-SONIC——这不是版本升级，是范式切换。**我们等于把控制栈底层从一个模块化拼接的系统，换成了一个端到端的基础模型**。对我们而言相当于整条控制栈要重写一遍。"
- 三阶段讲清楚（指着三阶段图）：
  - "**Plan 阶段** 人主导，跟 agent 一起做非常详尽的计划，每一步有明确的成/不成判据。"
  - "**Execute 阶段** 完全交给 coding agent，**工程师不在场**。每改一步，harness 自动跑一遍 grasp pipeline，agent 自己读 metric + 视觉证据，自己判断有没有性能回退。配 Codex CLI 的 `/goal` 命令——experimental feature，要在 codex-cli 0.128+ 里 `/experimental` 打开——把'完成 WBC→SONIC 迁移且抓取性能不回退'作为跨 turn 持久目标。"
  - "**Review 阶段** 人回来收尾。先看 metric，metric 没问题再人工看 harness 的视觉输出，最后人工做一遍全流程端到端测试。"
- "结果：架构级升级的执行阶段 agent 连续自主跑了几个小时，工程师**全程不在场**。"

**备注**
- `/goal` 实测核过：codex-cli 0.128.0+，目前 experimental，命令带斜杠
- Q&A 准备：如果有人问"为什么不是直接 N1.7"——是因为 GEAR-SONIC 2 月发布、N1.7 4 月才商业 release，时间线对得上

---

# 页 7 · 校准关键一句 [7:00-7:30]

**屏幕呈现**

```
        ❶ Agent 没有突然变聪明
        ❷ 给了它判断自己工作好坏的工具

让 agent 跑得久 ——
   不靠模型变神
   靠"越过边界时会被挡住、边界内时能自己往前走"
```

**讲稿要点**

- "我想特别强调这一句话——**这不是 agent 突然变聪明，是给了它判断自己工作好坏的工具**。"
- "让 agent 跑得久，靠的不是模型变神，靠'agent 越过边界时会被挡住、agent 在边界内时能自己往前走'。"
- 过渡："这件事在内部抓取项目跑通之后，我们发现——任何需要 agent 长时间自主跑机器人代码的场景都缺这一层。所以抽出来做了独立 repo——这就是 roboharness。下面看具体长什么样。"

---

# 页 8 · 案例 A · qpos 索引 [7:30-8:30]

**屏幕呈现**

```
案例 A · qpos 索引（仿真，commit 102a593）

   截图 ✓ 看起来抓住了
   Agent 自己 ✓ 判 PASS

   ──加上 --assert-success 物理约束 metric──

   metric ✗ cube z ≈ 0
                根本没抬起来

   根因：agent 假设 slide joint 在 qpos 前，用了 qpos[5]
        MuJoCo 实际 free joint 在前，正确是 qpos[2]

   教训方向：光看图不够，metric 是兜底
```

**讲稿要点**

- "0307 那版我们只让 agent 看截图自己说成没成——但我们实际遇到的失败，**两个方向都有**。先看第一个。"
- "截图看起来抓取成功，agent 自己也判 PASS。后来在 CI 里加了物理约束 metric，发现 cube z ≈ 0、根本没抬起来。"
- "根因：agent 写代码时假设 MuJoCo qpos 里 slide joint 在前，用了 `qpos[5]`；MuJoCo 实际 free joint 在前，正确是 `qpos[2]`。"
- "教训：**光看图不够，metric 是兜底**。"

**备注** 这页 60 秒以内讲完，主要为下一页（B 案例）做铺垫。

---

# 页 9 · 案例 B · 拇指朝下抓 bottle [8:30-10:00]

**屏幕呈现**

```
案例 B · 拇指朝下抓 bottle（真机 Unitree G1）

[图 #3：左侧正常抓 vs 右侧拇指朝下抓的对比照]

   初始时唯一的 metric：抓取中心点 vs bottle 中心点 3D 距离

   Agent 规划出的解：整个手反着抓，拇指朝下而不是朝上
                    → metric ✓ 完全 PASS
                    → 3D 中心点距离确实在阈值内

   ──但 visual harness 一打开图，谁都看得出来不对──

   第三幕：我们去问"为什么 agent 会这么做"
          发现是 metric 太弱
          加了新 metric：手部 / 拇指朝向
          → 这种失败模式永久关掉

   教训方向：光看 metric 不够
            visual 才能挡住"数学上对、物理上荒谬"的解
```

**讲稿要点**

- "第二个案例是真机上发生的。一开始我们只有一个 metric——抓取中心点 vs bottle 中心点的 3D 距离差。"
- "某次优化过程中，agent 规划出一个非常奇怪的解：**整个手是反着抓的，拇指朝下而不是朝上**。"（指图 #3）
- "metric 完全 PASS——3D 中心点距离确实在阈值内。但 visual harness 一打开图，谁都看得出来不对。"
- "第三幕，**我们做的事**：去问'为什么 agent 会这么做'——发现 metric 太弱。加了一个新 metric——手部朝向 / 拇指朝向必须在某个范围内。**这种失败模式以后 metric 自动挡住，不需要 visual 再去抓了**。"
- "教训：**光看 metric 不够，visual 才能挡住数学上对、物理上荒谬的解**。"
- 一句方法论："其实这是 reward hacking 的经典模式——agent 不只会犯错，更危险的是它会找出一种弱 metric 下数学 PASS 但语义荒谬的解。"

**备注**
- 案例稍长，留 90 秒给这页
- 图 #3 是这页的灵魂，必须有

---

# 页 10 · metric ↔ visual 的循环分工 [10:00-10:30]

**屏幕呈现**

```
metric 和 visual 不是冗余的两类规则
是 闭环里两个不同角色

  visual harness   →  发现新的失败模式（unknown unknowns）
                       ↓
                      人/agent 分析
                       ↓
  新 metric        ←  把这种失败永久转成可量化的护栏

每发现一个新坑，就从 visual 那侧搬到 metric 那侧
这就是 engineer the harness 字面意义在做的事
```

**讲稿要点**

- "这两个案例放一起说明的事：**metric 和 visual 是 harness 的两个不同角色，不是冗余的两类规则**。"
- 指流程图："visual 是**发现器**——抓那些我们预先没想到的失败模式。metric 是**稳态护栏**——精确、快、确定性。两者分工是动态的、迭代的——visual 抓到新坑，团队就把它编码成 metric 关掉。"
- "这就是 Mitchell Hashimoto 讲的 **engineer the harness** 字面意义在做的事。"

---

# 页 11 · 自主率分阶段说清楚 [10:30-12:00]

**屏幕呈现**

```
"我用你这套，得投入多少 / 之后能多自主？"

  Harness 调试期 ────────  一次性成本
                          · 写契约
                          · 跑 seeded good/bad/ambiguous corpus
                          · 调阈值

  Harness 稳态期 ────────  95%+ Agent 自己 PASS/FAIL
                          只有真正新颖/复杂的场景
                          才需要人最后兜底
                          ↑ SONIC 升级就属于这一类
                            所以有 Review 阶段

  ⇒ 长程任务（数小时）能跑下来
    是因为大部分 per-step 判断根本不需要人介入
```

**讲稿要点**

- "讲到这里，可能有人想问——'我也想用这套，得投入多少？之后能多自主？' 这两件事要分开讲。"
- "**调试期**是一次性成本：写契约、跑 seeded good/bad/ambiguous corpus、调阈值。"
- "**稳态期**：已经调过的场景上，**95%+ 任务 agent 自己 PASS/FAIL**。只有真正新颖、复杂的场景才需要人最后兜底——SONIC 升级就属于后者，所以才有那个 Review 阶段。"
- "这也是为什么 SONIC 那种几小时长程任务能跑——**大部分 per-step 判断根本不需要人介入**。"

---

# 页 12 · 现场 Demo [12:00-13:30]

**屏幕呈现**

```
现场切到浏览器，打开：

  https://miaodx.com/roboharness/grasp/

[备用截图 1：页面顶部 "PASS — 10/10 constraints satisfied" banner + "No surfaced cases"]
[备用截图 2：Phase Timeline 一格，含 front/side/top 三视角图]
[备用截图 3：Constraint Evaluation 10 个 metric 全 PASS 表格]
```

**讲稿要点（按这个顺序指）**

1. **顶部 "Run Decision" banner**（10 秒）：
   - "看这里——`PASS — 10/10 constraints satisfied`。这是 Run Decision banner，agent 跑完一轮，给出的整体判决。"

2. **"No surfaced cases" 区**（20 秒）：
   - "**这是核心**——'No surfaced cases. No material changes surfaced. Old baseline remains authoritative.'"
   - "翻译一下：这一轮跑完，没有任何需要人看的事情。**clean 的时候工程师根本不用打开这一页**，agent 自己继续推进下一步。"
   - "刚才说的'数小时无人值守'就是这么来的——大部分 case 都是 clean，clean 的时候人不介入。"

3. **Surfaced / Suppressed / Unchanged 计数**（10 秒）：
   - "右边这几个数字——Surfaced 0, Suppressed 1, Unchanged 1. 这是 escape hatch 的设计：只有 Surfaced > 0 才需要人看。"

4. **Hard Metric Results / Constraint Evaluation 表**（20 秒）：
   - 下拉到 Constraint Evaluation 表
   - "这就是 metric_gate 的具体形态——10 条物理约束，每条都有 Expected / Actual / Severity。`cube_height_mm > 5.0` actual `143.4`——cube 真的抬起来了。这就是 qpos 案例的护栏现在长什么样。"

5. **Phase Timeline + 截图**（20 秒）：
   - 下拉到 Phase Timeline
   - "每个 phase 都有 front/side/top 三视角截图，metric 数值（grip err / contacts），有没有 alarms。这就是 visual harness 的数据形态。"

6. **回到 Run Decision**（10 秒）：
   - "回到顶部——所有这些数据合起来，得出顶部那个 PASS 判决。**agent 自己读这些数据、自己做出这个判决**。clean 的时候人不打开。"

**备注**
- 网络/投屏不稳定时切到备用截图。备用截图按 1/2/3 顺序换页
- 这页是全场最有冲击力的，留够 90 秒

---

# 页 13 · 抽出来之后又长了什么 [13:30-15:00]

**屏幕呈现**

```
抽成 repo 之后（自 0429 至今 3 周）

  v0.3.0 / 0.3.1 已发到 PyPI

  ❶ Agent Visual Review v1（5-20 上线）
     同一个 coding agent 在 bounded manifest 边界内做视觉评审
     不另起 VLM 服务
     每个 visual dimension 必须声明 metric_fallback
       或 why_not_metricized，否则 fail-closed
     → 把"agent 看图"也做成有契约的、不能滥用的机制

  ❷ Python contract → SKILL.md 自动产出（5-20 上线）
     contract.py 是手写真值
     roboharness contract generate
       → SKILL.md / schemas / stubs
     → Agent 自己加载契约约束自己

  共同方向：判断边界从 run-time 扩到 design-time
            Agent 不是变聪明，是约束它的契约写得更清楚了
```

**讲稿要点**

- "抽成 repo 之后这 3 周，又长了两件事。"
- "**Agent Visual Review v1**：让同一个 coding agent 在 bounded manifest 边界内做视觉评审——不另起 VLM 服务。每个 visual dimension 必须声明对应的 metric_fallback 或 why_not_metricized，否则 prepare 阶段就 fail。把'agent 看图'这件事也做成了有契约的、不能滥用的机制。"
- "**Python contract → SKILL.md 自动产出**：`contract.py` 是手写真值，跑 `roboharness contract generate` 自动产出 SKILL.md + schemas + stubs。**agent 自己加载契约约束自己**。"
- "这两件事方向是一致的——**判断边界从 run-time 扩到 design-time**。再次回扣那句话——agent 不是变聪明了，是约束它的契约写得更清楚了。"

**备注**
- 这两个 feature 都是 5 月 20 日刚 commit 的，要诚实——"刚上、小规模验证中"
- 不需要进 demo

---

# 页 14 · Sidenote · 这个 repo 自己也是 agent 做的 [15:00-16:30]

**屏幕呈现**

```
顺便：这个 repo 自己也是 agent 做的

  139 commit · 100% AI 完成
       ├─ 65 次 AI solo (Claude/Codex 直接 author)
       ├─ 66 次 AI 协作 (co-authored-by trailer)
       └─ 8 次小改动漏加 trailer
         (gitignore / README 链接 / 几行 docs 修补)

  工作流：
    Opus 聊出 roadmap → 拆 GitHub issue
    → routine 每小时自动解 issue
    → 人 review PR
    → 手机上就能完成全套

  (4 routine 设计细节这里不展开)
```

**讲稿要点**

- "顺便说一句——**这个 repo 自己也是 AI agent 做的**。"
- "139 个 commit，100% AI 完成。其中 65 次 AI 直接 author，66 次 AI 协作（带 co-authored-by trailer），剩下 8 次是 gitignore、README 链接这种小改动漏加 trailer——也不是手写的。"
- "工作流速描：用 Opus 聊出 roadmap，拆成 GitHub issue，云端 routine 每小时自己解 issue，人只 review PR。手机上就能完成全套。"
- "4 routine 设计细节这里不展开，会后可以聊，公众号上有完整文章。"

**备注**
- 节奏快，2 分钟内
- Q&A prep：如果有人当场 `git log | grep claude` 发现 8 个没 trailer——"我们用 co-authored-by trailer 标记，偶尔小改动忘了加；这些都是 gitignore 之类的杂活"

---

# 页 15 · 点题 · 同一思路两层应用 [16:30-17:30]

**屏幕呈现**

```
这两件事 ── 用 agent 做 + 为 agent 做 ── 是同一个工程思路

           ┌──────────────────────────┐
           │      trust boundary      │
           └──────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
    For agents                    By agents
   (产品层)                       (开发层)

   contract                      branch namespace
   metric_gate                   GitHub comment as msgbus
   visual review manifest        PR template / CI gate
   approval report               4 routine 隔离

  Agent 越过边界 → 被挡住
  Agent 边界内  → 能自己往前走

  同一种 pattern · 两个不同层面
```

**讲稿要点**

- "我前面留了一个钩子——'用 agent 做'和'为 agent 做'背后是同一个工程思路。现在收一下。"
- "这件事叫 **trust boundary**。"
- "对外（for agents）：agent 在契约边界内自主，metric / visual / contract / approval report 这些都是边界上的'闸'。"
- "对内（by agents）：4 个 routine 在分支命名空间边界内并行工作，GitHub comment 当 message bus，没装中心调度器。"
- "本质都是同一件事：**让 agent 越过边界时被挡住、让 agent 在边界内时能自己往前走**。"
- "信任不是建立在'agent 足够好'上，是建立在'agent 越过边界时一定会被挡住'上。"

---

# 页 16 · Takeaway · 工程师向 [17:30-18:30]

**屏幕呈现**

```
给工程师同学：2 个能直接拿走的实践

  ❶ 契约先于 prompt
     把目标 / 不可改部分 / 异常行为
     写成结构化契约
     不能 ground 的 prompt 别让 agent 跑
     节省的不是 token, 是审查时间

  ❷ 给 Agent 真正的判断依据
     metric + visual 双轨, 缺一不可
     · qpos 案例 → visual 没救你, metric 救
     · 拇指朝下 → metric 没救你, visual 救
     发现一个新失败模式 → promote 到 metric
     engineer the harness 字面意思
```

**讲稿要点**

- "如果你也在做 AI coding 实践，两件事可以直接拿走。"
- "**第一，契约先于 prompt**。把目标、不可改部分、异常行为写成结构化契约。不能 ground 的 prompt 别让 agent 跑——这件事节省的不是 token，是审查时间。"
- "**第二，给 agent 真正的判断依据——metric + visual 双轨**。今天两个案例反向对称：qpos 是 metric 救了 visual，拇指朝下是 visual 救了 metric，缺一不可。"
- "每次 visual 抓到一个新的失败模式，就把它 promote 成 metric——这就是 engineer the harness 字面意义，也是我们日常的工作流。"

---

# 页 17 · Takeaway · 产品/管理向 [18:30-19:00]

**屏幕呈现**

```
给关心更宏观图景的同学：2 个判断

  ❶ 瓶颈已经换地方了
     从 "模型够不够好" → "Agent 能不能自己判断"
     这件事一旦解决, 研发流程长度直接解锁
     ──SONIC 升级就是证据

  ❷ 解决方向是 契约工程, 不是堆模型
     同模型 + 好契约    >    更强模型 + 没契约
     这是杠杆点所在
```

**讲稿要点**

- "如果你关心更宏观的图景——两个判断。"
- "**第一，瓶颈已经换地方了**。从'模型够不够好'切到了'agent 能不能自己判断'。这件事一旦解决，研发流程的长度直接解锁——SONIC 那个例子就是证据。"
- "**第二，解决方向是契约工程，不是堆模型**。同一个模型 + 好契约 vs 更强模型 + 没契约——前者赢。这是杠杆点所在。"

---

# 页 18 · 边界 · 当前限制 [19:00-19:30]

**屏幕呈现**

```
诚实说一下当前边界

  · 主要在 MuJoCo + G1 抓取 / 到达 / 移动 类任务上验证
  · Contract preset 数量还在补
  · Sim-to-real 真机 evidence 通道还在硬件验证
  · Agent Visual Review v1 刚上 (5-20), 小规模验证中
```

**讲稿要点**

- "最后诚实说一下当前边界。"
- "目前主要在 MuJoCo + G1 抓取/到达/移动类任务上验证过；contract preset 数量还在补；sim-to-real 真机 evidence 通道还在硬件验证；Agent Visual Review v1 刚上，小规模验证。"
- "**说这些是因为——如果你想在自己项目里用，这是我们目前真实的能力边界。**"

**备注** 这页 30 秒，快速诚实带过，不卖惨。

---

# 页 19 · 收尾 + 二维码 [19:30-20:00]

**屏幕呈现**

```
让 Agent 跑得久
不是 Agent 变神
是把"Agent 越过边界时会被挡住"这件事做扎实

  ────────────────────────────────────

  github.com/MiaoDX/roboharness
  miaodx.com/roboharness/

  [公众号二维码]

  Q&A
```

**讲稿要点**

- "用一句话收尾——**让 agent 跑得久，不是 agent 变神，是把'agent 越过边界时会被挡住'这件事做扎实**。这件事在产品层和开发层都是同一个工程思路。"
- "项目地址放在这里；公众号上有完整的 4 routine 设计文章和后续更新。"
- "谢谢，欢迎提问。"

---

# 📋 Q&A 备战

会上可能被问到的，准备好答案。

**Q1: `git log | grep claude` 发现 8 个 commit 没 trailer，真的 100% 吗？**
答："我们用 `co-authored-by: Codex` trailer 标记，偶尔小改动忘了加；那 8 个都是 gitignore、README 链接之类的杂活，不是手写的。"

**Q2: 95% 自主率是怎么测的？**
答："seeded good/bad/ambiguous corpus 上跑出来的 PASS/FAIL/AMBIGUOUS 分布。repo 里 `tests/regression/mujoco_grasp/` 有完整的 corpus。"

**Q3: 为什么不用一个独立的 VLM 做 judge？**
答："Opus 4.7 视觉分辨率 3× 之后，同一个 coding agent 做视觉评审已经够用。多加一个 VLM 增加复杂度、延迟、成本，还会破坏'同一个 agent 端到端'的信任边界。"

**Q4: SONIC 升级几小时无人值守是不是夸大了？**
答："分三阶段：人在 Plan 和 Review，只是 Execute 那段 agent 自主跑、人不在场。Execute 阶段是几个小时连续无人值守，这是事实。"

**Q5: 为什么不直接把 metric 写得更精细，还需要 visual 吗？**
答："**我们就是这么做的**——但每个新 metric 都来自一次 visual 抓到的真实失败。如果一开始就让我们写 metric，我们想不到拇指朝下这种情况。Visual 的价值就是发现这些 unknown unknowns，发现一个就 promote 一个到 metric。"

**Q6: 这套东西其他项目能不能用？**
答："我们抽成独立 repo 就是为这个。当前最容易接入的是 MuJoCo + Gymnasium 类的仿真环境。会后可以单独聊你们项目的具体情况。"

**Q7: GR00T 不是 4 月才发 SONIC 吗？**
答："4 月发的是 GR00T N1.7（VLA），SONIC 本身是 2 月 19 日 GEAR-SONIC release。我们做迁移的时候用的是 GEAR-SONIC，时间线对得上。"

---

# ⏱ 时长预算 (合计 20 min)

| 页 | 段 | 时长 | 累计 |
|---|---|---|---|
| 1 | 标题 | 0:30 | 0:30 |
| 2 | 一句话定义 | 1:30 | 2:00 |
| 3 | 旧 vision 图 | 1:00 | 3:00 |
| 4 | 当时为什么做不到 | 1:30 | 4:30 |
| 5 | 关键洞察（注意力） | 0:30 | 5:00 |
| 6 | SONIC 兑现 | 2:00 | 7:00 |
| 7 | 校准一句话 | 0:30 | 7:30 |
| 8 | 案例 A · qpos | 1:00 | 8:30 |
| 9 | 案例 B · 拇指朝下 | 1:30 | 10:00 |
| 10 | metric↔visual 循环 | 0:30 | 10:30 |
| 11 | 自主率分阶段 | 1:30 | 12:00 |
| 12 | 现场 demo | 1:30 | 13:30 |
| 13 | 抽出后又长什么 | 1:30 | 15:00 |
| 14 | Sidenote (by agents) | 1:30 | 16:30 |
| 15 | 点题 (trust boundary) | 1:00 | 17:30 |
| 16 | Takeaway 工程师 | 1:00 | 18:30 |
| 17 | Takeaway 管理 | 0:30 | 19:00 |
| 18 | 边界 | 0:30 | 19:30 |
| 19 | 收尾 + 二维码 | 0:30 | 20:00 |

**压缩到 18 min 的方案**：合并页 5→4（关键洞察并入"当时为什么做不到"，省 30 秒）、合并页 7→6（校准一句话并入 SONIC，省 30 秒）、合并页 17→16（两类 takeaway 合一页，省 60 秒）。共省 2 分钟。

---

# ✅ 演讲前 checklist

- [ ] 三张占位图替换完毕
- [ ] 公众号二维码到位
- [ ] 浏览器预先打开 https://miaodx.com/roboharness/grasp/ 并验证可访问
- [ ] 备用截图三张（顶部 banner / Phase Timeline / Constraint Evaluation 表）已截好放在本地，网络不稳时切换
- [ ] Codex CLI `/goal` 版本号确认：codex-cli 0.128.0+
- [ ] GR00T 时间线确认：Decoupled WBC 2025-11 / GEAR-SONIC 2026-02-19
- [ ] qpos commit 哈希确认：`102a593`
- [ ] 朗读一遍页 6 SONIC 那段，三阶段流程要讲顺
- [ ] 朗读一遍页 9 拇指朝下三幕，metric/visual/promote 的对称结构要清晰
- [ ] 朗读一遍页 15 trust boundary 收口
