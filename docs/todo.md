# Roboharness TODO

> 按优先级排序，每个任务独立可交付。通过 [GitHub Issues](https://github.com/MiaoDX/RobotHarness/issues) 跟踪进度。

## P0 — 核心体验（直接影响项目可用性）

### ~~1. Demo GIF / 视频~~ ✅

- ~~用本地已跑通的抓取任务截图制作 GIF~~
- ~~放到 README 顶部~~
- 已完成：两组抓取任务 front view GIF 已嵌入 README

### 2. MuJoCo + Meshcat 端到端可运行 Example — [#2](https://github.com/MiaoDX/RobotHarness/issues/2)

- 基于本地已有的抓取项目，提取一个最小可复现的 example
- 目标：`pip install roboharness[mujoco] && python examples/mujoco_grasp.py` 直接跑通
- 包含：加载模型 → 运行仿真 → 在检查点截图 → 保存到磁盘

### 3. PyPI 发布配置 — [#3](https://github.com/MiaoDX/RobotHarness/issues/3)

- 添加 GitHub Actions workflow 做自动发布
- 配置 `pypi` trusted publisher
- 让 `pip install roboharness` 真正可用

## P1 — 生态集成（扩大用户覆盖面）

### 4. Isaac Lab Gymnasium Wrapper 验证 — [#4](https://github.com/MiaoDX/RobotHarness/issues/4)

- 用 Isaac Lab 的标准环境（如 `Isaac-Reach-Franka-v0`）测试 `RobotHarnessWrapper`
- 记录接入过程，写一个 `examples/isaac_lab_integration.py`

### 5. ManiSkill Gymnasium Wrapper 验证 — [#5](https://github.com/MiaoDX/RobotHarness/issues/5)

- 用 ManiSkill 的 `PickCube-v1` + `CPUGymWrapper` 测试
- 写 `examples/maniskill_integration.py`

### 6. Rerun 集成 — [#6](https://github.com/MiaoDX/RobotHarness/issues/6)

- 在 capture 阶段同时写入 Rerun `.rrd` 文件
- 利用 Rerun 的 Blueprint 定义标准化的调试布局
- 可作为给 Rerun 社区提交的 example 来增加曝光

## P2 — 项目工程化（提升专业度）

### 7. GitHub Actions CI — [#7](https://github.com/MiaoDX/RobotHarness/issues/7)

- Python 3.9/3.10/3.11/3.12 矩阵测试
- ruff lint + pytest
- 在 README 加 CI badge

### 8. 多相机支持增强 — [#8](https://github.com/MiaoDX/RobotHarness/issues/8)

- Gymnasium Wrapper 目前只能通过 `env.render()` 获取单一视角
- 需要检测环境是否支持多相机（如 Isaac Lab 的 `TiledCamera`）
- 对支持的环境自动抓取多视角

### 9. CLI 工具 — [#9](https://github.com/MiaoDX/RobotHarness/issues/9)

- `roboharness inspect ./harness_output/` — 浏览截图和状态
- `roboharness report ./harness_output/` — 生成汇总报告
- 方便 Agent 和人类快速查看结果

## P3 — 社区推广（提升 Star 和 Reputation）

### 10. 英文 Blog 文章 — [#10](https://github.com/MiaoDX/RobotHarness/issues/10)

- 主题："Why AI Coding Agents Don't Need a Separate VLM for Robot Debugging"
- 发布到 Medium / 个人博客 / dev.to
- 在 Reddit (r/robotics, r/MachineLearning) 和 Twitter/X 分享

### 11. 给上游项目贡献 Example — [#11](https://github.com/MiaoDX/RobotHarness/issues/11)

- 给 Rerun 仓库提交一个 Roboharness 集成 example
- 给 ManiSkill 仓库提交一个 visual harness example
- 借助上游项目的流量获得曝光

### 12. 学术引用 — [#12](https://github.com/MiaoDX/RobotHarness/issues/12)

- 联系 AOR (arXiv:2603.04466) 作者，讨论协作或引用
- 如果积累了足够的实验结果，考虑写一个短论文/技术报告

---

## 完成记录

| 日期 | 完成项 |
|------|--------|
| 2026-04-02 | 项目骨架：pyproject.toml, 核心模块, Gymnasium Wrapper, GraspTaskStore, 22 个测试, 文档英文化 |
| 2026-04-02 | Demo GIF：两组抓取任务 front view GIF 嵌入 README（P0-1 ✅） |
| 2026-04-02 | Issues 跟踪：将 TODO 项关联至 GitHub Issues #2-#12 |
