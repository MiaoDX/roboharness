# 机器人控制开源项目技术调研与 Roboharness 适配分析

> Gymnasium Wrapper 方案能覆盖主流项目的 60%，但两大生态系统（legged_gym 系与 JAX/Brax 系）需要专门适配器。

在 2024–2026 年间活跃的社区共建机器人控制开源项目中，Isaac Lab、ManiSkill 和 LocoMuJoCo 原生支持 Gymnasium API，可低侵入接入；而 unitree_rl_gym（legged_gym 系）和 MuJoCo Playground（JAX/Brax 系）因架构根本差异，需编写定制化桥接层。

## 五大项目概览

| 项目 | Stars | 仿真器 | Gymnasium 兼容 | 许可证 | 关键机构 |
|------|-------|--------|---------------|--------|---------|
| Isaac Lab | ~6,700 | PhysX + Newton/MuJoCo-Warp | 原生 | BSD-3-Clause | NVIDIA |
| unitree_rl_gym | ~3,100 | Isaac Gym + MuJoCo | 自定义 API | BSD-3-Clause | Unitree Robotics |
| ManiSkill | ~2,400 | SAPIEN (PhysX) | 通过 Wrapper | Apache-2.0 | Hillbot/UCSD |
| MuJoCo Playground | ~1,800 | MuJoCo MJX/Warp | Brax 函数式 | Apache-2.0 | Google DeepMind |
| LocoMuJoCo | ~1,400 | MuJoCo + MJX | 原生 | MIT | TU Darmstadt |

---

## Isaac Lab：NVIDIA 生态的标准化框架

Isaac Lab 是目前最成熟的 Gymnasium 原生机器人学习框架，GitHub 拥有 376+ 贡献者和三种可视化后端（Kit/Newton/Rerun）。

其 `ManagerBasedRLEnv` 和 `DirectRLEnv` 两种环境工作流都直接继承 `gymnasium.Env`，通过 `gym.make("Isaac-Reach-Franka-v0")` 即可创建标准环境。

### 仿真架构

基于 PhysX GPU 加速物理引擎，通过 OmniPhysics Tensor API 实现零拷贝 CUDA → PyTorch 张量互通。3.0 版本引入了 Newton 后端（MuJoCo-Warp），支持无需 Isaac Sim 的 "kit-less" 模式运行。

仿真循环核心是 **decimation 机制** — 每个 RL step 内执行多个物理子步骤，典型配置为 `dt=1/120, decimation=2`，对应 RL 步长 1/60 秒。

`env.step()` 内部执行：预处理动作 → 多次 `sim.step()` → 渲染（可选） → 计算奖励/终止 → 重置已终止环境 → 计算观测。

### 配置系统

使用自定义 `@configclass` 装饰器（增强版 Python dataclass），支持 Hydra 风格的 CLI 覆盖（`env.a.b.param=value`）。环境通过标准 `gymnasium.register()` 注册，命名规范为 `Isaac-<Task>-<Robot>-v0`。

### 可视化

三种后端各有取舍：
- **Kit Visualizer**：RTX 光追渲染但依赖完整 Isaac Sim
- **Newton Visualizer**：轻量适合大规模环境
- **Rerun Visualizer**：支持 Web 和时间回溯，但在大规模环境中存在性能崩溃问题

视频录制支持 `gymnasium.wrappers.RecordVideo`（需 `--enable_cameras` 和 ffmpeg）以及 USD 动画烘焙。

已知问题：无头模式渲染卡死（#324）、启用相机后不渲染（#3250）、录制不包含调试标记（#2233）、Docker 中 WebRTC 错误（#3192）。

### Roboharness 接入可行性：极高

标准 Gymnasium Wrapper 模式直接适用：

```python
env = gym.make("Isaac-Cartpole-v0", render_mode="rgb_array")
env = RobotHarnessWrapper(env)  # 一行代码接入
env = Sb3VecEnvWrapper(env)     # RL 库 wrapper 必须在最后
```

关键注意事项：观测和动作是 GPU 上的 PyTorch 张量（形状 `(num_envs, ...)`），Wrapper 必须高效处理批量 GPU 数据，避免不必要的 CPU 传输。通过 `env.unwrapped` 可访问 scene、sim、cfg 等内部对象。

**Isaac Lab 用户代码改动量：零行。**

---

## unitree_rl_gym：最流行的人形 RL 训练仓库（需要深度适配）

拥有 3,100+ stars 和 520+ forks，是 Unitree Go2/H1/G1 机器人 RL 训练的事实标准。但它继承自 ETH Zurich 的 legged_gym，使用完全自定义的 API — 与 Gymnasium 完全不兼容。

### 仿真架构

分为两条路径：
- **训练路径**：使用 Isaac Gym 的 `self.gym.simulate(self.sim)` 驱动 PhysX 物理引擎，配合 `refresh_*_tensor()` 系列函数刷新 GPU 张量状态
- **验证路径**（sim-to-sim）：使用 MuJoCo 的 `mj_step()` 进行单实例 CPU 仿真

`step()` 返回 `(obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras)` 五元组 — 与 Gymnasium 的 `(obs, reward, terminated, truncated, info)` 完全不同。环境内部自动重置已终止的并行环境（`reset_idx()`），无需外部调用 `reset()`。

配置系统使用嵌套 Python 类（非 dataclass 也非 YAML），通过继承覆盖：`LeggedRobotCfg → G1Cfg`。环境注册通过自定义 `TaskRegistry` 而非 `gymnasium.register()`。

**无自动化测试，无 CI/CD** — 这是该项目的显著弱点。

### Roboharness 接入可行性：中等偏难

存在五个根本性不兼容：
1. **向量化 vs 单实例**：Isaac Gym 同时运行 4096+ 并行环境，Gymnasium 期望单实例
2. **返回值签名不匹配**：五元组 vs 标准五元组（字段含义不同）
3. **无 `reset()` 方法**：自动内部重置 vs 外部调用
4. **GPU 张量 vs NumPy 数组**：所有数据在 GPU 上
5. **无 Space 定义**：无 `observation_space` / `action_space`

推荐路径：包装 MuJoCo sim-to-sim 验证路径（`deploy/deploy_mujoco/deploy_mujoco.py`），这条路径是单实例、CPU 运行、使用标准 MuJoCo API，工作量约 1 周。

或者建议用户迁移到更新的 `unitree_rl_lab`（基于 Isaac Lab，已接近 Gymnasium 兼容）。

---

## MuJoCo Playground：DeepMind 的 JAX 函数式架构

Google DeepMind 的 GPU 加速机器人学习框架，覆盖 50+ 环境（DM Control Suite 25+、运动 19、操作 10），支持 MJX（JAX）和 MuJoCo Warp（NVIDIA）双后端。发表于 RSS 2025，已实现五个以上真实机器人平台的零样本 sim-to-real 迁移。

### 仿真架构

完全基于 JAX 函数式编程。环境是纯函数：`state = env.step(state, action)` 而非 Gymnasium 的 `obs, rew, ... = env.step(action)`。所有状态通过显式 State dataclass 传递，无隐藏内部状态 — 这是 JAX JIT 编译的要求。

批量并行通过 `jax.vmap()` 实现，训练循环通过 `jax.lax.scan` 在设备上完成完整 rollout。单线程 MJX 在 GPU 上比 CPU MuJoCo 慢 10 倍 — 其优势完全来自大规模并行（batch 1024–8192+）。

### Roboharness 接入可行性：困难

标准 Gymnasium Wrapper 无法直接使用，原因是：
1. **状态传递 vs 内部状态**：需在 Wrapper 内维护 `self._state`
2. **JAX 数组 vs NumPy**：每次 step 需 GPU → CPU 传输
3. **JIT 不兼容**：Gymnasium 的 Python 控制流（检查 done、调用 reset）破坏 JIT 编译
4. **重置语义冲突**：Brax 的 AutoResetWrapper 在 JIT 内处理重置

现有桥接模式 `RSLRLBraxWrapper`（通过 DLPack 实现 JAX → PyTorch 零拷贝传输）是最接近的参考。建议在 JAX 层面（MjxEnv 级别）拦截，在 Brax wrapper 链组装前插入，以避免破坏 JIT。但这意味着不能使用标准 Gymnasium Wrapper，需要编写 JAX-native 适配器。

---

## ManiSkill：GPU 并行操作基准的模范 Gymnasium 集成

ManiSkill3 基于 SAPIEN 引擎实现 GPU 并行仿真，在 RTX 4090 上达到 200K+ 状态 FPS、30K+ 渲染 FPS，比 Isaac Lab 节省 2–3 倍 GPU 内存。发表于 RSS 2025，包含桌面操作、灵巧手、绘画/清洁等丰富任务。

### 仿真架构

将所有并行环境的刚体/关节体放入单个 PhysX 场景，通过空间分区实现子场景隔离。支持异构并行仿真 — 不同子场景可包含不同物体和关节体。

### Gymnasium 兼容性（三种模式）

```python
# 模式 1：标准 gym.Env（CPU 单实例，NumPy）
env = gym.make("PickCube-v1", num_envs=1)
env = CPUGymWrapper(env)  # 标准 Gymnasium 接口

# 模式 2：gymnasium.vector.VectorEnv（GPU 并行，PyTorch 张量）
env = gym.make("PickCube-v1", num_envs=N)
env = ManiSkillVectorEnv(env, auto_reset=True)

# 模式 3：原始批量模式（非标准，torch 张量）
env = gym.make("PickCube-v1", num_envs=N)
```

### Roboharness 接入可行性：极高

CPUGymWrapper 模式直接适用标准 Gymnasium Wrapper。ManiSkillVectorEnv 模式需要处理批量 GPU 数据，与 Isaac Lab 类似。RecordEpisode wrapper 已内建录制功能。

---

## Roboharness 接入总结

| 项目 | 接入可行性 | 接入方式 | 预估工作量 |
|------|-----------|---------|-----------|
| Isaac Lab | 极高 | 标准 Gymnasium Wrapper | < 1 天 |
| ManiSkill | 极高 | CPUGymWrapper + 标准 Wrapper | < 1 天 |
| LocoMuJoCo | 高 | 标准 Gymnasium Wrapper | < 1 天 |
| unitree_rl_gym | 中等偏难 | 包装 MuJoCo sim-to-sim 路径 | ~1 周 |
| MuJoCo Playground | 困难 | JAX-native 适配器 | ~2 周 |
