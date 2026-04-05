# CI Strategy: CPU + GPU Testing

> 讨论背景：随着 Roboharness 集成 WBC 运控、cuRobo 规划、Policy 推理等组件，
> CI 需要从纯 CPU 扩展到 GPU 测试。本文档记录分层方案和平台选型。

## 调研结论（2026-04-02）

### GPU 需求分析

| 组件 | GPU 是否必须 | 说明 |
|------|-------------|------|
| **GR00T Decoupled WBC — 上半身 IK** | **不需要** | 核心是 Pinocchio + Pink + qpsolvers，纯 CPU |
| **GR00T Decoupled WBC — RL locomotion** | 可能需要 | TensorRT 需 GPU，但 ONNX Runtime CPU 推理或许够用 |
| **GR00T SONIC 全身控制** | **必须** | TensorRT + CUDA，无 CPU 路径 |
| **cuRobo 路径规划** | **必须** | 全部自定义 CUDA kernel，无 CPU fallback |
| **CPU 替代（路径规划）** | N/A | VAMP（35μs 中位规划）、OMPL/MoveIt 均可替代 cuRobo |

### 集成策略

**不整包依赖 `decoupled_wbc`**（torch ~2.5GB、锁 Python 3.10、仓库 1+GB LFS），
采用薄封装方案：只依赖 `pin` + `pin-pink` + `qpsolvers`（~200-300MB，支持 Python 3.10-3.12）。

**GPU CI 时间线**：
- Phase 1（WBC 上半身 IK）：不需要 GPU CI
- Phase 2（RL locomotion）：视 ONNX Runtime CPU 表现决定
- Phase 3（SONIC / cuRobo）：必须搭建 GPU CI

详见 [#34](https://github.com/MiaoDX/roboharness/issues/34)。

## 当前状态

- GitHub Actions `ubuntu-latest`，纯 CPU
- lint (ruff) + pytest 矩阵测试 (Python 3.9–3.12)
- 29 个单元测试，全部 mock SimulatorBackend，不涉及真实仿真器或 GPU

## 分层 CI 策略

### Layer 1: CPU CI（保持不变）

所有核心逻辑（Protocol 层、数据存储、Gymnasium Wrapper）继续在 GitHub-hosted runner 上测试。
**GPU 需求不应拖慢或阻塞基础 CI。**

```yaml
# 现有 ci.yml，无需改动
runs-on: ubuntu-latest
```

### Layer 2: GPU CI（新增）

针对需要 GPU 的测试（cuRobo IK/motion planning、Policy 推理、MuJoCo GPU 渲染等），
新增独立的 GPU job。

```yaml
gpu-test:
  runs-on: ubuntu-gpu  # GitHub GPU runner (示例)
  if: |
    github.event_name == 'push' && github.ref == 'refs/heads/main'
    || contains(github.event.pull_request.labels.*.name, 'gpu-test')
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.12"
    - name: Install GPU dependencies
      run: pip install -e ".[dev,mujoco]" torch curobo
    - name: Run GPU tests
      run: pytest -m gpu
```

**触发策略**（GPU 测试贵且慢，不需要每个 commit 都跑）：
- 合并到 main 时自动跑
- PR 通过 label `gpu-test` 手动触发
- 可选：nightly/weekly 定时跑

### Layer 3: pytest markers 分离 CPU/GPU 测试

```python
# conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires GPU hardware")
    config.addinivalue_line("markers", "slow: slow integration tests")
```

```python
# tests/test_curobo_planning.py
@pytest.mark.gpu
def test_curobo_ik_solver():
    """需要 CUDA 的 cuRobo 逆运动学测试"""
    ...

@pytest.mark.gpu
@pytest.mark.slow
def test_policy_inference():
    """Policy 推理端到端测试"""
    ...
```

```ini
# pyproject.toml 中排除 GPU 测试（CPU CI 默认行为）
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: requires GPU hardware",
    "slow: slow integration tests",
]
```

CPU CI 跑 `pytest`（自动跳过 gpu marked 测试），GPU CI 跑 `pytest -m gpu`。

## 云端 GPU CI 平台对比

| 平台 | 原理 | GPU 类型 | 大致价格 | 优点 | 缺点 |
|------|------|----------|----------|------|------|
| **Cirun.io** | 在你的云账户启 GPU 实例 | 任意 (AWS/GCP/Azure) | 云实例费 (g4dn.xlarge ~$0.53/hr)；**开源免费** | 灵活，按需启停，开源零平台费 | 需绑定云账户 |
| **RunsOn** | AWS 上自动启停 EC2 runner | 任意 EC2 GPU | T4 on-demand ~$0.009/min, spot ~$0.004/min；**非商业免费** | 最便宜，比 GitHub GPU Runner 便宜 85-94% | 仅 AWS，需申请 GPU quota |
| **GitHub GPU Runners** | GitHub 原生 larger runner | T4 (16GB) | $0.07/min (~$4.20/hr) | 零运维，改一行 `runs-on` | **需 Team/Enterprise plan**，个人账户不可用 |
| **Cirrus Runners** | 固定月费无限分钟 | Linux GPU | $150/mo (OSS 50% off → $75/mo) | 无限使用，成本可预测 | 偶尔跑不划算 |
| **Self-hosted Runner** | 自己的 GPU 机器 | 自有硬件 | 电费 | 完全控制 | 需维护硬件 |
| **Modal** | GPU serverless（非 GHA 生态） | T4–H100 | 按秒计费 | 极致灵活 | 需额外脚本集成到 CI |

> **注意**：BuildJet、Namespace.so、WarpBuild 目前**不提供 GPU runner**，仅有 CPU。
> AWS CodeBuild GPU 很贵（~$0.30/min），不推荐。

### 10 分钟 GPU 测试的单次成本对比

| 平台 | 单次成本 (10min) | 月成本 (50 次/月) |
|------|-------------------|-------------------|
| RunsOn (T4 spot) | $0.04 | ~$5 |
| Cirun + AWS T4 on-demand | $0.09 | ~$5 |
| RunsOn (T4 on-demand) | $0.09 | ~$8 |
| GitHub GPU Runner | $0.70 | ~$35 + Team plan 费用 |
| Cirrus Runners | N/A (月费) | $75–150 |
| AWS CodeBuild | $3.00 | $150 |

### 推荐路径

**阶段一（现在）**：先加 pytest markers，把 GPU 测试标记出来，CPU CI 不受影响。

**阶段二（有 GPU 测试时）**：
- **首选 Cirun.io + AWS**（开源免费，只付云实例费 ~$5/月）
- 或 **RunsOn + AWS spot**（最便宜，~$5/月）
- 有 Team plan 则可用 **GitHub GPU Runners**（最简单但最贵）

**阶段三（测试变多时）**：
- Docker 镜像固化 GPU 环境（CUDA + PyTorch + cuRobo）
- 考虑 nightly 全量 GPU 测试 + PR 只跑关键路径
- 如果 GPU 测试频率很高，考虑 Cirrus Runners 的无限分钟方案

### Cirun.io 配置示例（推荐方案）

```yaml
# .cirun.yml
runners:
  - name: gpu-runner
    cloud: aws
    instance_type: g4dn.xlarge  # T4 GPU, 4 vCPU, 16GB
    machine_image: ami-xxxxx    # Deep Learning AMI
    preemptible: true           # spot instance, 更便宜
    labels:
      - gpu
```

```yaml
# .github/workflows/ci.yml 中新增 GPU job
gpu-test:
  runs-on: [self-hosted, gpu]
  if: github.ref == 'refs/heads/main'
  steps:
    - uses: actions/checkout@v4
    - name: Run GPU tests
      run: |
        pip install -e ".[demo,dev]"
        pytest -m gpu
```

## 社区参考

- **ManiSkill**: self-hosted GPU runner
- **Isaac Lab**: NVIDIA 内部 GPU CI 集群
- **cuRobo**: NVIDIA 内部 CI，社区贡献者本地测试
- **PyTorch**: 大规模 self-hosted GPU 集群
- **Hugging Face Transformers**: 混合方案（CPU 在 GitHub-hosted，GPU 在 self-hosted）
- **cloud_gpu_build_agent**: 开源项目，用 Terraform 在 AWS/GCP/Azure 创建临时 GPU VM 跑 CI

大多数中小型机器人开源项目采用 self-hosted 单机方案或云端按需 GPU runner。

## 参考链接

- [GitHub GPU Runners (larger runners)](https://docs.github.com/en/actions/using-github-hosted-runners/using-larger-runners)
- [Cirun.io](https://cirun.io/) — 开源项目免费
- [RunsOn GPU Runners](https://runs-on.com/runners/gpu/) — 非商业免费
- [Cirrus Runners](https://cirrus-runners.app/pricing/)
- [Modal](https://modal.com/)
- [cloud_gpu_build_agent (Terraform GPU CI)](https://github.com/jfpanisset/cloud_gpu_build_agent)
