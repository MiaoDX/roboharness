# CI Strategy: CPU + GPU Testing

> 讨论背景：随着 Robot-Harness 集成 cuRobo 规划、WBC 运控、Policy 推理等 GPU 依赖组件，
> CI 需要从纯 CPU 扩展到 GPU 测试。本文档记录分层方案和平台选型。

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
| **GitHub GPU Runners** | GitHub 原生 larger runner | T4, A10G | ~$0.07–0.15/min | 零运维，改一行 `runs-on` | 需 Team/Enterprise plan |
| **BuildJet** | 托管 GitHub Actions runner | T4 | ~$0.07/min | 一行改动 | GPU 型号有限 |
| **Cirun.io** | 在你的云账户启 GPU 实例 | 任意 (AWS/GCP) | 云实例费 (g4dn.xlarge ~$0.53/hr) + Cirun 开源免费 | 灵活，按需启停 | 需绑定云账户 |
| **RunsOn** | AWS 上自动启停 EC2 runner | 任意 EC2 GPU | EC2 费用 | 开源免费 | 仅 AWS |
| **Self-hosted Runner** | 自己的 GPU 机器 | 自有硬件 | 电费 | 完全控制 | 需维护硬件 |
| **Modal** | GPU serverless（非 GHA 生态） | T4, A10G, A100, H100 | 按秒计费 | 极致灵活 | 需额外脚本集成 |

### 推荐路径

**阶段一（现在）**：先加 pytest markers，把 GPU 测试标记出来，CPU CI 不受影响。

**阶段二（有 GPU 测试时）**：
- 首选 **GitHub GPU Runners**（如果有 Team plan）或 **BuildJet** — 最小改动
- 预算敏感则用 **Cirun.io** + AWS spot instance — 成本最低

**阶段三（测试变多时）**：
- Docker 镜像固化 GPU 环境（CUDA + PyTorch + cuRobo）
- 考虑 nightly 全量 GPU 测试 + PR 只跑关键路径

## 成本估算

假设：
- GPU 测试跑 10 分钟
- 每周触发 5 次（合并到 main + 偶尔 PR label 触发）
- 使用 GitHub GPU Runner 或 BuildJet

**月成本 ≈ 10min × 5次/周 × 4周 × $0.10/min = $20/月**

对于开源项目完全可接受。用 Cirun + AWS spot 可以更便宜（约 $5–10/月）。

## 社区参考

- **ManiSkill**: self-hosted GPU runner
- **Isaac Lab**: NVIDIA 内部 GPU CI 集群
- **cuRobo**: NVIDIA 内部 CI
- **PyTorch**: 大规模 self-hosted GPU 集群
- **Hugging Face Transformers**: 混合方案（CPU 在 GitHub-hosted，GPU 在 self-hosted）

大多数中小型机器人开源项目采用 self-hosted 单机方案或 GitHub GPU runner。

## 参考链接

- [GitHub larger runners](https://docs.github.com/en/actions/using-github-hosted-runners/using-larger-runners)
- [BuildJet GPU runners](https://buildjet.com/for-github-actions)
- [Cirun.io](https://cirun.io/)
- [RunsOn](https://runs-on.com/)
- [Modal](https://modal.com/)
