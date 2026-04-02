# Contributing to Robot-Harness

Thank you for your interest in contributing! Robot-Harness is in early development and we welcome contributions of all kinds.

## How to Contribute

### Reporting Bugs

- Open an issue on [GitHub Issues](https://github.com/MiaoDX/RobotHarness/issues)
- Include your Python version, OS, and simulator versions
- Provide a minimal reproducible example if possible

### Suggesting Features

- Open a feature request issue
- Describe the use case and why it would be useful
- If you have a design in mind, share it — we appreciate thoughtful proposals

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Make your changes
5. Run linting and tests:
   ```bash
   ruff check src/ tests/
   pytest
   ```
6. Commit with a clear message
7. Open a Pull Request

### Adding a New Simulator Backend

We especially welcome new simulator backends! To add one:

1. Create a new file in `src/robot_harness/backends/`
2. Implement the `SimulatorBackend` protocol (see `core/harness.py`)
3. Add an example in `examples/`
4. Add the simulator's dependencies to `pyproject.toml` as an optional dependency group

### Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Type hints are encouraged (Python 3.10+ style)
- Keep it simple — avoid unnecessary abstractions

## Development Setup

```bash
git clone https://github.com/MiaoDX/RobotHarness.git
cd RobotHarness
pip install -e ".[dev]"
```

## AI Agent Contributions

We actively welcome contributions from AI coding agents! If you're using [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [OpenAI Codex](https://github.com/openai/codex), [OpenCode](https://github.com/nicepkg/OpenCode), or other autonomous coding tools to contribute, go for it — just make sure the tests pass.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
