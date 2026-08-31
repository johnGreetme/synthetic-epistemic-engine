# Contributing to Synthetic Epistemic Engine (SEE)

Thank you for your interest in contributing to the **Synthetic Epistemic Engine (SEE)**! We welcome contributions from researchers, roboticists, and software engineers.

---

## 1. Development Setup

### Prerequisites
- Python 3.10 or higher
- Git 2.30+

### Step-by-Step Installation
1. **Clone the repository recursively** (crucial for linking the `diana_core` formal verification microkernel submodule):
   ```bash
   git clone --recursive https://github.com/johnGreetme/synthetic-epistemic-engine.git
   cd synthetic-epistemic-engine
   ```
   *If you previously cloned without `--recursive`, run:*
   ```bash
   git submodule update --init --recursive
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies and developer tooling**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

---

## 2. Running Tests & Quality Checks

Before submitting changes, ensure all tests pass and code conforms to our style standards:

```bash
# Run test suite
pytest tests/ -v

# Run linting
ruff check .

# Run formatting check
ruff format --check .

# Auto-fix formatting and linting
ruff format .
ruff check --fix .
```

---

## 3. Pull Request Guidelines

1. **Create a Feature Branch**:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. **Follow Conventional Commits**:
   - `feat:` A new feature or capability (e.g., `feat: add SE(3) arc trajectory verification`)
   - `fix:` A bug fix (e.g., `fix: prevent race condition in metabolic triage`)
   - `docs:` Documentation updates
   - `chore:` Tooling, dependency, or configuration changes
   - `test:` Adding or updating test cases
   - `refactor:` Code improvements without behavioral changes

3. **Verify Submodule Integrity**:
   Ensure `diana_core` remains pinned to an immutable release commit and no accidental changes are made inside the submodule directory.

4. **Submit Your Pull Request**:
   Fill out the PR template completely, referencing any related issues.

---

## 4. Code of Conduct

All contributors are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).
