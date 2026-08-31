# Synthetic Epistemic Engine (SEE)

[![CI](https://github.com/johnGreetme/synthetic-epistemic-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/johnGreetme/synthetic-epistemic-engine/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: >=3.10](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Architecture: diana-os-core Verified Submodule](https://img.shields.io/badge/Architecture-diana--os--core%20Verified%20Submodule-success)](https://github.com/johnGreetme/diana-os-core)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A biologically-inspired, decentralized, zero-trust robotic swarm intelligence that implements **Active Inference**, **Causal Morphogenesis**, **LLM DreamSandbox Verification**, **Metabolic ZeroMQ Gossipsub**, and the **Ed25519 Tombstone Protocol**.

SEE builds upon the immutable `diana-os-core` formal verification microkernel.

---

## Architecture Overview

1. **Active Inference & Artificial Nociception (`see.active_inference`)**:
   - Stochastic Variational Inference (SVI) via NumPyro & JAX.
   - Negative Evidence Lower Bound (-ELBO) Free Energy (FE) tracking augmented with Expected Information Gain (EIG).
   - Sustained Free Energy $> 500.0$ over 3 consecutive ticks triggers `PAIN_THRESHOLD_EXCEEDED` to initiate neurogenesis.

2. **DreamSandbox & Z3 Crucible (`see.dream_sandbox`)**:
   - Dynamic submodule adapter linking to `diana_core` formal verification microkernel.
   - Dynamic extraction of `unsat_core()` when kinematic constraints or torque limits are violated, returning structured semantic bounding boxes for LLM re-prompting.
   - Pre-allocated `LatentArena` enabling zero-JIT-recompile structural mutation.

3. **Metabolic Triage & ZeroMQ Mesh (`see.mesh`)**:
   - Ingress: `zmq.PUSH` (Foragers) -> `zmq.PULL` (Queen) on port `5577`.
   - Metabolic Triage Priority Queue sorting by `(-pre_morph_fe, arrival_timestamp, payload)` with explicit tie-breaking.
   - Egress: `zmq.PUB` (Queen) -> `zmq.SUB` (Foragers) on port `5578` broadcasting `RESIN_SKILL` and `TOMBSTONE`.
   - Zero-trust Ed25519 cryptographic enclaves for all message envelopes.

4. **Clawhub Registry & Tombstone Protocol (`see.immunity`)**:
   - FAISS `IndexFlatL2` 6-dimensional anomaly vector memory.
   - Eureka collision deduplication: requires $>20\%$ mechanical efficiency gain (`fe_reduction > existing * 1.20`) when $L2 < 0.1$.
   - Physics anti-spoofing detection (`fe_reduction > 50000.0` or `pre_morph_fe > 100000.0`) triggers global key revocation and Apoptosis.

5. **Split-Brain Sovereignty (Limp Mode) (`see.nodes`)**:
   - Offline foragers clamp Z3 torque limits to conservative bound ($1.0\text{ N}\cdot\text{m}$) and reject unverified LLM mutations.

---

## Directory Layout

```text
synthetic-epistemic-engine/
├── .github/
│   ├── ISSUE_TEMPLATE/          # Structured Bug Report & Feature Request templates
│   ├── workflows/               # CI/CD (Multi-Python matrix, releases)
│   └── PULL_REQUEST_TEMPLATE.md # Standard PR checklist
├── .gitmodules                  # Submodule tracking diana-os-core
├── .pre-commit-config.yaml      # Code quality & formatting hooks
├── diana_core/                  # Git submodule tracking diana-os-core
├── see/
│   ├── __init__.py
│   ├── active_inference/        # NumPyro SVI, Free Energy scalar calculation
│   │   ├── __init__.py
│   │   └── nociception.py
│   ├── dream_sandbox/           # LLM bypass generation & Z3 adapter
│   │   ├── __init__.py
│   │   ├── morphogenetic_agent.py
│   │   └── crucible_adapter.py
│   ├── mesh/                    # ZeroMQ Gossipsub, Ed25519 cryptographic enclaves
│   │   ├── __init__.py
│   │   ├── transport.py
│   │   ├── crypto_enclave.py
│   │   └── triage.py            # PriorityQueue sorted by (-pre_morph_fe)
│   ├── immunity/                # FAISS vector registry & Tombstone protocol
│   │   ├── __init__.py
│   │   ├── clawhub_registry.py
│   │   └── apoptosis.py
│   └── nodes/
│       ├── __init__.py
│       ├── queen_node.py
│       └── forager_node.py
├── tests/
│   ├── test_svi_fe.py
│   ├── test_crucible_unsat.py
│   ├── test_zmq_triage.py
│   └── test_tombstone.py
├── CHANGELOG.md                 # Semantic versioning history
├── CODE_OF_CONDUCT.md           # Contributor Covenant v2.1
├── CONTRIBUTING.md              # Developer setup & contribution guidelines
├── LICENSE                      # Apache 2.0 License
├── pyproject.toml               # Package build configuration & tool settings
├── README.md
└── SECURITY.md                  # Vulnerability disclosure policy
```

---

## Quickstart & Testing

### Installation
```bash
# Clone recursively to include the diana_core submodule
git clone --recursive https://github.com/johnGreetme/synthetic-epistemic-engine.git
cd synthetic-epistemic-engine

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting & Linting
```bash
ruff check .
ruff format --check .
```
