# Synthetic Epistemic Engine (SEE)

[![CI](https://github.com/johnGreetme/synthetic-epistemic-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/johnGreetme/synthetic-epistemic-engine/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: >=3.10](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Architecture: diana-os-core Verified Submodule](https://img.shields.io/badge/Architecture-diana--os--core%20Verified%20Submodule-success)](https://github.com/johnGreetme/diana-os-core)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A biologically-inspired, hardware-enforced robotic swarm intelligence built to execute the **State-Locked Protocol**.

SEE transforms passive sensors into sovereign, self-resolving nodes. By combining **Active Inference**, **Causal Morphogenesis**, and **LLM DreamSandbox Verification** on top of the immutable `diana-os-core` Z3 microkernel, SEE mathematically guarantees that AI-generated physical actions are safe before they ever reach physical actuators. It is the cognitive engine powering the **Kytin Swarm**.

---

## The Genesis: Moving Beyond the "Zombie" Transformer

> *"Every major artificial intelligence today is fundamentally passive. Modern foundation models are autoregressive Transformers—static mathematical objects frozen in latent space. They do not perceive time. They possess no intrinsic motivation. They only wake up for a fraction of a second when fed a prompt, predict the next statistically likely token, and immediately return to dormancy."*

Industry is attempting to achieve Artificial General Intelligence (AGI) through brute-force scale—burning megawatts of datacenter power to simulate what biological neural systems accomplish on **20 watts**. Piling on clusters of GPUs to refine a statistical next-token guessing game will never yield true, autonomous cyber-physical intelligence.

To build something fundamentally new, one does not compete on brute-force scale. **One competes on architecture.**

### The Paradigm Shift: From Passive Prediction to Continuous Active Inference

The **Synthetic Epistemic Engine (SEE)** abandons passive prompt-response mechanics in favor of a **Continuous Active Inference Substrate**:

| Dimension | Legacy AI (Transformer / LLM) | Synthetic Epistemic Engine (SEE) |
| :--- | :--- | :--- |
| **Temporal State** | Frozen in latent space; awakens only on prompt | Continuously ticking ($100\text{ Hz}$ SVI inference loop) |
| **Motivation** | Zero intrinsic drive; passive token completion | **Epistemic Drive**: Maximizes Expected Information Gain (EIG) |
| **Error Handling** | Hallucinates or crashes with discrete error codes | **Artificial Nociception**: Measures divergence as physical **Pain** ($FE$) |
| **Evolution** | Static weights requiring expensive retraining | **Causal Morphogenesis**: Grows new latent dimensions at runtime |
| **Safety** | Probabilistic RLHF alignment (jailbreakable) | **State-Locked Protocol**: Deterministic Z3 formal theorem proving |

---

## How It Works: The 4-Stage Cognitive Loop

```text
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. CONTINUOUS ACTIVE INFERENCE (NumPyro + JAX)                         │
  │    Runs continuous SVI over sensory flux. If reality diverges from     │
  │    expected priors, Free Energy spikes (Physical Pain).                │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │ (FE > 500.0 sustained for 3 ticks)
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 2. CAUSAL MORPHOGENESIS & DREAMSANDBOX                                 │
  │    Agony forces the node to grow new latent dimensions (neurogenesis)  │
  │    or query LLMs as sandboxed subconscious dream simulators.           │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │ (Generated physical bypass)
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 3. THE Z3 CRUCIBLE (D.I.A.N.A. OS Microkernel)                        │
  │    LLM hallucinations are proven in Z3 SMT solvers. Unsafe torque or   │
  │    kinematic collisions are vetoed; only SAT mutations become .resin.  │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │ (Master Ed25519 Seal)
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 4. ZERO-TRUST SWARM IMMUNITY (Clawhub & ZeroMQ)                        │
  │    Validated skills propagate globally via ZeroMQ Gossipsub into local │
  │    FAISS registries, granting all swarm nodes instant immunity.        │
  └────────────────────────────────────────────────────────────────────────┘
```

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
   - Downloads and deduplicates verified Skills from **Clawhub** via a FAISS `IndexFlatL2` 6-dimensional anomaly vector memory.
   - Eureka collision deduplication: requires $>20\%$ mechanical efficiency gain (`fe_reduction > existing * 1.20`) when $L2 < 0.1$.
   - Physics anti-spoofing detection triggers global key revocation and Apoptosis via the Tombstone broadcast.

5. **Split-Brain Sovereignty (Limp Mode) (`see.nodes`)**:
   - Offline foragers clamp Z3 torque limits to conservative bound ($1.0\text{ N}\cdot\text{m}$) and reject unverified LLM mutations.

---

## Kytin Swarm: Physical Hardware Topology

SEE is designed to bridge probabilistic edge AI with deterministic physical hardware, specifically optimized for the following cluster architecture (the Apiary):

*   **The Forager Node (Edge/Drone):** Powered by the **NVIDIA Jetson AGX Thor**. Runs the SVI nociception engine, processes real-time telemetry, and clamps physical actuators into Limp Mode if the network drops.
*   **The Queen Node (Cluster Brain):** Powered by the **NVIDIA RTX 6000 Ada Generation**. Handles heavy LLM causal morphogenesis, mathematically verifies `.resin` mutations via the Z3 Crucible, and broadcasts cryptographic signatures.
*   **The Hardware Interlock:** Integrates via USB/UART with hardware security modules (e.g., Lilygo T-Dongle S3) and flight controllers (e.g., Pixhawk) to act as a physical guillotine, executing Apoptosis if spoofed telemetry is detected.

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
