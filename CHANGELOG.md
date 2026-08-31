# Changelog

All notable changes to the **Synthetic Epistemic Engine (SEE)** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-08-31

### Added
- **Active Inference & Artificial Nociception (`see.active_inference`)**:
  - SVI formulation using NumPyro and JAX to compute Free Energy (-ELBO).
  - Continuous tracking of physical pain divergence with `PAIN_THRESHOLD_EXCEEDED` event triggered on 3 sustained ticks ($FE > 500.0$).
- **DreamSandbox & Z3 Crucible (`see.dream_sandbox`)**:
  - Dynamic submodule adapter linking to `diana_core` formal verification microkernel.
  - Tracked invariant checking and `unsat_core()` extraction into `SemanticBoundingBox` veto prompts for LLM re-prompting.
  - Pre-allocated `LatentArena` boolean mask enabling zero-JIT-recompile neurogenesis and structural morphogenesis.
- **Metabolic Triage & ZeroMQ Mesh (`see.mesh`)**:
  - Ingress `zmq.PUSH` -> `zmq.PULL` (:5577) feeding a thread-safe `MetabolicTriageQueue` ordered by `(-pre_morph_fe, timestamp, sequence_id)` with explicit tie-breaking.
  - Egress `zmq.PUB` -> `zmq.SUB` (:5578) broadcasting `RESIN_SKILL` and `TOMBSTONE` topics.
  - Zero-trust Ed25519 cryptographic enclaves for signing and verifying all network payloads.
- **Clawhub FAISS Registry & Eureka Deduplication (`see.immunity`)**:
  - FAISS `IndexFlatL2` 6-dimensional anomaly vector memory.
  - Eureka collision deduplication requiring $>20\%$ mechanical efficiency gain (`fe_reduction > existing * 1.20`) when $L2 < 0.1$.
  - `.resin` Domain Specific Language serialization.
  - Tombstone Protocol & Apoptosis memory sanitization for revoked identities.
- **Nodes & Split-Brain Sovereignty (`see.nodes`)**:
  - `QueenNode` cluster coordinator and `ForagerNode` edge robotic agent.
  - Split-Brain Sovereignty (Limp Mode) clamping torque limits to $1.0\text{ N}\cdot\text{m}$ when offline.
- **Packaging & CI/CD**:
  - `pyproject.toml` with Apache-2.0 license, dependency specs, Ruff, Mypy, and Pytest configs.
  - GitHub Actions workflows for automated multi-Python CI matrix and package releases.
  - Contributor guidelines, Code of Conduct, and Security disclosure policy.
