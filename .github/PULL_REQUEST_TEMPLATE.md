## Description

<!-- Provide a brief description of the problem solved, feature added, or refactoring performed. -->

## Type of Change

- [ ] 🚀 New feature (`feat:`)
- [ ] 🐛 Bug fix (`fix:`)
- [ ] 📚 Documentation update (`docs:`)
- [ ] 🛠️ Refactoring / Performance (`refactor:`)
- [ ] 🧪 Test coverage improvement (`test:`)
- [ ] ⚙️ CI/CD / Infrastructure (`chore:`)

## Subsystems Impacted

- [ ] `see.active_inference` (NumPyro / JAX SVI, Nociception)
- [ ] `see.dream_sandbox` (Z3 Crucible Adapter, Morphogenetic Agent)
- [ ] `see.mesh` (ZeroMQ Mesh, Metabolic Triage, Ed25519 Enclave)
- [ ] `see.immunity` (FAISS Clawhub Registry, Tombstone Protocol, Apoptosis)
- [ ] `see.nodes` (QueenNode, ForagerNode)
- [ ] `diana_core` (Submodule linking / Invariants)

## Checklist

- [ ] I have run `pytest tests/ -v` and all tests pass locally.
- [ ] I have run `ruff check .` and `ruff format --check .` with zero errors.
- [ ] I have added/updated unit tests for the new code.
- [ ] I have updated relevant docstrings and type annotations.
- [ ] Submodule `diana_core` remains pinned to an immutable commit (no unintended submodule modifications).
