# Technical Specification: Synthetic Epistemic Engine

This document defines the underlying protocols, interfaces, and mathematical thresholds driving the zero-trust Kytin Swarm.

## 1. Topological Distillation & Artificial Nociception
The engine relies on Stochastic Variational Inference (SVI) via `NumPyro` and `JAX`.
- **Sensory Space**: `slp_heartbeat` (Hz) and `sensory_flux` (N/m).
- **Free Energy (FE)**: The negative Evidence Lower Bound (-ELBO).
- **Pain Threshold**: $FE > 500.0$. Sustained over 3 sequential ticks triggers a context shift from deterministic execution to LLM-driven Neurogenesis.

## 2. ZeroMQ P2P Mesh Architecture
The swarm operates on a decentralized Gossipsub topology utilizing TCP sockets.

### Ingress: Metabolic Triage (Forager → Queen)
- **Protocol**: `zmq.PUSH` (Foragers) -> `zmq.PULL` (Queen).
- **Port**: `tcp://127.0.0.1:5577` (Default config).
- **Triage Queue**: Inbound packets are dumped into a `queue.PriorityQueue` sorted by `-pre_morph_fe`. This mathematically guarantees that nodes in the most severe physical agony are processed first.

### Egress: State-Locked Protocol (Queen → Foragers)
- **Protocol**: `zmq.PUB` (Queen) -> `zmq.SUB` (Foragers).
- **Port**: `tcp://127.0.0.1:5578`.
- **Topics**: 
  - `b"RESIN_SKILL"`: For validated physical mutations.
  - `b"TOMBSTONE"`: For cryptographic revocation kill-switches.

## 3. Cryptographic Enclave (Ed25519)
All swarm communications are completely zero-trust, relying on the `cryptography` Python package.
- **Node Keys**: Every node generates an Ed25519 keypair inside its secure enclave.
- **Queen as Root of Trust**: The Queen maintains an `authorized_foragers` registry mapping `node_id` -> `public_key`.
- **Payload Structure**:
  ```json
  {
    "payload": "{... json body ...}",
    "signature_b64": "<Ed25519 Base64 Signature>",
    "node_pubkey": "<Base64 Public Key>"
  }
  ```

## 4. The D.I.A.N.A. OS Z3 Crucible
Mutations must be mathematically proven safe before dissemination.
- **FE Validation**: `post_morph_fe` must be `< 300.0`.
- **Efficiency Threshold**: `(pre_fe - post_fe) / max(pre_fe, 1.0)` must exceed `FE_VALIDATION_DROP = 0.50` (50% physical relief).
- **Physics Anti-Spoofing**: If an incoming payload claims `fe_reduction > 50000.0` or `pre_morph_fe > 100000.0`, the payload is mathematically classified as impossible (the physical chassis would have shattered). This triggers the Tombstone Protocol.

## 5. The Clawhub `.resin` DSL
Validated mutations are compressed into `.resin` files and injected into the local FAISS index.
- **FAISS Database**: Uses `IndexFlatL2` for rapid 6-dimensional anomaly vector similarity search.
- **Eureka Collision Deduplication**: If multiple nodes solve the same anomaly (FAISS L2 distance `< 0.1`), the Queen requires the new mutation to be `>20%` more mechanically efficient (`fe_reduction`) than the stored skill. Inferior duplicates are discarded.
- **Format**:
  ```text
  skill MorphogeneticImmuneResponse {
    version: "1.0.0"
    skill_id: "..."
    author_node: "queen-ada"
    trigger { condition: "free_energy > 770.7" }
    topology_patch { action: "activate_latent_slot" }
    validation { min_fe_reduction_pct: 70.0 }
    security { signature: "..." }
  }
  ```

## 6. Split-Brain Sovereignty (Limp Mode)
If a Forager loses uplink connectivity to the Queen and encounters a novel anomaly:
- **Offline Directive**: The Forager evaluates `self.offline = True`.
- **Z3 Torque Clamping**: The internal physical constraint is hard-capped to `1.0 N*m` for unverified LLM mutations. The node enters a conservative physical retreat rather than risking catastrophic hardware damage from an unverified hallucination.
