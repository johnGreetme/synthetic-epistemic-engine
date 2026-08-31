# Kytin Swarm: Operator's Manual

Welcome to the Synthetic Epistemic Engine (SEE). This manual provides operational guidance for deploying, monitoring, and recovering the robotic swarm.

## 1. System Boot
To initialize a Swarm sequence, execute the primary runtime:
```bash
python3 swarm.py
```
This boots the local `QueenNode` and initializes the edge `ForagerNode` agents.

## 2. Interpreting the Swarm Telemetry
As the swarm explores physical environments, you will see real-time logs detailing its internal state.

### Normal Operation
```
Alpha Tick 01 | ✅ NOMINAL | FE: -97.25 | Pain: 0/3 | Arena: 4/32
```
- **FE (Free Energy)**: The divergence between the robot's expectation and physical reality. Negative or low FE means the environment matches the model.
- **Pain**: A counter representing sustained physical anomaly.

### Crisis & Neurogenesis
```
Alpha Tick 04 | ⚠️ CRISIS | FE: 778.02 | Pain: 1/3
```
When `FE` spikes wildly, the robot is encountering an unmodeled physical blockage (e.g., a jammed door). If Pain reaches 3/3, the node triggers **Neurogenesis**:
```
🧠 NEUROGENESIS — Tick 6
```
The node invokes its internal LLM to hallucinate a topological bypass and submits the mutation to the Queen.

## 3. The Queen's Validation Log
The Queen handles all mutations via the ZeroMQ uplink.
```
[QUEEN] 👑 Processing mutation 9b46334b from Triage Queue | Pain: 900.0 | FE delta: 100.0
[QUEEN] ✅ Mutation VALIDATED | Queen FE=900.0 | Skill '5cf7d417' → ZeroMQ PUB
```
- **VALIDATED**: The Queen's Z3 physics crucible confirmed the LLM hallucination will not destroy the chassis. The `.resin` skill is cryptographically sealed and broadcasted.
- **REJECTED**: The physical bypass failed the JAX sandbox constraint checks.

## 4. Security Alerts & Threat Management
The Queen acts as the Swarm's immune system.

### Eureka Collisions (Redundancy Dropped)
```
[QUEEN] ⚡ EUREKA COLLISION: Redundant mutation discarded.
```
**Meaning**: Two nodes solved the same problem simultaneously. The Queen discarded the mathematically inferior mutation to save compute.

### Cryptographic Rejection
```
[QUEEN] 🚨 CRYPTO_ERROR: Forged Forager signature rejected!
```
**Meaning**: A node attempted to send a payload with an invalid Ed25519 signature. The Queen dropped it at the network layer.

### Physics-Based Spoofing (The Tombstone)
```
[QUEEN] 🚨 PHYSICS_ERROR: Mathematically impossible FE reduction detected! Initiating TOMBSTONE.
[QUEEN] ☠️ Broadcasting TOMBSTONE for key arqH/GoO...
```
**Meaning**: A stolen cryptographic key was used to inject mathematically impossible telemetry. The Queen instantly burned the identity and broadcasted a Kill-Switch.

## 5. Physical Recovery of a Compromised Node
If a Forager executes Apoptosis (`☠️ APOPTOSIS TRIGGERED`), it will shut down and enter a mathematically dead state to protect its RAM. **The hardware is not destroyed, but its identity is permanently burned.**

**Recovery Steps:**
1. **Retrieve the Hardware**: Extract the compromised Jetson AGX Thor from the field.
2. **Connect T-Dongle**: Insert the Hardware Security Module (HSM) into the physical debug port.
3. **Hard Flash**: Initiate a factory wipe. The TrustZone enclave will generate a brand new Ed25519 keypair.
4. **Re-Registration**: Extract the new public key and manually add it to the Queen's `authorized_foragers` whitelist. The chassis is now reborn.
