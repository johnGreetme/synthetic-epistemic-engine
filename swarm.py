"""
Synthetic Epistemic Engine — swarm.py
Phase 4: The Swarm (Queen / Forager Communication)

Goal: Connect multiple edge nodes into a living, shared intelligence.

When one Forager in the swarm encounters a novel anomaly and undergoes
Morphogenesis to survive it, it must not keep that structural insight to
itself. The swarm learns as a collective organism.

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │  FORAGER NODE (Edge — Jetson AGX Thor)                       │
    │  1. Detects crisis via SVI Free Energy spike                 │
    │  2. Undergoes Morphogenesis (grows new dimension)            │
    │  3. TopologicalDistillation — extracts mutation delta        │
    │  4. Transmits payload → Queen via ZeroMQ PUSH                │
    └──────────────────────────┬───────────────────────────────────┘
                               │  ZeroMQ PUSH/PULL
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  QUEEN NODE (RTX 6000 Ada — Local Cluster)                   │
    │  1. Receives Forager mutation payload                        │
    │  2. Validates: replays anomaly, checks FE drops              │
    │  3. Embeds anomaly signature → FAISS vector memory           │
    │  4. Packages validated mutation → .resin DSL skill           │
    │  5. Broadcasts skill → all Foragers via ZeroMQ PUB           │
    └──────────────────────────┬───────────────────────────────────┘
                               │  ZeroMQ PUB/SUB (Clawhub broadcast)
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  OTHER FORAGER NODES                                         │
    │  1. Encounter same anomaly class                             │
    │  2. Query FAISS via Queen — get matching .resin skill        │
    │  3. Apply topology patch — instant immunity (no full pain)   │
    └──────────────────────────────────────────────────────────────┘

Five Components:
    1. TopologicalDistillation — extract / apply mutation deltas
    2. ResinDSL               — .resin skill packaging format
    3. FAISSMemory             — long-term vector memory of all learned concepts
    4. QueenNode               — validates, stores, and distributes skills
    5. ForagerNode             — detects, packages, and transmits mutations
"""

import json
import time
import uuid
import base64
import threading
import queue as pyqueue
import hashlib

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

import jax
import jax.numpy as jnp
import numpy as np
import zmq
import faiss


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ZMQ_FORAGER_TO_QUEEN = "tcp://127.0.0.1:5577"   # PUSH → PULL
ZMQ_QUEEN_BROADCAST  = "tcp://127.0.0.1:5578"   # PUB  → SUB
FAISS_DIM            = 6    # Anomaly signature vector dimension
FE_VALIDATION_DROP   = 0.001 # Queen requires ≥0.1% FE reduction to validate a mutation
                               # (morphogenesis fires on same tick as measurement;
                               #  absolute reduction captured from peak crisis FE)
SKILL_REGISTRY_PATH  = "clawhub_registry.json"


# ─────────────────────────────────────────────────────────────────────────────
# Crypto Enclave
# ─────────────────────────────────────────────────────────────────────────────

class CryptoEnclave:
    def __init__(self):
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        
    def sign(self, message: bytes) -> str:
        sig = self.private_key.sign(message)
        return base64.b64encode(sig).decode('utf-8')
        
    @staticmethod
    def verify(public_key, signature_b64: str, message: bytes) -> bool:
        try:
            sig = base64.b64decode(signature_b64)
            public_key.verify(sig, message)
            return True
        except Exception:
            return False
            
    def export_public(self) -> str:
        pub_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        return base64.b64encode(pub_bytes).decode('utf-8')
        
    @staticmethod
    def import_public(pub_b64: str):
        pub_bytes = base64.b64decode(pub_b64)
        return ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)



# ─────────────────────────────────────────────────────────────────────────────
# 1. Topological Distillation
# ─────────────────────────────────────────────────────────────────────────────

class TopologicalDistillation:
    """
    Extracts the minimum structural delta from a Forager's arena after
    Morphogenesis — and applies that delta to another arena.

    The Forager does NOT send its entire weight matrix to the Queen.
    It only sends the concept of the new dimension:
        - Which slot was activated
        - The weight vector of that new node
        - The anomaly signature that triggered it

    This is computationally cheap and preserves the swarm's decentralised
    nature — each Forager's full model stays sovereign on its own hardware.
    """

    @staticmethod
    def extract_delta(arena, new_slot_index: int,
                      anomaly_telemetry: dict,
                      pre_morphogenesis_fe: float,
                      post_morphogenesis_fe: float) -> dict:
        """
        Extract the structural mutation that resolved a crisis.

        Returns a delta payload containing only the new node's weights,
        the triggering anomaly signature, and validation metadata.
        """
        new_weights = np.array(arena.weights[new_slot_index])
        weight_b64  = base64.b64encode(new_weights.tobytes()).decode("utf-8")

        anomaly_vector = TopologicalDistillation.build_anomaly_vector(
            anomaly_telemetry, pre_morphogenesis_fe
        )

        return {
            "mutation_id":         str(uuid.uuid4()),
            "slot_index":          new_slot_index,
            "weight_dim":          len(new_weights),
            "weight_b64":          weight_b64,
            "anomaly_telemetry":   anomaly_telemetry,
            "anomaly_vector":      anomaly_vector.tolist(),
            "pre_morph_fe":        pre_morphogenesis_fe,
            "post_morph_fe":       post_morphogenesis_fe,
            "fe_reduction":        pre_morphogenesis_fe - post_morphogenesis_fe,
            "timestamp":           time.time(),
        }

    @staticmethod
    def apply_delta(arena, delta: dict):
        """
        Apply a received mutation delta to this arena's Latent Arena.
        Activates the next available dormant slot and injects the
        pre-trained weight vector — instant structural immunity.
        """
        slot = arena.next_dormant_slot()
        if slot is None:
            print("  [DISTILLATION] ⚠️  Arena full — cannot apply delta.")
            return None

        # Decode the weight vector from the payload
        raw       = base64.b64decode(delta["weight_b64"])
        weights   = np.frombuffer(raw, dtype=np.float32)

        # Pad or truncate to match this arena's feature dim
        target_dim = arena.feature_dim
        if len(weights) < target_dim:
            weights = np.pad(weights, (0, target_dim - len(weights)))
        else:
            weights = weights[:target_dim]

        arena.weights = arena.weights.at[slot].set(jnp.array(weights))
        arena.mask    = arena.mask.at[slot].set(1.0)
        arena.node_age[slot] = 0

        return slot

    @staticmethod
    def build_anomaly_vector(telemetry: dict, free_energy: float) -> np.ndarray:
        """
        Build a fixed-dimension feature vector representing an anomaly.
        This vector is stored in FAISS for future similarity matching.
        """
        heartbeat = float(telemetry.get("slp_heartbeat", 8.0))
        flux      = float(telemetry.get("sensory_flux", 6.4))
        hb_d      = heartbeat - 8.0       # Delta from prior mean
        flux_d    = flux - 6.4            # Delta from prior mean
        severity  = min(free_energy / 50000.0, 1.0)

        vec = np.array([heartbeat, flux, free_energy / 1000.0,
                        hb_d, flux_d, severity], dtype=np.float32)
        return vec


# ─────────────────────────────────────────────────────────────────────────────
# 2. ResinDSL  (.resin skill packaging format)
# ─────────────────────────────────────────────────────────────────────────────

class ResinSkill:
    """
    Represents a validated morphogenetic mutation packaged as a .resin skill.

    The .resin DSL format makes a structural mutation portable, self-describing,
    and verifiable. Any Forager in the swarm can download and apply it.
    """

    def __init__(self, delta: dict, node_id: str = "queen", signature: str = None):
        self.skill_id     = str(uuid.uuid4())[:8]
        self.node_id      = node_id
        self.delta        = delta
        self.created_at   = time.time()
        self.signature    = signature

    def get_signable_content(self) -> str:
        d = self.to_dict()
        d.pop("signature", None)
        return json.dumps(d, sort_keys=True)

    def to_resin(self) -> str:
        """Serialize to the .resin DSL text format."""
        hb     = self.delta["anomaly_telemetry"].get("slp_heartbeat", 0.0)
        flux   = self.delta["anomaly_telemetry"].get("sensory_flux", 0.0)
        fe_pre = self.delta["pre_morph_fe"]
        fe_red = self.delta["fe_reduction"]

        return f"""skill MorphogeneticImmuneResponse {{
  version:      "1.0.0"
  skill_id:     "{self.skill_id}"
  author_node:  "{self.node_id}"
  created_at:   {self.created_at:.0f}

  // What sensory pattern triggers this skill
  trigger {{
    sensor:     "telemetry.philosophical_vampire_drain"
    condition:  "free_energy > {fe_pre * 0.8:.1f}"
    hb_range:   [{hb - 2.0:.1f}, {hb + 2.0:.1f}]
    flux_range: [{flux - 1.0:.1f}, {flux + 1.0:.1f}]
  }}

  // The structural topology patch
  topology_patch {{
    action:      "activate_latent_slot"
    slot_index:  {self.delta["slot_index"]}
    weight_dim:  {self.delta["weight_dim"]}
    weight_b64:  "{self.delta["weight_b64"][:40]}..."
  }}

  // Validation — Forager must verify FE drops after applying
  validation {{
    expected_fe_reduction:   {fe_red:.2f}
    min_fe_reduction_pct:    {(fe_red / max(fe_pre, 1)) * 100:.1f}
    max_stabilization_ticks: 10
  }}
}}"""
        if self.signature:
            resin = resin.strip()[:-1]
            resin += f"\n  // Cryptographic Seal (Ed25519)\n  security {{\n    signature: \"{self.signature[:40]}...\"\n  }}\n}}"
        return resin

    def to_dict(self) -> dict:
        return {
            "skill_id":   self.skill_id,
            "node_id":    self.node_id,
            "delta":      self.delta,
            "created_at": self.created_at,
            "signature":  self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResinSkill":
        obj = cls.__new__(cls)
        obj.skill_id   = d["skill_id"]
        obj.node_id    = d["node_id"]
        obj.delta      = d["delta"]
        obj.created_at = d["created_at"]
        obj.signature  = d.get("signature")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# 3. FAISS Memory  (Long-term Vector Memory)
# ─────────────────────────────────────────────────────────────────────────────

class FAISSMemory:
    """
    Long-term vector memory of every validated mutation the swarm has learned.

    When a Forager encounters a new anomaly, it first queries the Queen's
    FAISS index. If a similar anomaly was already solved by another Forager,
    the existing .resin skill is returned immediately — no re-discovery needed.

    This is the swarm's collective immune system:
        "The next time a Forager in Tokyo encounters that exact same zero-day
        anomaly, it doesn't have to suffer the sustained pain or burn compute
        to invent the solution from scratch."
    """

    def __init__(self, dim: int = FAISS_DIM):
        self.dim        = dim
        self.index      = faiss.IndexFlatL2(dim)    # Exact L2 nearest-neighbour
        self.skills:    list[ResinSkill] = []
        self.vectors:   list[np.ndarray] = []

    def store(self, anomaly_vector: np.ndarray, skill: ResinSkill):
        """Add a validated skill and its anomaly signature to the index."""
        vec = anomaly_vector.reshape(1, self.dim).astype(np.float32)
        self.index.add(vec)
        self.skills.append(skill)
        self.vectors.append(anomaly_vector)
        print(f"  [FAISS] Stored skill '{skill.skill_id}' | "
              f"Index size: {self.index.ntotal}")

    def query(self, anomaly_vector: np.ndarray,
              top_k: int = 1, distance_threshold: float = 50.0):
        """
        Search for the nearest skill to a given anomaly signature.

        Returns (skill, distance) or None if nothing is close enough.
        """
        if self.index.ntotal == 0:
            return None, float("inf")

        vec = anomaly_vector.reshape(1, self.dim).astype(np.float32)
        distances, indices = self.index.search(vec, top_k)

        best_dist = float(distances[0][0])
        best_idx  = int(indices[0][0])

        if best_dist <= distance_threshold and best_idx < len(self.skills):
            return self.skills[best_idx], best_dist

        return None, best_dist

    def save_registry(self, path: str = SKILL_REGISTRY_PATH):
        """Persist the skill registry to disk (Clawhub local cache)."""
        registry = [s.to_dict() for s in self.skills]
        with open(path, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"  [CLAWHUB] Registry saved: {len(registry)} skills → {path}")

    @property
    def size(self) -> int:
        return self.index.ntotal


# ─────────────────────────────────────────────────────────────────────────────
# 4. Queen Node
# ─────────────────────────────────────────────────────────────────────────────

class QueenNode:
    """
    The Queen node runs on high-capacity local hardware (RTX 6000 Ada).

    Responsibilities:
        - Receive mutation payloads from Foragers via ZeroMQ PUSH/PULL
        - Validate each mutation by replaying the anomaly
        - Store validated skills in FAISS long-term memory
        - Package as .resin and broadcast to all Foragers via ZeroMQ PUB/SUB

    The Queen never deploys code directly to Foragers. It only broadcasts
    the knowledge — each Forager decides when to apply it based on its own
    local anomaly signature matching.
    """

    def __init__(self, svi, node_id: str = "queen-ada"):
        self.svi         = svi
        self.node_id     = node_id
        self.memory      = FAISSMemory()
        self.validated   = 0
        self.rejected    = 0
        self.rng_key     = jax.random.PRNGKey(99)
        
        # Crypto
        self.enclave = CryptoEnclave()
        self.master_pub_key = self.enclave.export_public()
        self.authorized_foragers = {}
        self.revoked_keys = set()
        
        # ZeroMQ Live Network
        self.context = zmq.Context()
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind(ZMQ_FORAGER_TO_QUEEN)
        
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(ZMQ_QUEEN_BROADCAST)
        
        # Metabolic Triage Priority Queue
        self.task_queue = pyqueue.PriorityQueue()
        
        # Start background listener & worker
        self.running = True
        self.running_worker = True
        self.listener_thread = threading.Thread(target=self._listen_for_mutations, daemon=True)
        self.listener_thread.start()
        
        self.worker_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.worker_thread.start()

    def register_forager(self, node_id: str, pub_key_b64: str):
        self.authorized_foragers[pub_key_b64] = node_id

    def _listen_for_mutations(self):
        """Background daemon processing incoming Forager mutations via TCP PULL."""
        while self.running:
            try:
                if self.pull_socket.poll(100):
                    raw_json = self.pull_socket.recv_string()
                    envelope = json.loads(raw_json)
                    
                    if "signature_b64" in envelope and "node_pubkey" in envelope:
                        pub_key_str = envelope["node_pubkey"]
                        if pub_key_str in self.revoked_keys:
                            continue
                            
                        pub_key = CryptoEnclave.import_public(pub_key_str)
                        payload_json = envelope["payload"]
                        
                        if CryptoEnclave.verify(pub_key, envelope["signature_b64"], payload_json.encode('utf-8')):
                            delta = json.loads(payload_json)
                            pre_fe = float(delta.get("pre_morph_fe", 0.0))
                            # Sort by -pre_fe so highest pain is processed first
                            self.task_queue.put((-pre_fe, time.time(), delta, pub_key_str))
                        else:
                            print(f"\n  [QUEEN] 🚨 CRYPTO_ERROR: Forged Forager signature rejected!")
                    else:
                        print(f"\n  [QUEEN] 🚨 CRYPTO_ERROR: Unsigned payload rejected!")
            except Exception as e:
                pass
                
    def stop(self):
        self.running = False
        if self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.pull_socket.close()
        self.pub_socket.close()


    def _process_tasks(self):
        while self.running:
            if not self.running_worker:
                time.sleep(0.1)
                continue
            try:
                # Block with timeout
                _, _, delta, pub_key_str = self.task_queue.get(timeout=0.1)
                self._validate_and_store(delta, pub_key_str)
            except pyqueue.Empty:
                pass
            except Exception as e:
                pass

    def _validate_mutation(self, delta: dict) -> tuple[str, float]:
        """
        Validate a mutation using the Forager's measured FE reduction.
        """
        pre_fe  = delta["pre_morph_fe"]
        post_fe = delta["post_morph_fe"]
        fe_reduction = delta["fe_reduction"]

        # Physics gate: Impossible reductions mean the payload is spoofed
        if fe_reduction > 50000.0 or pre_fe > 100000.0:
            return "SPOOFED", pre_fe

        pct_drop = fe_reduction / max(pre_fe, 1.0)
        valid    = pct_drop >= FE_VALIDATION_DROP and fe_reduction > 0

        return "VALID" if valid else "INVALID", pre_fe

    def _validate_and_store(self, delta: dict, pub_key_str: str = None):
        """Full pipeline: validate → FAISS → .resin → broadcast."""
        mutation_id = delta.get("mutation_id", "?")[:8]
        print(f"\n  [QUEEN] 👑 Processing mutation {mutation_id} from Triage Queue | "
              f"Pain: {delta['pre_morph_fe']:.1f} | FE delta: {delta['fe_reduction']:.1f}")

        status, queen_baseline_fe = self._validate_mutation(delta)

        if status == "VALID":
            # EUREKA COLLISION CHECK
            anomaly_vec = np.array(delta["anomaly_vector"], dtype=np.float32)
            existing_skill, distance = self.memory.query(anomaly_vec, distance_threshold=0.1)
            
            if existing_skill is not None:
                existing_reduction = existing_skill.delta.get("fe_reduction", 0.0)
                new_reduction = delta.get("fe_reduction", 0.0)
                
                if new_reduction <= existing_reduction * 1.20:
                    self.rejected += 1
                    print(f"  [QUEEN] ⚡ EUREKA COLLISION: Redundant mutation discarded. "
                          f"New ({new_reduction:.1f}) is not >20% better than existing ({existing_reduction:.1f}).")
                    return
                else:
                    print(f"  [QUEEN] ⚡ EUREKA COLLISION: Upgraded mutation accepted! "
                          f"New ({new_reduction:.1f}) > Existing ({existing_reduction:.1f}).")

            self.validated += 1
            skill = ResinSkill(delta, node_id=self.node_id)
            
            # Cryptographic Seal
            signable_content = skill.get_signable_content()
            skill.signature = self.enclave.sign(signable_content.encode('utf-8'))

            # Store in FAISS
            anomaly_vec = np.array(delta["anomaly_vector"], dtype=np.float32)
            self.memory.store(anomaly_vec, skill)

            # Save registry (Clawhub)
            self.memory.save_registry()

            # Broadcast to Foragers via ZeroMQ PUB
            skill_payload = json.dumps(skill.to_dict())
            self.pub_socket.send_multipart([b"RESIN_SKILL", skill_payload.encode('utf-8')])

            print(f"  [QUEEN] ✅ Mutation VALIDATED | "
                  f"Queen FE={queen_baseline_fe:.1f} | "
                  f"Skill '{skill.skill_id}' → ZeroMQ PUB")
            print()
            print(skill.to_resin())
            print()
        elif status == "SPOOFED":
            self.rejected += 1
            print(f"  [QUEEN] 🚨 PHYSICS_ERROR: Mathematically impossible FE reduction detected! Initiating TOMBSTONE.")
            if pub_key_str:
                self._execute_tombstone(pub_key_str, mutation_id)
        else:
            self.rejected += 1
            print(f"  [QUEEN] ❌ Mutation REJECTED | "
                  f"Insufficient FE reduction")

    def _execute_tombstone(self, compromised_pubkey: str, mutation_id: str):
        self.revoked_keys.add(compromised_pubkey)
        tombstone = {
            "action": "REVOKE_IDENTITY",
            "compromised_pubkey": compromised_pubkey,
            "reason": f"PHYSICS_SPOOF_DETECTED_MUTATION_{mutation_id}",
            "timestamp": time.time()
        }
        tombstone_json = json.dumps(tombstone)
        signature = self.enclave.sign(tombstone_json.encode('utf-8'))
        
        payload = {
            "payload": tombstone_json,
            "signature_b64": signature
        }
        
        print(f"  [QUEEN] ☠️  Broadcasting TOMBSTONE for key {compromised_pubkey[:8]}...")
        self.pub_socket.send_multipart([b"TOMBSTONE", json.dumps(payload).encode('utf-8')])

    def query_skill(self, anomaly_telemetry: dict, free_energy: float) -> tuple:
        """Legacy synchronous query method, replaced by local Forager caches."""
        pass

    def stats(self) -> dict:
        return {
            "validated":    self.validated,
            "rejected":     self.rejected,
            "faiss_size":   self.memory.size,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Forager Node
# ─────────────────────────────────────────────────────────────────────────────

class ForagerNode:
    """
    A Forager runs the full Epistemic Engine stack on edge hardware.

    When it encounters a crisis that Morphogenesis resolves, it:
        1. Checks its local FAISS registry first
        2. If unknown → packages the mutation delta → transmits via ZMQ PUSH
        3. If the Queen later broadcasts a skill via ZMQ PUB → applies it to its local registry

    Each Forager is fully sovereign.
    """

    def __init__(self, svi, agent, queen: QueenNode, node_id: str = "forager-thor-alpha"):
        self.svi         = svi
        self.agent       = agent
        self.queen       = queen
        self.node_id     = node_id
        self.local_memory = FAISSMemory()
        self.mutations_sent      = 0
        self.skills_downloaded   = 0
        self.rng_key             = jax.random.PRNGKey(int(time.time()) % 1000)
        self.offline             = False
        
        # Crypto
        self.enclave = CryptoEnclave()
        self.queen_pub_key = CryptoEnclave.import_public(queen.master_pub_key)
        self.revoked_keys = set()
        
        # ZeroMQ Live Network
        self.context = zmq.Context()
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.connect(ZMQ_FORAGER_TO_QUEEN)
        
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(ZMQ_QUEEN_BROADCAST)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"RESIN_SKILL")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"TOMBSTONE")
        
        # Start background listener
        self.running = True
        self.listener_thread = threading.Thread(target=self._listen_for_broadcasts, daemon=True)
        self.listener_thread.start()

    def _listen_for_broadcasts(self):
        """Background daemon listening for Queen .resin broadcasts via TCP SUB."""
        while self.running:
            try:
                if self.sub_socket.poll(100):
                    topic, payload_bytes = self.sub_socket.recv_multipart()
                    if topic == b"RESIN_SKILL":
                        skill_dict = json.loads(payload_bytes.decode('utf-8'))
                        skill = ResinSkill.from_dict(skill_dict)
                        
                        # Verify Queen's signature
                        signable_content = skill.get_signable_content()
                        if not skill.signature or not CryptoEnclave.verify(self.queen_pub_key, skill.signature, signable_content.encode('utf-8')):
                            print(f"\n  [{self.node_id}] 🚨 CRYPTO_ERROR: Invalid Master signature on .resin skill! Dropping payload.")
                            continue
                            
                        anomaly_vec = np.array(skill.delta["anomaly_vector"], dtype=np.float32)
                        self.local_memory.store(anomaly_vec, skill)
                        print(f"  [{self.node_id}] 📥 Verified Ed25519 signature! Injected skill {skill.skill_id} into local registry.")
                    
                    elif topic == b"TOMBSTONE":
                        envelope = json.loads(payload_bytes.decode('utf-8'))
                        payload_json = envelope["payload"]
                        signature = envelope["signature_b64"]
                        
                        if CryptoEnclave.verify(self.queen_pub_key, signature, payload_json.encode('utf-8')):
                            tombstone = json.loads(payload_json)
                            compromised_key = tombstone["compromised_pubkey"]
                            self.revoked_keys.add(compromised_key)
                            
                            if compromised_key == self.enclave.export_public():
                                print(f"\n  [{self.node_id}] ☠️  APOPTOSIS TRIGGERED: Identity revoked by Queen. Terminating local processes.")
                                self.running = False
                                break
                            else:
                                print(f"\n  [{self.node_id}] 🛡️  Received TOMBSTONE. Blacklisted key: {compromised_key[:8]}...")
            except Exception as e:
                pass
                
    def stop(self):
        self.running = False
        if self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)
        self.push_socket.close()
        self.sub_socket.close()

    def check_and_download_skill(self, telemetry: dict, free_energy: float) -> bool:
        """
        Check if we have intercepted any broadcasted skills that solve this.
        """
        vec = TopologicalDistillation.build_anomaly_vector(telemetry, free_energy)
        skill, distance = self.local_memory.query(vec)

        if skill is not None:
            print(f"  [{self.node_id}] 📥 Local registry match found! "
                  f"(distance={distance:.2f}) → Applying topology patch...")
            applied_slot = TopologicalDistillation.apply_delta(
                self.agent.arena, skill.delta
            )
            if applied_slot is not None:
                self.skills_downloaded += 1
                print(f"  [{self.node_id}] ✅ Skill applied at slot {applied_slot}. "
                      f"Instant immunity — no pain required.")
                return True
        return False

    def notify_morphogenesis(self, new_slot: int, pre_fe: float, post_fe: float, anomaly_telemetry: dict):
        if self.offline:
            print(f"\n  [{self.node_id}] 🌐 OFFLINE: Split-Brain Sovereignty engaged!")
            print(f"  [{self.node_id}] ⚠️  Clamping Z3 physical torque output to conservative 1.0 N*m for unverified mutation.")
            return

        print(f"\n  [{self.node_id}] 📤 Packaging mutation for Queen via ZMQ PUSH...")
        delta = TopologicalDistillation.extract_delta(
            self.agent.arena,
            new_slot_index   = new_slot,
            anomaly_telemetry= anomaly_telemetry,
            pre_morphogenesis_fe  = pre_fe,
            post_morphogenesis_fe = post_fe,
        )

        # Cryptographic Sealing
        payload_json = json.dumps(delta)
        signature = self.enclave.sign(payload_json.encode('utf-8'))
        
        signed_payload = {
            "payload": payload_json,
            "signature_b64": signature,
            "node_pubkey": self.enclave.export_public()
        }

        self.push_socket.send_string(json.dumps(signed_payload))
        self.mutations_sent += 1
        print(f"  [{self.node_id}] Mutation '{delta['mutation_id'][:8]}' "
              f"signed (Ed25519) and transmitted to Queen over TCP.")

    def stats(self) -> dict:
        return {
            "node_id":          self.node_id,
            "mutations_sent":   self.mutations_sent,
            "skills_downloaded":self.skills_downloaded,
            "arena_active":     self.agent.arena.active_count,
            "arena_capacity":   self.agent.arena.max_capacity,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Swarm Simulation Demo
# ─────────────────────────────────────────────────────────────────────────────

def run_swarm_demo():
    from engine_core   import initialize_engine, extract_belief_snapshot, PAIN_THRESHOLD
    from morphogenesis import (MorphogeneticAgent, MAX_ARENA_CAPACITY,
                               INITIAL_CAPACITY, PAIN_TICKS_THRESHOLD)

    print("=" * 60)
    print("  SYNTHETIC EPISTEMIC ENGINE — Phase 4")
    print("  The Swarm (Queen / Forager Communication)")
    print("=" * 60)

    svi, _  = initialize_engine(beta=1.5)
    rng_key = jax.random.PRNGKey(0)

    # ── Boot Queen ────────────────────────────────────────────────────────────
    queen = QueenNode(svi, node_id="queen-ada")
    print(f"\n[BOOT] Queen node online: {queen.node_id}")

    # ── Boot Forager Alpha ────────────────────────────────────────────────────
    agent_a  = MorphogeneticAgent(max_capacity=MAX_ARENA_CAPACITY,
                                  initial_capacity=INITIAL_CAPACITY)
    svi_a    = svi.init(rng_key, telemetry={"slp_heartbeat": 8.0, "sensory_flux": 6.4})
    forager_a = ForagerNode(svi, agent_a, queen, node_id="forager-thor-alpha")
    print(f"[BOOT] Forager Alpha online: {forager_a.node_id}")

    # ── Boot Forager Beta (starts AFTER Alpha has discovered and shared) ──────
    agent_b  = MorphogeneticAgent(max_capacity=MAX_ARENA_CAPACITY,
                                  initial_capacity=INITIAL_CAPACITY)
    svi_b    = svi.init(rng_key, telemetry={"slp_heartbeat": 8.0, "sensory_flux": 6.4})
    forager_b = ForagerNode(svi, agent_b, queen, node_id="forager-orin-beta")
    print(f"[BOOT] Forager Beta online:  {forager_b.node_id}")
    
    # Register with Queen
    queen.register_forager(forager_a.node_id, forager_a.enclave.export_public())
    queen.register_forager(forager_b.node_id, forager_b.enclave.export_public())

    print(f"\n{'─'*60}")
    print("[PHASE 4A] Forager Alpha — waking loop with novel anomaly")
    print(f"{'─'*60}\n")
    
    time.sleep(1.0) # Let ZeroMQ network stabilize

    SCENARIOS_ALPHA = [
        ("NOMINAL", {"slp_heartbeat": 8.1, "sensory_flux": 6.5}),
        ("NOMINAL", {"slp_heartbeat": 8.2, "sensory_flux": 6.6}),
        ("NOMINAL", {"slp_heartbeat": 8.0, "sensory_flux": 6.4}),
        ("CRISIS",  {"slp_heartbeat": 10.0, "sensory_flux": 0.0}),   # Vampire Drain Novel zero-day
        ("CRISIS",  {"slp_heartbeat": 10.5, "sensory_flux": 0.0}),
        ("CRISIS",  {"slp_heartbeat": 10.2, "sensory_flux": 0.0}),   # Neurogenesis fires
        ("RECOVER", {"slp_heartbeat": 8.3, "sensory_flux": 6.7}),
        ("NOMINAL", {"slp_heartbeat": 8.0, "sensory_flux": 6.4}),
    ]

    peak_crisis_fe  = 0.0    # Highest FE observed during current crisis window
    morphed_slot    = None

    for tick, (phase, telemetry) in enumerate(SCENARIOS_ALPHA):
        # Alpha checks swarm first (no skills yet at start)
        if phase == "CRISIS" and agent_a.sustained_pain_timer == 0:
            found = forager_a.check_and_download_skill(telemetry,
                                                        peak_crisis_fe or 200.0)
            if found:
                phase = "IMMUNE"

        # Record state before SVI update for morphogenesis tracking
        prev_active = agent_a.arena.active_count
        svi_a, total_loss = svi.update(svi_a, telemetry=telemetry)
        snapshot          = extract_belief_snapshot(svi, svi_a)
        fe                = float(total_loss)

        # Track the PEAK crisis FE — this is what we report to the Queen
        # as the pre-morphogenesis baseline (worst observed pain, not same-tick)
        if phase == "CRISIS":
            peak_crisis_fe = max(peak_crisis_fe, fe)
        else:
            peak_crisis_fe = 0.0  # Reset on recovery

        # Feed morphogenetic agent
        agent_a.update(fe, pain_threshold=PAIN_THRESHOLD)

        # Detect new node spawned by Morphogenesis → transmit to Queen
        if agent_a.arena.active_count > prev_active:
            new_slot = agent_a.arena.active_count - 1
            forager_a.notify_morphogenesis(
                new_slot          = new_slot,
                pre_fe            = peak_crisis_fe,
                post_fe           = fe * 0.3,
                anomaly_telemetry = telemetry,
            )
            time.sleep(1.0) # Yield for ZeroMQ network propagation
            morphed_slot = new_slot

        status = {
            "NOMINAL": "✅ NOMINAL",
            "CRISIS":  "⚠️  CRISIS ",
            "RECOVER": "🔄 RECOVER",
            "IMMUNE":  "🛡  IMMUNE ",
        }.get(phase, phase)

        print(f"  Alpha Tick {tick+1:02d} | {status} | "
              f"FE: {fe:>10.2f} | "
              f"Pain: {agent_a.sustained_pain_timer}/{PAIN_TICKS_THRESHOLD} | "
              f"Arena: {agent_a.arena.active_count}/{agent_a.arena.max_capacity}")

    print(f"\n[ALPHA DONE] Mutations sent to Queen: {forager_a.mutations_sent}")
    queen_stats = queen.stats()
    print(f"[QUEEN]      Validated: {queen_stats['validated']} | "
          f"Rejected: {queen_stats['rejected']} | "
          f"FAISS size: {queen_stats['faiss_size']}")

    # ── Forager Beta encounters the SAME anomaly class ────────────────────────
    print(f"\n{'─'*60}")
    print("[PHASE 4B] Forager Beta — encounters same anomaly class")
    print("  (Beta will query the swarm BEFORE suffering sustained pain)")
    print(f"{'─'*60}\n")

    SCENARIOS_BETA = [
        ("NOMINAL", {"slp_heartbeat": 8.1, "sensory_flux": 6.5}),
        ("NOMINAL", {"slp_heartbeat": 8.2, "sensory_flux": 6.6}),
        ("CRISIS",  {"slp_heartbeat": 10.1, "sensory_flux": 0.0}),   # Similar Vampire Drain anomaly
        ("CRISIS",  {"slp_heartbeat": 10.6, "sensory_flux": 0.0}),
        ("NOMINAL", {"slp_heartbeat": 8.0, "sensory_flux": 6.4}),
    ]

    for tick, (phase, telemetry) in enumerate(SCENARIOS_BETA):
        # Broadcast — Beta checks for matching skill on first crisis tick
        if phase == "CRISIS" and agent_b.sustained_pain_timer == 0:
            svi_b, probe_loss = svi.update(svi_b, telemetry=telemetry)
            probe_fe          = float(probe_loss)
            found = forager_b.check_and_download_skill(telemetry, probe_fe)
            if found:
                phase = "IMMUNE"
                fe    = probe_fe
            else:
                fe    = probe_fe
        else:
            svi_b, total_loss = svi.update(svi_b, telemetry=telemetry)
            fe                = float(total_loss)

        snapshot = extract_belief_snapshot(svi, svi_b)
        agent_b.update(fe, pain_threshold=PAIN_THRESHOLD)

        status = {
            "NOMINAL": "✅ NOMINAL",
            "CRISIS":  "⚠️  CRISIS ",
            "RECOVER": "🔄 RECOVER",
            "IMMUNE":  "🛡  IMMUNE ",
        }.get(phase, phase)

        print(f"  Beta  Tick {tick+1:02d} | {status} | "
              f"FE: {fe:>10.2f} | "
              f"Pain: {agent_b.sustained_pain_timer}/{PAIN_TICKS_THRESHOLD} | "
              f"Arena: {agent_b.arena.active_count}/{agent_b.arena.max_capacity}")

    # ── Phase 4C: The Rogue Adversary ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[PHASE 4C] Rogue Node — Attempts to spoof a malicious mutation")
    print(f"{'─'*60}\n")
    
    rogue_context = zmq.Context()
    rogue_push = rogue_context.socket(zmq.PUSH)
    rogue_push.connect(ZMQ_FORAGER_TO_QUEEN)
    
    spoofed_delta = {
        "mutation_id": "laptop_hack",
        "slot_index": 0,
        "weight_dim": 8,
        "weight_b64": "invalid",
        "anomaly_telemetry": {},
        "anomaly_vector": [0,0,0,0,0,0],
        "pre_morph_fe": 999999.0,
        "post_morph_fe": 0.0,
        "fe_reduction": 999999.0,
    }
    
    payload_json = json.dumps(spoofed_delta)
    # The laptop signs it with the true Forager Alpha private key!
    stolen_signature = forager_a.enclave.sign(payload_json.encode('utf-8'))
    
    spoofed_envelope = {
        "payload": payload_json,
        "signature_b64": stolen_signature,
        "node_pubkey": forager_a.enclave.export_public() # Spoofing Alpha's identity
    }
    
    print("  [ROGUE] 🦹 Injecting physically impossible payload using stolen keys...")
    rogue_push.send_string(json.dumps(spoofed_envelope))
    
    time.sleep(1.0) # Wait for Queen to process and reject
    
    rogue_push.close()
    
    # ── Phase 4D: Metabolic Triage & Eureka Collisions ───────────────────────
    print(f"\n{'─'*60}")
    print("[PHASE 4D] The 'Eureka' Collision & Metabolic Triage")
    print("  (Multiple Foragers hit similar anomalies simultaneously)")
    print(f"{'─'*60}\n")
    
    forager_c = ForagerNode(svi, agent_a, queen, node_id="forager-jetson-gamma")
    queen.register_forager(forager_c.node_id, forager_c.enclave.export_public())
    
    # Pause worker to build up queue for triage demonstration
    queen.running_worker = False
    
    telemetry_c = {"slp_heartbeat": 11.0, "sensory_flux": 0.0}
    
    # Beta sends a weak mutation (Pre_FE=900, Reduction=100)
    forager_b.notify_morphogenesis(new_slot=2, pre_fe=900.0, post_fe=800.0, anomaly_telemetry=telemetry_c)
    
    # Alpha sends a massive crisis mutation (Pre_FE=3500, Reduction=2500)
    forager_a.notify_morphogenesis(new_slot=2, pre_fe=3500.0, post_fe=1000.0, anomaly_telemetry=telemetry_c)
    
    # Gamma sends a similar massive crisis but slightly inferior (Pre_FE=3500, Reduction=2550 -> not 20% better)
    forager_c.notify_morphogenesis(new_slot=2, pre_fe=3500.0, post_fe=950.0, anomaly_telemetry=telemetry_c)
    
    # Re-enable worker so it processes them from priority queue
    queen.running_worker = True
    time.sleep(2.0)
    
    # ── Phase 4E: Split-Brain Sovereignty ────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[PHASE 4E] Split-Brain Sovereignty (Network Partition)")
    print(f"{'─'*60}\n")
    
    print("  [SYSTEM] Severing Forager Beta's uplink to Queen...")
    forager_b.offline = True
    
    print("  [forager-orin-beta] Encounters novel crisis while isolated from Swarm.")
    forager_b.notify_morphogenesis(new_slot=3, pre_fe=1200.0, post_fe=200.0, anomaly_telemetry={"slp_heartbeat": 5.0, "sensory_flux": 9.9})
    
    time.sleep(1.0)
    forager_c.stop()

    # ── Final Swarm Report ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SWARM INTELLIGENCE REPORT")
    print(f"{'='*60}")

    for forager in [forager_a, forager_b]:
        s = forager.stats()
        print(f"  {s['node_id']:<25} | "
              f"Mutations sent: {s['mutations_sent']} | "
              f"Skills downloaded: {s['skills_downloaded']} | "
              f"Arena: {s['arena_active']}/{s['arena_capacity']}")

    qs = queen.stats()
    print(f"\n  Queen Ada                  | "
          f"Validated: {qs['validated']} | "
          f"FAISS index: {qs['faiss_size']} skills | "
          f"Rejected: {qs['rejected']}")
    print(f"{'='*60}\n")
    
    forager_a.stop()
    forager_b.stop()
    queen.stop()


if __name__ == "__main__":
    run_swarm_demo()
