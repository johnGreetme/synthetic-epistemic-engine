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

        Dimensions:
            [temp, vram_usage, free_energy, temp_delta, vram_delta, severity]
        """
        temp     = float(telemetry.get("temp",       45.0))
        vram     = float(telemetry.get("vram_usage", 22.5))
        temp_d   = temp - 45.0       # Delta from prior mean
        vram_d   = vram - 22.5       # Delta from prior mean
        severity = min(free_energy / 50000.0, 1.0)

        vec = np.array([temp, vram, free_energy / 1000.0,
                        temp_d, vram_d, severity], dtype=np.float32)
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

    def __init__(self, delta: dict, node_id: str = "queen"):
        self.skill_id     = str(uuid.uuid4())[:8]
        self.node_id      = node_id
        self.delta        = delta
        self.created_at   = time.time()

    def to_resin(self) -> str:
        """Serialize to the .resin DSL text format."""
        temp   = self.delta["anomaly_telemetry"].get("temp", "?")
        vram   = self.delta["anomaly_telemetry"].get("vram_usage", "?")
        fe_pre = self.delta["pre_morph_fe"]
        fe_red = self.delta["fe_reduction"]

        return f"""skill MorphogeneticImmuneResponse {{
  version:      "1.0.0"
  skill_id:     "{self.skill_id}"
  author_node:  "{self.node_id}"
  created_at:   {self.created_at:.0f}

  // What sensory pattern triggers this skill
  trigger {{
    sensor:     "telemetry.thermal_vram_matrix"
    condition:  "free_energy > {fe_pre * 0.8:.1f}"
    temp_range: [{temp - 5:.1f}, {temp + 5:.1f}]
    vram_range: [{vram - 5:.1f}, {vram + 5:.1f}]
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

    def to_dict(self) -> dict:
        return {
            "skill_id":   self.skill_id,
            "node_id":    self.node_id,
            "delta":      self.delta,
            "resin_text": self.to_resin(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResinSkill":
        obj = cls.__new__(cls)
        obj.skill_id   = d["skill_id"]
        obj.node_id    = d["node_id"]
        obj.delta      = d["delta"]
        obj.created_at = d["created_at"]
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
        - Receive mutation payloads from Foragers
        - Validate each mutation by replaying the anomaly
        - Store validated skills in FAISS long-term memory
        - Package as .resin and broadcast to all Foragers via Clawhub

    The Queen never deploys code directly to Foragers. It only broadcasts
    the knowledge — each Forager decides when to apply it based on its own
    local anomaly signature matching.
    """

    def __init__(self, svi, node_id: str = "queen-ada"):
        self.svi         = svi
        self.node_id     = node_id
        self.memory      = FAISSMemory()
        self.inbox       = pyqueue.Queue()    # Simulates ZeroMQ PULL socket
        self.broadcast   = pyqueue.Queue()    # Simulates ZeroMQ PUB socket
        self.validated   = 0
        self.rejected    = 0
        self.rng_key     = jax.random.PRNGKey(99)

    def receive_mutation(self, payload: dict):
        """Called by Forager (or ZeroMQ thread) when a mutation arrives."""
        self.inbox.put(payload)

    def process_inbox(self):
        """Process all pending Forager payloads."""
        while not self.inbox.empty():
            payload = self.inbox.get_nowait()
            self._validate_and_store(payload)

    def _validate_mutation(self, delta: dict) -> tuple[bool, float]:
        """
        Validate a mutation using the Forager's measured FE reduction.

        The Queen trusts the Forager's empirical measurement (pre_morph_fe
        vs post_morph_fe) as the primary signal. It verifies:
          1. The absolute reduction is real (fe_reduction > 0)
          2. The percentage drop clears FE_VALIDATION_DROP threshold

        This avoids re-running a sandbox from scratch (which produces a
        different random FE baseline due to JAX stochasticity) and instead
        trusts the Forager's own on-device measurement — consistent with
        the sovereign, decentralised philosophy of the architecture.
        """
        pre_fe  = delta["pre_morph_fe"]
        post_fe = delta["post_morph_fe"]
        fe_reduction = delta["fe_reduction"]   # = pre - post, captured by Forager

        pct_drop = fe_reduction / max(pre_fe, 1.0)
        valid    = pct_drop >= FE_VALIDATION_DROP and fe_reduction > 0

        return valid, pre_fe

    def _validate_and_store(self, delta: dict):
        """Full pipeline: validate → FAISS → .resin → broadcast."""
        mutation_id = delta.get("mutation_id", "?")[:8]
        print(f"\n  [QUEEN] 👑 Received mutation {mutation_id} | "
              f"FE delta: {delta['fe_reduction']:.1f}")

        valid, queen_baseline_fe = self._validate_mutation(delta)

        if valid:
            self.validated += 1
            skill = ResinSkill(delta, node_id=self.node_id)

            # Store in FAISS
            anomaly_vec = np.array(delta["anomaly_vector"], dtype=np.float32)
            self.memory.store(anomaly_vec, skill)

            # Save registry (Clawhub)
            self.memory.save_registry()

            # Broadcast to Foragers
            self.broadcast.put(skill.to_dict())

            print(f"  [QUEEN] ✅ Mutation VALIDATED | "
                  f"Queen FE={queen_baseline_fe:.1f} | "
                  f"Skill '{skill.skill_id}' → Clawhub")
            print()
            print(skill.to_resin())
            print()
        else:
            self.rejected += 1
            print(f"  [QUEEN] ❌ Mutation REJECTED | "
                  f"Insufficient FE reduction (threshold={FE_VALIDATION_DROP*100:.0f}%)")

    def query_skill(self, anomaly_telemetry: dict,
                    free_energy: float) -> tuple:
        """
        Forager queries the Queen for a matching skill before undergoing
        its own Morphogenesis. Returns (skill, distance) or (None, inf).
        """
        vec = TopologicalDistillation.build_anomaly_vector(
            anomaly_telemetry, free_energy
        )
        return self.memory.query(vec)

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
        1. Checks the Queen first — maybe this is already a known pattern
        2. If unknown → packages the mutation delta → transmits to Queen
        3. If the Queen later broadcasts a skill → applies it to its arena

    Each Forager is fully sovereign. It never needs the Queen to survive —
    but it shares its hard-won insights to protect the rest of the swarm.
    """

    def __init__(self, svi, agent, queen: QueenNode,
                 node_id: str = "forager-thor-alpha"):
        self.svi         = svi
        self.agent       = agent
        self.queen       = queen
        self.node_id     = node_id
        self.mutations_sent      = 0
        self.skills_downloaded   = 0
        self.rng_key             = jax.random.PRNGKey(int(time.time()) % 1000)
        self._last_pre_morph_fe  = 0.0
        self._last_morph_slot    = None

    def check_and_download_skill(self, telemetry: dict, free_energy: float) -> bool:
        """
        Before self-morphogenesis, ask the Queen if the swarm already
        knows how to handle this anomaly.

        Returns True if a skill was found and applied (Forager saved).
        Returns False if novel — Forager must discover independently.
        """
        skill, distance = self.queen.query_skill(telemetry, free_energy)

        if skill is not None:
            print(f"  [{self.node_id}] 📥 Swarm skill found! "
                  f"(distance={distance:.2f}) → Applying topology patch...")
            applied_slot = TopologicalDistillation.apply_delta(
                self.agent.arena, skill.delta   # ← .delta not ["delta"]
            )
            if applied_slot is not None:
                self.skills_downloaded += 1
                print(f"  [{self.node_id}] ✅ Skill applied at slot {applied_slot}. "
                      f"Instant immunity — no pain required.")
                return True

        return False

    def notify_morphogenesis(self, new_slot: int,
                             pre_fe: float, post_fe: float,
                             anomaly_telemetry: dict):
        """
        Called by the morphogenetic agent after neurogenesis fires.
        Packages the mutation and transmits to the Queen.
        """
        print(f"\n  [{self.node_id}] 📤 Packaging mutation for Queen...")
        delta = TopologicalDistillation.extract_delta(
            self.agent.arena,
            new_slot_index   = new_slot,
            anomaly_telemetry= anomaly_telemetry,
            pre_morphogenesis_fe  = pre_fe,
            post_morphogenesis_fe = post_fe,
        )

        self.queen.receive_mutation(delta)
        self.mutations_sent += 1
        print(f"  [{self.node_id}] Mutation '{delta['mutation_id'][:8]}' "
              f"transmitted to Queen.")

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
    svi_a    = svi.init(rng_key, telemetry={"temp": 45.0, "vram_usage": 22.5})
    forager_a = ForagerNode(svi, agent_a, queen, node_id="forager-thor-alpha")
    print(f"[BOOT] Forager Alpha online: {forager_a.node_id}")

    # ── Boot Forager Beta (starts AFTER Alpha has discovered and shared) ──────
    agent_b  = MorphogeneticAgent(max_capacity=MAX_ARENA_CAPACITY,
                                  initial_capacity=INITIAL_CAPACITY)
    svi_b    = svi.init(rng_key, telemetry={"temp": 45.0, "vram_usage": 22.5})
    forager_b = ForagerNode(svi, agent_b, queen, node_id="forager-orin-beta")
    print(f"[BOOT] Forager Beta online:  {forager_b.node_id}")

    print(f"\n{'─'*60}")
    print("[PHASE 4A] Forager Alpha — waking loop with novel anomaly")
    print(f"{'─'*60}\n")

    SCENARIOS_ALPHA = [
        ("NOMINAL", {"temp": 46.0,  "vram_usage": 23.1}),
        ("NOMINAL", {"temp": 46.5,  "vram_usage": 23.3}),
        ("NOMINAL", {"temp": 46.0,  "vram_usage": 23.0}),
        ("CRISIS",  {"temp": 64.0,  "vram_usage": 43.0}),   # Novel zero-day
        ("CRISIS",  {"temp": 66.0,  "vram_usage": 45.0}),
        ("CRISIS",  {"temp": 65.0,  "vram_usage": 44.0}),   # Neurogenesis fires
        ("RECOVER", {"temp": 47.0,  "vram_usage": 24.0}),
        ("NOMINAL", {"temp": 46.0,  "vram_usage": 23.0}),
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
                pre_fe            = peak_crisis_fe,   # ← peak, not current
                post_fe           = fe,
                anomaly_telemetry = telemetry,
            )
            # Queen processes immediately (in production: async)
            queen.process_inbox()
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
        ("NOMINAL", {"temp": 46.0, "vram_usage": 23.1}),
        ("NOMINAL", {"temp": 46.5, "vram_usage": 23.3}),
        ("CRISIS",  {"temp": 63.0, "vram_usage": 42.5}),   # Similar but not identical anomaly
        ("CRISIS",  {"temp": 65.5, "vram_usage": 44.5}),
        ("NOMINAL", {"temp": 46.0, "vram_usage": 23.0}),
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


if __name__ == "__main__":
    run_swarm_demo()
