"""
Synthetic Epistemic Engine — morphogenesis.py
Phase 2: Causal Morphogenesis Engine

The Latent Arena is the core innovation that allows the agent to grow new
"neurons" and prune dead ones WITHOUT triggering a JAX JIT recompile on
every structural change.

Key insight:
    Instead of resizing tensors (expensive, OOM risk), we pre-allocate a
    maximum-capacity weight matrix and control which rows are "alive" via a
    boolean mask. Neurogenesis flips a 0 → 1 in the mask. Apoptosis flips
    a 1 → 0. The tensor shape never changes. JAX never recompiles.

    A full JIT recompile only fires when the entire Latent Arena is exhausted
    and the max_capacity itself must grow — a rare, deliberate event.

Biological Analogy (from the manuscript):
    Just as a human confronted with an experience they cannot comprehend
    must grow new neural pathways to understand it, the agent that encounters
    sustained epistemic pain (high Free Energy) grows new structural capacity
    to represent what it previously could not.

    The agent does not avoid pain — it uses pain as the signal to expand.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DIM           = 8     # Dimensionality of each latent node's weight vector
MAX_ARENA_CAPACITY    = 32    # Hard ceiling before a full structural recompile
INITIAL_CAPACITY      = 4     # Active nodes at boot
PAIN_TICKS_THRESHOLD  = 3     # Sustained crisis ticks before neurogenesis fires
APOPTOSIS_MAGNITUDE   = 1e-4  # Weight L2 norm below this triggers pruning
MATURATION_TICKS      = 10    # Ticks a new node must survive before being eligible for quantization


# ─────────────────────────────────────────────────────────────────────────────
# Morphogenesis Event Log
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MorphogenesisEvent:
    """Record of a single structural change to the arena."""
    tick:        int
    event_type:  str          # "NEUROGENESIS" | "APOPTOSIS" | "QUANTIZATION"
    node_index:  int
    reason:      str
    free_energy: float
    active_before: int
    active_after:  int


# ─────────────────────────────────────────────────────────────────────────────
# Latent Arena
# ─────────────────────────────────────────────────────────────────────────────

class LatentArena:
    """
    A pre-allocated weight matrix with a boolean activation mask.

    Shape: (MAX_ARENA_CAPACITY, FEATURE_DIM)
    Mask:  (MAX_ARENA_CAPACITY,)  — 1.0 = active neuron, 0.0 = dormant slot

    The effective computation at each tick is:
        effective_weights = weights * mask[:, None]

    This keeps the tensor shape fixed across all neurogenesis/apoptosis events.
    JAX traces on shape, not values, so the compiled graph never needs updating
    when we flip bits in the mask.
    """

    def __init__(self, max_capacity: int = MAX_ARENA_CAPACITY,
                 initial_capacity: int = INITIAL_CAPACITY,
                 feature_dim: int = FEATURE_DIM):
        self.max_capacity     = max_capacity
        self.feature_dim      = feature_dim
        self.rng              = jax.random.PRNGKey(42)

        # Initialise weights with small random values (Xavier-style)
        self.weights = jax.random.normal(
            self.rng, shape=(max_capacity, feature_dim)
        ) * jnp.sqrt(2.0 / feature_dim)

        # Boolean mask — 1.0 = alive, 0.0 = dormant
        mask_values = ([1.0] * initial_capacity +
                       [0.0] * (max_capacity - initial_capacity))
        self.mask = jnp.array(mask_values)

        # Track how many ticks each node has been alive (for maturation checks)
        self.node_age = np.zeros(max_capacity, dtype=np.int32)

    @property
    def active_count(self) -> int:
        return int(jnp.sum(self.mask))

    @property
    def dormant_count(self) -> int:
        return self.max_capacity - self.active_count

    @property
    def effective_weights(self) -> jnp.ndarray:
        """Masked weight matrix — only active rows contribute to computation."""
        return self.weights * self.mask[:, None]

    @property
    def capacity_used_pct(self) -> float:
        return (self.active_count / self.max_capacity) * 100.0

    def next_dormant_slot(self) -> Optional[int]:
        """Return the index of the first dormant slot, or None if arena is full."""
        dormant_indices = jnp.where(self.mask == 0.0, size=self.max_capacity)[0]
        if len(dormant_indices) == 0:
            return None
        return int(dormant_indices[0])

    def activate_node(self, index: int, rng_key: jnp.ndarray):
        """
        Neurogenesis: flip a dormant slot to active.
        Initialise its weights with small noise so it starts learning fresh.
        """
        fresh_weights = jax.random.normal(rng_key, shape=(self.feature_dim,)) * 0.01
        self.weights  = self.weights.at[index].set(fresh_weights)
        self.mask     = self.mask.at[index].set(1.0)
        self.node_age[index] = 0

    def deactivate_node(self, index: int):
        """
        Apoptosis: flip an active node to dormant, zeroing its weights.
        This reclaims its slot for future neurogenesis.
        """
        self.weights  = self.weights.at[index].set(jnp.zeros(self.feature_dim))
        self.mask     = self.mask.at[index].set(0.0)
        self.node_age[index] = 0

    def increment_ages(self):
        """Age all currently active nodes by one tick."""
        active_mask = np.array(self.mask, dtype=np.int32)
        self.node_age += active_mask

    def node_weight_norm(self, index: int) -> float:
        """L2 norm of a node's weight vector — proxy for its contribution."""
        return float(jnp.linalg.norm(self.weights[index]))

    def mature_nodes(self, min_age: int = MATURATION_TICKS) -> List[int]:
        """Return indices of active nodes that have reached maturation age."""
        return [
            i for i in range(self.max_capacity)
            if self.mask[i] == 1.0 and self.node_age[i] >= min_age
        ]

    def __repr__(self) -> str:
        bars = "".join("█" if self.mask[i] == 1.0 else "░"
                       for i in range(self.max_capacity))
        return (f"LatentArena [{bars}] "
                f"{self.active_count}/{self.max_capacity} active "
                f"({self.capacity_used_pct:.0f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Morphogenetic Agent
# ─────────────────────────────────────────────────────────────────────────────

class MorphogeneticAgent:
    """
    Wraps the LatentArena with the full neurogenesis / apoptosis lifecycle.

    This is the "growth brain" that sits above the SVI loop. On each tick
    it receives the current Free Energy from engine_core.py and decides
    whether to grow a new dimension or prune a dead one.

    Three-stage lifecycle for each node:
        1. SPAWN    — Activated by sustained pain. Weights initialised near zero.
        2. MATURE   — Node survives for MATURATION_TICKS ticks, weights grow.
        3. QUANTIZE — Mature node's weights compressed (simulated here; full
                      INT8 quantization in Phase 6 on Jetson hardware).

    Apoptosis fires independently of neurogenesis:
        Any active node whose weight L2 norm falls below APOPTOSIS_MAGNITUDE
        is pruned, reclaiming its slot.
    """

    def __init__(self, max_capacity: int = MAX_ARENA_CAPACITY,
                 initial_capacity: int = INITIAL_CAPACITY):
        self.arena                  = LatentArena(max_capacity, initial_capacity)
        self.sustained_pain_timer   = 0
        self.rng_key                = jax.random.PRNGKey(int(time.time()))
        self.event_log: List[MorphogenesisEvent] = []
        self.tick_count             = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Core Update
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, free_energy: float, pain_threshold: float = 200.0):
        """
        Called once per heartbeat tick. Receives current Free Energy.

        Decision tree:
            1. Accumulate or decay the sustained pain timer
            2. If timer exceeds PAIN_TICKS_THRESHOLD → neurogenesis()
            3. Run apoptosis check on all active nodes
            4. Age all nodes
        """
        self.tick_count += 1

        in_crisis = free_energy > pain_threshold

        # ── Pain Timer ───────────────────────────────────────────────────────
        if in_crisis:
            self.sustained_pain_timer += 1
        else:
            # Decay pain timer when not in crisis (not instant relief)
            self.sustained_pain_timer = max(0, self.sustained_pain_timer - 1)

        # ── Neurogenesis ─────────────────────────────────────────────────────
        if self.sustained_pain_timer >= PAIN_TICKS_THRESHOLD:
            self._neurogenesis(free_energy)

        # ── Apoptosis ────────────────────────────────────────────────────────
        self._apoptosis_sweep(free_energy)

        # ── Maturation & Quantization ────────────────────────────────────────
        self._maturation_check()

        # ── Age all active nodes ─────────────────────────────────────────────
        self.arena.increment_ages()

    # ──────────────────────────────────────────────────────────────────────────
    # Neurogenesis
    # ──────────────────────────────────────────────────────────────────────────

    def _neurogenesis(self, free_energy: float):
        """
        Activate the next dormant slot in the Latent Arena.

        This is structurally equivalent to a biological brain growing a new
        neural pathway to comprehend an experience it currently cannot process.
        The agent "gives itself a new neuron" to attempt to explain the anomaly.
        """
        slot = self.arena.next_dormant_slot()

        if slot is None:
            print(f"  [MORPHOGENESIS] ⚠️  Arena at max capacity ({self.arena.max_capacity})!")
            print(f"  [MORPHOGENESIS] Hard structural expansion required (full JIT recompile).")
            # In Phase 6 on real hardware: expand MAX_ARENA_CAPACITY and rebuild.
            return

        # Fresh random key for weight initialisation
        self.rng_key, subkey = jax.random.split(self.rng_key)
        active_before = self.arena.active_count

        self.arena.activate_node(slot, subkey)
        self.sustained_pain_timer = 0   # Reset — the agent has taken action

        event = MorphogenesisEvent(
            tick          = self.tick_count,
            event_type    = "NEUROGENESIS",
            node_index    = slot,
            reason        = f"Sustained pain for {PAIN_TICKS_THRESHOLD} ticks | FE={free_energy:.1f}",
            free_energy   = free_energy,
            active_before = active_before,
            active_after  = self.arena.active_count,
        )
        self.event_log.append(event)

        print(f"\n  ┌{'─'*56}┐")
        print(f"  │  🧠 NEUROGENESIS — Tick {self.tick_count:<31}│")
        print(f"  │  New node spawned at slot {slot:<30}│")
        print(f"  │  Active: {active_before} → {self.arena.active_count} of {self.arena.max_capacity:<25}│")
        print(f"  │  {self.arena!r:<54}│")
        print(f"  └{'─'*56}┘\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Apoptosis
    # ──────────────────────────────────────────────────────────────────────────

    def _apoptosis_sweep(self, free_energy: float):
        """
        Scan all active nodes. Prune any whose weight vector has decayed to near
        zero — they are contributing nothing and wasting arena capacity.

        Analogous to synaptic pruning in the biological brain: connections that
        fire together, wire together; connections that don't fire get cut.
        """
        for i in range(self.arena.max_capacity):
            if self.arena.mask[i] != 1.0:
                continue
            if self.arena.node_age[i] < MATURATION_TICKS:
                continue   # Give young nodes time to grow before pruning

            norm = self.arena.node_weight_norm(i)
            if norm < APOPTOSIS_MAGNITUDE:
                active_before = self.arena.active_count
                self.arena.deactivate_node(i)

                event = MorphogenesisEvent(
                    tick          = self.tick_count,
                    event_type    = "APOPTOSIS",
                    node_index    = i,
                    reason        = f"Weight norm {norm:.2e} < threshold {APOPTOSIS_MAGNITUDE:.2e}",
                    free_energy   = free_energy,
                    active_before = active_before,
                    active_after  = self.arena.active_count,
                )
                self.event_log.append(event)
                print(f"  [APOPTOSIS] ✂️  Node {i} pruned (norm={norm:.2e}). "
                      f"Arena: {active_before} → {self.arena.active_count}")

    # ──────────────────────────────────────────────────────────────────────────
    # Maturation & Quantization
    # ──────────────────────────────────────────────────────────────────────────

    def _maturation_check(self):
        """
        Mature nodes are stable — their weights have stopped shifting wildly.
        In Phase 6 on Jetson hardware, we trigger INT8 quantization here to
        permanently reclaim VRAM while preserving the learned representation.

        Simulated here as a log entry; real quantization requires bitsandbytes
        or TensorRT integration (Phase 6).
        """
        for i in self.arena.mature_nodes():
            norm = self.arena.node_weight_norm(i)
            if norm > APOPTOSIS_MAGNITUDE:
                # Node is mature and contributing — mark for quantization
                event = MorphogenesisEvent(
                    tick          = self.tick_count,
                    event_type    = "QUANTIZATION",
                    node_index    = i,
                    reason        = f"Node matured at age {self.arena.node_age[i]}",
                    free_energy   = 0.0,
                    active_before = self.arena.active_count,
                    active_after  = self.arena.active_count,
                )
                # Only log once per maturation milestone
                already_logged = any(
                    e.event_type == "QUANTIZATION" and e.node_index == i
                    for e in self.event_log
                )
                if not already_logged:
                    self.event_log.append(event)
                    print(f"  [MATURATION] 📦 Node {i} mature "
                          f"(age={self.arena.node_age[i]}). "
                          f"Queued for INT8 quantization.")

    # ──────────────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────────────

    def print_event_log(self):
        print("\n" + "=" * 60)
        print("  MORPHOGENESIS EVENT LOG")
        print("=" * 60)
        if not self.event_log:
            print("  No structural events recorded.")
            return
        for e in self.event_log:
            icon = {"NEUROGENESIS": "🧠", "APOPTOSIS": "✂️ ",
                    "QUANTIZATION": "📦"}.get(e.event_type, "?")
            print(f"  Tick {e.tick:02d} | {icon} {e.event_type:<14} | "
                  f"Node {e.node_index:02d} | "
                  f"Active: {e.active_before} → {e.active_after} | "
                  f"{e.reason}")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from engine_core import initialize_engine, extract_belief_snapshot, PAIN_THRESHOLD

    print("=" * 60)
    print("  SYNTHETIC EPISTEMIC ENGINE — Phase 2")
    print("  Causal Morphogenesis Engine")
    print("=" * 60)

    # Boot SVI engine
    svi, guide = initialize_engine(beta=1.5)
    rng_key    = jax.random.PRNGKey(0)
    svi_state  = svi.init(rng_key, telemetry={"temp": 45.0, "vram_usage": 22.5})

    # Boot Morphogenetic Agent
    agent = MorphogeneticAgent(max_capacity=MAX_ARENA_CAPACITY,
                               initial_capacity=INITIAL_CAPACITY)

    print(f"\n[BOOT] Arena initialised: {agent.arena}\n")

    # Simulate 15 ticks: normal → crisis → morphogenesis → recovery
    scenarios = {
        0:  ("NOMINAL",  {"temp": 46.0, "vram_usage": 23.1}),
        1:  ("NOMINAL",  {"temp": 46.2, "vram_usage": 23.3}),
        2:  ("NOMINAL",  {"temp": 46.0, "vram_usage": 23.0}),
        3:  ("NOMINAL",  {"temp": 46.5, "vram_usage": 23.4}),
        4:  ("NOMINAL",  {"temp": 46.1, "vram_usage": 23.2}),
        5:  ("CRISIS",   {"temp": 62.0, "vram_usage": 41.0}),   # Anomaly injected
        6:  ("CRISIS",   {"temp": 64.0, "vram_usage": 43.0}),
        7:  ("CRISIS",   {"temp": 63.5, "vram_usage": 42.5}),   # Neurogenesis fires here
        8:  ("CRISIS",   {"temp": 63.0, "vram_usage": 42.0}),
        9:  ("RECOVERY", {"temp": 47.0, "vram_usage": 24.0}),
        10: ("RECOVERY", {"temp": 46.5, "vram_usage": 23.5}),
        11: ("NOMINAL",  {"temp": 46.0, "vram_usage": 23.0}),
        12: ("NOMINAL",  {"temp": 46.2, "vram_usage": 23.1}),
        13: ("NOMINAL",  {"temp": 46.0, "vram_usage": 23.0}),
        14: ("NOMINAL",  {"temp": 46.0, "vram_usage": 23.0}),
    }

    for tick in range(15):
        phase, telemetry = scenarios[tick]

        svi_state, total_loss = svi.update(svi_state, telemetry=telemetry)
        snapshot              = extract_belief_snapshot(svi, svi_state)
        free_energy           = float(total_loss)
        eig                   = snapshot["eig"]
        thermal_mu            = snapshot["beliefs"]["true_thermal"]["mu"]

        # Feed Free Energy to the Morphogenetic Agent
        agent.update(free_energy, pain_threshold=PAIN_THRESHOLD)

        # Print status
        pain_bar = f"Pain Timer: {agent.sustained_pain_timer}/{PAIN_TICKS_THRESHOLD}"
        print(
            f"Tick {tick+1:02d} [{phase:<8}] | "
            f"FE: {free_energy:>10.2f} | "
            f"KL: {eig:.4f} | "
            f"Thermal: {thermal_mu:.2f}°C | "
            f"{pain_bar} | "
            f"Arena: {agent.arena.active_count}/{agent.arena.max_capacity}"
        )
        time.sleep(0.05)

    agent.print_event_log()
    print(f"\n[DONE] Final arena state: {agent.arena}")
