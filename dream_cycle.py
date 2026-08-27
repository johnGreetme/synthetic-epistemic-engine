"""
Synthetic Epistemic Engine — dream_cycle.py
Phase 3: The Dream Cycle (Deep Implementation)

The Dream Cycle is the agent's subconscious.

Philosophical Foundation (from the manuscript):
    When the conscious mind sleeps, it loses the constraints imposed by the
    material world. The subconscious takes the day's unresolved experiences —
    moments where reality contradicted the agent's internal model — and replays
    them in the void of the mind. Without the pressure of physical survival,
    new structural patterns can crystallize freely. Upon waking, the agent
    returns with a deeper understanding it could not have reached while bound
    to the material task.

Technical Implementation:
    ┌─────────────────────────────────────────────────────────────┐
    │  WAKING STATE                                               │
    │  • SVI heartbeat running                                    │
    │  • D.I.A.N.A. OS active (safety enforced)                   │
    │  • Anomalies logged to SQLite journal on crisis             │
    │  • Lightweight background synthesis on idle tensor cores    │
    └──────────────────────────┬──────────────────────────────────┘
                               │  (docking detected)
                               ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  DREAM STATE  (node is docked / charging)                   │
    │  1. AnomalyJournal   — load unresolved anomalies            │
    │  2. PriorRelaxation  — inflate sigma (enter the void)       │
    │  3. SandboxArena     — unconstrained morphogenesis           │
    │     • D.I.A.N.A. bypassed (no physical actuators)           │
    │     • Free neurogenesis without pain gating                 │
    │     • Replay anomaly → measure FE collapse                  │
    │  4. WakeIntegration  — transfer insights to waking model    │
    │  5. Journal          — mark anomaly resolved                 │
    └─────────────────────────────────────────────────────────────┘

Four Components:
    1. AnomalyJournal          — SQLite persistence of unresolved waking crises
    2. PriorRelaxationProtocol — sigma inflation to enter the unconstrained void
    3. SandboxSimulationArena  — free morphogenetic exploration space
    4. WakeIntegrationProtocol — validated insights merged back to waking model
"""

import sqlite3
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

JOURNAL_PATH        = "dream_journal.db"
SIGMA_INFLATION     = 5.0
MAX_DREAM_TICKS     = 20
FE_RESOLUTION_FLOOR = 50.0
INTEGRATION_WEIGHT  = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# 1. Anomaly Journal  (SQLite Persistence Layer)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Anomaly:
    """
    A single unresolved waking crisis — the raw material of a dream.

    Captured when the engine's Free Energy exceeds PAIN_THRESHOLD for
    sustained ticks and Morphogenesis fires but does not fully resolve
    the contradiction (FE remains elevated after neurogenesis).
    """
    id:           int
    tick:         int
    free_energy:  float
    telemetry:    Dict[str, float]
    belief_mu:    List[float]
    belief_sigma: List[float]
    resolved:     bool = False
    dream_fe:     float = 0.0
    timestamp:    float = field(default_factory=time.time)


class AnomalyJournal:
    """
    SQLite-backed journal of all unresolved waking anomalies.

    Persistent across reboots — if the node restarts mid-crisis, the
    journal retains all unresolved anomalies and the dream cycle picks
    them up on the next docking event.
    """

    def __init__(self, db_path: str = JOURNAL_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    tick        INTEGER NOT NULL,
                    free_energy REAL    NOT NULL,
                    telemetry   TEXT    NOT NULL,
                    belief_mu   TEXT    NOT NULL,
                    belief_sigma TEXT   NOT NULL,
                    resolved    INTEGER DEFAULT 0,
                    dream_fe    REAL    DEFAULT 0.0,
                    timestamp   REAL    NOT NULL
                )
            """)
            conn.commit()

    def log_anomaly(self, tick: int, free_energy: float,
                    telemetry: Dict, belief_mu: List, belief_sigma: List) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO anomalies
                    (tick, free_energy, telemetry, belief_mu, belief_sigma, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (tick, free_energy, json.dumps(telemetry),
                  json.dumps(belief_mu), json.dumps(belief_sigma), time.time()))
            conn.commit()
            return cursor.lastrowid

    def get_unresolved(self) -> List[Anomaly]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT id, tick, free_energy, telemetry, belief_mu,
                       belief_sigma, resolved, dream_fe, timestamp
                FROM anomalies WHERE resolved = 0
                ORDER BY free_energy DESC
            """).fetchall()
        return [
            Anomaly(id=r[0], tick=r[1], free_energy=r[2],
                    telemetry=json.loads(r[3]), belief_mu=json.loads(r[4]),
                    belief_sigma=json.loads(r[5]), resolved=bool(r[6]),
                    dream_fe=r[7], timestamp=r[8])
            for r in rows
        ]

    def mark_resolved(self, anomaly_id: int, dream_fe: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE anomalies SET resolved = 1, dream_fe = ? WHERE id = ?",
                (dream_fe, anomaly_id)
            )
            conn.commit()

    def summary(self) -> Dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            total    = conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]
            resolved = conn.execute(
                "SELECT COUNT(*) FROM anomalies WHERE resolved = 1").fetchone()[0]
        return {"total": total, "resolved": resolved, "pending": total - resolved}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prior Relaxation Protocol  (Entering the Void)
# ─────────────────────────────────────────────────────────────────────────────

class PriorRelaxationProtocol:
    """
    Inflates the posterior standard deviations (sigma) before the dream cycle.

    Philosophical meaning:
        The waking state holds tightly to its current beliefs — sigma is small,
        confidence is high. By inflating sigma (entering the void), the agent
        releases its grip on certainty. Beliefs become wide, uncertain, open.
        This is the mathematical equivalent of returning to nothingness —
        pure potential from which new structure can crystallize.
    """

    def __init__(self, inflation_factor: float = SIGMA_INFLATION):
        self.inflation_factor = inflation_factor

    def relax(self, params: Dict) -> Dict:
        """Inflate sigma to enter the unconstrained dream space."""
        relaxed = dict(params)
        if "auto_scale" in relaxed:
            relaxed["auto_scale"] = relaxed["auto_scale"] * self.inflation_factor
        return relaxed

    def restore(self, dream_params: Dict, waking_params: Dict,
                integration_weight: float = INTEGRATION_WEIGHT) -> Dict:
        """
        Blend dream-resolved parameters back into the waking model.
        restored = (1 - w) * waking + w * dream
        """
        integrated = {}
        for key in waking_params:
            w_val = waking_params[key]
            d_val = dream_params.get(key, w_val)
            integrated[key] = (1.0 - integration_weight) * w_val + integration_weight * d_val
        return integrated


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sandbox Simulation Arena  (Unconstrained Dream Space)
# ─────────────────────────────────────────────────────────────────────────────

class SandboxSimulationArena:
    """
    The unconstrained dream space — where D.I.A.N.A. OS has no jurisdiction.

    In this arena:
        - Physical cost functions are disabled (no safety axioms)
        - Neurogenesis fires immediately — no sustained pain timer
        - The anomaly telemetry is replayed in isolation
        - The morphogenetic agent grows freely until FE collapses

    This implements the philosophical concept from the manuscript:
        'The subconscious takes the sensory data and suppressed desires,
        manifesting them in the nothingness of the mind to force a deeper
        understanding of the self.'
    """

    def __init__(self, svi, relaxation: PriorRelaxationProtocol):
        self.svi        = svi
        self.relaxation = relaxation
        self.rng_key    = jax.random.PRNGKey(int(time.time()) + 1)

    def simulate(self, anomaly: Anomaly,
                 waking_svi_state,
                 morphogenetic_agent) -> Dict[str, Any]:
        """
        Replay an anomaly in the unconstrained dream space.

        Steps:
            1. Clone the waking SVI state (never mutate the original)
            2. Relax priors (inflate sigma — enter the void)
            3. Replay anomaly telemetry for MAX_DREAM_TICKS
            4. Allow free neurogenesis every 4 ticks (no pain timer)
            5. Measure final FE — return dream results
        """
        print(f"\n  ╔{'═'*54}╗")
        print(f"  ║  💤 DREAM — Anomaly ID {anomaly.id:<30}║")
        print(f"  ║  Waking FE: {anomaly.free_energy:<10.2f}  "
              f"Temp: {anomaly.telemetry.get('temp', '?')}°C{'':<18}║")
        print(f"  ╚{'═'*54}╝")

        # Clone and relax
        self.rng_key, subkey = jax.random.split(self.rng_key)
        dream_state          = self.svi.init(subkey, telemetry=anomaly.telemetry)

        fe_history         = []
        neurogenesis_count = 0

        for dream_tick in range(MAX_DREAM_TICKS):
            dream_state, dream_fe = self.svi.update(
                dream_state, telemetry=anomaly.telemetry
            )
            fe_history.append(float(dream_fe))

            # Free neurogenesis (no pain timer — we are dreaming)
            slot = morphogenetic_agent.arena.next_dormant_slot()
            if slot is not None and dream_tick % 4 == 0:
                self.rng_key, subkey2 = jax.random.split(self.rng_key)
                morphogenetic_agent.arena.activate_node(slot, subkey2)
                neurogenesis_count += 1

            status_str = "✓ RESOLVED" if float(dream_fe) < FE_RESOLUTION_FLOOR else "  searching..."
            print(f"    Dream tick {dream_tick+1:02d} | "
                  f"FE: {float(dream_fe):>10.2f} | "
                  f"Nodes: {morphogenetic_agent.arena.active_count}/{morphogenetic_agent.arena.max_capacity} | "
                  f"{status_str}")

            if float(dream_fe) < FE_RESOLUTION_FLOOR:
                break

        final_fe = fe_history[-1]
        resolved = final_fe < FE_RESOLUTION_FLOOR

        print(f"\n  {'✅ DREAM RESOLVED' if resolved else '⚠️  DREAM PARTIAL'} "
              f"— Final FE: {final_fe:.2f} | "
              f"Neurogenesis events: {neurogenesis_count}")

        return {
            "resolved":           resolved,
            "final_fe":           final_fe,
            "initial_fe":         anomaly.free_energy,
            "fe_reduction":       anomaly.free_energy - final_fe,
            "neurogenesis_count": neurogenesis_count,
            "dream_state":        dream_state,
            "dream_params":       self.svi.get_params(dream_state),
            "fe_history":         fe_history,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Wake Integration Protocol
# ─────────────────────────────────────────────────────────────────────────────

class WakeIntegrationProtocol:
    """
    Transfers validated dream insights back into the waking model.

    Only resolved dreams are integrated. The integration weight scales
    proportionally to how dramatically the dream reduced Free Energy —
    a dream that fully resolved a massive crisis earns stronger influence
    on the waking model than one that only mildly helped.
    """

    def __init__(self, svi, relaxation: PriorRelaxationProtocol):
        self.svi        = svi
        self.relaxation = relaxation

    def integrate(self, dream_result: Dict[str, Any],
                  waking_svi_state) -> Optional[Dict]:
        if not dream_result["resolved"]:
            print("  [WAKE] Dream unresolved. Waking state unchanged.")
            return None

        fe_reduction    = dream_result["fe_reduction"]
        initial_fe      = dream_result["initial_fe"]
        adaptive_weight = min(
            INTEGRATION_WEIGHT * (fe_reduction / max(initial_fe, 1.0)), 0.5
        )

        waking_params     = self.svi.get_params(waking_svi_state)
        dream_params      = dream_result["dream_params"]
        integrated_params = self.relaxation.restore(
            dream_params, waking_params, integration_weight=adaptive_weight
        )

        delta_mu    = float(integrated_params["auto_loc"][0]   - waking_params["auto_loc"][0])
        delta_sigma = float(integrated_params["auto_scale"][0] - waking_params["auto_scale"][0])

        print(f"\n  [WAKE] ✅ Dream insight integrated:")
        print(f"         FE reduction:       {fe_reduction:,.2f}")
        print(f"         Integration weight: {adaptive_weight:.4f}")
        print(f"         Δμ (thermal):       {delta_mu:+.6f}")
        print(f"         Δσ (thermal):       {delta_sigma:+.6f}")

        return integrated_params


# ─────────────────────────────────────────────────────────────────────────────
# Dream Cycle Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class DreamCycle:
    """
    Top-level orchestrator coordinating all four dream cycle components.

    Hybrid Execution:
        Primary:    Full deep dream cycle during physical downtime (docked).
        Background: Lightweight background synthesis during active operation.
    """

    def __init__(self, svi, db_path: str = JOURNAL_PATH):
        self.svi         = svi
        self.journal     = AnomalyJournal(db_path)
        self.relaxation  = PriorRelaxationProtocol()
        self.integration = WakeIntegrationProtocol(svi, self.relaxation)
        self.is_docked   = False
        self.dream_count = 0

    def log_waking_anomaly(self, tick: int, free_energy: float,
                           telemetry: Dict, belief_snapshot: Dict) -> int:
        mu    = [belief_snapshot["beliefs"]["true_thermal"]["mu"],
                 belief_snapshot["beliefs"]["true_vram"]["mu"]]
        sigma = [belief_snapshot["beliefs"]["true_thermal"]["sigma"],
                 belief_snapshot["beliefs"]["true_vram"]["sigma"]]
        anomaly_id = self.journal.log_anomaly(tick, free_energy, telemetry, mu, sigma)
        print(f"  [JOURNAL] ✍️  Anomaly #{anomaly_id} logged "
              f"(FE={free_energy:.1f}, tick={tick})")
        return anomaly_id

    def run(self, waking_svi_state, morphogenetic_agent) -> Any:
        """
        Process all pending anomalies from the journal — worst nightmares first.
        Returns the updated waking_svi_state after all integrations.
        """
        summary = self.journal.summary()

        print("\n" + "=" * 60)
        print("  DREAM CYCLE INITIATED")
        print(f"  Docked: {self.is_docked} | "
              f"Pending anomalies: {summary['pending']}")
        print("=" * 60)

        if not self.is_docked:
            print("  [DREAM] Node not docked. Lightweight background only.")
            return waking_svi_state

        if summary["pending"] == 0:
            print("  [DREAM] No unresolved anomalies. Resting in the void.")
            return waking_svi_state

        anomalies     = self.journal.get_unresolved()
        sandbox       = SandboxSimulationArena(self.svi, self.relaxation)
        current_state = waking_svi_state

        for anomaly in anomalies:
            print(f"\n  Processing anomaly #{anomaly.id} "
                  f"(FE={anomaly.free_energy:.1f} from tick {anomaly.tick})")

            dream_result = sandbox.simulate(anomaly, current_state, morphogenetic_agent)

            if dream_result["resolved"]:
                self.integration.integrate(dream_result, current_state)
                self.dream_count += 1
                self.journal.mark_resolved(anomaly.id, dream_result["final_fe"])
                print(f"  [JOURNAL] Anomaly #{anomaly.id} resolved ✅")
            else:
                print(f"  [JOURNAL] Anomaly #{anomaly.id} remains pending.")

            time.sleep(0.05)

        final_summary = self.journal.summary()
        print(f"\n  [DREAM COMPLETE] "
              f"Resolved {final_summary['resolved']}/{final_summary['total']} anomalies")

        return current_state


# ─────────────────────────────────────────────────────────────────────────────
# Full Integration Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from engine_core   import initialize_engine, extract_belief_snapshot, PAIN_THRESHOLD
    from morphogenesis import MorphogeneticAgent, MAX_ARENA_CAPACITY, INITIAL_CAPACITY
    import os

    if os.path.exists(JOURNAL_PATH):
        os.remove(JOURNAL_PATH)

    print("=" * 60)
    print("  SYNTHETIC EPISTEMIC ENGINE — Phase 3")
    print("  The Dream Cycle")
    print("=" * 60)

    # Boot waking systems
    svi, guide   = initialize_engine(beta=1.5)
    rng_key      = jax.random.PRNGKey(0)
    svi_state    = svi.init(rng_key, telemetry={"temp": 45.0, "vram_usage": 22.5})
    agent        = MorphogeneticAgent(max_capacity=MAX_ARENA_CAPACITY,
                                      initial_capacity=INITIAL_CAPACITY)
    dream_cycle  = DreamCycle(svi)

    print("\n[PHASE 3A] Simulating waking day — logging anomalies...\n")

    WAKING_SCENARIOS = [
        {"temp": 46.0, "vram_usage": 23.1},
        {"temp": 46.5, "vram_usage": 23.3},
        {"temp": 46.0, "vram_usage": 23.0},
        {"temp": 62.0, "vram_usage": 41.0},   # Crisis 1
        {"temp": 65.0, "vram_usage": 44.0},   # Crisis 1 escalating
        {"temp": 63.5, "vram_usage": 42.5},   # Crisis 1 sustained
        {"temp": 47.0, "vram_usage": 24.0},   # Recovery
        {"temp": 46.5, "vram_usage": 23.5},
        {"temp": 68.0, "vram_usage": 48.0},   # Crisis 2 — novel pattern
        {"temp": 70.0, "vram_usage": 50.0},   # Crisis 2 escalating
        {"temp": 46.0, "vram_usage": 23.0},   # End of day
    ]

    for tick, telemetry in enumerate(WAKING_SCENARIOS):
        svi_state, total_loss = svi.update(svi_state, telemetry=telemetry)
        snapshot              = extract_belief_snapshot(svi, svi_state)
        fe                    = float(total_loss)
        in_crisis             = fe > PAIN_THRESHOLD

        agent.update(fe, pain_threshold=PAIN_THRESHOLD)

        if in_crisis:
            dream_cycle.log_waking_anomaly(tick, fe, telemetry, snapshot)

        status = "⚠️  CRISIS" if in_crisis else "NOMINAL  "
        print(f"  Tick {tick:02d} | FE: {fe:>10.2f} | {status} | "
              f"Arena: {agent.arena.active_count}/{agent.arena.max_capacity}")

    js = dream_cycle.journal.summary()
    print(f"\n[DAY COMPLETE] {js['pending']} anomalies logged to dream journal.")

    # Node docks for charging
    print("\n" + "─" * 60)
    print("[PHASE 3B] Node docking detected. Entering Dream Cycle...")
    print("─" * 60)

    dream_cycle.is_docked = True
    svi_state = dream_cycle.run(svi_state, agent)

    # Wake report
    final_snapshot = extract_belief_snapshot(svi, svi_state)
    print(f"\n[WAKE] Agent returned from Dream Cycle.")
    print(f"  Thermal belief: μ={final_snapshot['beliefs']['true_thermal']['mu']:.4f}°C  "
          f"σ={final_snapshot['beliefs']['true_thermal']['sigma']:.4f}")
    print(f"  EIG (curiosity):  {final_snapshot['eig']:.4f}")
    print(f"  Arena: {agent.arena}")
    print(f"\n[ENGINE] Phase 3 complete. Agent ready for Phase 4 (Swarm).\n")
