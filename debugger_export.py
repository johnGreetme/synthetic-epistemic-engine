"""
Synthetic Epistemic Engine — debugger_export.py
Phase 5: Epistemic Debugger — Data Export Layer

Runs a full simulation and captures every tick's belief state, Free Energy,
morphogenesis events, and arena topology into a structured JSON file
that the Three.js visualizer consumes.

Output: debugger_data.json
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from engine_core import PAIN_THRESHOLD, PRIORS, extract_belief_snapshot, initialize_engine
from morphogenesis import (
    INITIAL_CAPACITY,
    MAX_ARENA_CAPACITY,
    PAIN_TICKS_THRESHOLD,
    MorphogeneticAgent,
)


def run_and_export(output_path: str = "debugger_data.json"):
    print("=" * 60)
    print("  EPISTEMIC DEBUGGER — Capture Run")
    print("=" * 60)

    svi, guide = initialize_engine(beta=1.5)
    rng_key = jax.random.PRNGKey(0)
    svi_state = svi.init(rng_key, telemetry={"temp": 45.0, "vram_usage": 22.5})
    agent = MorphogeneticAgent(max_capacity=MAX_ARENA_CAPACITY, initial_capacity=INITIAL_CAPACITY)

    SCENARIOS = [
        {"temp": 46.0, "vram_usage": 23.1},
        {"temp": 46.2, "vram_usage": 23.3},
        {"temp": 46.0, "vram_usage": 23.0},
        {"temp": 46.5, "vram_usage": 23.4},
        {"temp": 46.0, "vram_usage": 23.0},
        {"temp": 46.3, "vram_usage": 23.2},
        {"temp": 62.0, "vram_usage": 41.0},  # Crisis begins
        {"temp": 64.0, "vram_usage": 43.0},  # Crisis escalates
        {"temp": 65.0, "vram_usage": 44.0},  # Morphogenesis fires
        {"temp": 63.5, "vram_usage": 42.5},
        {"temp": 60.0, "vram_usage": 38.0},
        {"temp": 47.0, "vram_usage": 24.0},  # Recovery begins
        {"temp": 46.5, "vram_usage": 23.5},
        {"temp": 46.0, "vram_usage": 23.0},
        {"temp": 46.2, "vram_usage": 23.1},
        {"temp": 46.0, "vram_usage": 23.0},
        {"temp": 46.0, "vram_usage": 23.0},
        {"temp": 68.0, "vram_usage": 48.0},  # Second crisis
        {"temp": 70.0, "vram_usage": 50.0},  # Second crisis escalates
        {"temp": 71.0, "vram_usage": 51.0},  # Morphogenesis fires again
        {"temp": 69.0, "vram_usage": 49.0},
        {"temp": 48.0, "vram_usage": 25.0},  # Recovery
        {"temp": 46.5, "vram_usage": 23.5},
        {"temp": 46.0, "vram_usage": 23.0},
        {"temp": 46.0, "vram_usage": 23.0},
    ]

    frames = []

    for tick, telemetry in enumerate(SCENARIOS):
        prev_active = agent.arena.active_count

        svi_state, total_loss = svi.update(svi_state, telemetry=telemetry)
        snapshot = extract_belief_snapshot(svi, svi_state)
        fe = float(total_loss)

        agent.update(fe, pain_threshold=PAIN_THRESHOLD)

        morphogenesis_event = agent.arena.active_count > prev_active

        params = svi.get_params(svi_state)
        mu_all = np.array(params.get("auto_loc", jnp.array([45.0, 22.5])))
        sig_all = np.array(params.get("auto_scale", jnp.array([2.0, 1.5])))

        # V2 Epistemic Debugger: Dynamic PCA/SVD Projection
        # Extract the full N-dimensional covariance matrix of the agent's beliefs
        cov_matrix = jnp.diag(jnp.array(sig_all) ** 2)

        # Run SVD to find the principal components
        U, S, Vh = jax.numpy.linalg.svd(cov_matrix)

        # Map the top 2 principal components to the X and Y axes
        projection_axes = U[:, :2].tolist()

        # Arena mask as list
        mask = [int(agent.arena.mask[i]) for i in range(agent.arena.max_capacity)]

        frame = {
            "tick": tick,
            "telemetry": telemetry,
            "free_energy": fe,
            "eig": snapshot["eig"],
            "beliefs": {
                "thermal_mu": snapshot["beliefs"]["true_thermal"]["mu"],
                "thermal_sigma": snapshot["beliefs"]["true_thermal"]["sigma"],
                "vram_mu": snapshot["beliefs"]["true_vram"]["mu"],
                "vram_sigma": snapshot["beliefs"]["true_vram"]["sigma"],
            },
            "posterior_mu": mu_all.tolist(),
            "posterior_sigma": sig_all.tolist(),
            "svd_projection_axes": projection_axes,
            "pain_timer": agent.sustained_pain_timer,
            "arena_active": agent.arena.active_count,
            "arena_mask": mask,
            "morphogenesis": morphogenesis_event,
            "status": (
                "CRISIS"
                if fe > PAIN_THRESHOLD
                else "MORPHOGENESIS"
                if morphogenesis_event
                else "NOMINAL"
            ),
        }
        frames.append(frame)
        print(
            f"  Tick {tick:02d} | FE: {fe:>10.2f} | "
            f"Status: {frame['status']:<14} | "
            f"Arena: {agent.arena.active_count}/{agent.arena.max_capacity}"
        )

    export = {
        "engine": "Synthetic Epistemic Engine",
        "phase": 5,
        "priors": PRIORS,
        "thresholds": {
            "pain": PAIN_THRESHOLD,
            "morphogenesis_ticks": PAIN_TICKS_THRESHOLD,
        },
        "arena": {
            "max_capacity": MAX_ARENA_CAPACITY,
            "initial_capacity": INITIAL_CAPACITY,
        },
        "total_ticks": len(frames),
        "frames": frames,
    }

    with open(output_path, "w") as f:
        json.dump(export, f, indent=2)

    print(f"\n[EXPORT] {len(frames)} frames → {output_path}")
    morph_count = sum(1 for f in frames if f["morphogenesis"])
    crisis_count = sum(1 for f in frames if f["status"] == "CRISIS")
    print(f"[EXPORT] Crisis ticks: {crisis_count} | Morphogenesis events: {morph_count}")


if __name__ == "__main__":
    run_and_export()
