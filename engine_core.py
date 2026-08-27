"""
Synthetic Epistemic Engine — engine_core.py
Phase 1: Mathematical Foundation

Core Mathematical Drive:
    Total Loss = ELBO - β · EIG

Where EIG (Expected Information Gain) is the analytic KL Divergence
between the posterior Q(x|o) and prior P(x) for each latent variable:

    KL(Q ‖ P) = Σᵢ [ log(σ_p,i / σ_q,i) + (σ_q,i² + (μ_q,i - μ_p,i)²) / (2σ_p,i²) - ½ ]

A high KL means the agent's beliefs have shifted far from its priors — it has
learned something. Maximising this term creates the "Desire to Understand."
"""

import os
import time
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, init_to_mean
from numpyro.infer.autoguide import AutoDiagonalNormal
import numpyro.optim as optim

# ─────────────────────────────────────────────────────────────────────────────
# JAX Allocator Override (Edge VRAM Sovereignty)
# ─────────────────────────────────────────────────────────────────────────────
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"

# ─────────────────────────────────────────────────────────────────────────────
# Model Priors Registry
# These are the agent's "prior beliefs" — what it expects reality to look like
# before receiving any sensory evidence. EIG is measured against these.
#
# Encoding your book's concept: before observing the world, the agent holds
# a "subjective truth." The further its posterior moves from this prior,
# the more it has understood of objective reality.
# ─────────────────────────────────────────────────────────────────────────────
PRIORS = {
    "true_thermal": {"mu": 45.0, "sigma": 2.0},    # Expected CPU/GPU temp (°C)
    "true_vram":    {"mu": 22.5, "sigma": 1.5},     # Expected VRAM baseline (GB)
}


# ─────────────────────────────────────────────────────────────────────────────
# KL Divergence (Analytic, Closed-Form)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def kl_divergence_diagonal_gaussian(
    mu_q:    jnp.ndarray,   # Posterior mean  (from SVI guide)
    sigma_q: jnp.ndarray,   # Posterior std   (from SVI guide)
    mu_p:    jnp.ndarray,   # Prior mean      (from PRIORS registry)
    sigma_p: jnp.ndarray,   # Prior std       (from PRIORS registry)
) -> jnp.ndarray:
    """
    Analytic KL(Q ‖ P) for diagonal Gaussians.

    KL = Σᵢ [ log(σ_p,i/σ_q,i) + (σ_q,i² + (μ_q,i - μ_p,i)²) / (2σ_p,i²) - ½ ]

    Returns a scalar: total information gain across all latent dimensions.
    High value  → posterior is far from prior → agent has learned something new.
    Low value   → posterior collapsed back to prior → nothing new was understood.
    """
    log_ratio  = jnp.log(sigma_p / sigma_q)
    numerator  = sigma_q ** 2 + (mu_q - mu_p) ** 2
    denominator = 2.0 * sigma_p ** 2
    return jnp.sum(log_ratio + (numerator / denominator) - 0.5)


@jax.jit
def compute_total_eig(param_map: dict) -> jnp.ndarray:
    """
    Extract posterior parameters from the NumPyro AutoDiagonalNormal guide,
    compare against every prior in the PRIORS registry, and sum the KL
    divergences to produce a single scalar EIG value.

    AutoDiagonalNormal stores parameters as:
        param_map["auto_loc"]   → concatenated posterior means  [N]
        param_map["auto_scale"] → concatenated posterior scales [N]

    We split these back into per-latent-variable slices mapped to PRIORS.
    """
    mu_q_all    = param_map.get("auto_loc",   jnp.array([45.0, 22.5]))
    sigma_q_all = param_map.get("auto_scale", jnp.array([2.0,  1.5]))

    # Build prior tensors in the same order as the guide's variable ordering
    prior_names = list(PRIORS.keys())                       # ["true_thermal", "true_vram"]
    mu_p    = jnp.array([PRIORS[n]["mu"]    for n in prior_names])
    sigma_p = jnp.array([PRIORS[n]["sigma"] for n in prior_names])

    # Slice the guide params to match the number of registered priors
    n = len(prior_names)
    mu_q    = mu_q_all[:n]
    sigma_q = sigma_q_all[:n]

    # Clamp sigma to avoid log(0) — numerically safe floor
    sigma_q = jnp.clip(sigma_q, a_min=1e-6)
    sigma_p = jnp.clip(sigma_p, a_min=1e-6)

    return kl_divergence_diagonal_gaussian(mu_q, sigma_q, mu_p, sigma_p)


# ─────────────────────────────────────────────────────────────────────────────
# Epistemic Trace ELBO
# The agent's total objective function.
#
#   Total Loss = ELBO - β · KL(Q ‖ P)
#
# Minimising this loss means the agent simultaneously:
#   - Minimises surprise (ELBO term — Pragmatic Survival / D.I.A.N.A. territory)
#   - Maximises information gain (KL term — Epistemic Drive / Desire to Understand)
#
# β (beta) is the "curiosity coefficient." Higher β = stronger desire to
# understand. Lower β = more conservative, survival-dominant behaviour.
# ─────────────────────────────────────────────────────────────────────────────

class EpistemicTraceELBO(Trace_ELBO):
    """
    Replaces the standard Trace_ELBO with an epistemically-augmented objective.

    The inherited .loss() computes the standard negative ELBO (Free Energy).
    We subtract β * EIG to inject the Desire to Understand.
    """

    def __init__(self, beta: float = 1.5, num_particles: int = 1):
        super().__init__(num_particles=num_particles)
        self.beta = beta

    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        # Pragmatic survival component (standard Free Energy)
        pragmatic_loss = super().loss(rng_key, param_map, model, guide, *args, **kwargs)

        # Epistemic drive component (real KL divergence)
        eig = compute_total_eig(param_map)

        # Final objective — the agent is penalised for being incurious
        total_loss = pragmatic_loss - (self.beta * eig)
        return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# Generative Model (The Agent's Belief System / Subjective Truth)
# ─────────────────────────────────────────────────────────────────────────────

def epistemic_model(telemetry=None):
    """
    The agent's internal generative model of reality.

    It encodes the agent's causal understanding of the world:
        Thermal state  →  drives VRAM load
        VRAM load      →  observed by sensors

    When telemetry is provided, the agent's beliefs are updated by
    reconciling this subjective model with the objective sensor data.
    """
    # --- Latent Variables (Subjective Truth) ---
    true_thermal = numpyro.sample(
        "true_thermal",
        dist.Normal(loc=PRIORS["true_thermal"]["mu"],
                    scale=PRIORS["true_thermal"]["sigma"])
    )

    # Internal causal physics: thermal pressure drives VRAM consumption
    expected_vram = true_thermal * 0.5
    true_vram = numpyro.sample(
        "true_vram",
        dist.Normal(loc=expected_vram,
                    scale=PRIORS["true_vram"]["sigma"])
    )

    # --- Sensory Observations (Objective Reality) ---
    if telemetry is not None:
        numpyro.sample(
            "obs_temp",
            dist.Normal(true_thermal, scale=0.5),
            obs=jnp.array(telemetry["temp"])
        )
        numpyro.sample(
            "obs_vram",
            dist.Normal(true_vram, scale=0.1),
            obs=jnp.array(telemetry["vram_usage"])
        )


# ─────────────────────────────────────────────────────────────────────────────
# Engine Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def initialize_engine(beta: float = 1.5):
    """
    Bootstrap the SVI engine with the Epistemic objective.

    init_to_mean warm-starts the guide's auto_loc at each latent variable's
    prior mean (e.g. 45.0°C for true_thermal) rather than at zero.
    This ensures the initial KL divergence is near 0 and the loss is
    meaningful from tick 1 — the agent begins from a sensible belief, not
    from a state of total ignorance.
    """
    guide     = AutoDiagonalNormal(epistemic_model, init_loc_fn=init_to_mean)
    optimizer = optim.Adam(step_size=0.01)
    svi       = SVI(epistemic_model, guide, optimizer,
                    loss=EpistemicTraceELBO(beta=beta))
    return svi, guide


@jax.jit
def _jit_update(svi, state, telemetry_array):
    """JIT-compiled inner update. Receives telemetry as a pre-packed array."""
    return svi.update(state, telemetry=telemetry_array)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic Helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_belief_snapshot(svi, svi_state) -> dict:
    """
    Pull the agent's current posterior distribution parameters.
    These are the exact values needed by the Epistemic Debugger (Phase 5).

    Returns:
        mu    — posterior means  (what the agent currently believes to be true)
        sigma — posterior stds   (how confident the agent is)
        eig   — current KL from prior (how much it has learned from baseline)
    """
    params = svi.get_params(svi_state)
    mu     = params.get("auto_loc",   jnp.array([45.0, 22.5]))
    sigma  = params.get("auto_scale", jnp.array([2.0,  1.5]))
    eig    = compute_total_eig(params)

    return {
        "beliefs": {
            "true_thermal": {"mu": float(mu[0]), "sigma": float(sigma[0])},
            "true_vram":    {"mu": float(mu[1]), "sigma": float(sigma[1])},
        },
        "eig": float(eig),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Boot Loop
# ─────────────────────────────────────────────────────────────────────────────

# After warm-start initialisation the baseline loss sits near 5–30 depending
# on how well the observations match the priors. A threshold of 200 gives
# comfortable headroom before the crisis flag fires.
PAIN_THRESHOLD  = 200.0      # Free Energy above this triggers crisis response
CURIOSITY_FLOOR = 0.01       # KL/EIG below this means the agent is stagnant


def boot_engine(beta: float = 1.5, ticks: int = 10):
    """
    Run the Synthetic Epistemic Engine heartbeat loop.

    Each tick:
        1. Ingest live telemetry (replace mock with real sensor reads in Phase 6)
        2. Update SVI state (reconcile beliefs with reality)
        3. Compute total epistemic loss = ELBO - β·KL
        4. Extract and print belief snapshot (feeds Epistemic Debugger in Phase 5)
        5. Flag crises and epistemic stagnation for Morphogenesis (Phase 2)
    """
    print("=" * 60)
    print("  SYNTHETIC EPISTEMIC ENGINE — Phase 1")
    print(f"  Curiosity Coefficient (β): {beta}")
    print(f"  Pain Threshold:            {PAIN_THRESHOLD}")
    print("=" * 60)

    svi, guide = initialize_engine(beta=beta)
    rng_key    = jax.random.PRNGKey(0)

    # Warm-start at the prior (baseline normal operation)
    baseline_telemetry = {"temp": 45.0, "vram_usage": 22.5}
    svi_state = svi.init(rng_key, telemetry=baseline_telemetry)
    print("[BOOT] Engine initialised at prior baseline.\n")

    for tick in range(ticks):
        # ── TELEMETRY (replace with hardware reads in Phase 6) ──────────────
        # Tick 6 simulates a thermal anomaly to demonstrate the epistemic shift
        if tick >= 6:
            current_telemetry = {"temp": 62.0, "vram_usage": 41.0}   # Anomaly
        else:
            current_telemetry = {"temp": 46.0, "vram_usage": 23.1}   # Normal

        # ── SVI UPDATE ───────────────────────────────────────────────────────
        svi_state, total_loss = svi.update(svi_state, telemetry=current_telemetry)

        # ── BELIEF SNAPSHOT ──────────────────────────────────────────────────
        snapshot = extract_belief_snapshot(svi, svi_state)
        eig      = snapshot["eig"]
        beliefs  = snapshot["beliefs"]

        # ── DIAGNOSTICS ──────────────────────────────────────────────────────
        status = "NOMINAL"
        if float(total_loss) > PAIN_THRESHOLD:
            status = "⚠️  CRISIS — Morphogenesis candidate"
        elif eig < CURIOSITY_FLOOR:
            status = "💤 STAGNANT — Epistemic drive weakening"

        print(
            f"Tick {tick+1:02d} | "
            f"Loss: {float(total_loss):+8.3f} | "
            f"KL/EIG: {eig:6.4f} | "
            f"Thermal μ: {beliefs['true_thermal']['mu']:.2f}°C "
            f"(σ={beliefs['true_thermal']['sigma']:.4f}) | "
            f"{status}"
        )
        time.sleep(0.05)

    print("\n[ENGINE] Heartbeat loop complete.")
    print("[ENGINE] Anomaly queue available for dream_cycle.py ingestion.")


if __name__ == "__main__":
    boot_engine(beta=1.5, ticks=10)
