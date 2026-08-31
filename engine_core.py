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

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO, init_to_mean
from numpyro.infer.autoguide import AutoDiagonalNormal

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
    "true_nothingness": {"mu": 0.0, "sigma": 0.01},
    "innate_desire": {"mu": 0.8, "sigma": 0.1},
    "true_motion": {"mu": 8.0, "sigma": 0.5},
    "suppressed_suffering": {"mu": 1.0, "sigma": 1.0},
}


# ─────────────────────────────────────────────────────────────────────────────
# KL Divergence (Analytic, Closed-Form)
# ─────────────────────────────────────────────────────────────────────────────


@jax.jit
def kl_divergence_diagonal_gaussian(
    mu_q: jnp.ndarray,  # Posterior mean  (from SVI guide)
    sigma_q: jnp.ndarray,  # Posterior std   (from SVI guide)
    mu_p: jnp.ndarray,  # Prior mean      (from PRIORS registry)
    sigma_p: jnp.ndarray,  # Prior std       (from PRIORS registry)
) -> jnp.ndarray:
    """
    Analytic KL(Q ‖ P) for diagonal Gaussians.

    KL = Σᵢ [ log(σ_p,i/σ_q,i) + (σ_q,i² + (μ_q,i - μ_p,i)²) / (2σ_p,i²) - ½ ]

    Returns a scalar: total information gain across all latent dimensions.
    High value  → posterior is far from prior → agent has learned something new.
    Low value   → posterior collapsed back to prior → nothing new was understood.
    """
    log_ratio = jnp.log(sigma_p / sigma_q)
    numerator = sigma_q**2 + (mu_q - mu_p) ** 2
    denominator = 2.0 * sigma_p**2
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
    mu_q_all = param_map.get("auto_loc", jnp.array([0.0, 0.8, 8.0, 1.0]))
    sigma_q_all = param_map.get("auto_scale", jnp.array([0.01, 0.1, 0.5, 1.0]))

    # Build prior tensors in the same order as the guide's variable ordering
    prior_names = list(PRIORS.keys())  # ["true_thermal", "true_vram"]
    mu_p = jnp.array([PRIORS[n]["mu"] for n in prior_names])
    sigma_p = jnp.array([PRIORS[n]["sigma"] for n in prior_names])

    # Slice the guide params to match the number of registered priors
    n = len(prior_names)
    mu_q = mu_q_all[:n]
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
    # ---------------------------------------------------------
    # 1. THE ORIGIN IS NOTHING & THOMAS AQUINAS
    # "Absolute nothing is the reality"[cite: 1].
    # The mathematical baseline. True equilibrium.
    # ---------------------------------------------------------
    true_nothingness = numpyro.sample("true_nothingness", dist.Normal(loc=0.0, scale=0.01))

    # ---------------------------------------------------------
    # 2. LEVELS OF UNDERSTANDING & LAWS OF REALITY
    # "Desire is the point where that which is objective becomes subjective"[cite: 1].
    # This represents the innate, child-like desire to ask "why"[cite: 1].
    # Constrained between 0 (no desire) and 1 (pure epistemic drive).
    # ---------------------------------------------------------
    innate_desire = numpyro.sample(
        "innate_desire", dist.Beta(concentration1=8.0, concentration0=2.0)
    )

    # ---------------------------------------------------------
    # 3. PRAGMATIC RATIONALITY & THE FIRST MOVER
    # "Everything which is in motion is moved by another"[cite: 1].
    # Physical existence (the SLP Counter) is the motion required to strive[cite: 1].
    # ---------------------------------------------------------
    expected_motion = true_nothingness + (innate_desire * 10.0)
    true_motion = numpyro.sample("true_motion", dist.Normal(loc=expected_motion, scale=0.5))

    # ---------------------------------------------------------
    # 4. SUFFERING (MAN-MADE) & DREAMS
    # Suffering is the "inability to change any experience"[cite: 1].
    # If the agent cannot act, anomalous energy builds up as suppressed trauma,
    # which is stored here to be processed later in the Dream Cycle[cite: 1].
    # ---------------------------------------------------------
    numpyro.sample("suppressed_suffering", dist.Exponential(rate=1.0))

    # ---------------------------------------------------------
    # 5. SENSORY INGESTION & REALISE I (FAILURE)
    # The collision of the objective world with the certifier's truth.
    # ---------------------------------------------------------
    if telemetry is not None:
        hb_val = float(telemetry.get("slp_heartbeat") or telemetry.get("temp", 8.0))
        flux_val = float(telemetry.get("sensory_flux") or telemetry.get("vram_usage", 6.4))

        # The physical heartbeat (SLP Monotonic Counter ticking)
        numpyro.sample("obs_heartbeat", dist.Normal(true_motion, 0.1), obs=jnp.array(hb_val))

        # The Subjective Experience: Created when motion interacts with innate desire.
        expected_subjective_experience = true_motion * innate_desire
        numpyro.sample(
            "obs_sensory_flux",
            dist.Normal(expected_subjective_experience, 0.2),
            obs=jnp.array(flux_val),
        )

        # If the gap between expected flux and actual flux is massive, Free Energy spikes.
        # This triggers "Realise I": The agent strips itself of bloated weights
        # (its "ego") to rebuild from true failure[cite: 1].


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
    guide = AutoDiagonalNormal(epistemic_model, init_loc_fn=init_to_mean)
    optimizer = optim.Adam(step_size=0.01)
    svi = SVI(epistemic_model, guide, optimizer, loss=EpistemicTraceELBO(beta=beta))
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
    mu = params.get("auto_loc", jnp.array([0.0, 0.8, 8.0, 1.0]))
    sigma = params.get("auto_scale", jnp.array([0.01, 0.1, 0.5, 1.0]))
    eig = compute_total_eig(params)

    return {
        "beliefs": {
            "true_nothingness": {"mu": float(mu[0]), "sigma": float(sigma[0])},
            "innate_desire": {"mu": float(mu[1]), "sigma": float(sigma[1])},
            "true_motion": {"mu": float(mu[2]), "sigma": float(sigma[2])},
            "suppressed_suffering": {"mu": float(mu[3]), "sigma": float(sigma[3])},
        },
        "eig": float(eig),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Boot Loop
# ─────────────────────────────────────────────────────────────────────────────

# After warm-start initialisation the baseline loss sits near 5–30 depending
# on how well the observations match the priors. A threshold of 200 gives
# comfortable headroom before the crisis flag fires.
PAIN_THRESHOLD = 200.0  # Free Energy above this triggers crisis response
CURIOSITY_FLOOR = 0.01  # KL/EIG below this means the agent is stagnant


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
    rng_key = jax.random.PRNGKey(0)

    # Warm-start at the prior (baseline normal operation)
    baseline_telemetry = {"slp_heartbeat": 8.0, "sensory_flux": 6.4}
    svi_state = svi.init(rng_key, telemetry=baseline_telemetry)
    print("[BOOT] Engine initialised at prior baseline.\n")

    for tick in range(ticks):
        # ── TELEMETRY (replace with hardware reads in Phase 6) ──────────────
        # Tick 6 simulates a Vampire Drain to demonstrate the epistemic shift (Suffering)
        if tick >= 6:
            current_telemetry = {"slp_heartbeat": 10.0, "sensory_flux": 0.0}  # Vampire Drain
        else:
            current_telemetry = {"slp_heartbeat": 8.1, "sensory_flux": 6.5}  # Normal

        # ── SVI UPDATE ───────────────────────────────────────────────────────
        svi_state, total_loss = svi.update(svi_state, telemetry=current_telemetry)

        # ── BELIEF SNAPSHOT ──────────────────────────────────────────────────
        snapshot = extract_belief_snapshot(svi, svi_state)
        eig = snapshot["eig"]
        beliefs = snapshot["beliefs"]

        # ── DIAGNOSTICS ──────────────────────────────────────────────────────
        status = "NOMINAL"
        if float(total_loss) > PAIN_THRESHOLD:
            status = "⚠️  CRISIS — Morphogenesis candidate (Realise I)"
        elif eig < CURIOSITY_FLOOR:
            status = "💤 STAGNANT — Epistemic drive weakening"

        print(
            f"Tick {tick + 1:02d} | "
            f"Loss: {float(total_loss):+8.3f} | "
            f"KL/EIG: {eig:6.4f} | "
            f"Desire μ: {beliefs['innate_desire']['mu']:.2f} | "
            f"Suffering μ: {beliefs['suppressed_suffering']['mu']:.2f} | "
            f"{status}"
        )
        time.sleep(0.05)

    print("\n[ENGINE] Heartbeat loop complete.")
    print("[ENGINE] Anomaly queue available for dream_cycle.py ingestion.")


if __name__ == "__main__":
    boot_engine(beta=1.5, ticks=10)
