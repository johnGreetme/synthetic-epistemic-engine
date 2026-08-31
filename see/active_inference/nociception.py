"""Synthetic Epistemic Engine — Artificial Nociception & Active Inference.

Implements Stochastic Variational Inference (SVI) using NumPyro and JAX to continuously
evaluate Free Energy (FE) as the negative Evidence Lower Bound (-ELBO).
Monitors physical sensory flux divergence and triggers PAIN_THRESHOLD_EXCEEDED
events when FE > 500.0 sustained over 3 consecutive ticks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO, init_to_mean
from numpyro.infer.autoguide import AutoDiagonalNormal

# JAX VRAM Allocator Guardrails for Edge Hardware
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")

PAIN_THRESHOLD: float = 500.0
SUSTAINED_PAIN_TICKS: int = 3

PRIORS: dict[str, dict[str, float]] = {
    "true_nothingness": {"mu": 0.0, "sigma": 0.01},
    "innate_desire": {"mu": 0.8, "sigma": 0.1},
    "true_motion": {"mu": 8.0, "sigma": 0.5},
    "suppressed_suffering": {"mu": 1.0, "sigma": 1.0},
}


@jax.jit
def kl_divergence_diagonal_gaussian(
    mu_q: jnp.ndarray,
    sigma_q: jnp.ndarray,
    mu_p: jnp.ndarray,
    sigma_p: jnp.ndarray,
) -> jnp.ndarray:
    """Computes closed-form analytic KL(Q || P) for diagonal Gaussian distributions."""
    log_ratio = jnp.log(sigma_p / sigma_q)
    numerator = sigma_q**2 + (mu_q - mu_p) ** 2
    denominator = 2.0 * sigma_p**2
    return jnp.sum(log_ratio + (numerator / denominator) - 0.5)


@jax.jit
def compute_total_eig(param_map: dict[str, Any]) -> jnp.ndarray:
    """Computes total Expected Information Gain (EIG) across active latent priors."""
    mu_q_all = param_map.get("auto_loc", jnp.array([0.0, 0.8, 8.0, 1.0]))
    sigma_q_all = param_map.get("auto_scale", jnp.array([0.01, 0.1, 0.5, 1.0]))

    prior_names = list(PRIORS.keys())
    mu_p = jnp.array([PRIORS[n]["mu"] for n in prior_names])
    sigma_p = jnp.array([PRIORS[n]["sigma"] for n in prior_names])

    n = len(prior_names)
    mu_q = mu_q_all[:n]
    sigma_q = sigma_q_all[:n]

    sigma_q = jnp.clip(sigma_q, a_min=1e-6)
    sigma_p = jnp.clip(sigma_p, a_min=1e-6)

    return kl_divergence_diagonal_gaussian(mu_q, sigma_q, mu_p, sigma_p)


class EpistemicTraceELBO(Trace_ELBO):
    """Augmented SVI Trace ELBO incorporating the epistemic drive (EIG bonus)."""

    def __init__(self, beta: float = 1.5, num_particles: int = 1) -> None:
        super().__init__(num_particles=num_particles)
        self.beta = beta

    def loss(
        self,
        rng_key: jax.Array,
        param_map: dict[str, Any],
        model: Any,
        guide: Any,
        *args: Any,
        **kwargs: Any,
    ) -> jnp.ndarray:
        pragmatic_loss = super().loss(rng_key, param_map, model, guide, *args, **kwargs)
        eig = compute_total_eig(param_map)
        total_loss = pragmatic_loss - (self.beta * eig)
        return total_loss


def epistemic_model(telemetry: dict[str, float] | None = None) -> None:
    """Probabilistic generative model of robot sensory expectations."""
    true_nothingness = numpyro.sample("true_nothingness", dist.Normal(loc=0.0, scale=0.01))
    innate_desire = numpyro.sample(
        "innate_desire", dist.Beta(concentration1=8.0, concentration0=2.0)
    )
    expected_motion = true_nothingness + (innate_desire * 10.0)
    true_motion = numpyro.sample("true_motion", dist.Normal(loc=expected_motion, scale=0.5))
    numpyro.sample("suppressed_suffering", dist.Exponential(rate=1.0))

    if telemetry is not None:
        heartbeat_obs = telemetry.get("slp_heartbeat", 8.0)
        sensory_flux_obs = telemetry.get("sensory_flux", 6.4)

        numpyro.sample(
            "obs_heartbeat",
            dist.Normal(true_motion, 0.1),
            obs=jnp.array(float(heartbeat_obs)),
        )

        expected_subjective_experience = true_motion * innate_desire
        numpyro.sample(
            "obs_sensory_flux",
            dist.Normal(expected_subjective_experience, 0.2),
            obs=jnp.array(float(sensory_flux_obs)),
        )


@dataclass
class NociceptionEvent:
    """Represents a discrete sensory nociception evaluation result."""

    tick: int
    free_energy: float
    pain_level: float
    sustained_ticks: int
    pain_threshold_exceeded: bool
    telemetry: dict[str, float]
    belief_snapshot: dict[str, Any]
    event_name: str | None = None


class NociceptionEngine:
    """Stochastic Variational Inference engine tracking artificial nociception."""

    def __init__(
        self,
        beta: float = 1.5,
        pain_threshold: float = PAIN_THRESHOLD,
        sustained_ticks_threshold: int = SUSTAINED_PAIN_TICKS,
        seed: int = 0,
    ) -> None:
        self.beta = beta
        self.pain_threshold = pain_threshold
        self.sustained_ticks_threshold = sustained_ticks_threshold
        self.rng_key = jax.random.PRNGKey(seed)

        self.guide = AutoDiagonalNormal(epistemic_model, init_loc_fn=init_to_mean)
        self.optimizer = optim.Adam(step_size=0.01)
        self.svi = SVI(
            epistemic_model,
            self.guide,
            self.optimizer,
            loss=EpistemicTraceELBO(beta=self.beta),
        )

        # Warm start state
        self.rng_key, init_key = jax.random.split(self.rng_key)
        self.svi_state = self.svi.init(
            init_key, telemetry={"slp_heartbeat": 8.0, "sensory_flux": 6.4}
        )

        self.tick_count: int = 0
        self.consecutive_pain_ticks: int = 0
        self.history: list[NociceptionEvent] = []

    def update(self, telemetry: dict[str, float]) -> NociceptionEvent:
        """Runs a single SVI inference step with incoming telemetry."""
        self.tick_count += 1
        self.svi_state, loss = self.svi.update(self.svi_state, telemetry=telemetry)
        fe_scalar = float(loss)

        if fe_scalar > self.pain_threshold:
            self.consecutive_pain_ticks += 1
        else:
            self.consecutive_pain_ticks = 0

        pain_exceeded = self.consecutive_pain_ticks >= self.sustained_ticks_threshold
        event_name = "PAIN_THRESHOLD_EXCEEDED" if pain_exceeded else None

        beliefs = self.extract_belief_snapshot()

        event = NociceptionEvent(
            tick=self.tick_count,
            free_energy=fe_scalar,
            pain_level=max(0.0, fe_scalar - self.pain_threshold),
            sustained_ticks=self.consecutive_pain_ticks,
            pain_threshold_exceeded=pain_exceeded,
            telemetry=dict(telemetry),
            belief_snapshot=beliefs,
            event_name=event_name,
        )

        self.history.append(event)
        return event

    def extract_belief_snapshot(self) -> dict[str, Any]:
        """Extracts current posterior distribution parameters from SVI state."""
        params = self.svi.get_params(self.svi_state)
        mu = params.get("auto_loc", jnp.array([0.0, 0.8, 8.0, 1.0]))
        sigma = params.get("auto_scale", jnp.array([0.01, 0.1, 0.5, 1.0]))
        eig = float(compute_total_eig(params))

        return {
            "beliefs": {
                "true_nothingness": {"mu": float(mu[0]), "sigma": float(sigma[0])},
                "innate_desire": {"mu": float(mu[1]), "sigma": float(sigma[1])},
                "true_motion": {"mu": float(mu[2]), "sigma": float(sigma[2])},
                "suppressed_suffering": {"mu": float(mu[3]), "sigma": float(sigma[3])},
            },
            "eig": eig,
        }
