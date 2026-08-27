import os
import time
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal
import numpyro.optim as optim

# 1. Hijack the JAX Allocator for Edge VRAM Sovereignty
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"

# 2. Epistemic Trace ELBO (Injecting Desire to Understand)
class Epistemic_Trace_ELBO(Trace_ELBO):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta  # The strength of the "Desire to Understand"

    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        # Calculate standard ELBO (Pragmatic Survival)
        standard_elbo = super().loss(rng_key, param_map, model, guide, *args, **kwargs)
        
        # Calculate Expected Information Gain (EIG) - Epistemic Drive
        eig = self._calculate_eig(param_map)
        
        # Total Loss = Pragmatic (ELBO) - beta * Epistemic (EIG)
        total_loss = standard_elbo - (self.beta * eig)
        return total_loss
        
    def _calculate_eig(self, param_map):
        # Simulate a "curiosity reward"
        variance_sum = 0.0
        for name, value in param_map.items():
            if "scale" in name:
                variance_sum += jnp.sum(value)
        return variance_sum * 0.1

# 3. Define the Generative Model (The Agent's Belief System)
def epistemic_homeostasis_model(telemetry=None):
    true_thermal = numpyro.sample(
        "true_thermal", dist.Normal(loc=45.0, scale=2.0)
    )
    expected_vram = true_thermal * 0.5
    true_vram = numpyro.sample(
        "true_vram", dist.Normal(loc=expected_vram, scale=1.5)
    )

    if telemetry is not None:
        numpyro.sample(
            "obs_temp", dist.Normal(true_thermal, 0.5), obs=telemetry.get("temp")
        )
        numpyro.sample(
            "obs_vram", dist.Normal(true_vram, 0.1), obs=telemetry.get("vram_usage")
        )

# 4. Initialize the Epistemic SVI Engine
def initialize_epistemic_engine():
    guide = AutoDiagonalNormal(epistemic_homeostasis_model)
    optimizer = optim.Adam(step_size=0.01)
    svi = SVI(epistemic_homeostasis_model, guide, optimizer, loss=Epistemic_Trace_ELBO(beta=1.5))
    return svi, guide

@jax.jit
def update_step(svi, state, data):
    return svi.update(state, telemetry=data)

def boot_engine():
    print("[EPISTEMIC OS] Booting Synthetic Epistemic Engine...")
    svi, guide = initialize_epistemic_engine()
    rng_key = jax.random.PRNGKey(0)
    
    baseline_telemetry = {"temp": 45.0, "vram_usage": 32.5}
    svi_state = svi.init(rng_key, telemetry=baseline_telemetry)
    
    print("[EPISTEMIC OS] Engine ONLINE. Beginning epistemic loop.")
    pain_threshold = 150.0
    
    for _ in range(5):
        current_telemetry = {"temp": 48.0, "vram_usage": 36.1}
        svi_state, epistemic_loss = update_step(svi, svi_state, current_telemetry)
        
        print(f"Heartbeat | Epistemic Loss (Safety - Curiosity): {float(epistemic_loss):.2f}")
        time.sleep(0.1)
        
if __name__ == "__main__":
    boot_engine()
