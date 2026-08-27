import sys
import os
import pytest
import jax

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine_core import initialize_engine, PAIN_THRESHOLD
from morphogenesis import MorphogeneticAgent, INITIAL_CAPACITY, MAX_ARENA_CAPACITY

def test_epistemic_drive_generates_action_in_boredom():
    """
    Test 1: The Epistemic Drive (Desire to Understand)
    Even in a perfectly safe, homeostatic environment, the EIG (Expected Information Gain)
    component of the Epistemic Trace ELBO should force the agent to act to seek novelty.
    """
    svi, _ = initialize_engine(beta=1.5)
    rng_key = jax.random.PRNGKey(0)
    
    # Perfect homeostasis
    telemetry = {"temp": 45.0, "vram_usage": 22.0}
    svi_state = svi.init(rng_key, telemetry=telemetry)
    
    # Run a few updates
    for _ in range(5):
        svi_state, loss = svi.update(svi_state, telemetry=telemetry)
        
    fe = float(loss)
    
    # The agent should still generate internal free energy (from EIG penalty)
    # ensuring it doesn't just sit in a coma.
    assert fe > 0.0, "Agent fell into a coma. Epistemic drive failed to generate novelty-seeking EIG."


def test_morphogenesis_convergence():
    """
    Test 2: Morphogenesis Convergence
    When subjected to sustained high Free Energy (pain), the LatentArena
    should trigger neurogenesis to expand its capacity.
    """
    agent = MorphogeneticAgent(max_capacity=32, initial_capacity=4)
    
    initial_active = agent.arena.active_count
    assert initial_active == 4
    
    # Simulate sustained pain (Free Energy > Threshold)
    sustained_pain = PAIN_THRESHOLD * 2
    
    # Run multiple ticks of pain
    for _ in range(5):
        agent.update(free_energy=sustained_pain, pain_threshold=PAIN_THRESHOLD)
        
    # The arena should have expanded its capacity
    assert agent.arena.active_count > initial_active, "Morphogenesis failed to trigger under sustained pain."
    assert agent.arena.active_count <= 32, "Morphogenesis exceeded max capacity."
