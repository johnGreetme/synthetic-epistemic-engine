"""Unit tests for epistemic drive and morphogenetic convergence."""

from see.active_inference.nociception import NociceptionEngine
from see.dream_sandbox.morphogenetic_agent import MorphogeneticAgent


def test_epistemic_drive_generates_action_in_boredom():
    """
    Test 1: The Epistemic Drive (Desire to Understand)
    Even in a homeostatic environment, the EIG (Expected Information Gain)
    component of the Epistemic Trace ELBO ensures beliefs and EIG are actively computed.
    """
    engine = NociceptionEngine(beta=1.5, seed=42)
    telemetry = {"slp_heartbeat": 8.0, "sensory_flux": 6.4}

    # Run updates
    for _ in range(5):
        event = engine.update(telemetry)

    beliefs = engine.extract_belief_snapshot()
    assert "eig" in beliefs
    assert beliefs["eig"] >= 0.0
    assert event.free_energy is not None


def test_morphogenesis_convergence():
    """
    Test 2: Morphogenesis Convergence
    When subjected to sustained high Free Energy (pain), the LatentArena
    triggers neurogenesis to expand its capacity.
    """
    agent = MorphogeneticAgent(max_capacity=32, initial_capacity=4)

    initial_active = agent.arena.active_count
    assert initial_active == 4

    sustained_pain = 800.0  # > 500.0 threshold

    # Run ticks of pain
    for _ in range(5):
        agent.update(free_energy=sustained_pain, pain_threshold=500.0)

    # The arena should have expanded its capacity
    assert agent.arena.active_count > initial_active, (
        "Morphogenesis failed to trigger under sustained pain."
    )
    assert agent.arena.active_count <= 32, "Morphogenesis exceeded max capacity."
