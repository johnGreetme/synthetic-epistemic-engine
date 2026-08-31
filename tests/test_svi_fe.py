"""Unit tests for NumPyro/JAX SVI Free Energy calculation and Nociception pain events."""

from see.active_inference.nociception import (
    PAIN_THRESHOLD,
    NociceptionEngine,
)


def test_svi_initialization_and_belief_snapshot():
    """Verifies SVI engine initializes properly and computes non-zero EIG and beliefs."""
    engine = NociceptionEngine(beta=1.5, seed=42)
    beliefs = engine.extract_belief_snapshot()

    assert "beliefs" in beliefs
    assert "eig" in beliefs
    assert "true_motion" in beliefs["beliefs"]
    assert isinstance(beliefs["beliefs"]["true_motion"]["mu"], float)
    assert beliefs["eig"] >= 0.0


def test_homeostatic_telemetry_low_free_energy():
    """Verifies that expected sensory observations keep Free Energy below pain threshold."""
    engine = NociceptionEngine(beta=1.5, seed=42)
    telemetry = {"slp_heartbeat": 8.0, "sensory_flux": 6.4}

    event = engine.update(telemetry)
    assert event.tick == 1
    assert event.free_energy < PAIN_THRESHOLD
    assert event.pain_threshold_exceeded is False
    assert event.sustained_ticks == 0


def test_pain_threshold_exceeded_sustained_three_ticks():
    """Verifies that severe sensory anomalies spike Free Energy > 500.0 and trigger pain event on 3rd tick."""
    engine = NociceptionEngine(
        beta=1.5,
        pain_threshold=500.0,
        sustained_ticks_threshold=3,
        seed=42,
    )

    # Severe unmodeled anomaly causing large SVI divergence
    crisis_telemetry = {"slp_heartbeat": 25.0, "sensory_flux": 85.0}

    # Tick 1: Spike occurs
    ev1 = engine.update(crisis_telemetry)
    assert ev1.free_energy > 500.0
    assert ev1.sustained_ticks == 1
    assert ev1.pain_threshold_exceeded is False
    assert ev1.event_name is None

    # Tick 2: Pain sustained
    ev2 = engine.update(crisis_telemetry)
    assert ev2.free_energy > 500.0
    assert ev2.sustained_ticks == 2
    assert ev2.pain_threshold_exceeded is False
    assert ev2.event_name is None

    # Tick 3: Threshold exceeded -> PAIN_THRESHOLD_EXCEEDED fired
    ev3 = engine.update(crisis_telemetry)
    assert ev3.free_energy > 500.0
    assert ev3.sustained_ticks == 3
    assert ev3.pain_threshold_exceeded is True
    assert ev3.event_name == "PAIN_THRESHOLD_EXCEEDED"

    # Tick 4: Return to homeostasis -> resets counter
    homeo_telemetry = {"slp_heartbeat": 8.0, "sensory_flux": 6.4}
    ev4 = engine.update(homeo_telemetry)
    assert ev4.sustained_ticks == 0
    assert ev4.pain_threshold_exceeded is False
