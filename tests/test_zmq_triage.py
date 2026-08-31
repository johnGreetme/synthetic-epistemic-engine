"""Unit tests for Metabolic Triage priority queue sorting and ZeroMQ transport."""

import time
import pytest
from see.mesh.triage import MetabolicTriageQueue, TriageItem
from see.mesh.crypto_enclave import CryptoEnclave, SignedEnvelope


def test_metabolic_triage_priority_ordering():
    """Verifies that nodes with highest physical pain (-pre_morph_fe) are popped first."""
    triage = MetabolicTriageQueue()

    low_pain_payload = {"node_id": "forager-1", "action": "low_pain"}
    med_pain_payload = {"node_id": "forager-2", "action": "med_pain"}
    critical_pain_payload = {"node_id": "forager-3", "action": "critical_pain"}

    # Insert in random order
    triage.push(payload=med_pain_payload, pre_morph_fe=650.0)
    triage.push(payload=low_pain_payload, pre_morph_fe=120.0)
    triage.push(payload=critical_pain_payload, pre_morph_fe=1850.0)

    # 1st: Critical pain (1850.0)
    item1 = triage.pop()
    assert item1.payload["node_id"] == "forager-3"
    assert item1.free_energy == 1850.0

    # 2nd: Medium pain (650.0)
    item2 = triage.pop()
    assert item2.payload["node_id"] == "forager-2"
    assert item2.free_energy == 650.0

    # 3rd: Low pain (120.0)
    item3 = triage.pop()
    assert item3.payload["node_id"] == "forager-1"
    assert item3.free_energy == 120.0

    assert triage.empty() is True


def test_metabolic_triage_explicit_tie_breaking():
    """Verifies that identical Free Energy values resolve via timestamp/sequence without TypeError on dicts."""
    triage = MetabolicTriageQueue()

    payload_a = {"node_id": "forager-A", "complex_data": {"alpha": [1, 2, 3]}}
    payload_b = {"node_id": "forager-B", "complex_data": {"beta": [4, 5, 6]}}

    # Both have the exact same FE: 800.0
    t0 = time.time()
    triage.push(payload=payload_a, pre_morph_fe=800.0, timestamp=t0)
    triage.push(payload=payload_b, pre_morph_fe=800.0, timestamp=t0 + 0.001)

    # Should pop without TypeError
    first = triage.pop()
    second = triage.pop()

    assert first.payload["node_id"] == "forager-A"
    assert second.payload["node_id"] == "forager-B"


def test_signed_payload_envelope_verification():
    """Verifies that signed envelopes are tamper-evident and can be parsed."""
    sender_enclave = CryptoEnclave()
    payload = {"slot_index": 5, "fe_reduction": 450.0}

    envelope = sender_enclave.wrap_payload(payload)
    assert isinstance(envelope, SignedEnvelope)

    is_valid, parsed = CryptoEnclave.unwrap_payload(envelope)
    assert is_valid is True
    assert parsed["slot_index"] == 5

    # Tampered payload fails verification
    tampered_envelope = SignedEnvelope(
        payload='{"slot_index": 5, "fe_reduction": 999999.0}',
        signature_b64=envelope.signature_b64,
        node_pubkey=envelope.node_pubkey,
    )
    is_valid_tampered, _ = CryptoEnclave.unwrap_payload(tampered_envelope)
    assert is_valid_tampered is False
