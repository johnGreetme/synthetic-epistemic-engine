"""Unit tests for the Ed25519 Tombstone Protocol and Forager Apoptosis."""

import pytest
from see.nodes.queen_node import QueenNode
from see.nodes.forager_node import ForagerNode
from see.immunity.apoptosis import ApoptosisManager
from see.mesh.crypto_enclave import CryptoEnclave


def test_queen_detects_spoofed_fe_and_initiates_tombstone():
    """Verifies that impossible Free Energy reduction triggers SPOOFED classification and Tombstone."""
    queen = QueenNode(bind_zmq=False)
    attacker_enclave = CryptoEnclave()
    attacker_pubkey = attacker_enclave.export_public()

    # Impossible physical claims
    spoofed_payload = {
        "node_id": "captured-node",
        "pre_morph_fe": 150000.0,      # > 100,000 max bound
        "post_morph_fe": 10.0,
        "fe_reduction": 149990.0,      # > 50,000 max bound
    }

    envelope = attacker_enclave.wrap_payload(spoofed_payload)
    success = queen.ingest_mutation_envelope(envelope.to_json())
    assert success is True

    # Process task through triage
    result = queen.process_next_triage_task()
    assert result is not None
    assert result["status"] == "SPOOFED"

    # Queen should have revoked attacker's public key
    assert attacker_pubkey in queen.revoked_keys
    assert queen.stats()["spoofed"] == 1

    # Subsequent packets from this key must be rejected at ingest
    subsequent_payload = {"node_id": "captured-node", "pre_morph_fe": 400.0}
    env2 = attacker_enclave.wrap_payload(subsequent_payload)
    ingest_result = queen.ingest_mutation_envelope(env2.to_json())
    assert ingest_result is False


def test_forager_blacklists_foreign_tombstoned_key():
    """Verifies that healthy foragers add foreign tombstoned keys to their blacklist without self-terminating."""
    forager = ForagerNode(node_id="healthy-forager", connect_zmq=False)
    foreign_key = CryptoEnclave().export_public()

    tombstone_payload = {
        "action": "REVOKE_IDENTITY",
        "compromised_pubkey": foreign_key,
        "reason": "PHYSICS_SPOOF_MUTATION_1234",
    }

    event = forager.apoptosis.handle_tombstone(tombstone_payload)
    assert event.status == "KEY_BLACKLISTED"
    assert event.is_self is False
    assert forager.apoptosis.is_revoked(foreign_key) is True
    assert forager.apoptosis.is_dead is False


def test_forager_executes_self_apoptosis_on_identity_revocation():
    """Verifies that a node receiving a Tombstone targeting its own key triggers Apoptosis and halts."""
    forager = ForagerNode(node_id="compromised-forager", connect_zmq=False)
    own_key = forager.pubkey

    tombstone_payload = {
        "action": "REVOKE_IDENTITY",
        "compromised_pubkey": own_key,
        "reason": "PHYSICS_SPOOF_DETECTED",
    }

    event = forager.apoptosis.handle_tombstone(tombstone_payload)
    assert event.status == "APOPTOSIS_EXECUTED"
    assert event.is_self is True
    assert forager.apoptosis.is_dead is True

    # Processing telemetry after apoptosis must raise RuntimeError
    with pytest.raises(RuntimeError, match="dead due to Apoptosis"):
        forager.process_telemetry_tick({"slp_heartbeat": 8.0, "sensory_flux": 6.4})
