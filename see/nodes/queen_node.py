"""Synthetic Epistemic Engine — Queen Node.

Orchestrates the swarm's collective cognitive synthesis:
1. Ingests signed mutation payloads over ZeroMQ PULL into a thread-safe Metabolic Triage Queue.
2. Validates physical telemetry and enforces kinematic continuity (triggers Tombstones on spoofed FE).
3. Performs Eureka Collision deduplication against FAISS vector memory.
4. Packages validated mutations into .resin DSL skills, applies the Swarm Master Ed25519 signature,
   and broadcasts to all Foragers over ZeroMQ PUB.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import numpy as np
import zmq

from see.dream_sandbox.crucible_adapter import CrucibleAdapter
from see.immunity.clawhub_registry import ClawhubRegistry, ResinSkill, build_anomaly_vector
from see.mesh.crypto_enclave import CryptoEnclave, SignedEnvelope
from see.mesh.transport import (
    TOPIC_RESIN_SKILL,
    TOPIC_TOMBSTONE,
    ZMQ_FORAGER_TO_QUEEN,
    ZMQ_QUEEN_BROADCAST,
)
from see.mesh.triage import MetabolicTriageQueue

PHYSICS_MAX_FE_REDUCTION: float = 50000.0
PHYSICS_MAX_PRE_FE: float = 100000.0
FE_VALIDATION_DROP_PCT: float = 0.001


class QueenNode:
    """Master cluster coordinator node managing verification, triage, and skill distribution."""

    def __init__(
        self,
        node_id: str = "queen-ada",
        bind_zmq: bool = True,
        ingress_addr: str = ZMQ_FORAGER_TO_QUEEN,
        egress_addr: str = ZMQ_QUEEN_BROADCAST,
        registry_path: str | None = None,
    ) -> None:
        self.node_id = node_id
        self.bind_zmq = bind_zmq
        self.ingress_addr = ingress_addr
        self.egress_addr = egress_addr

        # Cryptographic Identity (Swarm Master)
        self.enclave = CryptoEnclave()
        self.master_pubkey = self.enclave.export_public()

        # Storage & Memory
        self.registry = ClawhubRegistry()
        if registry_path:
            self.registry.load_registry(registry_path)

        # Verification Engine
        self.crucible = CrucibleAdapter()

        # Metabolic Triage
        self.triage_queue = MetabolicTriageQueue()

        # Access Control & Blacklists
        self.authorized_foragers: dict[str, str] = {}  # pubkey -> node_id
        self.revoked_keys: set[str] = set()

        # Counters
        self.validated_count: int = 0
        self.rejected_count: int = 0
        self.spoofed_count: int = 0

        # Networking & Threads
        self.context: zmq.Context | None = None
        self.pull_sock: zmq.Socket | None = None
        self.pub_sock: zmq.Socket | None = None

        self._running = False
        self._listener_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None

        if self.bind_zmq:
            self._init_zmq()

    def _init_zmq(self) -> None:
        """Binds ZeroMQ PULL and PUB sockets."""
        self.context = zmq.Context()
        self.pull_sock = self.context.socket(zmq.PULL)
        self.pull_sock.bind(self.ingress_addr)

        self.pub_sock = self.context.socket(zmq.PUB)
        self.pub_sock.bind(self.egress_addr)

    def register_forager(self, node_id: str, pubkey_b64: str) -> None:
        """Authorizes a Forager node public key."""
        self.authorized_foragers[pubkey_b64] = node_id

    def start(self) -> None:
        """Starts background ZeroMQ listener and triage worker threads."""
        if not self.bind_zmq or self._running:
            return

        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listener_loop, daemon=True, name="QueenListener"
        )
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="QueenWorker"
        )
        self._listener_thread.start()
        self._worker_thread.start()

    def stop(self) -> None:
        """Stops threads and closes ZeroMQ sockets cleanly."""
        self._running = False
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=1.0)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

        if self.pull_sock:
            self.pull_sock.close()
        if self.pub_sock:
            self.pub_sock.close()
        if self.context:
            self.context.term()

    def _listener_loop(self) -> None:
        """Background loop receiving incoming Forager mutations over ZeroMQ PULL."""
        while self._running and self.pull_sock:
            try:
                if self.pull_sock.poll(100):
                    raw_str = self.pull_sock.recv_string()
                    self.ingest_mutation_envelope(raw_str)
            except Exception:
                pass

    def _worker_loop(self) -> None:
        """Background loop processing mutations from the Metabolic Triage Queue."""
        while self._running:
            try:
                self.process_next_triage_task(block=True, timeout=0.1)
            except Exception:
                pass

    def ingest_mutation_envelope(self, envelope_json: str) -> bool:
        """Verifies signature, checks revocation blacklist, and enqueues to Metabolic Triage."""
        try:
            envelope = SignedEnvelope.from_json(envelope_json)
            pubkey = envelope.node_pubkey

            if pubkey in self.revoked_keys:
                return False

            is_valid, delta = CryptoEnclave.unwrap_payload(envelope)
            if not is_valid or delta is None:
                return False

            pre_fe = float(delta.get("pre_morph_fe", 0.0))
            self.triage_queue.push(
                payload=delta,
                pre_morph_fe=pre_fe,
                pub_key_b64=pubkey,
            )
            return True
        except Exception:
            return False

    def validate_mutation(self, delta: dict[str, Any]) -> tuple[str, str]:
        """Validates Free Energy physics drop and guards against telemetry spoofing.

        Returns:
            ("VALID" | "INVALID" | "SPOOFED", reason)
        """
        pre_fe = float(delta.get("pre_morph_fe", 0.0))
        post_fe = float(delta.get("post_morph_fe", 0.0))
        fe_reduction = float(delta.get("fe_reduction", pre_fe - post_fe))

        # Physics Anti-Spoofing Check
        if fe_reduction > PHYSICS_MAX_FE_REDUCTION or pre_fe > PHYSICS_MAX_PRE_FE:
            return "SPOOFED", f"PHYSICS_ANOMALY: Impossible FE delta ({fe_reduction:.1f})"

        pct_drop = fe_reduction / max(pre_fe, 1.0)
        if pct_drop >= FE_VALIDATION_DROP_PCT and fe_reduction > 0:
            return "VALID", "FE_DROP_SATISFIED"

        return "INVALID", f"INSUFFICIENT_FE_DROP: {pct_drop:.4f} < {FE_VALIDATION_DROP_PCT}"

    def process_next_triage_task(
        self, block: bool = False, timeout: float | None = None
    ) -> dict[str, Any] | None:
        """Pops highest pain task, validates, checks Eureka collisions, and broadcasts skills."""
        if self.triage_queue.empty() and not block:
            return None

        try:
            item = self.triage_queue.pop(block=block, timeout=timeout)
        except Exception:
            return None

        delta = item.payload
        pubkey = item.pub_key_b64
        status, reason = self.validate_mutation(delta)

        if status == "SPOOFED":
            self.spoofed_count += 1
            if pubkey:
                self.broadcast_tombstone(
                    compromised_pubkey=pubkey,
                    reason=f"SPOOFED_TELEMETRY: {reason}",
                )
            return {"status": "SPOOFED", "reason": reason, "delta": delta}

        elif status == "VALID":
            # Anomaly Vector Extraction
            if "anomaly_vector" in delta:
                anomaly_vec = np.array(delta["anomaly_vector"], dtype=np.float32)
            else:
                anomaly_vec = build_anomaly_vector(
                    delta.get("anomaly_telemetry", {}),
                    float(delta.get("pre_morph_fe", 0.0)),
                )

            fe_red = float(delta.get("fe_reduction", 0.0))
            is_accepted, eureka_reason, existing_skill = self.registry.evaluate_eureka_collision(
                anomaly_vec, fe_red
            )

            if not is_accepted:
                self.rejected_count += 1
                return {
                    "status": "REJECTED_EUREKA_COLLISION",
                    "reason": eureka_reason,
                    "delta": delta,
                }

            # Compile into .resin skill
            skill = ResinSkill(delta=delta, node_id=self.node_id)
            signable_str = skill.get_signable_content()
            skill.signature = self.enclave.sign(signable_str)

            # Store in FAISS
            self.registry.store(anomaly_vec, skill)
            self.validated_count += 1

            # Broadcast to Swarm
            self.broadcast_skill(skill)

            return {
                "status": "VALIDATED",
                "skill_id": skill.skill_id,
                "skill": skill,
                "delta": delta,
            }

        else:
            self.rejected_count += 1
            return {"status": "REJECTED", "reason": reason, "delta": delta}

    def broadcast_tombstone(
        self, compromised_pubkey: str, reason: str = "PHYSICS_SPOOF_DETECTED"
    ) -> dict[str, Any]:
        """Revokes a compromised public key and broadcasts a signed Tombstone."""
        self.revoked_keys.add(compromised_pubkey)

        tombstone_data = {
            "action": "REVOKE_IDENTITY",
            "compromised_pubkey": compromised_pubkey,
            "reason": reason,
            "timestamp": time.time(),
        }
        envelope = self.enclave.wrap_payload(tombstone_data)
        envelope_bytes = envelope.to_json().encode("utf-8")

        if self.pub_sock:
            self.pub_sock.send_multipart([TOPIC_TOMBSTONE, envelope_bytes])

        return tombstone_data

    def broadcast_skill(self, skill: ResinSkill) -> None:
        """Broadcasts validated .resin skill over ZeroMQ PUB."""
        payload_bytes = json.dumps(skill.to_dict()).encode("utf-8")
        if self.pub_sock:
            self.pub_sock.send_multipart([TOPIC_RESIN_SKILL, payload_bytes])

    def stats(self) -> dict[str, Any]:
        """Returns runtime statistics for the Queen coordinator."""
        return {
            "node_id": self.node_id,
            "validated": self.validated_count,
            "rejected": self.rejected_count,
            "spoofed": self.spoofed_count,
            "triage_queue_size": self.triage_queue.qsize(),
            "faiss_size": self.registry.size,
            "revoked_keys_count": len(self.revoked_keys),
        }
