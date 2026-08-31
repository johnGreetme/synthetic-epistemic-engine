"""Synthetic Epistemic Engine — Forager Node.

Represents an edge robotic agent operating on Jetson AGX Thor hardware:
1. Runs local SVI Artificial Nociception to monitor Free Energy and physical pain.
2. Checks local FAISS memory for instant immunity before undergoing neurogenesis.
3. Packages and signs novel morphogenetic mutations and pushes them to the Queen over ZeroMQ PUSH.
4. Listens for Queen broadcasts (RESIN_SKILL and TOMBSTONE) over ZeroMQ SUB.
5. Enforces Split-Brain Sovereignty (Limp Mode) when offline, clamping torque limits to 1.0 N*m.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import numpy as np
import zmq

from see.active_inference.nociception import NociceptionEngine, NociceptionEvent
from see.dream_sandbox.crucible_adapter import CrucibleAdapter
from see.dream_sandbox.morphogenetic_agent import MorphogeneticAgent
from see.immunity.apoptosis import ApoptosisManager
from see.immunity.clawhub_registry import ClawhubRegistry, ResinSkill, build_anomaly_vector
from see.mesh.crypto_enclave import CryptoEnclave, SignedEnvelope
from see.mesh.transport import (
    TOPIC_RESIN_SKILL,
    TOPIC_TOMBSTONE,
    ZMQ_FORAGER_TO_QUEEN,
    ZMQ_QUEEN_BROADCAST,
)


class ForagerNode:
    """Sovereign edge robotic node executing active inference, morphogenetic bypasses, and ZMQ mesh."""

    LIMP_MODE_TORQUE_LIMIT: float = 1.0
    NORMAL_TORQUE_LIMIT: float = 5.0

    def __init__(
        self,
        node_id: str = "forager-thor-alpha",
        queen_pubkey: str | None = None,
        connect_zmq: bool = True,
        ingress_addr: str = ZMQ_FORAGER_TO_QUEEN,
        egress_addr: str = ZMQ_QUEEN_BROADCAST,
    ) -> None:
        self.node_id = node_id
        self.connect_zmq = connect_zmq
        self.ingress_addr = ingress_addr
        self.egress_addr = egress_addr

        # Cryptographic Identity & Enclave
        self.enclave = CryptoEnclave()
        self.pubkey = self.enclave.export_public()
        self.queen_pubkey = queen_pubkey

        # Subsystems
        self.nociception = NociceptionEngine()
        self.agent = MorphogeneticAgent()
        self.local_registry = ClawhubRegistry()
        self.crucible = CrucibleAdapter()

        # Apoptosis
        self.apoptosis = ApoptosisManager(
            node_id=self.node_id,
            own_pubkey=self.pubkey,
            on_self_apoptosis_callback=self._on_apoptosis,
        )

        # State flags
        self.offline: bool = False
        self.mutations_sent: int = 0
        self.skills_applied: int = 0

        # Networking & Background Listener
        self.context: zmq.Context | None = None
        self.push_sock: zmq.Socket | None = None
        self.sub_sock: zmq.Socket | None = None

        self._running = False
        self._listener_thread: threading.Thread | None = None

        if self.connect_zmq:
            self._init_zmq()

    def _init_zmq(self) -> None:
        """Connects ZeroMQ PUSH and SUB sockets."""
        self.context = zmq.Context()
        self.push_sock = self.context.socket(zmq.PUSH)
        self.push_sock.connect(self.ingress_addr)

        self.sub_sock = self.context.socket(zmq.SUB)
        self.sub_sock.connect(self.egress_addr)
        self.sub_sock.setsockopt(zmq.SUBSCRIBE, TOPIC_RESIN_SKILL)
        self.sub_sock.setsockopt(zmq.SUBSCRIBE, TOPIC_TOMBSTONE)

    def start(self) -> None:
        """Starts background ZeroMQ broadcast subscriber thread."""
        if not self.connect_zmq or self._running or self.apoptosis.is_dead:
            return

        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listener_loop, daemon=True, name=f"ForagerListener-{self.node_id}"
        )
        self._listener_thread.start()

    def stop(self) -> None:
        """Stops background threads and closes ZeroMQ sockets."""
        self._running = False
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=1.0)

        if self.push_sock:
            self.push_sock.close()
        if self.sub_sock:
            self.sub_sock.close()
        if self.context:
            self.context.term()

    def _on_apoptosis(self) -> None:
        """Callback invoked when this node's identity is tombstoned."""
        self._running = False
        if self.push_sock:
            self.push_sock.close()
        if self.sub_sock:
            self.sub_sock.close()

    def _listener_loop(self) -> None:
        """Processes incoming Queen broadcasts."""
        while self._running and self.sub_sock and not self.apoptosis.is_dead:
            try:
                if self.sub_sock.poll(100):
                    topic, payload_bytes = self.sub_sock.recv_multipart()
                    if topic == TOPIC_RESIN_SKILL:
                        self._handle_skill_broadcast(payload_bytes)
                    elif topic == TOPIC_TOMBSTONE:
                        self._handle_tombstone_broadcast(payload_bytes)
            except Exception:
                pass

    def _handle_skill_broadcast(self, payload_bytes: bytes) -> bool:
        """Validates Queen Ed25519 signature and injects skill into local FAISS registry."""
        try:
            skill_dict = json.loads(payload_bytes.decode("utf-8"))
            skill = ResinSkill.from_dict(skill_dict)

            # Signature verification if Queen public key is registered
            if self.queen_pubkey:
                signable_str = skill.get_signable_content()
                if not skill.signature or not CryptoEnclave.verify(
                    self.queen_pubkey, skill.signature, signable_str
                ):
                    return False

            delta = skill.delta
            if "anomaly_vector" in delta:
                anomaly_vec = np.array(delta["anomaly_vector"], dtype=np.float32)
            else:
                anomaly_vec = build_anomaly_vector(
                    delta.get("anomaly_telemetry", {}),
                    float(delta.get("pre_morph_fe", 0.0)),
                )

            self.local_registry.store(anomaly_vec, skill)
            return True
        except Exception:
            return False

    def _handle_tombstone_broadcast(self, payload_bytes: bytes) -> None:
        """Processes Tombstone revocation envelope."""
        try:
            envelope = SignedEnvelope.from_json(payload_bytes.decode("utf-8"))
            if self.queen_pubkey:
                is_valid = CryptoEnclave.verify(
                    self.queen_pubkey, envelope.signature_b64, envelope.payload
                )
                if not is_valid:
                    return

            tombstone_data = json.loads(envelope.payload)
            self.apoptosis.handle_tombstone(tombstone_data)
        except Exception:
            pass

    def check_local_immunity(
        self, telemetry: dict[str, float], free_energy: float
    ) -> tuple[bool, ResinSkill | None]:
        """Queries local FAISS registry for a previously learned structural patch."""
        vec = build_anomaly_vector(telemetry, free_energy)
        skill, distance = self.local_registry.query(vec, distance_threshold=0.1)

        if skill is not None:
            slot = self.agent.arena.next_dormant_slot()
            if slot is not None:
                self.agent.arena.activate_slot(slot)
                self.skills_applied += 1
                return True, skill

        return False, None

    def process_telemetry_tick(
        self, telemetry: dict[str, float]
    ) -> tuple[NociceptionEvent, dict[str, Any] | None]:
        """Runs a tick of active inference nociception and coordinates immunity or morphogenesis."""
        if self.apoptosis.is_dead:
            raise RuntimeError(f"Node {self.node_id} is dead due to Apoptosis.")

        event = self.nociception.update(telemetry)
        result_action: dict[str, Any] | None = None

        if event.pain_threshold_exceeded:
            # 1. Check local FAISS immunity
            has_immunity, skill = self.check_local_immunity(telemetry, event.free_energy)
            if has_immunity and skill:
                result_action = {
                    "action": "APPLIED_LOCAL_IMMUNITY",
                    "skill_id": skill.skill_id,
                    "free_energy": event.free_energy,
                }
                return event, result_action

            # 2. Trigger Morphogenesis
            morph_event = self.agent.update(
                event.free_energy, pain_threshold=self.nociception.pain_threshold
            )
            if morph_event:
                # 3. Check Split-Brain Sovereignty (Limp Mode)
                if self.offline:
                    # Enforce conservative torque clamping
                    limp_verification = self.crucible.verify_kinematics(
                        target_state={"required_torque": self.LIMP_MODE_TORQUE_LIMIT},
                        torque_limit=self.LIMP_MODE_TORQUE_LIMIT,
                    )
                    result_action = {
                        "action": "SPLIT_BRAIN_LIMP_MODE",
                        "torque_limit": self.LIMP_MODE_TORQUE_LIMIT,
                        "verification": limp_verification,
                        "slot_index": morph_event.node_index,
                    }
                    return event, result_action

                # 4. Transmit Mutation to Queen
                pre_fe = event.free_energy
                post_fe = max(0.0, pre_fe * 0.2)  # Projected reduction
                delta = self._create_mutation_delta(
                    slot_index=morph_event.node_index,
                    telemetry=telemetry,
                    pre_fe=pre_fe,
                    post_fe=post_fe,
                )
                self.transmit_mutation(delta)
                result_action = {
                    "action": "TRANSMITTED_MUTATION_TO_QUEEN",
                    "delta": delta,
                }

        return event, result_action

    def _create_mutation_delta(
        self,
        slot_index: int,
        telemetry: dict[str, float],
        pre_fe: float,
        post_fe: float,
    ) -> dict[str, Any]:
        """Packages local structural weights and anomaly telemetry into mutation payload."""
        weights = np.array(self.agent.arena.weights[slot_index])
        vec = build_anomaly_vector(telemetry, pre_fe)

        return {
            "node_id": self.node_id,
            "slot_index": slot_index,
            "weight_dim": len(weights),
            "weight_b64": "MOCK_WEIGHTS_B64",
            "anomaly_telemetry": telemetry,
            "anomaly_vector": vec.tolist(),
            "pre_morph_fe": pre_fe,
            "post_morph_fe": post_fe,
            "fe_reduction": pre_fe - post_fe,
            "timestamp": time.time(),
        }

    def transmit_mutation(self, delta: dict[str, Any]) -> SignedEnvelope | None:
        """Signs and transmits mutation payload to Queen over ZeroMQ PUSH."""
        if self.offline or self.apoptosis.is_dead:
            return None

        envelope = self.enclave.wrap_payload(delta)
        if self.push_sock:
            self.push_sock.send_string(envelope.to_json())
            self.mutations_sent += 1

        return envelope

    def stats(self) -> dict[str, Any]:
        """Returns runtime telemetry for the Forager node."""
        return {
            "node_id": self.node_id,
            "is_dead": self.apoptosis.is_dead,
            "offline": self.offline,
            "mutations_sent": self.mutations_sent,
            "skills_applied": self.skills_applied,
            "arena_active": self.agent.arena.active_count,
            "faiss_size": self.local_registry.size,
        }
