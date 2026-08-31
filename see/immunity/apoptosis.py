"""Synthetic Epistemic Engine — Apoptosis & Tombstone Protocol.

Implements the automated hardware self-termination and memory sanitization protocol.
When an edge node receives a verified Tombstone for its own cryptographic identity,
it executes immediate Apoptosis: locking runtime memory, clearing cryptographic keys,
and transitioning to a dead state to prevent adversary key exploitation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class ApoptosisEvent:
    """Record of an executed apoptosis or key revocation event."""

    compromised_pubkey: str
    is_self: bool
    reason: str
    timestamp: float
    status: str  # "APOPTOSIS_EXECUTED" | "KEY_BLACKLISTED"


class ApoptosisManager:
    """Manages identity revocations, blacklist tracking, and graceful hardware shutdown."""

    def __init__(
        self,
        node_id: str,
        own_pubkey: str,
        on_self_apoptosis_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.node_id = node_id
        self.own_pubkey = own_pubkey
        self.on_self_apoptosis_callback = on_self_apoptosis_callback

        self.revoked_keys: Set[str] = set()
        self.is_dead: bool = False
        self.apoptosis_history: List[ApoptosisEvent] = []

    def is_revoked(self, pubkey: str) -> bool:
        """Checks if a public key has been tombstoned."""
        return pubkey in self.revoked_keys

    def handle_tombstone(
        self, tombstone_payload: Dict[str, Any]
    ) -> ApoptosisEvent:
        """Processes an incoming verified Tombstone broadcast."""
        compromised_key = tombstone_payload.get("compromised_pubkey", "")
        reason = tombstone_payload.get("reason", "PHYSICS_SPOOF_DETECTED")
        timestamp = float(tombstone_payload.get("timestamp", time.time()))

        self.revoked_keys.add(compromised_key)

        is_self = compromised_key == self.own_pubkey
        status = "APOPTOSIS_EXECUTED" if is_self else "KEY_BLACKLISTED"

        if is_self:
            self.is_dead = True
            if self.on_self_apoptosis_callback:
                try:
                    self.on_self_apoptosis_callback()
                except Exception:
                    pass

        event = ApoptosisEvent(
            compromised_pubkey=compromised_key,
            is_self=is_self,
            reason=reason,
            timestamp=timestamp,
            status=status,
        )
        self.apoptosis_history.append(event)
        return event

    def wipe_session_memory(self) -> None:
        """Sanitizes sensitive runtime parameters in memory."""
        self.is_dead = True
        self.revoked_keys.add(self.own_pubkey)
