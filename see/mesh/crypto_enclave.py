"""Synthetic Epistemic Engine — Ed25519 Cryptographic Enclave.

Provides hardware-isolated zero-trust identity and cryptographic signing for all
swarm messages using Ed25519 keypairs. Ensures zero-trust payload verification across
both Forager PUSH streams and Queen Gossipsub broadcasts.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


@dataclass
class SignedEnvelope:
    """Standard cryptographic envelope wrapping serialized JSON payload and Ed25519 signature."""

    payload: str
    signature_b64: str
    node_pubkey: str

    def to_json(self) -> str:
        """Serializes envelope to JSON string."""
        return json.dumps(
            {
                "payload": self.payload,
                "signature_b64": self.signature_b64,
                "node_pubkey": self.node_pubkey,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> SignedEnvelope:
        """Parses envelope from JSON string."""
        data = json.loads(json_str)
        return cls(
            payload=data["payload"],
            signature_b64=data["signature_b64"],
            node_pubkey=data["node_pubkey"],
        )


class CryptoEnclave:
    """Manages Ed25519 private/public key operations and payload verification."""

    def __init__(self, private_key: ed25519.Ed25519PrivateKey | None = None) -> None:
        if private_key is None:
            self._private_key = ed25519.Ed25519PrivateKey.generate()
        else:
            self._private_key = private_key
        self._public_key = self._private_key.public_key()

    @property
    def public_key(self) -> ed25519.Ed25519PublicKey:
        """Returns internal Ed25519 public key instance."""
        return self._public_key

    def sign(self, message: bytes | str) -> str:
        """Signs a message using the enclave's Ed25519 private key, returning Base64 string."""
        data = message.encode("utf-8") if isinstance(message, str) else message
        sig_bytes = self._private_key.sign(data)
        return base64.b64encode(sig_bytes).decode("utf-8")

    @staticmethod
    def verify(
        public_key: ed25519.Ed25519PublicKey | str,
        signature_b64: str,
        message: bytes | str,
    ) -> bool:
        """Verifies an Ed25519 signature against a message and public key."""
        try:
            pub = (
                CryptoEnclave.import_public(public_key)
                if isinstance(public_key, str)
                else public_key
            )
            sig_bytes = base64.b64decode(signature_b64)
            data = message.encode("utf-8") if isinstance(message, str) else message
            pub.verify(sig_bytes, data)
            return True
        except (InvalidSignature, Exception):
            return False

    def export_public(self) -> str:
        """Exports the raw public key as a Base64-encoded string."""
        raw_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return base64.b64encode(raw_bytes).decode("utf-8")

    @staticmethod
    def import_public(pub_b64: str) -> ed25519.Ed25519PublicKey:
        """Imports an Ed25519 public key from a Base64 string."""
        raw_bytes = base64.b64decode(pub_b64)
        return ed25519.Ed25519PublicKey.from_public_bytes(raw_bytes)

    def wrap_payload(self, payload_obj: dict[str, Any]) -> SignedEnvelope:
        """Serializes dictionary to canonical JSON string and generates a signed envelope."""
        canonical_json = json.dumps(payload_obj, sort_keys=True)
        sig = self.sign(canonical_json)
        return SignedEnvelope(
            payload=canonical_json,
            signature_b64=sig,
            node_pubkey=self.export_public(),
        )

    @staticmethod
    def unwrap_payload(envelope: SignedEnvelope) -> tuple[bool, dict[str, Any] | None]:
        """Validates envelope signature and returns (is_valid, parsed_dict)."""
        is_valid = CryptoEnclave.verify(
            public_key=envelope.node_pubkey,
            signature_b64=envelope.signature_b64,
            message=envelope.payload,
        )
        if not is_valid:
            return False, None
        try:
            parsed = json.loads(envelope.payload)
            return True, parsed
        except Exception:
            return False, None
