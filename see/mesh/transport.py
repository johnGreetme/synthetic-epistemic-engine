"""Synthetic Epistemic Engine — ZeroMQ Mesh Transport & Gossipsub.

Implements asynchronous ZeroMQ network transport primitives:
- Ingress (Foragers -> Queen): zmq.PUSH -> zmq.PULL on port 5577
- Egress (Queen -> Foragers): zmq.PUB -> zmq.SUB on port 5578
"""

from __future__ import annotations

import zmq

ZMQ_FORAGER_TO_QUEEN: str = "tcp://127.0.0.1:5577"
ZMQ_QUEEN_BROADCAST: str = "tcp://127.0.0.1:5578"

TOPIC_RESIN_SKILL: bytes = b"RESIN_SKILL"
TOPIC_TOMBSTONE: bytes = b"TOMBSTONE"


class MeshTransport:
    """Manages ZeroMQ sockets and messaging for Queen and Forager nodes."""

    def __init__(
        self,
        context: zmq.Context | None = None,
        ingress_addr: str = ZMQ_FORAGER_TO_QUEEN,
        egress_addr: str = ZMQ_QUEEN_BROADCAST,
    ) -> None:
        self.context = context or zmq.Context()
        self.ingress_addr = ingress_addr
        self.egress_addr = egress_addr

    def create_queen_sockets(self) -> tuple[zmq.Socket, zmq.Socket]:
        """Binds Queen Ingress (PULL) and Egress (PUB) sockets."""
        pull_sock = self.context.socket(zmq.PULL)
        pull_sock.bind(self.ingress_addr)

        pub_sock = self.context.socket(zmq.PUB)
        pub_sock.bind(self.egress_addr)

        return pull_sock, pub_sock

    def create_forager_sockets(
        self, topics: list[bytes] | None = None
    ) -> tuple[zmq.Socket, zmq.Socket]:
        """Connects Forager Egress (PUSH) and Ingress (SUB) sockets."""
        push_sock = self.context.socket(zmq.PUSH)
        push_sock.connect(self.ingress_addr)

        sub_sock = self.context.socket(zmq.SUB)
        sub_sock.connect(self.egress_addr)

        sub_topics = topics or [TOPIC_RESIN_SKILL, TOPIC_TOMBSTONE]
        for t in sub_topics:
            sub_sock.setsockopt(zmq.SUBSCRIBE, t)

        return push_sock, sub_sock

    @staticmethod
    def send_broadcast(pub_socket: zmq.Socket, topic: bytes, payload_bytes: bytes) -> None:
        """Publishes a multipart message to subscribers."""
        pub_socket.send_multipart([topic, payload_bytes])

    @staticmethod
    def poll_and_recv(
        socket: zmq.Socket, timeout_ms: int = 100
    ) -> str | tuple[bytes, bytes] | None:
        """Polls socket and receives single or multipart message."""
        if socket.poll(timeout_ms):
            if socket.type == zmq.SUB:
                topic, data = socket.recv_multipart()
                return topic, data
            else:
                return socket.recv_string()
        return None
