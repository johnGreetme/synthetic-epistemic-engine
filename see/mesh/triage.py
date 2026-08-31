"""Synthetic Epistemic Engine — Metabolic Triage Queue.

Implements a thread-safe priority queue ordered strictly by mathematical pain (-pre_morph_fe).
Nodes in the most severe physical agony are processed first by the Queen's verification workers.
Includes explicit tie-breaking (timestamp and monotonic sequence counter) to ensure identical
Free Energy values never trigger comparison errors on arbitrary payload dicts.
"""

from __future__ import annotations

import itertools
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(order=True)
class TriageItem:
    """Entry stored in the metabolic triage priority queue with explicit tie-breaking."""

    priority: float
    timestamp: float
    sequence_id: int
    payload: Dict[str, Any] = field(compare=False)
    pub_key_b64: Optional[str] = field(default=None, compare=False)

    @property
    def free_energy(self) -> float:
        """Returns the original positive Free Energy value."""
        return -self.priority


class MetabolicTriageQueue:
    """Thread-safe priority queue sorting incoming Forager mutations by highest pain."""

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: queue.PriorityQueue[TriageItem] = queue.PriorityQueue(
            maxsize=maxsize
        )
        self._counter = itertools.count()

    def push(
        self,
        payload: Dict[str, Any],
        pre_morph_fe: float,
        pub_key_b64: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> TriageItem:
        """Pushes a mutation payload into the priority queue ordered by -pre_morph_fe."""
        ts = time.time() if timestamp is None else float(timestamp)
        seq = next(self._counter)
        priority = -float(pre_morph_fe)

        item = TriageItem(
            priority=priority,
            timestamp=ts,
            sequence_id=seq,
            payload=payload,
            pub_key_b64=pub_key_b64,
        )
        self._queue.put(item)
        return item

    def pop(self, block: bool = True, timeout: Optional[float] = None) -> TriageItem:
        """Pops the highest-priority (highest physical pain) mutation from the queue."""
        return self._queue.get(block=block, timeout=timeout)

    def pop_payload(
        self, block: bool = True, timeout: Optional[float] = None
    ) -> Tuple[Dict[str, Any], float, Optional[str]]:
        """Convenience method returning (payload, pre_morph_fe, pub_key_b64)."""
        item = self.pop(block=block, timeout=timeout)
        return item.payload, item.free_energy, item.pub_key_b64

    def empty(self) -> bool:
        """Returns True if queue is empty."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Returns current number of queued items."""
        return self._queue.qsize()

    def task_done(self) -> None:
        """Marks a previously popped task as processed."""
        self._queue.task_done()
