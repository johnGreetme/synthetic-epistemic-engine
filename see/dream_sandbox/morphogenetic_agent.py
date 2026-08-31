"""Synthetic Epistemic Engine — Morphogenetic Agent & Latent Arena.

Maintains pre-allocated weight matrices with boolean activation masks to allow
rapid edge neurogenesis and structural morphogenesis without triggering JAX JIT recompilation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

FEATURE_DIM: int = 8
MAX_ARENA_CAPACITY: int = 32
INITIAL_CAPACITY: int = 4
PAIN_TICKS_THRESHOLD: int = 3
APOPTOSIS_MAGNITUDE: float = 1e-4


@dataclass
class MorphogenesisEvent:
    """Record of a structural change to the latent arena."""

    tick: int
    event_type: str  # "NEUROGENESIS" | "APOPTOSIS" | "QUANTIZATION"
    node_index: int
    reason: str
    free_energy: float
    active_before: int
    active_after: int


class LatentArena:
    """Fixed-shape latent weight matrix governed by a dynamic boolean activation mask."""

    def __init__(
        self,
        max_capacity: int = MAX_ARENA_CAPACITY,
        initial_capacity: int = INITIAL_CAPACITY,
        feature_dim: int = FEATURE_DIM,
        seed: int = 42,
    ) -> None:
        self.max_capacity = max_capacity
        self.feature_dim = feature_dim
        self.rng = jax.random.PRNGKey(seed)

        self.weights = jax.random.normal(self.rng, shape=(max_capacity, feature_dim)) * jnp.sqrt(
            2.0 / feature_dim
        )

        mask_values = [1.0] * initial_capacity + [0.0] * (max_capacity - initial_capacity)
        self.mask = jnp.array(mask_values)
        self.node_age = np.zeros(max_capacity, dtype=np.int32)

    @property
    def active_count(self) -> int:
        """Returns number of active slots."""
        return int(jnp.sum(self.mask))

    def next_dormant_slot(self) -> int | None:
        """Finds the lowest index of an inactive dormant slot."""
        for i in range(self.max_capacity):
            if float(self.mask[i]) == 0.0:
                return i
        return None

    def activate_slot(
        self,
        slot_index: int,
        initial_weights: np.ndarray | jnp.ndarray | None = None,
    ) -> bool:
        """Activates a dormant slot with optional weight initialization."""
        if slot_index < 0 or slot_index >= self.max_capacity:
            return False

        if initial_weights is not None:
            w_arr = np.array(initial_weights, dtype=np.float32)
            if len(w_arr) < self.feature_dim:
                w_arr = np.pad(w_arr, (0, self.feature_dim - len(w_arr)))
            else:
                w_arr = w_arr[: self.feature_dim]
            self.weights = self.weights.at[slot_index].set(jnp.array(w_arr))

        self.mask = self.mask.at[slot_index].set(1.0)
        self.node_age[slot_index] = 0
        return True

    def deactivate_slot(self, slot_index: int) -> bool:
        """Deactivates a slot, marking it dormant."""
        if slot_index < 0 or slot_index >= self.max_capacity:
            return False
        self.mask = self.mask.at[slot_index].set(0.0)
        self.node_age[slot_index] = 0
        return True


class MorphogeneticAgent:
    """Edge agent that monitors Free Energy and triggers latent neurogenesis."""

    def __init__(
        self,
        max_capacity: int = MAX_ARENA_CAPACITY,
        initial_capacity: int = INITIAL_CAPACITY,
        feature_dim: int = FEATURE_DIM,
        pain_ticks_threshold: int = PAIN_TICKS_THRESHOLD,
    ) -> None:
        self.arena = LatentArena(
            max_capacity=max_capacity,
            initial_capacity=initial_capacity,
            feature_dim=feature_dim,
        )
        self.pain_ticks_threshold = pain_ticks_threshold
        self.consecutive_pain_ticks: int = 0
        self.tick_count: int = 0
        self.events: list[MorphogenesisEvent] = []

    def update(
        self, free_energy: float, pain_threshold: float = 500.0
    ) -> MorphogenesisEvent | None:
        """Processes a tick of Free Energy and triggers neurogenesis if pain is sustained."""
        self.tick_count += 1

        if free_energy > pain_threshold:
            self.consecutive_pain_ticks += 1
        else:
            self.consecutive_pain_ticks = 0

        if self.consecutive_pain_ticks >= self.pain_ticks_threshold:
            slot = self.arena.next_dormant_slot()
            if slot is not None:
                active_before = self.arena.active_count
                self.arena.activate_slot(slot)
                active_after = self.arena.active_count

                event = MorphogenesisEvent(
                    tick=self.tick_count,
                    event_type="NEUROGENESIS",
                    node_index=slot,
                    reason=f"Sustained FE ({free_energy:.1f}) exceeded threshold ({pain_threshold:.1f}) for {self.consecutive_pain_ticks} ticks",
                    free_energy=free_energy,
                    active_before=active_before,
                    active_after=active_after,
                )
                self.events.append(event)
                # Reset counter after triggering
                self.consecutive_pain_ticks = 0
                return event

        return None
