"""Synthetic Epistemic Engine — FAISS Clawhub Skill Registry & Eureka Deduplication.

Maintains an exact L2 nearest-neighbour vector database (IndexFlatL2) of all learned
physical anomaly bypasses across the swarm. Implements Eureka Collision deduplication:
if an incoming mutation matches an existing anomaly vector (L2 < 0.1), the new mutation
must provide >20% higher mechanical efficiency (FE reduction) than the stored skill,
or it is discarded as a redundant duplicate.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

FAISS_ANOMALY_DIM: int = 6
EUREKA_L2_THRESHOLD: float = 0.1
EUREKA_EFFICIENCY_MARGIN: float = 1.20  # Requires > 20% improvement
DEFAULT_REGISTRY_PATH: str = "clawhub_registry.json"


def build_anomaly_vector(
    telemetry: Dict[str, float], free_energy: float
) -> np.ndarray:
    """Extracts a 6-dimensional normalized anomaly feature vector for FAISS indexing."""
    heartbeat = float(telemetry.get("slp_heartbeat", 8.0))
    flux = float(telemetry.get("sensory_flux", 6.4))
    hb_delta = heartbeat - 8.0
    flux_delta = flux - 6.4
    fe_normalized = free_energy / 1000.0
    severity = min(max(free_energy / 50000.0, 0.0), 1.0)

    vec = np.array(
        [heartbeat, flux, fe_normalized, hb_delta, flux_delta, severity],
        dtype=np.float32,
    )
    return vec


class ResinSkill:
    """Represents a validated morphogenetic mutation packaged as a portable .resin skill."""

    def __init__(
        self,
        delta: Dict[str, Any],
        node_id: str = "queen-ada",
        skill_id: Optional[str] = None,
        signature: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> None:
        self.skill_id = skill_id or str(uuid.uuid4())[:8]
        self.node_id = node_id
        self.delta = delta
        self.signature = signature
        self.created_at = created_at or time.time()

    def get_signable_content(self) -> str:
        """Returns deterministic JSON string representation without signature for cryptographic signing."""
        d = self.to_dict()
        d.pop("signature", None)
        return json.dumps(d, sort_keys=True)

    def to_resin(self) -> str:
        """Formats the skill into human-readable, domain-specific .resin DSL."""
        anomaly_telemetry = self.delta.get("anomaly_telemetry", {})
        hb = float(anomaly_telemetry.get("slp_heartbeat", 8.0))
        flux = float(anomaly_telemetry.get("sensory_flux", 6.4))
        fe_pre = float(self.delta.get("pre_morph_fe", 0.0))
        fe_red = float(self.delta.get("fe_reduction", 0.0))
        slot_idx = self.delta.get("slot_index", 0)
        weight_dim = self.delta.get("weight_dim", 8)
        w_b64 = str(self.delta.get("weight_b64", ""))[:40]

        pct_drop = (fe_red / max(fe_pre, 1.0)) * 100.0

        resin_body = f"""skill MorphogeneticImmuneResponse {{
  version:      "1.0.0"
  skill_id:     "{self.skill_id}"
  author_node:  "{self.node_id}"
  created_at:   {self.created_at:.0f}

  // Sensory trigger pattern
  trigger {{
    sensor:     "telemetry.sensory_flux"
    condition:  "free_energy > {fe_pre * 0.8:.1f}"
    hb_range:   [{hb - 2.0:.1f}, {hb + 2.0:.1f}]
    flux_range: [{flux - 1.0:.1f}, {flux + 1.0:.1f}]
  }}

  // Structural topology patch
  topology_patch {{
    action:      "activate_latent_slot"
    slot_index:  {slot_idx}
    weight_dim:  {weight_dim}
    weight_b64:  "{w_b64}..."
  }}

  // Mechanical efficiency validation
  validation {{
    expected_fe_reduction:   {fe_red:.2f}
    min_fe_reduction_pct:    {pct_drop:.1f}
    max_stabilization_ticks: 10
  }}"""

        if self.signature:
            resin_body += f"""
  // Cryptographic Seal (Ed25519)
  security {{
    signature: "{self.signature[:40]}..."
  }}"""

        resin_body += "\n}"
        return resin_body

    def to_dict(self) -> Dict[str, Any]:
        """Converts skill to dictionary for JSON persistence/network transmission."""
        return {
            "skill_id": self.skill_id,
            "node_id": self.node_id,
            "delta": self.delta,
            "created_at": self.created_at,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ResinSkill:
        """Instantiates skill from parsed dictionary."""
        return cls(
            delta=d["delta"],
            node_id=d.get("node_id", "queen-ada"),
            skill_id=d.get("skill_id"),
            signature=d.get("signature"),
            created_at=d.get("created_at"),
        )


class ClawhubRegistry:
    """FAISS-backed local vector memory for morphogenetic skills and Eureka deduplication."""

    def __init__(self, dim: int = FAISS_ANOMALY_DIM) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.skills: List[ResinSkill] = []
        self.vectors: List[np.ndarray] = []

    @property
    def size(self) -> int:
        """Returns total number of registered skills."""
        return self.index.ntotal

    def store(self, anomaly_vector: np.ndarray, skill: ResinSkill) -> None:
        """Stores a verified skill and its anomaly vector in the FAISS index."""
        vec = anomaly_vector.reshape(1, self.dim).astype(np.float32)
        self.index.add(vec)
        self.skills.append(skill)
        self.vectors.append(anomaly_vector)

    def query(
        self,
        anomaly_vector: np.ndarray,
        top_k: int = 1,
        distance_threshold: float = 50.0,
    ) -> Tuple[Optional[ResinSkill], float]:
        """Finds closest skill within the distance threshold, or returns (None, distance)."""
        if self.index.ntotal == 0:
            return None, float("inf")

        vec = anomaly_vector.reshape(1, self.dim).astype(np.float32)
        distances, indices = self.index.search(vec, top_k)

        best_dist = float(distances[0][0])
        best_idx = int(indices[0][0])

        if best_dist <= distance_threshold and best_idx < len(self.skills):
            return self.skills[best_idx], best_dist

        return None, best_dist

    def evaluate_eureka_collision(
        self, anomaly_vector: np.ndarray, new_fe_reduction: float
    ) -> Tuple[bool, Optional[str], Optional[ResinSkill]]:
        """Evaluates whether a new mutation is a redundant Eureka collision or a viable upgrade.

        Returns:
            (is_accepted, reason, existing_skill_or_none)
        """
        existing_skill, dist = self.query(
            anomaly_vector, distance_threshold=EUREKA_L2_THRESHOLD
        )
        if existing_skill is None:
            return True, "NO_COLLISION", None

        existing_reduction = float(existing_skill.delta.get("fe_reduction", 0.0))
        required_reduction = existing_reduction * EUREKA_EFFICIENCY_MARGIN

        if new_fe_reduction > required_reduction:
            reason = (
                f"EUREKA_UPGRADE_ACCEPTED: New FE reduction ({new_fe_reduction:.1f}) "
                f"> 120% of existing ({existing_reduction:.1f})"
            )
            return True, reason, existing_skill
        else:
            reason = (
                f"EUREKA_COLLISION_REJECTED: New FE reduction ({new_fe_reduction:.1f}) "
                f"is not >20% better than existing ({existing_reduction:.1f})"
            )
            return False, reason, existing_skill

    def save_registry(self, path: str = DEFAULT_REGISTRY_PATH) -> None:
        """Persists all skills to JSON file."""
        data = [s.to_dict() for s in self.skills]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_registry(self, path: str = DEFAULT_REGISTRY_PATH) -> int:
        """Loads skills from JSON file and populates the FAISS index."""
        p = Path(path)
        if not p.exists():
            return 0

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_count = 0
        for item in data:
            skill = ResinSkill.from_dict(item)
            delta = skill.delta
            if "anomaly_vector" in delta:
                vec = np.array(delta["anomaly_vector"], dtype=np.float32)
            else:
                telemetry = delta.get("anomaly_telemetry", {})
                fe = float(delta.get("pre_morph_fe", 0.0))
                vec = build_anomaly_vector(telemetry, fe)
            self.store(vec, skill)
            loaded_count += 1

        return loaded_count
