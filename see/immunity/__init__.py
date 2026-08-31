"""Immunity, FAISS Vector Registry, and Tombstone Apoptosis module for SEE."""

from see.immunity.clawhub_registry import (
    ClawhubRegistry,
    ResinSkill,
    build_anomaly_vector,
    FAISS_ANOMALY_DIM,
    EUREKA_L2_THRESHOLD,
    EUREKA_EFFICIENCY_MARGIN,
)
from see.immunity.apoptosis import ApoptosisManager, ApoptosisEvent

__all__ = [
    "ClawhubRegistry",
    "ResinSkill",
    "build_anomaly_vector",
    "FAISS_ANOMALY_DIM",
    "EUREKA_L2_THRESHOLD",
    "EUREKA_EFFICIENCY_MARGIN",
    "ApoptosisManager",
    "ApoptosisEvent",
]
