"""Immunity, FAISS Vector Registry, and Tombstone Apoptosis module for SEE."""

from see.immunity.apoptosis import ApoptosisEvent, ApoptosisManager
from see.immunity.clawhub_registry import (
    EUREKA_EFFICIENCY_MARGIN,
    EUREKA_L2_THRESHOLD,
    FAISS_ANOMALY_DIM,
    ClawhubRegistry,
    ResinSkill,
    build_anomaly_vector,
)

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
