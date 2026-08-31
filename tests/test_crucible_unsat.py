"""Unit tests for Z3 Crucible dynamic submodule adapter and UNSAT core extraction."""

import pytest
from see.dream_sandbox.crucible_adapter import (
    CrucibleAdapter,
    SemanticBoundingBox,
    CrucibleVerificationResult,
    _resolve_diana_core,
)


def test_diana_core_submodule_dynamic_resolution():
    """Verifies that diana_core microkernel engine is located and loaded."""
    module = _resolve_diana_core()
    assert module is not None, "Failed to resolve diana_core submodule or z3_crucible engine"
    assert hasattr(module, "verify_invariants")


def test_z3_crucible_sat_verification():
    """Verifies that valid kinematics within physical bounds return SAT (is_safe=True)."""
    adapter = CrucibleAdapter()
    safe_state = {
        "required_torque": 3.5,
        "position_delta_rad": 0.5,
    }

    result = adapter.verify_kinematics(safe_state, torque_limit=5.0)
    assert result.is_safe is True
    assert "SATISFIABLE" in result.status
    assert result.bounding_box is None


def test_z3_crucible_unsat_core_and_semantic_bounding_box():
    """Verifies that excessive torque violates invariants, producing UNSAT core and semantic bounding box."""
    adapter = CrucibleAdapter()
    unsafe_state = {
        "required_torque": 25.0,  # Exceeds max 5.0 N*m limit
        "position_delta_rad": 0.1,
    }

    result = adapter.verify_kinematics(unsafe_state, torque_limit=5.0)
    assert result.is_safe is False
    assert "UNSATISFIABLE" in result.status
    assert len(result.unsat_core) > 0

    bbox = result.bounding_box
    assert isinstance(bbox, SemanticBoundingBox)
    assert "UNSAT" in bbox.prompt_feedback
    assert "5.0 N*m" in bbox.prompt_feedback
    assert bbox.suggested_clamps.get("required_torque") == 5.0


def test_z3_crucible_limp_mode_torque_clamping():
    """Verifies that Split-Brain Limp Mode clamps torque limits to 1.0 N*m."""
    adapter = CrucibleAdapter()

    # 2.0 N*m is safe in normal mode (<= 5.0) but UNSAT in Limp Mode (<= 1.0)
    moderate_state = {"required_torque": 2.0}

    normal_result = adapter.verify_kinematics(moderate_state, torque_limit=5.0)
    assert normal_result.is_safe is True

    limp_result = adapter.verify_kinematics(moderate_state, torque_limit=1.0)
    assert limp_result.is_safe is False
    assert limp_result.bounding_box is not None
    assert limp_result.bounding_box.suggested_clamps.get("required_torque") == 1.0
