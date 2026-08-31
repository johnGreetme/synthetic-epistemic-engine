"""Synthetic Epistemic Engine — Z3 Crucible Submodule Adapter.

Provides dynamic path resolution to import the immutable DIANA OS Z3 Crucible
microkernel from diana_core. Intercepts LLM-generated physical parameter mutations,
verifies cyber-physical invariants using Z3 theorem proving, and extracts
unsat_core() constraint tags into clean semantic bounding boxes for LLM re-prompting.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from z3 import (
        And,
        Bool,
        Implies,
        Int,
        Not,
        Or,
        Real,
        Solver,
        sat,
        simplify,
        unknown,
        unsat,
    )
except ImportError:
    Solver = None
    Int = Real = Bool = Not = And = Or = Implies = simplify = None
    sat = "sat"
    unsat = "unsat"
    unknown = "unknown"


def _resolve_diana_core() -> Any:
    """Dynamically locates and imports z3_crucible from diana_core or diana-os-core."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "diana_core",
        Path(__file__).resolve().parent.parent.parent / "diana-os-core",
        Path.cwd() / "diana_core",
        Path.cwd() / "diana-os-core",
    ]

    for root_candidate in candidates:
        engine_path = root_candidate / "engine" / "z3_crucible.py"
        if engine_path.exists():
            resolved_root = str(root_candidate.resolve())
            if resolved_root not in sys.path:
                sys.path.insert(0, resolved_root)

            spec = importlib.util.spec_from_file_location(
                "diana_core.engine.z3_crucible", str(engine_path)
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    try:
        import engine.z3_crucible as crucible

        return crucible
    except ImportError:
        pass

    return None


_CRUCIBLE_MODULE = _resolve_diana_core()


@dataclass
class SemanticBoundingBox:
    """Semantic bounding box explaining Z3 invariant violations for LLM re-prompting."""

    target_state: Dict[str, Any]
    unsat_core: List[str]
    prompt_feedback: str
    suggested_clamps: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrucibleVerificationResult:
    """Formal result of a Z3 invariant verification pass."""

    is_safe: bool
    status: str
    target_state: Dict[str, Any]
    unsat_core: List[str] = field(default_factory=list)
    bounding_box: Optional[SemanticBoundingBox] = None
    error_message: Optional[str] = None


class CrucibleAdapter:
    """Adapter bridging SEE DreamSandbox LLM mutations with the DIANA OS Z3 Crucible."""

    DEFAULT_TORQUE_LIMIT: float = 5.0
    LIMP_MODE_TORQUE_LIMIT: float = 1.0

    def __init__(self, timeout_ms: int = 50) -> None:
        self.timeout_ms = timeout_ms
        self.crucible_module = _CRUCIBLE_MODULE

    def get_solver(self) -> Any:
        """Initializes a Z3 solver configured with unsat_core tracking and timeout."""
        if Solver is None:
            raise ImportError(
                "z3-solver is required. Install via `pip install z3-solver`."
            )
        solver = Solver()
        try:
            solver.set("timeout", self.timeout_ms)
            solver.set(unsat_core=True)
        except Exception:
            pass
        return solver

    def verify_kinematics(
        self,
        target_state: Dict[str, Any],
        current_state: Optional[Dict[str, Any]] = None,
        custom_invariants: Optional[List[Dict[str, Any]]] = None,
        torque_limit: Optional[float] = None,
    ) -> CrucibleVerificationResult:
        """Formally verifies cyber-physical invariants and torque limits using Z3."""
        if torque_limit is None:
            torque_limit = self.DEFAULT_TORQUE_LIMIT

        invariants = list(custom_invariants or [])

        if "required_torque" in target_state or "torque" in target_state:
            t_var = "required_torque" if "required_torque" in target_state else "torque"
            invariants.append({
                "type": "range",
                "variable": t_var,
                "max": torque_limit,
                "label": f"Kinematic Bound: {t_var} <= {torque_limit} N*m",
            })

        if Solver is None:
            return CrucibleVerificationResult(
                is_safe=False,
                status="FAIL_CLOSED",
                target_state=target_state,
                error_message="z3-solver is not installed",
            )

        solver = self.get_solver()
        z3_vars: Dict[str, Any] = {}

        try:
            # 1. Assert target states with tracked tracking labels
            for key, val in target_state.items():
                s_key = re.sub(r"[^a-zA-Z0-9_]", "_", str(key))
                label = f"Target Assignment: {s_key} == {val}"
                if isinstance(val, bool):
                    v = Bool(s_key)
                    z3_vars[s_key] = v
                    solver.assert_and_track(v == val, Bool(label))
                elif isinstance(val, int):
                    v = Int(s_key)
                    z3_vars[s_key] = v
                    solver.assert_and_track(v == val, Bool(label))
                elif isinstance(val, float):
                    v = Real(s_key)
                    z3_vars[s_key] = v
                    solver.assert_and_track(v == val, Bool(label))

            # 2. Standard robotics kinematics checks
            pos_delta = z3_vars.get("position_delta_rad")
            if pos_delta is not None:
                solver.assert_and_track(
                    pos_delta >= -3.14159265,
                    Bool("Rule D: Robotics Kinematics Constraints (>= -PI)"),
                )
                solver.assert_and_track(
                    pos_delta <= 3.14159265,
                    Bool("Rule D: Robotics Kinematics Constraints (<= PI)"),
                )

            # 3. Custom Tracked Invariants
            for idx, inv in enumerate(invariants):
                inv_type = inv.get("type")
                var_name = inv.get("variable")
                z_var = z3_vars.get(var_name)
                if z_var is None:
                    continue

                lbl_prefix = inv.get("label", f"Custom Invariant [{idx}]")

                if inv_type == "range":
                    min_v = inv.get("min")
                    max_v = inv.get("max")
                    if min_v is not None:
                        solver.assert_and_track(
                            z_var >= min_v,
                            Bool(f"{lbl_prefix}: {var_name} >= {min_v}"),
                        )
                    if max_v is not None:
                        solver.assert_and_track(
                            z_var <= max_v,
                            Bool(f"{lbl_prefix}: {var_name} <= {max_v}"),
                        )
                elif (
                    inv_type == "max_delta"
                    and current_state
                    and var_name in current_state
                ):
                    curr_v = current_state[var_name]
                    max_d = inv.get("delta", 50)
                    diff = z_var - curr_v
                    solver.assert_and_track(
                        diff <= max_d,
                        Bool(f"{lbl_prefix}: {var_name} delta <= {max_d}"),
                    )
                    solver.assert_and_track(
                        diff >= -max_d,
                        Bool(f"{lbl_prefix}: {var_name} delta >= {-max_d}"),
                    )
                elif inv_type == "mutex":
                    other_name = inv.get("with_variable")
                    other_var = z3_vars.get(other_name)
                    if other_var is not None:
                        solver.assert_and_track(
                            Not(And(z_var, other_var)),
                            Bool(f"{lbl_prefix}: Mutex({var_name}, {other_name})"),
                        )

            # 4. SMT Evaluation
            check_result = solver.check()
            is_safe = check_result == sat

            if is_safe:
                return CrucibleVerificationResult(
                    is_safe=True,
                    status="SATISFIABLE (SAFE)",
                    target_state=target_state,
                )

            core_exprs = solver.unsat_core()
            unsat_core_tags = [str(c) for c in core_exprs]

            bounding_box = self.format_semantic_bounding_box(
                target_state=target_state,
                unsat_core_tags=unsat_core_tags,
                torque_limit=torque_limit,
            )

            return CrucibleVerificationResult(
                is_safe=False,
                status="UNSATISFIABLE (BLOCKED)",
                target_state=target_state,
                unsat_core=unsat_core_tags,
                bounding_box=bounding_box,
            )

        except Exception as exc:
            return CrucibleVerificationResult(
                is_safe=False,
                status="CRUCIBLE_FAULT",
                target_state=target_state,
                error_message=str(exc),
            )

    def format_semantic_bounding_box(
        self,
        target_state: Dict[str, Any],
        unsat_core_tags: List[str],
        torque_limit: float,
    ) -> SemanticBoundingBox:
        """Formats UNSAT core constraints into a structured semantic veto prompt for LLM regeneration."""
        core_summary = (
            ", ".join(unsat_core_tags)
            if unsat_core_tags
            else "Kinematic Boundary Violation"
        )
        suggested_clamps: Dict[str, Any] = {}

        for k, v in target_state.items():
            if "torque" in k and isinstance(v, (int, float)) and v > torque_limit:
                suggested_clamps[k] = torque_limit

        prompt = (
            "Your previous mutation payload was vetoed by the DIANA OS Z3 Crucible.\n"
            f"Evaluated Target State: {target_state}\n"
            "Mathematical Contradiction: UNSAT\n"
            f"Violated Axioms / Invariants: [{core_summary}]\n"
            f"Strict Physical Bounds: Maximum permitted torque is {torque_limit:.1f} N*m.\n"
            "Please recalculate the kinematic and impedance parameters within this semantic bounding box."
        )

        return SemanticBoundingBox(
            target_state=target_state,
            unsat_core=unsat_core_tags,
            prompt_feedback=prompt,
            suggested_clamps=suggested_clamps,
        )
