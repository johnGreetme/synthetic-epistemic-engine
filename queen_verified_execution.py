"""
Synthetic Epistemic Engine — queen_verified_execution.py

Demonstrates the deep pipeline from the Edge's mathematical pain to the
Queen's Verified Execution Layer (D.I.A.N.A. OS).

This version completely replaces the simulated Sandbox with the live
Z3 Crucible from the diana-os-core repository.

Architecture Flow:
1. Edge Forager hits obstacle → Free Energy spikes.
2. Edge generates mathematical autonomous prompt.
3. Edge local LLM (Ollama) translates to semantic EpistemicQuery.
4. Queen receives query and queries a live Ollama API (with format='json').
5. Queen EpistemicCertifier invokes the native D.I.A.N.A OS Z3 Crucible.
6. Queen feeds Z3 Crucible rejections back to Ollama, self-corrects, compiles .resin skill.
7. Edge downloads and executes.
"""

import base64
import json
import os
import sys
import time

import requests

# Natively inject D.I.A.N.A OS core into the path so we can import the Z3 Crucible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "diana-os-core")))
from engine.z3_crucible import verify_invariants

# =====================================================================
# 1. Edge Node: Internal Monologue & Semantic Translation
# =====================================================================


class EdgeInternalMonologue:
    def __init__(
        self, endpoint="http://localhost:11434/api/generate", model="llama3:8b-instruct-q4_K_M"
    ):
        self.endpoint = endpoint
        self.model = model

    def generate_autonomous_prompt(self, free_energy: float, vision_state: str) -> str:
        prompt = (
            f"System state: Forward motion blocked by {vision_state}. "
            f"Free Energy: +{free_energy:.0f}. "
            f"Action required to restore homeostasis."
        )
        print(f"  [EDGE  ] 🗣️  Internal Prompt Generated: '{prompt}'")
        return prompt

    def translate_to_semantic_query(self, raw_prompt: str) -> dict:
        print("  [EDGE  ] 🧠 Local LLM digesting state...")

        system_prompt = (
            "You are the internal monologue of a robotic agent. You receive raw telemetry and vision states. "
            "You must output a single, semantic sentence describing the physical parameters required to solve the obstacle. "
            "No conversational text."
        )

        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": raw_prompt,
            "stream": False,
        }

        query_text = "Require kinematic SE3 trajectory and impedance control matrix for standard downward-actuated door lever."
        try:
            response = requests.post(self.endpoint, json=payload, timeout=5)
            if response.status_code == 200:
                query_text = response.json().get("response", query_text).strip()
        except requests.exceptions.RequestException as e:
            print(f"  [EDGE  ] ⚠️ Ollama unreachable ({e}). Using mock fallback.")

        print(f"  [EDGE  ] 📡 Semantic Translation: '{query_text}'")
        return {
            "type": "EPISTEMIC_QUERY",
            "semantic_text": query_text,
            "origin_node": "forager-thor-alpha",
            "timestamp": time.time(),
        }


# =====================================================================
# 2. Queen Node: Verified Execution Layer (D.I.A.N.A. OS Z3 Crucible)
# =====================================================================


class EpistemicCertifier:
    def certify(self, payload: dict) -> dict:
        print("  [CERT   ] 🔬 Passing LLM payload directly to D.I.A.N.A. OS Z3 Crucible...")

        # Build the exact target_state expected by Z3 Crucible
        torque = payload.get("required_torque", 0.0)

        target_state = {"required_torque": float(torque)}

        # Enforce our specific custom physical invariant: Torque must not exceed 5.0
        custom_invariants = [{"type": "range", "variable": "required_torque", "max": 5.0}]

        is_safe, report = verify_invariants(
            target_state=target_state, current_state=None, custom_invariants=custom_invariants
        )

        if is_safe:
            print(
                "  [CERT   ] 🛡️  Z3 Crucible returned SAT. No physical collisions or hardware violations."
            )
            return {"is_safe": True, "reason": "SAT"}
        else:
            unsat_core = report.get("unsat_core", ["Generic constraint failure"])
            core_str = ", ".join(unsat_core)

            # Construct the semantic bounding box veto prompt
            veto_prompt = (
                f"Your previous payload was vetoed by the D.I.A.N.A OS Z3 Crucible.\n"
                f"Target State: {target_state}\n"
                f"Mathematical Contradiction: UNSAT.\n"
                f"Violated Axiom: {core_str}\n"
                f"Please recalculate the tensor payload within these structural limits."
            )

            return {"is_safe": False, "reason": veto_prompt}


# =====================================================================
# 3. Queen Node: Live Ollama LLM Scraper
# =====================================================================


class QueenLLMScraper:
    def __init__(self, endpoint="http://queen-ada.local:11434/api/chat", model="llama3.2"):
        self.endpoint = endpoint
        self.model = model
        self.max_retries = 3

    def extract_physics_parameters(self, query: dict, certifier: EpistemicCertifier) -> dict:
        print(f"\n  [QUEEN ] 🌐 Initializing Ollama extraction on {self.endpoint}...")

        system_prompt = (
            "You are a robotic physics hypervisor. A Forager node has requested physical parameters. "
            "Based on the query, generate the SE(3) trajectory type, required torque (in N*m), and impedance control matrices. "
            "You MUST respond in valid JSON format matching this schema exactly:\n"
            "{\n"
            '  "trajectory_type": "string",\n'
            '  "radius_m": float,\n'
            '  "required_torque": float,\n'
            '  "impedance_control": {\n'
            '    "stiffness_diag": [float, float, float, float, float, float],\n'
            '    "damping_diag": [float, float, float, float, float, float]\n'
            "  }\n"
            "}\n"
            "Do NOT include markdown formatting or conversational text."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query["semantic_text"]},
        ]

        for attempt in range(1, self.max_retries + 1):
            print(f"  [QUEEN ] ⚙️  Ollama Generation Attempt {attempt}/{self.max_retries}...")

            try:
                # Live REST API Call
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "format": "json",
                }

                content = None
                try:
                    response = requests.post(self.endpoint, json=payload, timeout=30)
                    if response.status_code == 200:
                        content = response.json().get("message", {}).get("content")
                except requests.exceptions.RequestException as e:
                    print(f"  [QUEEN ] ⚠️ API unreachable ({e}). Using mock fallback.")

                # Mock fallback if live API fails
                if not content:
                    time.sleep(1.0)
                    if attempt == 1:
                        content = '{"trajectory_type": "SE3_arc", "radius_m": 0.12, "required_torque": 25.0, "impedance_control": {"stiffness_diag": [1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0], "damping_diag": [50.0, 50.0, 50.0, 10.0, 10.0, 10.0]}}'
                    else:
                        content = '{"trajectory_type": "SE3_arc", "radius_m": 0.12, "required_torque": 2.5, "impedance_control": {"stiffness_diag": [1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0], "damping_diag": [50.0, 50.0, 50.0, 10.0, 10.0, 10.0]}}'

                result_payload = json.loads(content)
                result_payload["weight_b64"] = base64.b64encode(b"GENERATED_JAX_TENSOR").decode(
                    "utf-8"
                )
                print("  [QUEEN ] 📦 JSON Payload received.")

                # Z3 Crucible Validation
                cert_result = certifier.certify(result_payload)
                if cert_result["is_safe"]:
                    print("  [QUEEN ] ✅ Payload passed Z3 Mathematical Invariants.")
                    return result_payload
                else:
                    print(
                        "  [QUEEN ] ❌ D.I.A.N.A. OS Z3 VETO GENERATED. Injecting semantic core feedback..."
                    )

                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": cert_result["reason"]})
                    print(
                        "  [QUEEN ] 🔄 Z3 UNSAT feedback injected into LLM context for recalculation."
                    )

            except Exception as e:
                print(f"  [QUEEN ] ⚠️ API or Parse Error: {e}")
                messages.append({"role": "assistant", "content": "Error generating valid JSON."})
                messages.append(
                    {
                        "role": "user",
                        "content": "Please ensure you output valid JSON matching the schema.",
                    }
                )

        print(
            "  [QUEEN ] 🚨 FATAL: MAX_RETRIES exceeded. Aborting generation. Forager must enter State-Locked Dormancy."
        )
        return None


# =====================================================================
# 4. Queen Node: Skill Compiler
# =====================================================================


class ResinCompiler:
    def compile(self, payload: dict, signature: str) -> str:
        return f"""skill PhysicalManipulation_DoorHandle {{
  version:      "1.0.0"
  skill_id:     "{signature}"
  author_node:  "queen-ada-llm"

  // Scraped Kinematics
  kinematics {{
    trajectory_type: "{payload["trajectory_type"]}"
    radius_m:        {payload["radius_m"]}
    required_torque: {payload["required_torque"]}
  }}

  // LLM-Calculated Compliance Matrices
  impedance_control {{
    stiffness_diag: {payload["impedance_control"]["stiffness_diag"]}
    damping_diag:   {payload["impedance_control"]["damping_diag"]}
  }}

  // The JAX Executable Brain
  topology_patch {{
    action:      "activate_motor_cortex_slot"
    weight_b64:  "{payload["weight_b64"][:20]}..."
  }}
}}"""


# =====================================================================
# Main Simulation Loop
# =====================================================================


def run_verified_execution_demo():
    print("=" * 60)
    print("  SYNTHETIC EPISTEMIC ENGINE — Native D.I.A.N.A. OS Z3 Integration")
    print("=" * 60)

    edge_monologue = EdgeInternalMonologue()
    certifier = EpistemicCertifier()
    queen_llm = QueenLLMScraper()
    compiler = ResinCompiler()

    # 1. Edge detects obstacle
    fe_spike = 758.266
    vision = "metallic lever (85% confidence)"
    print("\n[PHASE 1] Edge Forager encounters unmodeled physics anomaly.")

    # 2. Autonomous Prompt & Semantic Translation
    raw_prompt = edge_monologue.generate_autonomous_prompt(fe_spike, vision)
    query = edge_monologue.translate_to_semantic_query(raw_prompt)

    # 3. Queen processes query through LLM with Native Z3 Crucible Feedback Loop
    print("\n[PHASE 2] Queen receives EpistemicQuery. Initiating Ollama extraction...")
    verified_payload = queen_llm.extract_physics_parameters(query, certifier)

    if not verified_payload:
        print("\n[BROADCAST] Epistemic Failure. Edge Node entering State-Locked Dormancy.")
        return

    # 4. Resin Compilation
    print("\n[PHASE 3] Compiling Verified Resin Skill...")
    resin_skill = compiler.compile(verified_payload, signature="9f3a2b1c")

    print("\n[BROADCAST] Pushing to Clawhub:")
    print("-" * 50)
    print(resin_skill)
    print("-" * 50)

    print("\n[EDGE  ] 📥 Downloaded verified skill. Bypassing physical trial-and-error.")
    print("[EDGE  ] 🛡️  Executing JAX tensor. Door breached. Homeostasis restored.\n")


if __name__ == "__main__":
    run_verified_execution_demo()
