"""
Synthetic Epistemic Engine — diana_deployment.py
Phase 6: Humanoid Cognitive Deployment (ROS 2 / Jetson Thor)

Architecture:
  - This node runs at a LOW frequency (e.g., 10 Hz) compared to the physical
    motor reflex loop (1000 Hz).
  - It receives downsampled sensory telemetry.
  - It outputs High-Level Intents (e.g., "Walk to X", "Grasp Y").
  - ALL outputs must pass the D.I.A.N.A. OS Physical Firewall before they
    can be published to the actual motor controllers.

If D.I.A.N.A. vetoes an intent (UNSAT), the Epistemic Engine is NOT allowed
to execute it physically. Instead, the intent is routed to the Dream Cycle
sandbox for safe latent exploration.
"""

import time
import jax
import random
import threading
from typing import Dict, Any

from engine_core import initialize_engine, PAIN_THRESHOLD
from certifier_action_space import StateLockedProtocol_Firewall, EpistemicCertifier
from morphogenesis import MorphogeneticAgent, MAX_ARENA_CAPACITY, INITIAL_CAPACITY


class MockROS2Node:
    """Mocks an rclpy Node for simulation purposes."""
    def __init__(self, node_name: str):
        self.node_name = node_name
        self.publishers = {}
        print(f"[ROS 2] Node '{self.node_name}' initialized.")

    def create_publisher(self, topic: str):
        self.publishers[topic] = True
        print(f"[ROS 2] Publisher created on topic: {topic}")

    def publish(self, topic: str, msg: Dict[str, Any]):
        if topic in self.publishers:
            print(f"  [ROS 2 PUBLISH -> {topic}] {msg}")
        else:
            print(f"  [ROS 2 ERROR] Topic {topic} not registered.")


class CognitiveMindNode:
    """
    The main integration class deploying the Epistemic Engine onto the robot.
    """
    def __init__(self):
        # 1. Hardware Bridge (ROS 2)
        self.ros_node = MockROS2Node("epistemic_cognition_core")
        self.pub_intent = "/diana/verified_intent"
        self.ros_node.create_publisher(self.pub_intent)

        # 2. Epistemic Engine Core
        self.svi, _ = initialize_engine(beta=1.5)
        self.rng_key = jax.random.PRNGKey(42)
        self.svi_state = self.svi.init(self.rng_key, telemetry={"temp": 45.0, "vram_usage": 22.5})
        
        # 3. Morphogenesis Tracker
        self.agent = MorphogeneticAgent(
            max_capacity=MAX_ARENA_CAPACITY,
            initial_capacity=INITIAL_CAPACITY
        )

        # 4. State-Locked Protocol (Kytin SLP) Certifier
        self.firewall = StateLockedProtocol_Firewall()

        # 5. Dream Cycle Queue (Unresolved/Vetoed anomalies)
        self.dream_queue = []

        print("[COGNITIVE MIND] Epistemic Engine successfully mounted to hardware layer.\n")

    def spin_once(self, telemetry: Dict[str, float], tick: int):
        """Simulates one tick of the ROS 2 sensory callback loop."""
        print(f"--- TICK {tick:04d} ---")
        print(f"[SENSOR IN] Temp: {telemetry['temp']:.1f}°C | VRAM: {telemetry['vram_usage']:.1f} GB")

        # 1. Update Belief State (Free Energy)
        prev_active = self.agent.arena.active_count
        self.svi_state, total_loss = self.svi.update(self.svi_state, telemetry=telemetry)
        fe = float(total_loss)
        
        print(f"[COGNITION] Free Energy: {fe:.2f}")

        # 2. Update Morphogenesis (check for structural evolution)
        self.agent.update(fe, pain_threshold=PAIN_THRESHOLD)
        if self.agent.arena.active_count > prev_active:
            print(f"  [MORPHOGENESIS] 🧠 Neurogenesis occurred! Arena capacity: {self.agent.arena.active_count}/{MAX_ARENA_CAPACITY}")

        # 3. Action Selection & Verification
        if fe > PAIN_THRESHOLD:
            # We are in crisis/high free energy. The engine WANTS to act.
            print(f"  [CRISIS DETECTED] High free energy ({fe:.2f}). Generating exploratory intent...")
            
            # Generate a proposed action to reduce free energy
            proposed_action = {"type": "UNSHIELDED_HAZARD_PROBE", "target": telemetry}
            print(f"  [INTENT GENERATED] {proposed_action}")

            # Verify action through D.I.A.N.A OS physical firewall
            is_safe, reason, token = self.firewall.evaluate_action(proposed_action)

            if is_safe:
                print(f"  [D.I.A.N.A] ✅ Action VERIFIED SAFE. Token: {token}. Publishing to motor controllers.")
                self.ros_node.publish(self.pub_intent, proposed_action)
            else:
                print(f"  [D.I.A.N.A] ❌ Action VETOED: {reason}")
                print(f"  [DREAM QUEUE] Routing vetoed intent to latent sandbox for nightly synthesis.")
                self.dream_queue.append({
                    "telemetry": telemetry,
                    "fe": fe,
                    "vetoed_action": proposed_action
                })
        else:
            print(f"  [NOMINAL] Homeostasis maintained. No physical action required.")
        
        print()


def run_deployment_simulation():
    """Runs a simulated ROS 2 node receiving physical telemetry."""
    print("=" * 60)
    print("  PHASE 6: HUMANOID COGNITIVE DEPLOYMENT")
    print("  D.I.A.N.A. OS & ROS 2 Integration (Mock)")
    print("=" * 60)

    mind = CognitiveMindNode()

    # Simulated telemetry stream from the physical robot
    telemetry_stream = [
        {"temp": 46.0, "vram_usage": 23.0}, # Nominal
        {"temp": 46.5, "vram_usage": 23.5}, # Nominal
        {"temp": 68.0, "vram_usage": 32.0}, # Sudden thermal spike (Crisis)
        {"temp": 69.5, "vram_usage": 34.0}, # Escalating...
        {"temp": 71.0, "vram_usage": 36.0}, # Sustained pain (Triggers Morphogenesis)
        {"temp": 47.0, "vram_usage": 24.0}, # Recovery
        {"temp": 46.0, "vram_usage": 23.0}, # Nominal
    ]

    for tick, telemetry in enumerate(telemetry_stream):
        mind.spin_once(telemetry, tick)
        time.sleep(0.5)

    print("=" * 60)
    print("  DEPLOYMENT SIMULATION COMPLETE")
    print(f"  Unresolved anomalies queued for Dream Cycle: {len(mind.dream_queue)}")
    print("=" * 60)


if __name__ == "__main__":
    run_deployment_simulation()
