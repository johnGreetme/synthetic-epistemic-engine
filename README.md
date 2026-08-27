# Synthetic Epistemic Engine
### The Sovereign Cognitive Core for Autonomous Humanoids

[![Status: Production](https://img.shields.io/badge/Status-Production-green)]()
[![Hardware: Kytin SLP](https://img.shields.io/badge/Hardware-Kytin%20SLP%20Ready-red)]()

## Overview
The Synthetic Epistemic Engine replaces traditional reinforcement learning and reward functions with a purely epistemic drive: **The Desire to Understand.** 
Instead of optimizing for survival or arbitrary external rewards, this engine minimizes a modified Free Energy objective (Epistemic Trace ELBO) that balances physical homeostasis with Expected Information Gain (EIG). 

The result is a system that actively seeks out anomalies, learns through unsupervised causal morphogenesis, and shares its structural "antibodies" across a swarm of edge nodes.

## Quickstart

### 1. Requirements
* Python 3.9+
* JAX & NumPyro
* ZeroMQ & FAISS (for swarm topologies)
* Kytin SLP SDK (for physical hardware deployment)

### 2. Running the Epistemic Debugger
To watch the agent's internal belief states in real-time as it traverses the Free Energy landscape:
```bash
python3 debugger_export.py
python3 -m http.server 8000
```
Open `http://localhost:8000/debugger.html` in your browser.

### 3. Running the Test Suite
Mathematically verify the epistemic philosophy:
```bash
pip install pytest
python3 -m pytest tests/
```

### 4. Deploying to Physical Hardware (ROS 2 & Kytin SLP)
The system is built to deploy on Jetson AGX Thor via Docker and Systemd (`kytin-slp.service`), with D.I.A.N.A OS ensuring absolute physical safety.
```bash
python3 diana_deployment.py
```
*(Ensure `USE_REAL_ROS = True` is set for physical deployments).*

## Deep Dive
For Builder Agents and advanced contributors, read the [ARCHITECTURE.md](ARCHITECTURE.md) to understand the underlying mathematics and structural topology.
