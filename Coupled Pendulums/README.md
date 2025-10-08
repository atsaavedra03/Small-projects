# Coupled Pendulum Simulation and Animation
This Python script simulates and optionally animates two pendulums connected by a spring using the **Runge–Kutta 4th-order method (RK4)** for numerical integration. The animation can be exported as a GIF using the Pillow library.

---

## Overview

The code numerically solves the motion of a coupled pendulum system defined by:
- Two pendulums of masses `m1`, `m2` and lengths `l1`, `l2`
- Connected by a spring of constant `k`
- Subject to gravity `g = 9.8 m/s²`

It supports two operation modes:
1. **Mode 1 – Numerical Output:**  
   Computes motion values (`θ₁`, `ω₁`, `θ₂`, `ω₂`) at specified time points.
2. **Mode 2 – Animation:**  
   Generates and saves an animated visualization (`.gif`) of the coupled pendulums.

---
## Dependencies

To run this script, you need:
```bash
pip install numpy matplotlib pillow
