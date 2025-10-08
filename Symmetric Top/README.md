# Symmetric Top Simulation using Runge–Kutta 4

This project simulates the motion of a **symmetric top** (rotating rigid body) using the **fourth-order Runge–Kutta (RK4)** method.  
It includes both a **Python script (`symmetric_top.py`)** for full automation and a **Jupyter notebook (`symmetric_top.ipynb`)** for interactive visualization and analysis.

---

## Overview

The simulation numerically integrates the angular motion equations of a symmetric top characterized by parameters such as:
- Moment of inertia components `i1`, `i3`
- Gravitational acceleration `g`
- Angular momentum parameters `p1`, `p2`
- Angular variables `x`, `y`, `z`, and their time derivatives

The **RK4 integration** is used to obtain stable, high-accuracy results for the angular motion over time.  

Results can be output as:
- Numerical data (`output.txt`)
- Animated `.gif` trajectories
- Static comparison plots across multiple parameters

---

## Dependencies

Make sure the following Python libraries are installed:

```bash
pip install numpy matplotlib pandas scipy pillow
```
## Files Description
symmetric_top.py

A full simulation and visualization script that:

Reads initial condition pairs (x0, p1) from input.txt.

Integrates the system using RK4 over a 5-second simulation time.

Outputs:

Numerical results (x, y) at t = 5 s to output.txt.

Animated .gif files of the motion (gif 1.gif, gif 2.gif, ...).

Provides several helper functions for visualization and analysis.

“#### Key Functions | Function | Purpose | | --- | --- | | `rk4_4()` | Core RK4 integrator for the four coupled differential equations | | `runrk4()` | Runs the RK4 integration for given initial conditions | | `rk4anim()` | Generates an animated `.gif` of the simulation | | `rk4graph1()` | Creates a static scatter plot for a single run | | `rk4graph()` | Compares multiple runs across different `p1` values | | `make_df_range()` | Builds a Pandas DataFrame analyzing nutation range | | `crosses_zero()` | Checks whether a test function crosses zero within a range | * * * ### `symmetric_top.ipynb` An interactive notebook that imports the functions from `symmetric_top.py` and allows users to **visualize and analyze** the system under different parameters. #### Cell Overview | Cell | Description | | --- | --- | | **1** | Imports all required libraries and functions from `symmetric_top.py` | | **2** | Runs `rk4graph()` to visualize motion for a list of `p1` values | | **3** | Uses `rk4graph1()` to display a single run at `p1 = -17` | | **4** | Generates a `.gif` animation (`example.gif`) of the symmetric top motion | * * * 📥 Input Format --------------- The `input.txt` file should contain one set of initial conditions per line: nginx Copiar código `x0 p1` Example: Copiar código `0.8 1.95 1.0 2.10 0.5 1.80` * * * 📤 Output Files --------------- * `output.txt` — Contains `x` and `y` values at t = 5 s for each input set * `gif N.gif` — Animated trajectory for each simulation (`N` = input line number) * `example.gif` — Example animation generated in the notebook * * * 🎞️ Example Usage ----------------- ### Run from Terminal bash Copiar código `python symmetric_top.py` ### Run Interactively Open `symmetric_top.ipynb` in Jupyter and execute the cells in order to: * Compare trajectories for multiple `p1` values * View a single case as a static plot * Generate an example `.gif` of the motion * * * 💡 Notes -------- * The simulation time step is set to `dt = 0.001` for numerical stability. * Initial angles are recommended to be within `[0, π]`. * If the generated `.gif` shows values beyond the visible axis range, try reducing the simulation time. * Results and visuals represent the qualitative behavior of a symmetric top under gravity. * * * 📚 Author --------- **Saavedra Torres Andres Tomas**”

