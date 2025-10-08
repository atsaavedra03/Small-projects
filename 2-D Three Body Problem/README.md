## Three-body Problem Simulator using Runge–Kutta 4

This program (`three_body.py`) simulates the **three-body problem in 2D** by numerically integrating the system of differential equations using the **fourth-order Runge–Kutta (RK4)** method.  

The code requires the following Python libraries:
- **Pillow** – to save the resulting animation (`.gif` file)
- **NumPy** – for numerical computation
- **Matplotlib** – for plotting and animation  

---

### 🧩 Input Overview

When running the code, the user must provide:
1. **Initial conditions** for each of the three bodies (3 lines total)  
2. A **simulation mode** (1 line)

Each line of initial conditions must include:
x y vx vy
For example:
0.0 0.1 0.5 0.8
sets the initial conditions of planet 1 as  
*(x = 0.0, y = 0.1, vx = 0.5, vy = 0.8)*.  

To ensure stable output, it is recommended that initial positions stay within the range **|x| < 1**.

---

### Mode 1 — Data Output

In this mode, the code outputs the **positions and velocities** of each body at specified times.

After selecting mode 1:
1
the system will ask:
1. The **number of times** to check  
2. Each **time value** to be checked (one per line)

For example:
3
0.0
0.1
0.5
will output the positions and velocities at *t = 0.0, 0.1,* and *0.5 s*.  
The minimum time resolution for mode 1 is **0.00001 s**.

---

### Mode 2 — Animation Output

Mode 2 generates an animated `.gif` showing the **orbital motion** of the three bodies.

After selecting mode 2:
2
the system will ask for:
simulation_time frame_interval
For example:
10 0.01
runs the simulation for **10 seconds** with a **frame interval of 0.01 s**.  
A frame interval of 0.01 is generally recommended for smooth playback.  

If the axes extend beyond ±10 and the planets leave the visible frame, the system likely diverged due to scattering.  
In that case, try reducing the total simulation time to observe earlier behavior before scattering occurs.

---

### Example Input — Mode 1
-0.97000436 0.24308753 0.4662036850 0.4323657300
0.0 0.0 -0.93240737 -0.86473146
0.97000436 -0.24308753 0.4662036850 0.4323657300
1
5
0.0
1.0
2.0
6.3259
8.0

---

### Example Input — Mode 2
-0.97000436 0.24308753 0.4662036850 0.4323657300
0.0 0.0 -0.93240737 -0.86473146
0.97000436 -0.24308753 0.4662036850 0.4323657300
2
10.0 0.01


---

### Notes
- Make sure `Pillow` is installed; otherwise, the animation cannot be saved.
- If using very small time steps or long simulations, computation time will increase.
- The simulation is purely Newtonian and does not include relativistic effects.

---



