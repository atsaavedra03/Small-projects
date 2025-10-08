

## Stamp Collection (Rutherford Scattering)

### Overview
This Python program simulates the scattering of alpha particles by gold nuclei using a numerical Runge–Kutta 4 (RK4) method with adaptive step size control. The simulation models Coulomb interactions between two charged particles and computes the relationship between the scattering angle and the impact parameter. 

The results are compared with theoretical predictions and are visualized through several graphs. The program also calculates the differential cross section and generates a histogram and probability density function (PDF) for the scattering angles.

The stamp collection name comes from the quote by Ernest Rutherford: "All science is either physics or stamp collecting". He then proceeded to get a Nobel Prize in chemistry.

### Dependencies
The code requires the following Python libraries:

- numpy
- matplotlib
- scipy


These can be installed with:
```bash
pip install numpy matplotlib scipy
```

### Description of Main Components 
#### Physical Constants
The script defines fundamental constants such as the mass of the proton, alpha particle, and gold nucleus, as well as Coulomb’s constant, elementary charge, and other physical parameters.

#### Initial Conditions
The center-of-mass (CoM) reference frame is established using:

- `initial\_conditions\_com\_rest()` to transform particle velocities.
- `initial\_values\_cartesian\_com()` to generate initial positions and velocities for the simulation.

#### Numerical Integration
The code uses a custom implementation of the fourth-order Runge–Kutta method **nrk4\_4** and an adaptive step-size control routine (\texttt{runrk4\_adaptive}) that adjusts the time step based on accuracy comparisons between half and full steps.

#### Scattering Angle and Cross Section
Functions such as `rk4\_theta\_adaptive()` and `scattering\_cross\_section()` compute:
- The scattering angle for each impact parameter.
- The differential cross section, both numerically and theoretically, using the Rutherford formula.

#### Monte Carlo Simulation
A Monte Carlo sampling of random impact parameters generates scattering angle distributions, which are used to compute histograms and probability density functions.

### Usage Instructions
To run the program, execute:
```bash
python project2_202319705_ATST.py
```

Upon running, you will be prompted to input a mode number:

- **Mode 1:** Generate plots and compare simulation with theoretical models.
- **Mode 2:** Compute a single scattering angle for a given impact parameter and display simulation time.


### Example Input
Below is an example of how to execute the program:

``` bash
Mode: 2
b: 1.0e-13
```

This will output the corresponding scattering angle (in degrees) and the time taken for the simulation (in milliseconds).

### Output Description

- **Scattering Angle vs. Impact Parameter:** Comparison between simulation and theoretical values.
- **Differential Cross Section:** Logarithmic plot of the numerically obtained cross section.
- **Histogram of Scattering Angles:** Frequency of scattering events.
- **Probability Density Function:** Comparison between Monte Carlo and theoretical distributions.

