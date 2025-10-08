## Three-body problem simulator using Range Kutta 4
The three_body.py Python file simulates the three-body problem in 2D by integrating the differential equations numerically using RK4. <br>
The code uses pillower to save the animation so it is required to run succesfully. Other than that numpy and matplotlib were used. <br>
When running the code it reads user input for initial conditions (3 lines) and simulation mode (1 line) <br>
Each line represents the initial x position and y position of the respective object. <br>
So 0.0 0.1 0.5 0.8 makes the initial conditions of planet 1: (x = 0.0, y = 0.1, vx = 0.5, vy = 0.8). <br>
The initial positions should be kept inside the |x|<1 range for optimal output. <br> 
<br>
Mode 1 outputs the different positions and velocities of each planet in the same format as the input. <br>
If mode 1 is selected the system will ask for how many different times will be checked and then will <br>
ask for that many inputs. <br>
So to check at time 0.0, 0.1 and 0.5 the input should look like this: <br>
3 <br>
0.0 <br>
0.1 <br>
0.5 <br>
The minimum time resolution for mode 1 is 0.00001s.<br>
<br>
Mode 2 will output a gif of the three body behavior for a required time and frame interval. <br>
So after inputing 2 the system will ask for both values in the same line, as such:<br>
10 0.01<br>
The recommended frame interval is 0.01<br>
If the axes extend to values above 10 and bodies can't be observed, the behavior probably has the planets fly away from scattering.<br>
In that case reducing the lenght of the gif will allow to observe the behavior befor scattering.<br>
<br>
# Example Input Mode 1:<br>
-0.97000436 0.24308753 0.4662036850 0.4323657300<br>
0.0 0.0 -0.93240737 -0.86473146<br>
0.97000436 -0.24308753 0.4662036850 0.4323657300<br>
1<br>
5<br>
0.0<br>
1.0<br>
2.0<br>
6.3259<br>
8.0<br>
<br>
# Example Input Mode 2:<br>
-0.97000436 0.24308753 0.4662036850 0.4323657300<br>
0.0 0.0 -0.93240737 -0.86473146<br>
0.97000436 -0.24308753 0.4662036850 0.4323657300<br>
2<br>
10.0 0.01<br>
