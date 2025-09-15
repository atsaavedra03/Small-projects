"""
Name:Saavedra Torres Andrés Tomas

The code uses pillower to save the animation so it is required to run succesfully. Other than that numpy and matplotlib were used.

References
I used the following resources
Rk4 method as provided by the TA
https://primer-computational-mathematics.github.io/book/c_mathematics/numerical_methods/5_Runge_Kutta_method.html

The matplotlib library
https://matplotlib.org/

This time around I asked Chat GPT for help with making the legend.
https://chatgpt.com/

Much of the coding structure is recycled from project 1. Specially the Input and Output sections.

I worked with 김동헌 from the 전기정보공학부.
"""

# Import libraries for numerical math, plotting, and animation

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# Read user input for initial conditions (3 lines) and simulation mode (1 line)
line1 = input()
line2 = input()
line3 = input()
line4 = input()

flist= []
# Parse initial positions and velocities for 3 particles from input
l1add  = line1.split(' ')
for x in range(0,len(l1add)):
    flist.append(l1add[x])

l2add  = line2.split(' ')
for x in range(0,len(l2add)):
    flist.append(l2add[x])

l3add  = line3.split(' ')
for x in range(0,len(l3add)):
    flist.append(l3add[x])

# Simulation mode: 1 = specific time checkpoints, 2 = custom time range
mode = float(line4)

flist = [float(i) for i in flist]
# Assign initial positions (x,y) and velocities (vx,vy) for 3 particles
x10, y10, vx10, vy10, x20, y20,vx20, vy20, x30, y30, vx30, vy30 = flist
max_t = 0
# Initialize time control variables (default dt = 1e-5)
dt = 0.00001
t_list = {'name':1}
n = 0

# Mode 1: read number of checkpoints and their times
if mode ==1:
    n = int(input())
    t_list = {}
    for i in range(1,n+1):
        t_number = "t" + str(i)
        t_list[t_number] = float(input())
    dt = 0.00001
    max_t = 10
# Mode 2: read max simulation time and timestep
elif mode ==2:
    line5 = input()
    l5 = line5.split(' ')
    l5 = [float(i) for i in l5]
    max_t,dt = l5

# Set starting time
t0 = 0

# Helper function: compute distance^3 between two points
def r2_3(x1,y1,x2,y2):
    return ((x1-x2)**2 + (y1-y2)**2)**(3/2)

# Define derivatives of positions/velocities for the 3-body system
# (forms the system of 12 coupled differential equations)

def f1(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3):
    return vx1

def f2(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3):
    return vy1

def f3(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3):
    value  = (x2-x1)/r2_3(x1,y1,x2,y2)  + (x3-x1)/r2_3(x1,y1,x3,y3)
    return value

def f4(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3):
    value  = (y2-y1)/r2_3(x1,y1,x2,y2)  + (y3-y1)/r2_3(x1,y1,x3,y3)
    return value

def f5(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3):
    return vx2

def f6(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3):
    return vy2

def f7(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3):
    value  = (x1-x2)/r2_3(x1,y1,x2,y2)  + (x3-x2)/r2_3(x2,y2,x3,y3)
    return value

def f8(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3):
    value  = (y1-y2)/r2_3(x1,y1,x2,y2)  + (y3-y2)/r2_3(x2,y2,x3,y3)
    return value

def f9(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3):
    return vx3

def f10(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3):
    return vy3

def f11(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3):
    value  = (x1-x3)/r2_3(x1,y1,x3,y3)  + (x2-x3)/r2_3(x2,y2,x3,y3)
    return value

def f12(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3):
    value  = (y1-y3)/r2_3(x1,y1,x3,y3)  + (y2-y3)/r2_3(x2,y2,x3,y3)
    return value

# Runge-Kutta 4th order integrator for the system (advances one timestep)
def rk4_4(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3,dt, f1, f2, f3, f4, f5, f6, f7, f8, f9 ,f10 ,f11, f12):
    k1 = dt*f1(t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3)
    h1 = dt*f2(t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3)
    m1 = dt*f3(t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3)
    n1 = dt*f4(t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3)
    q1 = dt*f5(t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3)
    w1 = dt*f6(t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3)
    e1 = dt*f7(t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3)
    r1 = dt*f8(t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3)
    u1 = dt*f9(t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3)
    i1 = dt*f10(t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3)
    o1 = dt*f11(t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3)
    p1 = dt*f12(t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3)


    k2 = dt*f1(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,vx1+0.5*m1,vy1+0.5*n1,x2+0.5*q1,y2+0.5*w1,vx2+0.5*e1,vy2+0.5*r1, x3+0.5*u1,y3+0.5*i1,vx3+0.5*o1,vy3+0.5*p1)
    h2 = dt*f2(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,vx1+0.5*m1,vy1+0.5*n1,x2+0.5*q1,y2+0.5*w1,vx2+0.5*e1,vy2+0.5*r1, x3+0.5*u1,y3+0.5*i1,vx3+0.5*o1,vy3+0.5*p1)
    m2 = dt*f3(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,vx1+0.5*m1,vy1+0.5*n1,x2+0.5*q1,y2+0.5*w1,vx2+0.5*e1,vy2+0.5*r1, x3+0.5*u1,y3+0.5*i1,vx3+0.5*o1,vy3+0.5*p1)
    n2 = dt*f4(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,vx1+0.5*m1,vy1+0.5*n1,x2+0.5*q1,y2+0.5*w1,vx2+0.5*e1,vy2+0.5*r1, x3+0.5*u1,y3+0.5*i1,vx3+0.5*o1,vy3+0.5*p1)
    q2 = dt*f5(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,vx1+0.5*m1,vy1+0.5*n1,x2+0.5*q1,y2+0.5*w1,vx2+0.5*e1,vy2+0.5*r1, x3+0.5*u1,y3+0.5*i1,vx3+0.5*o1,vy3+0.5*p1)
    w2 = dt*f6(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,vx1+0.5*m1,vy1+0.5*n1,x2+0.5*q1,y2+0.5*w1,vx2+0.5*e1,vy2+0.5*r1, x3+0.5*u1,y3+0.5*i1,vx3+0.5*o1,vy3+0.5*p1)
    e2 = dt*f7(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,vx1+0.5*m1,vy1+0.5*n1,x2+0.5*q1,y2+0.5*w1,vx2+0.5*e1,vy2+0.5*r1, x3+0.5*u1,y3+0.5*i1,vx3+0.5*o1,vy3+0.5*p1)
    r2 = dt*f8(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,vx1+0.5*m1,vy1+0.5*n1,x2+0.5*q1,y2+0.5*w1,vx2+0.5*e1,vy2+0.5*r1, x3+0.5*u1,y3+0.5*i1,vx3+0.5*o1,vy3+0.5*p1)
    u2 = dt*f9(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,vx1+0.5*m1,vy1+0.5*n1,x2+0.5*q1,y2+0.5*w1,vx2+0.5*e1,vy2+0.5*r1, x3+0.5*u1,y3+0.5*i1,vx3+0.5*o1,vy3+0.5*p1)
    i2 = dt*f10(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,vx1+0.5*m1,vy1+0.5*n1,x2+0.5*q1,y2+0.5*w1,vx2+0.5*e1,vy2+0.5*r1, x3+0.5*u1,y3+0.5*i1,vx3+0.5*o1,vy3+0.5*p1)
    o2 = dt*f11(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,vx1+0.5*m1,vy1+0.5*n1,x2+0.5*q1,y2+0.5*w1,vx2+0.5*e1,vy2+0.5*r1, x3+0.5*u1,y3+0.5*i1,vx3+0.5*o1,vy3+0.5*p1)
    p2 = dt*f12(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,vx1+0.5*m1,vy1+0.5*n1,x2+0.5*q1,y2+0.5*w1,vx2+0.5*e1,vy2+0.5*r1, x3+0.5*u1,y3+0.5*i1,vx3+0.5*o1,vy3+0.5*p1)


    k3 = dt*f1(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,vx1+0.5*m2,vy1+0.5*n2,x2+0.5*q2,y2+0.5*w2,vx2+0.5*e2,vy2+0.5*r2, x3+0.5*u2,y3+0.5*i2,vx3+0.5*o2,vy3+0.5*p2)
    h3 = dt*f2(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,vx1+0.5*m2,vy1+0.5*n2,x2+0.5*q2,y2+0.5*w2,vx2+0.5*e2,vy2+0.5*r2, x3+0.5*u2,y3+0.5*i2,vx3+0.5*o2,vy3+0.5*p2)
    m3 = dt*f3(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,vx1+0.5*m2,vy1+0.5*n2,x2+0.5*q2,y2+0.5*w2,vx2+0.5*e2,vy2+0.5*r2, x3+0.5*u2,y3+0.5*i2,vx3+0.5*o2,vy3+0.5*p2)
    n3 = dt*f4(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,vx1+0.5*m2,vy1+0.5*n2,x2+0.5*q2,y2+0.5*w2,vx2+0.5*e2,vy2+0.5*r2, x3+0.5*u2,y3+0.5*i2,vx3+0.5*o2,vy3+0.5*p2)
    q3 = dt*f5(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,vx1+0.5*m2,vy1+0.5*n2,x2+0.5*q2,y2+0.5*w2,vx2+0.5*e2,vy2+0.5*r2, x3+0.5*u2,y3+0.5*i2,vx3+0.5*o2,vy3+0.5*p2)
    w3 = dt*f6(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,vx1+0.5*m2,vy1+0.5*n2,x2+0.5*q2,y2+0.5*w2,vx2+0.5*e2,vy2+0.5*r2, x3+0.5*u2,y3+0.5*i2,vx3+0.5*o2,vy3+0.5*p2)
    e3 = dt*f7(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,vx1+0.5*m2,vy1+0.5*n2,x2+0.5*q2,y2+0.5*w2,vx2+0.5*e2,vy2+0.5*r2, x3+0.5*u2,y3+0.5*i2,vx3+0.5*o2,vy3+0.5*p2)
    r3 = dt*f8(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,vx1+0.5*m2,vy1+0.5*n2,x2+0.5*q2,y2+0.5*w2,vx2+0.5*e2,vy2+0.5*r2, x3+0.5*u2,y3+0.5*i2,vx3+0.5*o2,vy3+0.5*p2)
    u3 = dt*f9(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,vx1+0.5*m2,vy1+0.5*n2,x2+0.5*q2,y2+0.5*w2,vx2+0.5*e2,vy2+0.5*r2, x3+0.5*u2,y3+0.5*i2,vx3+0.5*o2,vy3+0.5*p2)
    i3 = dt*f10(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,vx1+0.5*m2,vy1+0.5*n2,x2+0.5*q2,y2+0.5*w2,vx2+0.5*e2,vy2+0.5*r2, x3+0.5*u2,y3+0.5*i2,vx3+0.5*o2,vy3+0.5*p2)
    o3 = dt*f11(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,vx1+0.5*m2,vy1+0.5*n2,x2+0.5*q2,y2+0.5*w2,vx2+0.5*e2,vy2+0.5*r2, x3+0.5*u2,y3+0.5*i2,vx3+0.5*o2,vy3+0.5*p2)
    p3 = dt*f12(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,vx1+0.5*m2,vy1+0.5*n2,x2+0.5*q2,y2+0.5*w2,vx2+0.5*e2,vy2+0.5*r2, x3+0.5*u2,y3+0.5*i2,vx3+0.5*o2,vy3+0.5*p2)


    k4 = dt*f1(t+dt,x1+k3,y1+h3,vx1+m3,vy1+n3,x2+q3,y2+w3,vx2+e3,vy2+r3, x3+u3,y3+i3,vx3+o3,vy3+p3)
    h4 = dt*f2(t+dt,x1+k3,y1+h3,vx1+m3,vy1+n3,x2+q3,y2+w3,vx2+e3,vy2+r3, x3+u3,y3+i3,vx3+o3,vy3+p3)
    m4 = dt*f3(t+dt,x1+k3,y1+h3,vx1+m3,vy1+n3,x2+q3,y2+w3,vx2+e3,vy2+r3, x3+u3,y3+i3,vx3+o3,vy3+p3)
    n4 = dt*f4(t+dt,x1+k3,y1+h3,vx1+m3,vy1+n3,x2+q3,y2+w3,vx2+e3,vy2+r3, x3+u3,y3+i3,vx3+o3,vy3+p3)
    q4 = dt*f5(t+dt,x1+k3,y1+h3,vx1+m3,vy1+n3,x2+q3,y2+w3,vx2+e3,vy2+r3, x3+u3,y3+i3,vx3+o3,vy3+p3)
    w4 = dt*f6(t+dt,x1+k3,y1+h3,vx1+m3,vy1+n3,x2+q3,y2+w3,vx2+e3,vy2+r3, x3+u3,y3+i3,vx3+o3,vy3+p3)
    e4 = dt*f7(t+dt,x1+k3,y1+h3,vx1+m3,vy1+n3,x2+q3,y2+w3,vx2+e3,vy2+r3, x3+u3,y3+i3,vx3+o3,vy3+p3)
    r4 = dt*f8(t+dt,x1+k3,y1+h3,vx1+m3,vy1+n3,x2+q3,y2+w3,vx2+e3,vy2+r3, x3+u3,y3+i3,vx3+o3,vy3+p3)
    u4 = dt*f9(t+dt,x1+k3,y1+h3,vx1+m3,vy1+n3,x2+q3,y2+w3,vx2+e3,vy2+r3, x3+u3,y3+i3,vx3+o3,vy3+p3)
    i4 = dt*f10(t+dt,x1+k3,y1+h3,vx1+m3,vy1+n3,x2+q3,y2+w3,vx2+e3,vy2+r3, x3+u3,y3+i3,vx3+o3,vy3+p3)
    o4 = dt*f11(t+dt,x1+k3,y1+h3,vx1+m3,vy1+n3,x2+q3,y2+w3,vx2+e3,vy2+r3, x3+u3,y3+i3,vx3+o3,vy3+p3)
    p4 = dt*f12(t+dt,x1+k3,y1+h3,vx1+m3,vy1+n3,x2+q3,y2+w3,vx2+e3,vy2+r3, x3+u3,y3+i3,vx3+o3,vy3+p3)


    t = t+dt
    x1 = x1 + (1/6) * (k1 + 2*k2 +2*k3 + k4)
    y1 = y1 + (1/6) * (h1 + 2*h2 +2*h3 + h4)
    vx1 = vx1 + (1/6) * (m1 + 2*m2 +2*m3 + m4)
    vy1 = vy1 + (1/6) * (n1 + 2*n2 +2*n3 + n4)
    x2 = x2 + (1/6) * (q1 + 2*q2 +2*q3 + q4)
    y2 = y2 + (1/6) * (w1 + 2*w2 +2*w3 + w4)
    vx2 = vx2 + (1/6) * (e1 + 2*e2 +2*e3 + e4)
    vy2 = vy2 + (1/6) * (r1 + 2*r2 +2*r3 + r4)
    x3 = x3 + (1/6) * (u1 + 2*u2 +2*u3 + u4)
    y3 = y3 + (1/6) * (i1 + 2*i2 +2*i3 + i4)
    vx3 = vx3 + (1/6) * (o1 + 2*o2 +2*o3 + o4)
    vy3 = vy3 + (1/6) * (p1 + 2*p2 +2*p3 + p4)
    return t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3


# Initialize simulation state (positions, velocities, and time)

t = t0
x1 = x10
y1 = y10
vx1 = vx10
vy1 = vy10
x2 = x20
y2 = y20
vx2 = vx20
vy2 = vy20
x3 = x30
y3 = y30
vx3 = vx30
vy3 = vy30

# Storage lists for positions, velocities, and times (for plotting & animation)

t_values = [t0]
x1_values = [x10]
y1_values = [y10]
vx1_values = [vx10]
vy1_values = [vy10]
x2_values = [x20]
y2_values = [y20]
vx2_values = [vx20]
vy2_values = [vy20]
x3_values = [x30]
y3_values = [y30]
vx3_values = [vx30]
vy3_values = [vy30]

# Main simulation loop: integrate system forward using RK4 and record results
while t <= max_t:
    t, x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = rk4_4(t, x1, y1,vx1,vy1,x2, y2,vx2,vy2,x3, y3,vx3,vy3,dt, f1, f2, f3, f4, f5, f6, f7, f8, f9 ,f10 ,f11, f12)

    t_values.append(t)
    x1_values.append(x1)
    y1_values.append(y1)
    vx1_values.append(vx1)
    vy1_values.append(vy1)
    x2_values.append(x2)
    y2_values.append(y2)
    vx2_values.append(vx2)
    vy2_values.append(vy2)
    x3_values.append(x3)
    y3_values.append(y3)
    vx3_values.append(vx3)
    vy3_values.append(vy3)


def d3_maker(str):
    if str[0] == '-':
        obj = 6
    else:
        obj = 5
    if len(str) != obj:
        missing = obj - len(str)
        for i in range(0,missing):
            str  = str + '0'
    return str


if mode == 1:
    keys = list(t_list.keys())
    l1 = [x1_values,y1_values,vx1_values,vy1_values]
    l2 = [x2_values,y2_values,vx2_values,vy2_values]
    l3 = [x3_values,y3_values,vx3_values,vy3_values]
    for i in range(0,n):
        name = keys[i]
        #rdc_term = int(t_list.get(name)/5)+1
        number = 0

        #number  = int( t_list.get(name) / dt)

        p1 = ''
        p2 = ''
        p3 = ''
        for x in l1:
            p1 = p1 + d3_maker(str(round(x[number],3))) + ' '
        for x in l2:
            p2 = p2 + d3_maker(str((round(x[number],3)))) + ' '
        for x in l3:
            p3 = p3 + d3_maker(str((round(x[number],3)))) + ' '
        print(p1)
        print(p2)
        print(p3)
        print()
# Define and run animation showing the motion of the three bodies
elif mode == 2:
    sd = 2
    
    x1, y1 = x10,y10
    x2, y2 = x20,y20
    x3, y3 = x30,y30

    max_x = max(max(x1_values),max(x2_values),max(x3_values),abs(max(x1_values)), abs(max(x2_values)),abs(max(x3_values)))
    max_y = max(max(y1_values),max(y2_values),max(y3_values),abs(max(y1_values)), abs(max(y2_values)),abs(max(y3_values)))

    fig_width = 5
    fig_height = 5

    fig, ax = plt.subplots(figsize = (fig_width, fig_height))
    ax.set_box_aspect(1)
    ax.set_aspect('equal')

    planetsize = 0.04

    #planet1 = ax.add_patch(plt.Circle((x1,y1),planetsize, fc = 'b', zorder = 3, label='Mass 1'))
    #planet2 = ax.add_patch(plt.Circle((x2,y2),planetsize, fc = 'r', zorder = 3, label='Mass 2'))
    #planet3 = ax.add_patch(plt.Circle((x3,y3),planetsize, fc = 'g', zorder = 3, label='Mass 3'))
    
    def animate(i):
        x1 = x1_values[i*sd]
        y1 = y1_values[i*sd]
        x2 = x2_values[i*sd]
        y2 = y2_values[i*sd]
        x3 = x3_values[i*sd]
        y3 = y3_values[i*sd]

        
        #planet1.set_center((x1,y1))
        #ax.add_patch(plt.Circle((x1,y1),0.005, fc = 'b', zorder = 1))
        #planet2.set_center((x2,y2))
        #ax.add_patch(plt.Circle((x2,y2),0.005, fc = 'r', zorder = 1))
        #planet3.set_center((x3,y3))
        #ax.add_patch(plt.Circle((x3,y3),0.005, fc = 'g', zorder = 1))
        
        time.set_text('t = '+ d3_maker(str(round(t_values[i*sd],3))))
    
    ax.set_xlim(-max_x-0.1, max_x+0.1)
    ax.set_ylim(-max_y-0.1, max_y+0.1)

    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')

    time =ax.text(-0.11, max_y+0.1*4, 't = '+ str(t0))

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='mass1', markerfacecolor='b', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='mass2', markerfacecolor='r', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='mass3', markerfacecolor='g', markersize=10)
    ]

    ax.legend(handles=legend_handles, loc='upper right')

    nsteps = int(len(x1_values)/sd)
    interval = dt
    ani = animation.FuncAnimation(fig, animate, frames = nsteps, repeat = True, interval = interval)
    # Export the animation to a file
    ani.save('threebody_ATST.gif', writer='pillow', fps=20)