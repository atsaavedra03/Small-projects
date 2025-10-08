
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq
import matplotlib.animation as animation

filei = open("input.txt", "r")
content = filei.read()
filei.close()
lines = content.splitlines()
ninputs = len(lines)
ivals = []
for i in lines:
    sep = i.split(' ')
    for x in (0,1):
        sep[x] = float(sep[x])
    ivals.append(sep)

#Initial Varibales#
m = 1
h = 1
i1 = 1
i2 = i1
i3 = 1.5
g = 9.81
w3 = 10
p2 = w3*i3
y0 = z0 = vx0 = 0
t0 = 0
dt = 0.001

p1 = 1.95
x0 = 0.8

# rk4 function
def rk4_4(t, x, y,z,vx,dt, f1, f2, f3, f4):
    k1 = dt*f1(t,x,y,z,vx)
    h1 = dt*f2(t,x,y,z,vx)
    m1 = dt*f3(t,x,y,z,vx)
    n1 = dt*f4(t,x,y,z,vx)


    k2 = dt*f1(t+0.5*dt,x+0.5*k1,y+0.5*h1,z+0.5*m1,vx+0.5*n1)
    h2 = dt*f2(t+0.5*dt,x+0.5*k1,y+0.5*h1,z+0.5*m1,vx+0.5*n1)
    m2 = dt*f3(t+0.5*dt,x+0.5*k1,y+0.5*h1,z+0.5*m1,vx+0.5*n1)
    n2 = dt*f4(t+0.5*dt,x+0.5*k1,y+0.5*h1,z+0.5*m1,vx+0.5*n1)

    k3 = dt*f1(t+0.5*dt,x+0.5*k2,y+0.5*h2,z+0.5*m2,vx+0.5*n2)
    h3 = dt*f2(t+0.5*dt,x+0.5*k2,y+0.5*h2,z+0.5*m2,vx+0.5*n2)
    m3 = dt*f3(t+0.5*dt,x+0.5*k2,y+0.5*h2,z+0.5*m2,vx+0.5*n2)
    n3 = dt*f4(t+0.5*dt,x+0.5*k2,y+0.5*h2,z+0.5*m2,vx+0.5*n2)

    k4 = dt*f1(t+dt,x+k3,y+h3,z+m3,vx+n3)
    h4 = dt*f2(t+dt,x+k3,y+h3,z+m3,vx+n3)
    m4 = dt*f3(t+dt,x+k3,y+h3,z+m3,vx+n3)
    n4 = dt*f4(t+dt,x+k3,y+h3,z+m3,vx+n3)

    t = t+dt
    x = x + (1/6) * (k1 + 2*k2 +2*k3 + k4)
    y = y + (1/6) * (h1 + 2*h2 +2*h3 + h4)
    z = z + (1/6) * (m1 + 2*m2 +2*m3 + m4)
    vx = vx + (1/6) * (n1 + 2*n2 +2*n3 + n4)

    return t, x, y, z, vx

#Define functions#
def f1(t,x,y,z,vx):
    return vx 
 
def f2(t,x,y,z,vx): 
    
    value = (p1 - p2*np.cos(x))/(i1*(np.sin(x)**2))
    
    return  value

def f3(t,x,y,z,vx):

    value = p2/i3 - f2(t,x,y,z,vx) * np.cos(x)

    return value

def f4(t,x,y,z,vx):

    value = (f2(t,x,y,z,vx)**2)*np.sin(x)*np.cos(x) - (i3*((f2(t,x,y,z,vx)*np.cos(x) + f3(t,x,y,z,vx))*f2(t,x,y,z,vx)*np.sin(x))/i1) + m*g*h*np.sin(x)/i1

    return value


#Function that calculates values with rk4 for different initial conditions
def runrk4(x0,p1):
    p1 = p1
    t = 0
    x = x0
    y = y0
    z = z0
    vx = vx0
    
    #Define functions to change p1#
    def f2(t,x,y,z,vx):
    
        value = (p1 - p2*np.cos(x))/(i1*(np.sin(x)**2))
    
        return  value

    def f3(t,x,y,z,vx):

        value = p2/i3 - f2(t,x,y,z,vx) * np.cos(x)

        return value

    def f4(t,x,y,z,vx):

        value = (f2(t,x,y,z,vx)**2)*np.sin(x)*np.cos(x) - (i3*((f2(t,x,y,z,vx)*np.cos(x) + f3(t,x,y,z,vx))*f2(t,x,y,z,vx)*np.sin(x))/i1) + m*g*h*np.sin(x)/i1

        return value


    t_values = [0]
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]
    vx_values = [vx0]

    max_t = 5

    while t <= max_t:
        t, x, y, z, vx = rk4_4(t, x, y,z,vx,dt, f1, f2, f3, f4)

        t_values.append(t)
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        vx_values.append(vx)
    #Make values beyong 2pi loop back.
    for i in range(len(y_values)):
        if y_values[i] <0:
            npi = abs(-y_values[i] // (2*np.pi)) + 1
            y_values[i] = y_values[i] + npi* 2*np.pi
        else:
            pass
    
    return x_values, y_values, t_values


#This function is now not needed after I found the f"{values:.2f}" notation
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

#Makes gif for a set of initial conditions. 
def rk4anim(x0,p1,sd=100, name = 'no_name.gif'):
    x_values, y_values, t_values  = runrk4(x0,p1)

    fig_width = 6
    fig_height = 5

    fig, ax = plt.subplots(figsize = (fig_width, fig_height))

    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0,np.pi)

    ax.set_xlabel('phy[rad]')
    ax.set_ylabel('theta[rad]')
    ax.invert_yaxis()

    animated_plot = ax.scatter([], [], s=0.8, c='black')

    time = ax.text(3, -0.1, 't = 0.00')
    
    sd = sd
    def animate(i):

        animated_plot.set_offsets(np.c_[y_values[:i*sd], x_values[:i*sd]])
    
        #time.set_text('t = '+ d3_maker(str(round(t_values[i*sd],3))))
        time.set_text('t = '+ f"{t_values[i*sd]:.2f}")

    nsteps = int(len(x_values)/sd)
    interval = dt
    ani = animation.FuncAnimation(fig, animate, frames = nsteps, repeat = True, interval = interval)
    fps = int(nsteps/5)

    ani.save(name, writer='pillow', fps=fps)


#This uses the information collected from the input file to write the values in output file and create the gifs.
index = int(5/dt)
open('output.txt', 'w').close()
for i in range(ninputs):
    x0, p1 = ivals[i]
    x_vals, y_vals, t_vals = runrk4(x0,p1)
    print(f"{x_vals[index]:.5e}", f"{y_vals[index]:.5e}")
    fileo = open("output.txt", "a")
    fileo.write(f"{x_vals[index]:.5e}"+" "+f"{y_vals[index]:.5e}"+"\n")
    fileo.close()
    number = str(i+1)
    gifname = "gif " + number +".gif"
    rk4anim(x0,p1,50,gifname)

#This allowed me to get a screenshot of the gif by plotting all the points as a scatter
def rk4graph1(x0,p1):
    x_values, y_values, t_values  = runrk4(x0,p1)

    fig_width = 6
    fig_height = 5

    fig, ax = plt.subplots(figsize = (fig_width, fig_height))

    time = ax.text(3, -0.1, 't = 4.95')

    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0,np.pi)
    ax.invert_yaxis()


    ax.set_xlabel('phy[rad]')
    ax.set_ylabel('theta[rad]')

    ax.scatter([y_values],[x_values], s=0.5, c='black')



# This function allowed me to judge manny different patterns of the graph for a range of p1.
def rk4graph(x0, p1_list, num_graphs):
    fig_width = 6 * num_graphs / 2
    fig_height = 5 / 2

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, num_graphs, figsize=(fig_width, fig_height))
    fig.tight_layout(pad=3)

    for i in range(num_graphs):
        # Get current axis
        ax = axes[i]

        # Run the RK4 simulation for current value of p1
        x_values, y_values, t_values = runrk4(x0, p1_list[i])

        # Plot the results
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, np.pi)
        ax.set_xlabel('psy [rad]')
        ax.set_ylabel('theta [rad]')
        ax.invert_yaxis()
        ax.scatter(y_values, x_values, s=0.5, c='black')

        # Set title for each subplot to indicate the p1 value
        ax.set_title(f'p1 = {p1_list[i]}')

    plt.show()


#This function was an initial attempt at finding what values had no nutation. It was developed with the help of Chat GPT, which I asked how to find roots of a function.
def crosses_zero(p1, x_min, x_max):
    # Define the function
    def f(x):
        return p1 - 15 * np.cos(x)
    
    # Check if the function crosses zero in the range [x_min, x_max]
    try:
        # Use brentq to find a root in the interval, if it exists
        root = brentq(f, x_min, x_max)
        return True
    except ValueError:
        # No crossing if brentq fails to find a root
        return False


#This is the method I actually used to find where there was no nutation. It simply makes a data frame for a range of values for p1 and calulates the range of theta.
def small(x):
    return abs(x)<0.1

def make_df_range(mi,ma,step,theta0 = np.pi/4):
    min_x = []
    max_x = []
    nut = []
    rang = []
    for i in np.arange(mi,ma,step):
        x_vals, y_vals, t_vals = runrk4(theta0,i)
        x_min = min(x_vals)
        x_max = max(x_vals)
        min_x.append(x_min)
        max_x.append(x_max)
        nut.append(small(x_min-x_max))
        rang.append(abs(x_min-x_max))
    return pd.DataFrame({'p1':np.arange(mi,ma,step), 'Min':min_x,'Max':max_x,'Nutation':nut, 'Range':rang})


