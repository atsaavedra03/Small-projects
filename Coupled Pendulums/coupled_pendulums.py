"""
Name: Saavedra Torres Andres Tomas
The code uses pillower to save the animation so it is required to run succesfully. Other than that numpy and matplotlib were used.

References
I used the following resources

Rk4 method as provided by the TA
https://primer-computational-mathematics.github.io/book/c_mathematics/numerical_methods/5_Runge_Kutta_method.html

Single pendulum animation from where I based the two pendulum animation
https://scipython.com/book2/chapter-7-matplotlib/problems/p77/animating-a-pendulum/

The matplotlib library
https://matplotlib.org/

I used help from the followiing page to download as a gif
https://holypython.com/how-to-save-matplotlib-animations-the-ultimate-guide/

Lastly I asked Chat GPT to detect errors in my code about 4 times in total
https://chatgpt.com/

I didn't work with any other classmate
"""


#Import libraries
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.patches as patch
import matplotlib.lines as lines

#Collect Inputs
line1 = input()
line2 = input()
line3 = input()

#Create storage list
flist= []

#Parse data for first two lines
l1add  = line1.split(' ')
for x in range(0,len(l1add)):
    flist.append(l1add[x])

l2add  = line2.split(' ')
for x in range(0,len(l2add)):
    flist.append(l2add[x])

mode = float(line3)
max_t = 0
dt = 0.001
t_list = {"name":3}
n = 1
#Depending on mode parse data and collect correct parameters
if mode ==1:
    n = int(input())
    t_list = {}
    for i in range(1,n+1):
        t_number = "t" + str(i)
        t_list[t_number] = float(input())
    dt = 0.001
    max_t = 40
elif mode ==2:
    line4 = input()
    l4 = line4.split(' ')
    l4 = [float(i) for i in l4]
    max_t,dt = l4

#Make values in storage into floats
flist = [float(i) for i in flist]

#Establish parameters from storage
m1, l1, m2, l2, deltax, x0, y0 = flist[0:7]

#Establish final initial parameters that are not defined by text
t0 = 0
vx0 = 0
vy0 = 0
g =9.8
k = 1
deltay = l1-l2

#Define the Runge Kutta method for 4 variables (x:theta, y:omega, vx: velocity theta, vy: velocity omega)
def rk4_4(t, x, y,vx,vy,dt, f1, f2, f3, f4):
    k1 = dt*f1(t,x,y,vx,vy)
    h1 = dt*f2(t,x,y,vx,vy)
    m1 = dt*f3(t,x,y,vx,vy)
    n1 = dt*f4(t,x,y,vx,vy)

    k2 = dt*f1(t+0.5*dt,x+0.5*k1,y+0.5*h1,vx+0.5*m1,vy+0.5*n1)
    h2 = dt*f2(t+0.5*dt,x+0.5*k1,y+0.5*h1,vx+0.5*m1,vy+0.5*n1)
    m2 = dt*f3(t+0.5*dt,x+0.5*k1,y+0.5*h1,vx+0.5*m1,vy+0.5*n1)
    n2 = dt*f4(t+0.5*dt,x+0.5*k1,y+0.5*h1,vx+0.5*m1,vy+0.5*n1)

    k3 = dt*f1(t+0.5*dt,x+0.5*k2,y+0.5*h2,vx+0.5*m2,vy+0.5*n2)
    h3 = dt*f2(t+0.5*dt,x+0.5*k2,y+0.5*h2,vx+0.5*m2,vy+0.5*n2)
    m3 = dt*f3(t+0.5*dt,x+0.5*k2,y+0.5*h2,vx+0.5*m2,vy+0.5*n2)
    n3 = dt*f4(t+0.5*dt,x+0.5*k2,y+0.5*h2,vx+0.5*m2,vy+0.5*n2)

    k4 = dt*f1(t+dt,x+k3,y+h3,vx+m3,vy+n3)
    h4 = dt*f2(t+dt,x+k3,y+h3,vx+m3,vy+n3)
    m4 = dt*f3(t+dt,x+k3,y+h3,vx+m3,vy+n3)
    n4 = dt*f4(t+dt,x+k3,y+h3,vx+m3,vy+n3)


    t = t+dt
    x = x + (1/6) * (k1 + 2*k2 +2*k3 + k4)
    y = y + (1/6) * (h1 + 2*h2 +2*h3 + h4)
    vx = vx + (1/6) * (m1 + 2*m2 +2*m3 + m4)
    vy = vy + (1/6) * (n1 + 2*n2 +2*n3 + n4)
    return t, x, y, vx, vy

#Define functions to apply in RK4
def f1(t,x,y,vx,vy):
    return vx

def f2(t,x,y,vx,vy):
    return vy



if l1>=l2:
    def f3(t,x,y,vx,vy):
        initial = np.sqrt(deltay**2 + deltax**2)
        changed = np.sqrt((deltay+l2*np.cos(y)-l1*np.cos(x))**2 + (deltax+l2*np.sin(y)-l1*np.sin(x))**2)
        cos = (deltax +l2*np.sin(y)-l1*np.sin(x))/changed
        base = (-g/l1 *np.sin(x) - (k/(l1*m1) * (initial - changed)*cos))
        # ax = (-g/l1 *np.sin(x) + k/(l1*m1) * ((np.sqrt(deltay**2 + deltax**2)-np.sqrt((deltay+l2*np.cos(y)-l1*np.cos(x))**2 + (deltax+l2*np.sin(x)-l1*np.sin(x))**2))*(deltax +l2*np.sin(y)*(np.sqrt((deltay+l2*np.cos(y)-l1*np.cos(x))**2 + (deltax+l2*np.sin(x)-l1*np.sin(x))**2))^-1)))
        return base

    def f4(t,x,y,vx,vy):
        initial = np.sqrt(deltay**2 + deltax**2)
        changed = np.sqrt((deltay+l2*np.cos(y)-l1*np.cos(x))**2 + (deltax+l2*np.sin(y)-l1*np.sin(x))**2)
        cos = (deltax +l2*np.sin(y)-l1*np.sin(x))/changed
        base = (-g/l2 *np.sin(y) + (k/(l2*m2) * (initial - changed)*cos))
        return base
elif l2< l1:
    def f3(t,x,y,vx,vy):
        initial = np.sqrt(deltay**2 + deltax**2)
        changed = np.sqrt((deltay-l2*np.cos(y)+l1*np.cos(x))**2 + (deltax+l2*np.sin(y)-l1*np.sin(x))**2)
        cos = (deltax +l2*np.sin(y)-l1*np.sin(x))/changed
        base = (-g/l1 *np.sin(x) - (k/(l1*m1) * (initial - changed)*cos))
        # ax = (-g/l1 *np.sin(x) + k/(l1*m1) * ((np.sqrt(deltay**2 + deltax**2)-np.sqrt((deltay+l2*np.cos(y)-l1*np.cos(x))**2 + (deltax+l2*np.sin(x)-l1*np.sin(x))**2))*(deltax +l2*np.sin(y)*(np.sqrt((deltay+l2*np.cos(y)-l1*np.cos(x))**2 + (deltax+l2*np.sin(x)-l1*np.sin(x))**2))^-1)))
        return base

    def f4(t,x,y,vx,vy):
        initial = np.sqrt(deltay**2 + deltax**2)
        changed = np.sqrt((deltay-l2*np.cos(y)+l1*np.cos(x))**2 + (deltax+l2*np.sin(y)-l1*np.sin(x))**2)
        cos = (deltax +l2*np.sin(y)-l1*np.sin(x))/changed
        base = (-g/l2 *np.sin(y) + (k/(l2*m2) * (initial - changed)*cos))
        return base


#The following are all attempts at defining the function
"""
def f3(t,x,y,vx,vy):
    base = -g/l1 *np.sin(x) + (k/(l1*m1) * (l2*np.sin(y)-l1*np.sin(x)))
    return base

def f4(t,x,y,vx,vy):
    base = -g/l2 *np.sin(y) - (k/(l2*m2) * (l2*np.sin(y)-l1*np.sin(x)))
    return base

def f3(t,x,y,vx,vy):
    ax = (-g/l - k/m1)*x + k/m1*y
    return ax

def f4(t,x,y,vx,vy):
    ay = (-g/l - k/m2)*y + k/m2*x
    return ay
"""

#Establish storage list for all 5 variables
t_values = [t0]
x_values = [x0]
y_values = [y0]
vx_values = [vx0]
vy_values = [vy0]


#Set up iterable variables as their intial values
t = t0
x = x0
y = y0
vx = vx0
vy = vy0

#Iterate till max_t
while t <= max_t:
    t, x, y, vx, vy = rk4_4(t,x,y,vx,vy,dt,f1,f2,f3,f4)

    t_values.append(t)
    x_values.append(x)
    y_values.append(y)
    vx_values.append(vx)
    vy_values.append(vy)


#Functions to change angles into cartesian coordinates for each pendulum
def ptoc(theta, l):
    return l * np.sin(theta), -l * np.cos(theta)
def ptoc2(theta, l):
    return l * np.sin(theta)+deltax, -l * np.cos(theta)

def divide(a,b):
    return a/b
#Depending on the mode give the apropiate output
if mode == 1:
    #Gets the title names from the dictionary
    keys = list(t_list.keys())
    #Added "slow down" term to account for the prediction overshooting
    for i in range(0,n):
        name = keys[i]
        rdc_term = int(divide(t_list.get(name),5))+1
        number  = int(divide(t_list.get(name),dt)) - rdc_term
        print(str(round(x_values[number],3)),str(round(vx_values[number], 3)),str(round(y_values[number],3)),str(round(vy_values[number], 3)))

elif mode == 2:
    # Create a still image of the two pendulums connected by a spring
    x1, y1 = ptoc(x0,l1)
    x2, y2 = ptoc2(y0,l2)

    fig_width = deltax * 10
    fig_height = abs(max(y1, y2)) * 10


    fig, ax = plt.subplots(figsize = (fig_width, fig_height))
    ax.set_aspect('equal')

    spring, = ax.plot([x1,x2],[y1,y2], lw = 1.5, c = 'g', linestyle = '--')
    line1pre = lines.Line2D([0,x1],[0,y1], lw = 1.5, c = 'b')
    line1 = ax.add_line(line1pre)
    line2pre = lines.Line2D([deltax,x2],[0,y2], lw = 1.5, c = 'r')
    line2 = ax.add_line(line2pre)
    ballsize = 0.015
    bob1 = ax.add_patch(patch.Circle(ptoc(x0, l1),ballsize, fc = 'b', zorder = 3))
    bob2 = ax.add_patch(patch.Circle(ptoc2(y0, l2),ballsize, fc = 'r', zorder = 3))

    pivot1 = ax.add_patch(patch.Circle((0,0),ballsize, fc = 'b', zorder = 3))
    pivot2 = ax.add_patch(patch.Circle((deltax,0),ballsize, fc = 'r', zorder = 3))

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.9, 0.1)

    #Add time label
    time =ax.text(-0.48, 0.05, 't = '+ str(t0))

    #Animate the images by modifying the x and y values calculated from the angle value
    #Function to change the position of objects in the still image
    def animate(i):
        x1,y1 = ptoc(x_values[i*10],l1)
        line1.set_data([0,x1],[0,y1])
        bob1.set(center=(x1,y1))
        x2,y2 = ptoc2(y_values[i*10],l2)
        line2.set_data([deltax,x2],[0,y2])
        bob2.set(center = (x2,y2))
        
        time.set_text('t = '+ str(round(t_values[i*10],2)))
        spring.set_data([x1,x2],[y1,y2])

    #Animate for the required values
    nsteps = int(len(x_values)/10)
    interval = dt*1000
    ani = animation.FuncAnimation(fig, animate, frames = nsteps, repeat = True, interval = interval)
    ani.save('Animated Coupled Pendulums ATST.gif', writer='pillow', fps=10)