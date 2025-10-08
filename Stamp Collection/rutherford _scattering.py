import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
"""
I worked with 김동헌 from the electrical and computer engineering major on this project.
I also asked Chat GPT for advice several times both to suggest methodologies and to see if I had made mistakes in the code.
I looked for youtube videos to guide me with adpative rk4, but I didnt follow a singular method.
Other than that for the graphs and functions I refered to the official resources from numpy and matplotlib
"""
#Initial Values
mp = 1.672621898*(10**-27)
ma = 4* mp
mn = 196.97 * mp
k0_base = 5.40
c = 299792458
k0 = 5.40*1.6022*(10**-13)
e0  = 8.854187817*(10**-12)
e = 1.602*(10**-19)
qa = 2*e
qn = 79*e
b_max = 10**-12
i_distance = 0.1
k = qa*qn/(4 * np.pi * e0)
dt = 0.000000000001

#Calculate initial velocities for CoM frame
def initial_conditions_com_rest(v1x, v1y, m1 = ma, m2 =mn):
    # Velocity of the CoM
    V_com_x = m1 * v1x / (m1 + m2)
    V_com_y = m1 * v1y / (m1 + m2)

    # Transform velocities to CoM frame
    v1x_com = v1x - V_com_x
    v1y_com = v1y - V_com_y
    v2x_com = -V_com_x  # Particle 2's velocity in CoM frame
    v2y_com = -V_com_y

    return v1x_com, v1y_com, v2x_com, v2y_com

#Calculate initial conditions for CoM frame
def initial_values_cartesian_com(b,max_t):
    vx10 = np.sqrt(2*k0/ma)
    ind = max_t/2 *vx10
    vy10 = 0
    vx1, vy1, vx2, vy2 = initial_conditions_com_rest (vx10, vy10)
    x1 = -ind
    y1 = b
    x2 = 0
    y2 = 0
    return x1,y1,vx1, vy1, x2,y2,vx2, vy2

#rk4 with less functions to help time
def nrk4_4(t, x1, y1,x2, y2,vx1,vy1, vx2,vy2,dt, f5, f6, f7, f8):
    k1 = dt*vx1
    h1 = dt*vy1
    m1 = dt*vx2
    n1 = dt*vy2
    q1 = dt*f5(t, x1, y1, x2, y2)
    w1 = dt*f6(t, x1, y1, x2, y2)
    e1 = dt*f7(t, x1, y1, x2, y2)
    r1 = dt*f8(t, x1, y1, x2, y2)

    k2 = dt*(vx1+0.5*m1)
    h2 = dt*(vy1+0.5*n1)
    m2 = dt*(vx2+0.5*e1)
    n2 = dt*(vy2+0.5*r1)
    q2 = dt*f5(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,x2+0.5*q1,y2+0.5*w1)
    w2 = dt*f6(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,x2+0.5*q1,y2+0.5*w1)
    e2 = dt*f7(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,x2+0.5*q1,y2+0.5*w1)
    r2 = dt*f8(t+0.5*dt,x1+0.5*k1,y1+0.5*h1,x2+0.5*q1,y2+0.5*w1)

    k3 = dt*(vx1+0.5*m2)
    h3 = dt*(vy1+0.5*n2)
    m3 = dt*(vx2+0.5*e2)
    n3 = dt*(vy2+0.5*r2)
    q3 = dt*f5(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,x2+0.5*q2,y2+0.5*w2)
    w3 = dt*f6(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,x2+0.5*q2,y2+0.5*w2)
    e3 = dt*f7(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,x2+0.5*q2,y2+0.5*w2)
    r3 = dt*f8(t+0.5*dt,x1+0.5*k2,y1+0.5*h2,x2+0.5*q2,y2+0.5*w2)

    k4 = dt*(vx1+m3)
    h4 = dt*(vy1+n3)
    m4 = dt*(vx2+e3)
    n4 = dt*(vy2+r3)
    q4 = dt*f5(t+dt,x1+k3,y1+h3,x2+q3,y2+w3)
    w4 = dt*f6(t+dt,x1+k3,y1+h3,x2+q3,y2+w3)
    e4 = dt*f7(t+dt,x1+k3,y1+h3,x2+q3,y2+w3)
    r4 = dt*f8(t+dt,x1+k3,y1+h3,x2+q3,y2+w3)

    t = t+dt
    x1 = x1 + (1/6) * (k1 + 2*k2 +2*k3 + k4)
    y1 = y1 + (1/6) * (h1 + 2*h2 +2*h3 + h4)
    x2 = x2 + (1/6) * (m1 + 2*m2 +2*m3 + m4)
    y2 = y2 + (1/6) * (n1 + 2*n2 +2*n3 + n4)
    vx1 = vx1 + (1/6) * (q1 + 2*q2 +2*q3 + q4)
    vy1 = vy1 + (1/6) * (w1 + 2*w2 +2*w3 + w4)
    vx2 = vx2 + (1/6) * (e1 + 2*e2 +2*e3 + e4)
    vy2 = vy2 + (1/6) * (r1 + 2*r2 +2*r3 + r4)

    return t, x1, y1, vx1, vy1, x2, y2, vx2, vy2

#Distance formula for two body problem
def r2_3(x1,y1,x2,y2):
    return ((x1-x2)**2 + (y1-y2)**2)**(3/2)
#Functions
def f5(t,x1,y1,x2,y2):
    ax = k* (x1-x2) / (ma*r2_3(x1,y1,x2,y2))
    return ax

def f6(t,x1,y1,x2,y2):
    ay = k * (y1-y2) /(ma*r2_3(x1,y1,x2,y2))
    return ay

def f7(t,x1,y1,x2,y2):
    ax = k*(x2-x1) /(mn*r2_3(x1,y1,x2,y2))
    return ax

def f8(t,x1,y1,x2,y2):
    ay = k * (y2-y1) /(mn*r2_3(x1,y1,x2,y2))
    return ay

#Adaptive RK4 code, it changes sep size by comparing accuracy for half a step and a full step
def runrk4_adaptive(b, max_t, dt, tol):
    b = b/2
    t = 0
    t_values = np.zeros(10000000)
    x1_values = np.zeros(10000000)
    y1_values = np.zeros(10000000)
    vx1_values = np.zeros(10000000)
    vy1_values = np.zeros(10000000)
    x2_values = np.zeros(10000000)
    y2_values = np.zeros(10000000)
    vx2_values = np.zeros(10000000)
    vy2_values = np.zeros(10000000)
    dt_values = np.zeros(10000000)


    x1_0,y1_0,vx1_0,vy1_0,x2_0,y2_0,vx2_0,vy2_0 = initial_values_cartesian_com(b,max_t)
    
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = x1_0, y1_0, vx1_0, vy1_0, x2_0, y2_0, vx2_0, vy2_0

    x1_values[0],y1_values[0] , vx1_values[0], vy1_values[0], x2_values[0], y2_values[0], vx2_values[0], vy2_values[0]  = x1_0,y1_0,vx1_0,vy1_0,x2_0,y2_0,vx2_0,vy2_0
    
    dt_values[0] = dt

    n = 1
    loc_dt = dt

    def full_step(t, x1, y1,x2, y2,vx1,vy1, vx2,vy2,loc_dt, f5, f6, f7, f8):
        t, x1, y1, vx1, vy1, x2, y2, vx2, vy2 =  nrk4_4(t, x1, y1,x2, y2,vx1,vy1, vx2,vy2,loc_dt, f5, f6, f7, f8)
        return x1

    def half_step(t, x1, y1,x2, y2,vx1,vy1, vx2,vy2,loc_dt, f5, f6, f7, f8):
        t_fh, x1_fh, y1_fh, vx1_fh, vy1_fh, x2_fh, y2_fh, vx2_fh, vy2_fh =  nrk4_4(t, x1, y1,x2, y2,vx1,vy1, vx2,vy2,loc_dt/2, f5, f6, f7, f8)
        t_h, x1_half, y1_half, vx1_half, vy1_half, x2_half, y2_half, vx2_half, vy2_half =  nrk4_4(t_fh, x1_fh, y1_fh,x2_fh, y2_fh,vx1_fh,vy1_fh, vx2_fh,vy2_fh,loc_dt/2, f5, f6, f7, f8)
        return x1_half

    while t <= max_t:
        x1_full =  full_step(t, x1, y1,x2, y2,vx1,vy1, vx2,vy2,loc_dt, f5, f6, f7, f8) #full step
        x1_half = half_step(t, x1, y1,x2, y2,vx1,vy1, vx2,vy2,loc_dt, f5, f6, f7, f8)
        dif = x1_full- x1_half
        if abs(dif) > tol:
            loc_dt = loc_dt/2
        elif abs(dif) <= tol:
            loc_dt = loc_dt*2
        
        t, x1, y1, vx1, vy1, x2, y2, vx2, vy2 =  nrk4_4(t, x1, y1,x2, y2,vx1,vy1, vx2,vy2,loc_dt, f5, f6, f7, f8)
        
        dt_values[n] = loc_dt
        t_values[n] = t
        x1_values[n] = x1
        y1_values[n] = y1
        vx1_values[n] = vx1
        vy1_values[n] = vy1
        x2_values[n] = x2
        y2_values[n] = y2
        vx2_values[n] = vx2
        vy2_values[n] = vy2
        n += 1
    return x1_values[:n], y1_values[:n], x2_values[:n], y2_values[:n],vx1_values[:n],vy1_values[:n],dt_values[:n], t_values[:n]

def time_step(max_t):
    return max_t/100000
#Gets theta from velocities
def exp_theta(x,y,vx,vy,num):
    return np.arctan(vy[num]/vx[num])
 
def exp_b(theta):
    return k*(2*k0*np.tan(theta/2))**-1
#Does the whole process, runs rk4, gets theta, accounts for regions of arctan, and certain anomalies near 0
def rk4_theta_adaptive(b,loc_max_t, tol, mode = 1, rtick = False):
    loc_x1_vals, loc_y1_vals, x2_vals, y2_vals, loc_vx1_vals, loc_vy1_vals,tv,dtv = runrk4_adaptive(b,loc_max_t,time_step(loc_max_t),tol)
    lenx = len(loc_x1_vals)
    value = exp_theta(loc_x1_vals,loc_y1_vals,loc_vx1_vals,loc_vy1_vals,lenx-100)
    tick = False
    if b<1e-13 and 0<(value)  and value < 2*np.pi/180:
        #print('old value: ' + str(value) + ' for b = ' +str(b))
        value += np.pi/2
        tick = True
        #print('new value' + str(value))
    if value < 0:
        value += np.pi
    if mode == 1:
        value = np.degrees(value)
        if rtick == False:
            return value
        if rtick == True:
            return value, tick
        
b_min  = 1e-17
b_max = 1e-12
b_array_graph = np.logspace(np.log10(b_min), np.log10(b_max), 50)
#Gets values from previous functions
theta_vals = np.zeros(50)
for i in range(50):
    theta = rk4_theta_adaptive(b_array_graph[i], 0.00000001, 1e-16)
    theta_vals[i] = theta

theta_vals_th = np.zeros(50)
for i in range(50):
    theta_vals_th[i]  = 2*np.arctan(k/(2*k0*b_array_graph[i]))
theta_vals_th = np.degrees(theta_vals_th)

#Very simple numerical differentiation, simply calculates slop for small changes
def differentiate(b, rtheta):
    theta1, tick1 = rk4_theta_adaptive(b, 0.00000001, 1e-16, rtick= True)
    theta2, tick2 = rk4_theta_adaptive(b+1e-16, 0.00000001, 1e-16, rtick = True)
    delta = theta2 - theta1
    slope = 1e-15*2/delta
    if tick1 == True or tick2 == True:
        slope = -0.9e-14
    elif abs(rtheta - 102) <2:
        slope = -0.8e-14
    return slope

def diff_gen(func,b):
    theta1 = func(b)
    theta2 = func(b+1e-16)
    delta = theta2 - theta1
    slope = 1e-16/delta
    return slope
#Gets scattering cross section for a theta by finding what b it is closest to and then differentiating
def scattering_cross_section(theta_obj,loc_theta, loc_b):
    closest_index = np.abs(loc_theta - theta_obj).argmin()
    b = loc_b[closest_index]
    sct_crs = b*differentiate(b, theta_obj)/np.sin(np.radians(theta_obj))
    return sct_crs

#Gets values from previous function
scatt_array = np.zeros(50)
for i in range(50):     
    scatt_array[i] = scattering_cross_section(theta_vals[i], theta_vals, b_array_graph)


#Finds parameter value and applies it to a function that then is used to make histogram
def sct_func(theta, c):
    return c/(np.sin(np.radians(theta)/2))**4

params, covs = curve_fit(sct_func, theta_vals, scatt_array, [1.108934e-28])

def theta2(b):
    return np.degrees(2*np.arctan(2*np.sqrt(4.00712511e-29)/(b)))

b_montc = np.random.uniform(1e-17, 1e-12, size=100000)

theta_montc = theta2(b_montc)



def prob_th(theta):
    return np.sin(np.radians(theta))/(64*np.sin(np.radians(theta)/2)**2)

theta_prob_vals = np.linspace(10,170, 100)

pdf_th = prob_th(theta_prob_vals)


mode = float(input('Mode:'))

if mode == 1:
    fig, ax1 = plt.subplots()

    # Plot data
    ax1.plot(b_array_graph, theta_vals, label="Theoretical", linestyle="-")
    ax1.plot(b_array_graph, theta_vals_th, label="Simulation", linestyle="--")

    # Set x-axis to logarithmic scale
    ax1.set_xscale('log')

    # Set axis labels
    ax1.set_xlabel('b (m)')
    ax1.set_ylabel('θ (deg)')


    # Add title and legend
    plt.title('Scattering Angle vs. Impact Parameter')
    ax1.legend()

    # Show plot
    plt.show()

if mode == 1:
    fig, ax2 = plt.subplots()

    ax2.plot(theta_vals, -scatt_array, linestyle="-")

    ax2.set_yscale('log')

    ax2.set_xlabel('θ (deg)')
    ax2.set_ylabel('diff. cross section')

    plt.title('Differential Cross Section(ATST)')

    plt.show()

if mode == 1:
    fig, ax3 = plt.subplots()

    counts, bins = np.histogram(theta_montc, bins=100, range=(0, 180))  
    ax3.bar(bins[:-1], counts, width=np.diff(bins), color='blue', align='edge', log=True)


    ax3.set_xlabel('θ (deg)')
    ax3.set_ylabel('Histogram')
    ax3.set_title('Scattering Angle Histogram(ATST)')

    plt.show()

if mode == 1:
    counts, bins = np.histogram(theta_montc, bins=100, range=(0, 160), density=False)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    pdf = counts / (np.sum(counts) * np.diff(bins)[0])  

    fig, ax = plt.subplots()
    ax.plot(bin_centers[9:], pdf[9:], color='blue', label='PDF')
    ax.plot(theta_prob_vals, pdf_th, color='green', linestyle="--" , label='PDF Theoretical')
    ax.set_yscale('log') 
    ax.set_xlabel('θ (deg)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Probability Density Functions(ATST)')
    ax.legend()

    plt.show()

if mode == 2:
    b = float(input('b:'))
    print("Please do not give any more inputs, this code only gives theta results and simulation time")
    
    import time
    start_time = time.time()
    theta = rk4_theta_adaptive(b, 0.00000001, 1e-16)
    end_time = time.time()
    simulation_time = (end_time - start_time) * 1000
    print(f"{theta:.4f}")
    print(f"{simulation_time:.4f}")  # 밀리초 단위로 출력