#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import scipy.signal as signal
from scipy.integrate import simpson

radius = 75e-9
kB = 1.380649e-23
T0 = 297
rho_SiO2 = 1850 # https://www.azom.com/properties.aspx?ArticleID=1387
eta_air = 18.27e-6 # Pa # (J.T.R.Watson (1995)).
d_gas = 0.372e-9 #m #(Sone (2007)), ρSiO2
pressure = 1 #mbar
#Define the time-step of the numerical integration and the max. time of integration
N = 500_000 #how many steps are simulated
dt_simulation = 1e-8 #simulation time step
tMax = N*dt_simulation
t = np.linspace(0,tMax,N)

n_particle = 1.42 #https://microparticles.de/eigenschaften
e0 = 8.85e-12
c = 3e8
wl = 1550e-9
k = 2*np.pi/wl
w0 = 0.5e-6
zr = np.pi*w0**2/wl
P = 0.8

V = 4*np.pi*radius**3/3
mass_particle = (4/3)*np.pi*radius**3*rho_SiO2
e_r = n_particle**2
alpha_cm = 3*V*e0*(e_r-1)/(e_r+2)
alpha_rad = alpha_cm/(1-((e_r-1)/(e_r+2))*((k*radius)**2 + 2j/3*(k*radius)**3))

w_x = 2*np.pi*150_000
w_y = 2*np.pi*180_000
w_z = 2*np.pi*50_000

def Gamma_env(Pressure_mbar):
    
    def mfp(P_gas):
        mfp_val = kB*T0/(2**0.5*np.pi*d_gas**2*P_gas)
        return mfp_val
    
    Pressure_pascals = 100*Pressure_mbar
    s = mfp(Pressure_pascals)
    K_n = s/radius
    c_K = 0.31*K_n/(0.785 + 1.152*K_n + K_n**2)
    gamma = 6*np.pi*eta_air*radius/mass_particle * 0.619/(0.619 + K_n) * (1+c_K)
    return gamma

if pressure == 0:
    gamma = 0
else:
    gamma = Gamma_env(pressure)

x_gauss = np.sqrt(kB*T0/(mass_particle*w_x**2))
y_gauss = np.sqrt(kB*T0/(mass_particle*w_y**2))
z_gauss = np.sqrt(kB*T0/(mass_particle*w_z**2))

v_gauss = np.sqrt(kB*T0/mass_particle)

#simulation parameters
pos0 = np.array((x_gauss,y_gauss,z_gauss))
vel0 = np.array((0,0,0))

@njit(fastmath=True)
def ode_x(x,v_x):
    
    ode_1 = v_x
    ode_2 = -w_x**2*x -gamma*v_x

    return np.array((ode_1,ode_2))

@njit(fastmath=True)
def ode_y(y,v_y):
    
    ode_1 = v_y
    ode_2 = -w_y**2*y -gamma*v_y

    return np.array((ode_1,ode_2))

@njit(fastmath=True)
def ode_z(z,v_z):
    
    Fs = 16*P*np.pi**2*np.abs(alpha_rad)**2/(3*wl**4*e0**2*c*w0**2*(1+z**2/zr**2))
    
    ode_1 = v_z
    ode_2 = -w_z**2*z -gamma*v_z + Fs/mass_particle

    return np.array((ode_1,ode_2))

#The next function applies an integration step perfoming Runge-Kutta 4th order
@njit(fastmath=True)
def rungeKuttaStep(pos,vel,dt):
    
    x = pos[0]
    y = pos[1]
    z = pos[2]
    
    v_x = vel[0]
    v_y = vel[1]
    v_z = vel[2]
    
    k1_x = dt*ode_x(x,v_x)
    k1_y = dt*ode_y(y,v_y)
    k1_z = dt*ode_z(z,v_z)
    
    k2_x = dt*ode_x(x+k1_x[0]/2,v_x+k1_x[1]/2)
    k2_y = dt*ode_y(y+k1_y[0]/2,v_y+k1_y[1]/2)
    k2_z = dt*ode_z(z+k1_z[0]/2,v_z+k1_z[1]/2)
    
    k3_x = dt*ode_x(x+k2_x[0]/2,v_x+k2_x[1]/2)
    k3_y = dt*ode_y(y+k2_y[0]/2,v_y+k2_y[1]/2)
    k3_z = dt*ode_z(z+k2_z[0]/2,v_z+k2_z[1]/2)
    
    k4_x = dt*ode_x(x+k3_x[0],v_x+k3_x[1])
    k4_y = dt*ode_y(y+k3_y[0],v_y+k3_y[1])
    k4_z = dt*ode_z(z+k3_z[0],v_z+k3_z[1])
    
    newPos = np.array((
        (x + (k1_x[0] + 2*k2_x[0] + 2*k3_x[0] + k4_x[0])/6),
        (y + (k1_y[0] + 2*k2_y[0] + 2*k3_y[0] + k4_y[0])/6),
        (z + (k1_z[0] + 2*k2_z[0] + 2*k3_z[0] + k4_z[0])/6)))
    
    
    newVel = np.array((
        (v_x + (k1_x[1] + 2*k2_x[1] + 2*k3_x[1] + k4_x[1])/6 + np.sqrt(2*kB *T0*gamma*mass_particle)*np.random.normal()*np.sqrt(dt)/mass_particle),
        (v_y + (k1_y[1] + 2*k2_y[1] + 2*k3_y[1] + k4_y[1])/6 + np.sqrt(2*kB *T0*gamma*mass_particle)*np.random.normal()*np.sqrt(dt)/mass_particle),
        (v_z + (k1_z[1] + 2*k2_z[1] + 2*k3_z[1] + k4_z[1])/6 + np.sqrt(2*kB *T0*gamma*mass_particle)*np.random.normal()*np.sqrt(dt)/mass_particle)))
    
    
    return newPos, newVel

#The next funcion applies the Runge-Kutta method the desired time interval
@njit(fastmath=True)
def rungeKutta(pos0,vel0,dt,steps):
    
    posSim = np.zeros((3,steps))
    velSim = np.zeros((3,steps))
                      
    posSim[:,0] = pos0
    velSim[:,0] = vel0
    
    for i in range(1,steps):
        
        newPos, newVel = rungeKuttaStep(posSim[:,i-1],velSim[:,i-1],dt)
        posSim[0,i] = newPos[0]
        posSim[1,i] = newPos[1]
        posSim[2,i] = newPos[2]
        velSim[0,i] = newVel[0]
        velSim[1,i] = newVel[1]
        velSim[2,i] = newVel[2]
        

    return posSim, velSim

#simulate the system!
positions, velocities = rungeKutta(pos0,vel0,dt_simulation,N)
