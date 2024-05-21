# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:24:54 2024

@author: tandeitnik
"""

from structeredLightSimulation import structeredLight as sl
from structeredLightSimulation import particle as pl
from structeredLightSimulation import simulation as sim
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# structered light properties
mode = 'laguerre'
indices_x = [[0,0]] #indices for the x polarization
coefs_x = [1] #coeficients of the indices
indices_y = [[0,0]] #indices for the y polarization
coefs_y = [0] #coeficients of the indices
polarizationVector = [1,0] #polarization vector

# particle properties
radius = 156e-9/2
wavelength = 1550e-9

#simulation parameters
dt = 1e-8
steps = 100_000
pos0 = [10e-9,10e-9,0]
vel0 = [0,0,0]
T0 = 293
pressure = 10
NA = 0.76
P = 0.1

# create objects
structeredLight = sl.structeredLight(mode,indices_x,coefs_x,indices_y,coefs_y,polarizationVector)
particle = pl.particle(radius = radius, wavelength = wavelength)
simulation = sim.simulation(particle = particle,structeredLight = structeredLight)

#print the intensity at a given z
x = np.linspace(-2e-6,2e-6,1000)
y = np.linspace(-2e-6,2e-6,1000)
z = 0
X,Y = np.meshgrid(x,y)
x0 = 0
y0 = 0
structeredLight.printIntensity(x = X, y = Y,z=z,x0=x0,y0=y0,NA=NA,wl=wavelength)

#simulate using Runge Kutta 4th order + Maruyama
positions_rk, velocities_rk = simulation.rungeKutta(dt, steps, pos0, vel0, T0, pressure,P,NA)

#simulate using Euler Maruyama (4x faster, but with worst 'precision')
positions_em, velocities_em = simulation.eulerMaruyama(dt, steps, pos0, vel0, T0, pressure,P,NA)

#plot results
timeStamps = np.linspace(0,dt*steps,steps)
plt.plot(timeStamps,positions_rk[0,:], label = 'Runge-Kutta 4th order')
plt.plot(timeStamps,positions_em[0,:], label = 'Euler-Maruyama')
plt.legend()

#evaluate the PSD amnd plot
fs = 1e6 #frequency sample of the oscilloscope
windows = 5
f, PSD_x = signal.welch(positions_rk[0][::int(1/dt/fs)], fs, window = 'hamming', nperseg = int(len(positions_rk[0][::int(1/dt/fs)])/windows))
f, PSD_y = signal.welch(positions_rk[1][::int(1/dt/fs)], fs, window = 'hamming', nperseg = int(len(positions_rk[1][::int(1/dt/fs)])/windows))
f, PSD_z = signal.welch(positions_rk[2][::int(1/dt/fs)], fs, window = 'hamming', nperseg = int(len(positions_rk[2][::int(1/dt/fs)])/windows))

plt.loglog(f,PSD_x,label = 'PSD x')
plt.loglog(f,PSD_y,label = 'PSD y')
plt.loglog(f,PSD_z,label = 'PSD z')
plt.legend()

"""
thing to do:
    
add fit to the PSD
add normalization of the coeficients
"""


