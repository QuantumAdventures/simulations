# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:24:54 2024

@author: tandeitnik
"""

from structeredLightSimulation import structeredLight as sl
from structeredLightSimulation import particle as pl
from structeredLightSimulation import simulation as sim
import matplotlib.pyplot as plt
import numpy as np

# structered light properties
mode = 'laguerre'
indices = [[0,0]]
coefs = [1]

# particle properties
radius = 156e-9/2
wavelength = 1550e-9

#simulation parameters
dt = 1e-7
steps = 10_000
pos0 = [100e-9,0,0]
vel0 = [0,0,0]
T0 = 293
pressure = 50
NA = 0.76
P = 0.1

# create objects
structeredLight = sl.structeredLight(mode,indices,coefs)
particle = pl.particle(radius = radius, wavelength = wavelength)
simulation = sim.simulation(particle = particle,structeredLight = structeredLight)

#simulate using Runge Kutta 4th order + Maruyama
positions, velocities = simulation.rungeKutta(dt, steps, pos0, vel0, T0, pressure,P,NA)

#plot results
timeStamps = np.linspace(0,dt*steps,steps)
plt.plot(timeStamps,positions[0,:])


