#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 19:40:56 2020

@author: thiagoguerreiro
Adapted from : http://hockygroup.hosting.nyu.edu/exercise/langevin-dynamics.html
"""


# Import useful stuff
import matplotlib.pyplot as plt
from glob import glob
import scipy, pylab
import scipy.optimize
import numpy as np
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.fftpack import fft, ifft
from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy import signal
import time as time
from scipy.stats import kurtosis
import scipy.stats as stats


######### Define functions to simulate trajectories and potentials ############
# Generate positions under the influence of a linear force
def linear_simulation(t_max, dt, initial_position, k_0, k_lin, gamma):
    """
    Simulate the brownian movement of a overdamped system under linear force 
    through Euler-Maruyama method 

    Parameters
    ----------
    t_max            : final simulation time, such that t \in (0, t_max)
    dt               : time increment
    initial_position : initial condition
    k_lin            : spring constant
    g                : damping constant

    Returns
    -------
    save_times : array with time stamps
    positions : array with simulated positions
    """
    
    global k_B , T, N_time, N_dimensions
    
    t = np.linspace(0, t_max, N_time)                                          # Array with time stamps
    
    size_system = (N_dimensions, N_time)                              # Arbitrary number of axes and N_time time stamps
    
    positions = np.zeros(size_system)                                          # Array to store calculated positions in each axis in each time
    positions[:, 0] = initial_position                                         # Define initial condition foe the three axes
    
    w = np.sqrt(2.0*k_B*T*dt/gamma) * np.random.normal(size=size_system)        # Wiener increment
    
    for i in range(N_time-1):                                                  # Loop through each time stamp and calculate current position
        grad_potencial = 2.0*np.multiply(k_lin, positions[:, i])
        
       # F_scatt = F_scatt_amplitude*V_lin(positions[0, i], positions[1, i], positions[2, i], k_0, k_lin)
       # F_scatt = F_scatt*np.array([0, 0, 1])
        positions[:, i+1] = positions[:, i] - grad_potencial*dt/gamma + w[:,i] # Euler-Maruyama method 
        #positions[:, i+1] = positions[:, i] - grad_potencial*dt/gamma + F_scatt*dt/gamma + w[:,i] # Euler-Maruyama method 
        
    return t, positions

# Generate positions under the influence of non-linearity (cubic force)
def non_linear_simulation(t_max, dt, initial_position, k_0, k_lin, k_cub, k_crossed, gamma):
    """
    Simulate the brownian movement of a overdamped system under the influence of non-linearity
    through Euler-Maruyama method 

    Parameters
    ----------
    t_max            : final simulation time, such that t \in (0, t_max)
    dt               : time increment
    initial_position : initial condition
    k_lin            : linear term constant
    k_cub            : cubic  term constant
    gamma            : damping constant

    Returns
    -------
    save_times : array with time stamps
    positions : array with simulated positions
    """
    global k_B , T, N_time, N_dimensions
    
    N_time = int(t_max/dt + 1)                                                 # Number of time stamps
    t = np.linspace(0, t_max, N_time)                                          # Array with time stamps
    
    size_system = (N_dimensions, N_time)                                       # Arbitrary number of axes and N_time time stamps
    
    positions = np.zeros(size_system)                                          # Array to store calculated positions in each axis in each time
    positions[:, 0] = initial_position                                         # Define initial condition foe the three axes
    
    w = np.sqrt(2.0 * k_B * T * dt/gamma) * np.random.normal(size=size_system) # Wiener increment
    
    for i in range(N_time-1):                                                  # Loop through each time stamp and calculate current position
        grad_potencial = 2.0*np.multiply(k_lin, positions[:, i]) + 4.0*np.multiply(k_cub,positions[:, i]**3)
        grad_potencial[0] +=  4.0*k_crossed*positions[0, i]*positions[1, i]**2
        grad_potencial[1] +=  4.0*k_crossed*positions[1, i]*positions[0, i]**2
        
        #F_scatt = F_scatt_amplitude*V_cub(positions[0, i], positions[1, i], positions[2, i], k_0, k_lin, k_cub, k_crossed)
        #F_scatt = F_scatt*np.array([0, 0, 1])
        positions[:, i+1] = positions[:, i] - grad_potencial*dt/gamma + w[:,i] # Euler-Maruyama method 
        #positions[:, i+1] = positions[:, i] - grad_potencial*dt/gamma + F_scatt*dt/gamma + w[:,i] # Euler-Maruyama method 
        
    return t, positions

# Non-linear potential
def V_cub(X, Y, k_0, k_lin, k_cub, k_cross):
    """
    Non-linear potential 

    Parameters
    ----------
    X       : x coordinate of the points where the potential is to be calculated
    Y       : y coordinate of the points where the potential is to be calculated
    Z       : z coordinate of the points where the potential is to be calculated
    k_0     : constant energy shift in the potential
    k_lin   : amplitude of the     linear term of the potential for each axis
    k_cub   : amplitude of the non-linear term of the potential for each axis
    k_cross : amplitude of the cross      term of the potential between x and y

    Returns
    -------
    V : potential at positions specified in input parameters
    """
    
    V =  k_0 + k_lin[0]*X**2 + k_lin[1]*Y**2 + k_cub[0]*X**4 + k_cub[1]*Y**4 + 2.0*k_cross*(X**2)*(Y**2)
    return V

# Linear potential
def V_lin(X, Y, k_0, k_lin):
    """
    Non-linear potential 

    Parameters
    ----------
    X     : x coordinate of the points where the potential is to be calculated
    Y     : y coordinate of the points where the potential is to be calculated
    Z     : z coordinate of the points where the potential is to be calculated
    k_0   : constant energy shift in the potential
    k_lin : amplitude of the linear term of the potential for each axis

    Returns
    -------
    V : potential at positions specified in input parameters
    """
    
    V = V =  k_0 + k_lin[0]*X**2 + k_lin[1]*Y**2
    return V
###############################################################################


############################## Define parameter ###############################
# Universal constant
k_B = 1.38e-23                                 # Boltzmann constant    [J/K]

# Thermal bath parameters
T = 300                                        # Bath Temperature      [K]
gamma = 1.29926e-9                             # Damping coefficient   [kg/s]   


# Simulation time
max_time = 1.0                                 # Final simulation time [s]
dt = 1e-4                                      # Time increment        [s]
N_time = int(max_time/dt + 1)                  # Number of time stamps


# Grid for plot of potential landscape
rho = np.linspace(-2e-6, 2e-6, 100)            # Radial       axis range [m]
z   = np.linspace(-2e-6, 2e-6, 100)            # Longitudinal axis range [m]
X, Y = np.meshgrid(rho, z)                     # 2D grid to plot potential landscape [m]


# Number of dimensions and initial condition
N_dimensions = 3                                  # Number of dimensions
x0 = np.zeros(N_dimensions)                       # Initial position in each axis [m]


# Extra parameters if we whish to consider the Scattering Force
n_p = 1.46                                        # Refractive index of the particle
n_m = 1.33                                        # Refractive index of the medium
m = n_p/n_m                                       # Relative refractive index of the particle

wavelength = 780e-9                               # Wavelength of the trapping beam [m]
R = 70e-9                                         # Radius of optically trapped nanosphere [m]

F_scatt_amplitude = -(m**2 - 1)/(m**2 + 2)*64*(np.pi**4)*(R**3)/(3*n_m*wavelength)

# Achar k_cub que faz kurtosis ser alta o suficiente para ver nao-linearidade
# Fazer caso extra só 2D


# Trap parameters for single example
k_0 = 0

k_lin_x = 1e-7                           # Spring constant on radial axis          [N/m^2]
k_lin_z = 1e-7                                    # Spring constant on longitudinal axis [N/m^2]

k_cub_x = (1e14) * k_lin_x               # Non-linearity strengths on the xy plane [N/m^4]
k_cub_z = (0.5*1e14) * k_lin_z                    # Non-linearity strengths on the z axis   [N/m^4]

k_lin   = np.array([k_lin_x, k_lin_x, k_lin_z])   # Linearity     strengths on each axis    [N/m^2]
k_cub   = np.array([k_cub_x, k_cub_x, k_cub_z])   # Non-linearity strengths on each axis    [N/m^4]
k_cross = 2*k_cub[0]


# Trap parameters for training data
N_examples = 100                                     # Number of examples to generate

k_cub_x_array = np.linspace(0, k_cub_x, N_examples)  # Array of non-linearity strengths on xy plane
k_cub_z_array = np.linspace(0, k_cub_z, N_examples)  # Array of non-linearity strengths on z axis

k_cub_array    = np.array([k_cub_x_array, k_cub_x_array, k_cub_z_array])   # Matrix of non-linearity strengths [N/m^4]
k_cross_array = 2*k_cub_x_array
###############################################################################


########################### Potential landscapes ##############################
# Calculate potentials in grid
potential_lin = V_lin(X, Y, 0, k_lin)
potential_cub = V_cub(X, Y, 0, k_lin, k_cub, k_cross)

min_potential = np.array([potential_lin.min(), potential_cub.min()]).min()
max_potential = np.array([potential_lin.max(), potential_cub.max()]).max()

order_lin = int(np.floor( np.log10( potential_lin.max() ) ))
order_cub = int(np.floor( np.log10( potential_cub.max() ) ))

# Create plotting figure for potential energy landscape and trace
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.linewidth"] = 1.0
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot potential energy landscape
plt_lin = axs[0].pcolor(X*1e6, Y*1e6, potential_lin/(10**order_lin), cmap='viridis')#, vmin=min_potential, vmax=max_potential)
axs[0].set_ylabel('z [$\mu$m]')
axs[0].set_xlabel('y [$\mu$m]')
axs[0].set_title("V_lin [$10^{" + str(order_lin) + "}$ N]")
fig.colorbar(plt_lin, ax=axs[0])

plt_cub = axs[1].pcolor(X*1e6, Y*1e6, potential_cub/(10**order_cub), cmap='viridis')#, vmin=min_potential, vmax=max_potential)
axs[1].set_ylabel('z [$\mu$m]')
axs[1].set_xlabel('y [$\mu$m]')
axs[1].set_title("V_cub [$10^{" + str(order_lin) + "}$ N]")
fig.colorbar(plt_cub, ax=axs[1])

fig.tight_layout() 
# fig.canvas.draw()
# fig.canvas.flush_events()
###############################################################################



###################### Calculate examples of trajectory #######################
# Perform simulation for linear potential
t_lin, positions_lin = linear_simulation(max_time, dt, x0, k_0, k_lin, gamma)

# Perform simulation for non-linear potential
t_non_lin, positions_non_lin = non_linear_simulation(max_time, dt, x0, k_0, k_lin, k_cub, k_cross, gamma)


# Create plotting figure for potential energy landscape and trace
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.linewidth"] = 1.0
fig, axs = plt.subplots(2, 2, figsize=(12, 6))

# Plot sample of non-linear trace
axs[0, 0].plot(t_lin, positions_lin[0,:])
axs[0, 0].grid()
axs[0, 0].set_ylabel(r"$\rho$ [m]")
axs[0, 0].set_xlabel('t [s]')
axs[0, 0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

axs[1, 0].plot(t_lin, positions_lin[1,:])
axs[1, 0].grid()
axs[1, 0].set_ylabel('z [m]')
axs[1, 0].set_xlabel('t [s]')
axs[1, 0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))


axs[0, 1].plot(t_non_lin, positions_non_lin[0,:])
axs[0, 1].grid()
axs[0, 1].set_ylabel(r"$\rho$ [m]")
axs[0, 1].set_xlabel('t [s]')
axs[0, 1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))


axs[1, 1].plot(t_non_lin, positions_non_lin[1,:])
axs[1, 1].grid()
axs[1, 1].set_ylabel('z [m]')
axs[1, 1].set_xlabel('t [s]')
axs[1, 1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

fig.tight_layout() 
# fig.canvas.draw()
# fig.canvas.flush_events()
###############################################################################



########################### Calculation of Kurtosis ###########################
delta_p = np.zeros( (N_dimensions, N_examples) )
for i in range(N_examples):                                                    # For each non-linear parameter
    k_cub   = k_cub_array[:, i]
    k_cross = k_cross_array[i]
    t_non_lin, positions_non_lin = non_linear_simulation(max_time, dt, x0, k_0, k_lin, k_cub, k_cross, gamma) # Simulate non-linear trace
    for j in range(N_dimensions):                                              # For each axis
        kurt = kurtosis(positions_non_lin[j,:], fisher=True)                   # Calculate the kurtosis
        delta_p[j, i] = kurt                                                   # Store it appropriately
    # del t_non_lin
    # del positions_non_lin
    print(i)



# Create plotting figure for potential energy landscape and trace
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.linewidth"] = 1.0
fig, axs = plt.subplots(len(x0), 1, figsize=(8, 6))

axis_name = ["x axis", "y axis", "z axis"]

# Plot of Kurtosis as a function of non-linear coupling strength
for j in range(N_dimensions):
    axs[j].plot(k_cub_array[j, :], delta_p[j, :], '-', alpha=.8, linewidth=1.0)
    axs[j].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    #axs[2].set_ylim([-2, 2])
    axs[j].grid()
    axs[j].set_ylabel('Kurtosis')
    axs[j].set_xlabel(r'k3 $[N/m^4]$')
    axs[j].set_title(axis_name[j])

fig.tight_layout() 
# fig.canvas.draw()
# fig.canvas.flush_events()
###############################################################################







# axs[0].plot(x, potential_lin,'-', alpha=.8, linewidth=1.0, label='nonlinear')
# axs[0].plot( x, VH(x,k),'--', alpha=.8, linewidth=1.0, label='linear')
# axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
# axs[0].grid()
# axs[0].legend()
# axs[0].set_ylabel('V(x)')
# axs[0].set_xlabel('x')