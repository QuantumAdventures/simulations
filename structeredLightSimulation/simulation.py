# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:24:54 2024

@author: tandeitnik
"""

import numpy as np
import sympy as sym

class simulation():
    
    def __init__(self,particle,structeredLight):
        
        self.particle = particle
        self.structeredLight = structeredLight
    
    def gammaEnv(self, pressure, T0):
        
        kB = 1.380649e-23
        eta_air = 18.27e-6 # Pa # (J.T.R.Watson (1995)).
        d_gas = 0.372e-9 #m #(Sone (2007)), ρSiO2
        pressurePascal = 100*pressure
        s = kB*T0/(2**0.5*np.pi*d_gas**2*pressurePascal)
        K_n = s/self.particle.radius
        c_K = 0.31*K_n/(0.785 + 1.152*K_n + K_n**2)
        gamma = 6*np.pi*eta_air*self.particle.radius/self.particle.massParticle * 0.619/(0.619 + K_n) * (1+c_K)
        
        return gamma
    
    # def eulerMaruyama():
    #     return

    def rungeKutta(self, dt, steps, pos0, vel0, T0, pressure,P,NA, gamma = None):
        
        kB = 1.380649e-23
        c  = 299_792_458
        e0 = 8.85e-12
        powerNorm = 2*P/(c*e0)
        
        fieldGradient = self.structeredLight.gradient()
        k_x = sym.lambdify(list(fieldGradient[0].free_symbols),powerNorm*np.real(self.particle.alpha_rad)*fieldGradient[0]/4)
        k_y = sym.lambdify(list(fieldGradient[1].free_symbols),powerNorm*np.real(self.particle.alpha_rad)*fieldGradient[1]/4)
        k_z = sym.lambdify(list(fieldGradient[2].free_symbols),powerNorm*np.real(self.particle.alpha_rad)*fieldGradient[2]/4)

        if gamma == None:
            
            if pressure == 0:
                gamma = 0
            else:
                gamma = self.gammaEnv(pressure,T0)
        
        sigma = np.sqrt(2*kB*T0*gamma*self.particle.massParticle)*np.sqrt(dt)/self.particle.massParticle
        
        positions  = np.zeros((3,steps))
        velocities =  np.zeros((3,steps))
        
        positions[:,0] = pos0
        velocities[:,0] = vel0
        
        for i in range(1,steps):
            
            newPos, newVel = self.rungeKuttaStep(positions[:,i-1],velocities[:,i-1],dt,k_x,k_y,k_z,NA,gamma,sigma)
            positions[0,i] = newPos[0]
            positions[1,i] = newPos[1]
            positions[2,i] = newPos[2]
            velocities[0,i] = newVel[0]
            velocities[1,i] = newVel[1]
            velocities[2,i] = newVel[2]
            
        return positions, velocities
    
    def ode(self,x,y,z,v,k,NA,gamma):
        
        ode_1 = v
        ode_2 = np.real(k(x = x,y = y, z = z,NA = NA, wl = self.particle.wavelength))/self.particle.massParticle - gamma*v

        return np.array((ode_1,ode_2))
    
    def rungeKuttaStep(self,pos,vel,dt,k_x,k_y,k_z,NA,gamma,sigma):
        
        x = pos[0]
        y = pos[1]
        z = pos[2]
        
        v_x = vel[0]
        v_y = vel[1]
        v_z = vel[2]
        
        k1_x = dt*self.ode(x,y,z,v_x,k_x,NA,gamma)
        k1_y = dt*self.ode(x,y,z,v_y,k_y,NA,gamma)
        k1_z = dt*self.ode(x,y,z,v_z,k_z,NA,gamma)
        
        k2_x = dt*self.ode(x+k1_x[0]/2,y+k1_y[0]/2,z+k1_z[0]/2,v_x+k1_x[1]/2,k_x,NA,gamma)
        k2_y = dt*self.ode(x+k1_x[0]/2,y+k1_y[0]/2,z+k1_z[0]/2,v_y+k1_y[1]/2,k_y,NA,gamma)
        k2_z = dt*self.ode(x+k1_x[0]/2,y+k1_y[0]/2,z+k1_z[0]/2,v_z+k1_z[1]/2,k_z,NA,gamma)
        
        k3_x = dt*self.ode(x+k2_x[0]/2,y+k2_y[0]/2,z+k2_z[0]/2,v_x+k2_x[1]/2,k_x,NA,gamma)
        k3_y = dt*self.ode(x+k2_x[0]/2,y+k2_y[0]/2,z+k2_z[0]/2,v_y+k2_y[1]/2,k_y,NA,gamma)
        k3_z = dt*self.ode(x+k2_x[0]/2,y+k2_y[0]/2,z+k2_z[0]/2,v_z+k2_z[1]/2,k_z,NA,gamma)
        
        k4_x = dt*self.ode(x+k3_x[0],y+k3_y[0]/2,z+k3_z[0],v_x+k3_x[1],k_x,NA,gamma)
        k4_y = dt*self.ode(x+k3_x[0],y+k3_y[0]/2,z+k3_z[0],v_y+k3_y[1],k_y,NA,gamma)
        k4_z = dt*self.ode(x+k3_x[0],y+k3_y[0]/2,z+k3_z[0],v_z+k3_z[1],k_z,NA,gamma)
        
        newPos = np.array((
            (x + (k1_x[0] + 2*k2_x[0] + 2*k3_x[0] + k4_x[0])/6),
            (y + (k1_y[0] + 2*k2_y[0] + 2*k3_y[0] + k4_y[0])/6),
            (z + (k1_z[0] + 2*k2_z[0] + 2*k3_z[0] + k4_z[0])/6)))
        
        newVel = np.array((
            (v_x + (k1_x[1] + 2*k2_x[1] + 2*k3_x[1] + k4_x[1])/6 + sigma*np.random.normal()),
            (v_y + (k1_y[1] + 2*k2_y[1] + 2*k3_y[1] + k4_y[1])/6 + sigma*np.random.normal()),
            (v_z + (k1_z[1] + 2*k2_z[1] + 2*k3_z[1] + k4_z[1])/6 + sigma*np.random.normal())))
        
        return newPos, newVel