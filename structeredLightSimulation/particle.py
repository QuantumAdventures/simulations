# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:07:04 2024

@author: tandeitnik
"""
import numpy as np

class particle():
    
    def __init__(self, radius, waveLength, n_particle = 1.42, n_medium = 1 ,rho = 1850):
        
        e0 = 8.85e-12
        
        self.radius = radius #particle radius [m]
        self.n_particle = n_particle #particle's refractive index
        self.n_medium = n_medium #medium's refractive index
        self.rho = rho #particle's density
        self.e_r = (n_particle/n_medium)**2
        self.volume = (4/3)*np.pi*radius**3
        self.massParticle = rho*(4/3)*np.pi*radius**3
        self.alpha_cm = 3*self.volume*e0*(self.e_r-1)/(self.e_r+2)
        self.alpha_rad = self.alpha_cm/(1-((self.e_r-1)/(self.e_r+2))*((2*np.pi*radius/waveLength)**2 + 2j/3*(2*np.pi*radius/waveLength)**3))
        self.sigma_ext = (2*np.pi/(e0*waveLength))*np.imag(self.alpha_rad)