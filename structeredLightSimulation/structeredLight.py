# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:28:31 2024

@author: tandeitnik
"""
import sympy as sym
import numpy as np


class structeredLight:
    
    def __init__(self, mode, indices, coefs):
        self.mode = mode
        self.indices = indices
        self.coefs = coefs
        
    def __str__(self):
        return f"{self.mode}({self.indices},self.coefs)"
   
    @staticmethod
    def symLaguerre(l,p,globalPhase = True):
        
        x, y, z, psi = sym.symbols('x y z psi', real = True)
        E = sym.symbols('E')
        k, phi, w, w0, wl, r, R, zR, NA, C = sym.symbols('k phi w w0 wl r R zR NA C', real = True, positive = True)
        
        r = sym.sqrt(x**2+y**2)
        k = 2*sym.pi/wl
        w0 = wl/(sym.pi*NA)
        zR = sym.pi*w0**2/wl
        phi = sym.atan2(y,x)
        w = w0*sym.sqrt(1+(z/zR)**2)
        R = z*(1+(zR/z)**2)
        C = sym.sqrt(2*np.math.factorial(p)/(sym.pi*np.math.factorial(p+np.abs(l))))
        psi = (np.abs(l)+2*p+1)*sym.atan2(z,zR)
        
        if globalPhase:
            E = (C/w)*(sym.sqrt(2)*r/w)**np.abs(l)*sym.assoc_laguerre(p, np.abs(l), 2*r**2/w**2)*sym.exp(-r**2/w**2 - 1j*(k*r**2/(2*R) + l*phi + psi + k*z))
        else:
            E = (C/w)*(sym.sqrt(2)*r/w)**np.abs(l)*sym.assoc_laguerre(p, np.abs(l), 2*r**2/w**2)*sym.exp(-r**2/w**2 - 1j*(l*phi + psi))
        return E
    
    @staticmethod
    def symHermite(n,m,globalPhase = True):
        
        x, y, z, psi = sym.symbols('x y z psi', real = True)
        E = sym.symbols('E')
        k, w, w0, wl, r, R, zR, NA, C = sym.symbols('k w w0 wl r R zR NA C', real = True, positive = True)
        
        r = sym.sqrt(x**2+y**2)
        k = 2*sym.pi/wl
        w0 = wl/(sym.pi*NA)
        zR = sym.pi*w0**2/wl
        w = w0*sym.sqrt(1+(z/zR)**2)
        R = z*(1+(zR/z)**2)
        C = sym.sqrt(2/(sym.pi*np.math.factorial(n)*np.math.factorial(m)*2**(n+m))) #https://jcmwave.com/docs/ParameterReference/0a2dd5b44fc46b68bd3c44031b2aecd4.html?version=4.4.0
        psi = (n+1)*sym.atan2(z,zR)
        
        if globalPhase:
            E = (C/w)*sym.hermite(n,sym.sqrt(2)*x/w)*sym.hermite(m,sym.sqrt(2)*y/w)*sym.exp(-r**2/w**2 - 1j*(k*r**2/(2*R) + k*z - psi))
        else:
            E = (C/w)*sym.hermite(n,sym.sqrt(2)*x/w)*sym.hermite(m,sym.sqrt(2)*y/w)*sym.exp(-r**2/w**2 - 1j*(psi))
        return E
    
    
    def electricField(self,globalPhase = True):
        
        E = sym.symbols('E')
        
        if self.mode == 'laguerre':
            
            for i, indice in enumerate(self.indices):
                
                if i == 0:
                    
                    E = self.coefs[i]*self.symLaguerre(indice[0],indice[1],globalPhase)
                    
                else:
                    
                    E = E + self.coefs[i]*self.symLaguerre(indice[0],indice[1],globalPhase)
                    
        elif self.mode == 'hermite':
            
            for i, indice in enumerate(self.indices):
                
                if i == 0:
                    
                    E = self.coefs[i]*self.symHermite(indice[0],indice[1],globalPhase)
                    
                else:
                    
                    E = E + self.coefs[i]*self.symHermite(indice[0],indice[1],globalPhase)
                    
        return E
    
    def intensity(self):
        
        E = self.electricField(globalPhase = False)
        
        return sym.conjugate(E)*E
    
    def gradient(self):
        """
        returns the gradient of the intensity
        """
        x, y, z = sym.symbols('x y z', real = True)
        
        return [sym.diff(self.intensity(),x),sym.diff(self.intensity(),y),sym.diff(self.intensity(),z)]