# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:34:27 2024

@author: tandeitnik
"""

import sympy as sym
import numpy as np

x, y, z, psi = sym.symbols('x y z psi', real = True)
E = sym.symbols('E')
k, phi, w, w0, wl, r, R, zR, C = sym.symbols('k phi w w0 wl r R zR C', real = True, positive = True)

r = sym.sqrt(x**2+y**2)
zR = sym.pi*w0**2/wl
R = z*(1+(zR/z)**2)
w = w0*sym.sqrt(1+(z/zR)**2)
k = 2*sym.pi/wl
phi = sym.atan2(y,x)

def symLaguerre(l,p):
    
    C = sym.sqrt(2*np.math.factorial(p)/(sym.pi*np.math.factorial(p+np.abs(l))))
    psi = (np.abs(l)+2*p+1)*sym.atan2(z,zR)
    E = (C/w)*(sym.sqrt(2)*r/w)**np.abs(l)*sym.assoc_laguerre(p, np.abs(l), 2*r**2/w**2)*sym.exp(-r**2/w**2 - 1j*(k*r**2/(2*R) + l*phi + psi + k*z))
    
    return E

def symHermite(n,m):
    
    C = sym.sqrt(2/(sym.pi*np.math.factorial(n)*np.math.factorial(m)*2**(n+m))) #https://jcmwave.com/docs/ParameterReference/0a2dd5b44fc46b68bd3c44031b2aecd4.html?version=4.4.0
    psi = (n+1)*sym.atan2(z,zR)
    E = (C/w)*sym.hermite(n,sym.sqrt(2)*x/w)*sym.hermite(m,sym.sqrt(2)*y/w)*sym.exp(-r**2/w**2 - 1j*(k*r**2/(2*R) + k*z - psi))
    
    return E

def superposition(mode,indices,coef):
    
    if mode == 'laguerre':
        
        for i, indice in enumerate(indices):
            
            if i == 0:
                
                E = coef[i]*symLaguerre(indice[0],indice[1])
                
            else:
                
                E = E + coef[i]*symLaguerre(indice[0],indice[1])
                
    elif mode == 'hermite':
        
        for i, indice in enumerate(indices):
            
            if i == 0:
                
                E = coef[i]*symHermite(indice[0],indice[1])
                
            else:
                
                E = E + coef[i]*symHermite(indice[0],indice[1])
                
    return E

def intensity(E):
    
    return sym.conjugate(E)*E

mode = 'laguerre'
indices = [[2,0],[-2,0]]
coef = [1/np.sqrt(2),1/np.sqrt(2)]

E = superposition(mode,indices,coef)

func = sym.lambdify([x,y,z,w0,wl],sym.simplify(sym.conjugate(E)*E))

z = 0
w0 = 2000e-9
wl = 1550e-9
x = np.linspace(-5*w0,5*w0,1000)
y = x
X,Y = np.meshgrid(x,y)

teste = func(X,Y,z,w0,wl)

