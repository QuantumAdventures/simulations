import numpy as np
from numba import njit
import sympy as sym

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
    
class particle():
    
    def __init__(self, radius, wavelength, n_particle = 1.42, n_medium = 1 ,rho = 1850):
        
        e0 = 8.85e-12
        
        self.radius = radius #particle radius [m]
        self.n_particle = n_particle #particle's refractive index
        self.n_medium = n_medium #medium's refractive index
        self.wavelength = wavelength
        self.rho = rho #particle's density
        self.e_r = (n_particle/n_medium)**2
        self.volume = (4/3)*np.pi*radius**3
        self.massParticle = rho*(4/3)*np.pi*radius**3
        self.alpha_cm = 3*self.volume*e0*(self.e_r-1)/(self.e_r+2)
        self.alpha_rad = self.alpha_cm/(1-((self.e_r-1)/(self.e_r+2))*((2*np.pi*radius/wavelength)**2 + 2j/3*(2*np.pi*radius/wavelength)**3))
        self.sigma_ext = (2*np.pi/(e0*wavelength))*np.imag(self.alpha_rad)
        
        
class simulation():
    
    def __init__(self,particle,structeredLight):
        
        self.particle = particle
        self.structeredLight = structeredLight
    
    @staticmethod
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
    
    
    @njit(fastmath=True)
    def rungeKutta(self, dt, steps, pos0, vel0, T0, pressure,NA, gamma = None):
        
        kB = 1.380649e-23
        
        fieldGradient = self.structeredLight.gradient
        k_x = sym.lambdify(list(fieldGradient[0].free_symbols),np.real(self.particle.alpha_rad)*fieldGradient[0]/4)
        k_y = sym.lambdify(list(fieldGradient[1].free_symbols),np.real(self.particle.alpha_rad)*fieldGradient[1]/4)
        k_z = sym.lambdify(list(fieldGradient[2].free_symbols),np.real(self.particle.alpha_rad)*fieldGradient[2]/4)

        if gamma != None:
            gamma = self.gammaEnv(pressure,T0)
        
        sigma = np.sqrt(2*kB *T0*gamma*self.particle.massParticle)*np.sqrt(dt)/self.particle.massParticle
        
        positions  = np.zeros((3,steps))
        velocities =  np.zeros((3,steps))
        
        positions[:,0] = pos0
        velocities[:,0] = vel0
        
        for i in range(1,steps-1):
            
            newPos, newVel = self.rungeKuttaStep(positions[:,i-1],velocities[:,i-1],dt,k_x,k_y,k_z,NA,gamma,sigma)
            positions[0,i] = newPos[0]
            positions[1,i] = newPos[1]
            positions[2,i] = newPos[2]
            velocities[0,i] = newVel[0]
            velocities[1,i] = newVel[1]
            velocities[2,i] = newVel[2]
            
        return positions, velocities
        
    @njit(fastmath=True)
    def ode(self,x,y,z,v,k,NA,gamma):
        
        ode_1 = v
        ode_2 = k(x = x,y = y, z = z,NA = NA, wl = self.particle.wavelength) - gamma*v

        return np.array((ode_1,ode_2))
    
    @staticmethod
    @njit(fastmath=True)
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