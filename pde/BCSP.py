# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:06:44 2022

@author: sebja
"""

import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve
from scipy.optimize import root
import pdb

class BCSP():
    
    def __init__(self, mu, sigma, pi, f=None, g=None, Z=None, Ndt = 1_001):
        
        self.mu = mu
        self.sigma = sigma
        self.f = f
        self.g = g
        self.pi = pi
        
        self.Z = Z
        
        self.t = np.linspace(0,1,Ndt)
        self.dt = self.t[1]-self.t[0]
        
        x = np.linspace(-10,10, 101)
        max_sigma = np.max(self.sigma(x))
        
        self.dx = max_sigma*np.sqrt(3*self.dt)
        
        self.x = np.arange(-50,51)*self.dx
        
    def mu_bar(self, t, x):
        
        result = 0
        for pi, mu in zip(self.pi, self.mu):
            result += pi*mu(t,x)
        
        return result        
        
    def var_sigma(self, t, x):
        
        result = 0
        mu_bar = self.mu_bar(t, x)
        for pi, mu in zip(self.pi,self.mu):
            result += pi*((mu(t,x)-mu_bar)/self.sigma(x))**2
        
        return result
            
        
    def SolveOmega(self, eta):
        
        w = np.zeros((len(self.t),len(self.x)))
        if self.f is not None:
            w[-1,:] = np.exp(-eta[1]*self.f(self.x))
        else:
            w[-1,:] = 1
        
        
        for i in range(len(self.t)-1, 0, -1):
            
            dw = (w[i,2:] - w[i,:-2])/(2*self.dx)
            ddw = (w[i,2:] - 2* w[i,1:-1] + w[i,:-2])/(self.dx**2)
            
            w[i-1, 1:-1] = w[i, 1:-1]\
                + self.dt*(self.mu_bar(self.t[i], self.x[1:-1])*dw\
                           + 0.5*self.sigma(self.x[1:-1])**2 * ddw)
            w[i-1, 1:-1] *= np.exp(-0.5*self.var_sigma(self.t[i], self.x[1:-1])*self.dt
                                   -eta[0]*self.g(self.t[i],self.x[1:-1])*self.dt)
                    
            w[i-1, 0] = 2*w[i-1, 1] - w[i-1, 2]
            w[i-1, -1] = 2*w[i-1, -2] - w[i-1, -3]
        
        
        mu_opt = np.zeros(w.shape)
        mu_opt[:] = np.nan
        dmu_opt  = np.zeros(w.shape)
        dmu_opt[:] = np.nan
        
        L = np.log(w)

        for i in range(w.shape[0]):

            mu_opt[i,1:-1] = self.mu_bar(self.t[i], self.x[1:-1]) \
                + self.sigma(self.x)**2*(L[i,2:]-L[i,:-2])/(2*self.dx)
            dmu_opt[i,1:-1] = self.sigma(self.x)**2*(L[i,2:]-L[i,:-2])/(2*self.dx)
            
            mu_opt[i,0] = self.mu_bar(self.t[i], self.x[0]) \
                + self.sigma(self.x)**2*(L[i,1]-L[i,0])/(self.dx)
            mu_opt[i,-1] = self.mu_bar(self.t[i], self.x[-1]) \
                + self.sigma(self.x)**2*(L[i,-1]-L[i,-2])/(self.dx)                
            
        self.w = w
        self.mu_opt = mu_opt
        self.dmu_opt = dmu_opt
        
        return w, mu_opt
    
    
    def QExpectation(self, eta):
        
        self.SolveOmega(eta)
        
        h = []
        h.append(np.zeros((len(self.t),len(self.x))))
        
        if self.f is not None:
            h[0][-1,:] = self.f(self.x)
            
            for i in range(len(self.t)-1, 0, -1):
                
                dh = (h[-1][i,2:] - h[-1][i,:-2])/(2*self.dx)
                ddh = (h[-1][i,2:] - 2* h[-1][i,1:-1] + h[-1][i,:-2])/(self.dx**2)
                
                h[-1][i-1, 1:-1] = h[-1][i, 1:-1]\
                    + self.dt*(self.mu_opt[i,1:-1]*dh\
                               + 0.5*self.sigma(self.x[1:-1])**2 * ddh)
                        
                h[-1][i-1, 0] = 2*h[-1][i-1, 1] - h[-1][i-1, 2]
                h[-1][i-1, -1] = 2*h[-1][i-1, -2] - h[-1][i-1, -3]
        
        h.append(np.zeros((len(self.t),len(self.x))))
        if self.g is not None:

            for i in range(len(self.t)-1, 0, -1):
                
                dh = (h[-1][i,2:] - h[-1][i,:-2])/(2*self.dx)
                ddh = (h[-1][i,2:] - 2* h[-1][i,1:-1] + h[-1][i,:-2])/(self.dx**2)
                
                h[-1][i-1, 1:-1] = h[-1][i, 1:-1]\
                    + self.dt*(self.mu_opt[i,1:-1]*dh\
                               + 0.5*self.sigma(self.x[1:-1])**2 * ddh
                               + self.g(self.t[i], self.x[1:-1]))
                        
                h[-1][i-1, 0] = 2*h[-1][i-1, 1] - h[-1][i-1, 2]
                h[-1][i-1, -1] = 2*h[-1][i-1, -2] - h[-1][i-1, -3]            
            
        return h
    
    def FindOptimalEta(self):
        
        if self.f is not None:
            
            transform = lambda a : 50*np.tanh(a)
                
            def error(a):
                
                eta = transform(a)
                
                f_err = interpolate.interp1d(self.x, self.QExpectation(eta)[0][0,:])
                g_err = interpolate.interp1d(self.x, self.QExpectation(eta)[1][0,:])
                print(eta, f_err(0), g_err(0))
                
                return np.array([f_err(0), g_err(0)])
                # return np.sqrt(f_err(0)**2+ g_err(0)**2)
            
            sol = fsolve(error, [0,0])
            eta = transform(sol)
        else:
            eta = 0
            
        self.SolveOmega(eta)
        
        return eta
    
    def Simulate(self):
        
        paths = []
        
        for mu in self.mu:
            paths.append(self.__Simulate__(mu))
            
        paths.append(self.__Simulate_opt__())
        
        return paths
    
    def Simulate_Qbar(self):
        
        X = np.zeros((len(self.t), self.Z.shape[1]))
        
        sqrt_dt = np.sqrt(self.dt)
        for i in range(len(self.t)-1):
            
            X[i+1,:] = X[i,:] \
                + self.mu(self.t[i],X[i,:])*self.dt \
                    + self.sigma(X[i,:])*sqrt_dt*self.Z[i,:]
    
        return X
    
    def __Simulate_opt__(self):
        
        X = np.zeros((len(self.t),self.Z.shape[1]))
        
        for i in range(len(self.t)-1):
            
            mu = interpolate.interp1d(self.x, 
                                      self.mu_opt[i,:], 
                                      fill_value='extrapolate')
            
            X[i+1,:] = X[i,:] + mu(X[i,:])*self.dt \
                + self.sigma(X[i,:])*np.sqrt(self.dt)*self.Z[i,:]
        
        return X
        
    def __Simulate__(self,mu):
        
        X = np.zeros((len(self.t),self.Z.shape[1]))
        
        for i in range(len(self.t)-1):
            X[i+1,:] = X[i,:] + mu(self.t[i], X[i,:])*self.dt \
                + self.sigma(X[i,:])*np.sqrt(self.dt)*self.Z[i,:]
            
        return X