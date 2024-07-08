# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:25:50 2024

@author: jaimunga
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.special as special
import torchquad
import pdb

class fda():
    
    def __init__(self, nbasis=5, data=None, delta=None):
        
        self.Lp = special.legendre_polynomial_p
        
        self.nbasis = nbasis
        
        self.data = data
        self.delta = delta
                        
        self.W = self.generate_W()
        self.x, self.phi = self.generate_phi()
        
        self.x_fine, self.phi_fine = self.generate_phi(n=1000)
        
        self.a = self.estimate_coeffs()
        
    def generate_phi(self, n=None):
        
        if n==None:
            phi = torch.zeros(self.data.shape[1], self.nbasis)
            x = torch.linspace(-1,1,self.data.shape[1])
        else:
            phi = torch.zeros(n, self.nbasis)
            x = torch.linspace(-1,1,n)
        
        for i in range(self.nbasis):
            phi[:,i] = self.Lp(x, i)
        
        return x, phi
        
    def generate_W(self):
        
        W = torch.zeros(self.nbasis, self.nbasis)
        
        x = torch.linspace(-1,1,1001)
        dx = x[1]-x[0]
        
        gl=torchquad.GaussLegendre()
        
        for i in range(self.nbasis):

            phi_i = self.Lp(x, i)
            
            for j in range(i, self.nbasis):  
                f = lambda x : self.Lp(x, i)*self.Lp(x,j)
                W[i,j] = gl.integrate(f, dim=1, N=1001, integration_domain=[[-1,1]]) 
                W[j,i] = W[i,j]
                
        W[torch.abs(W)<1e-6] = 0
                
        return W
    
    def matrix_to_one_half(self, A):
        
        evals, evecs = torch.linalg.eig(A)  # get eigendecomposition
        
        # get real parts
        evecs = torch.real(evecs)
        evals = torch.real(evals)
        
        # raise to power elementwise
        evpow = evals**0.5
        
        mpow = torch.matmul (evecs, torch.matmul (torch.diag (evpow), torch.inverse (evecs)))
        
        return mpow
    
    def estimate_coeffs(self):
        
        a = torch.zeros(self.data.shape[0], self.nbasis)
        
        U, Omega, Vh = torch.linalg.svd(self.phi, full_matrices=False)
        
        Omega_inv = 1.0/Omega
        
        for i in range(self.data.shape[0]):
            
            B = torch.matmul(U.T,self.data[i].reshape(-1,1))
            C = torch.matmul(torch.diag(Omega_inv), B)
            a[i,:] = torch.matmul(Vh.T, C).reshape(-1)
             
        return a
    
    def fit(self, a):
        # return torch.matmul(self.phi_fine, a.reshape(-1,1))
        return torch.matmul(self.phi_fine, a.unsqueeze(-1))
        
    def plot_fits(self, idx=None):
        
        if idx is None:
            idx = torch.randint(self.data.shape[0],(5,))
        for i in idx:
            plt.scatter(self.x, self.data[i], s=10, color='red')
            plt.plot(self.x_fine, self.fit(self.a[i]))
            
        plt.xticks([])
        plt.yticks([])
        plt.savefig('IV_projection.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        fig, ax = plt.subplots(1,self.nbasis, figsize=(10,3))
        for i in range(self.nbasis):
            ax[i].plot(self.a[:,i], linewidth=0.5)
        plt.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots(self.nbasis-1,self.nbasis, figsize=(5,5), sharex=True,sharey=True)
        
        for i in range(self.nbasis-1):
            for j in range(self.nbasis):
                if j > i:
                    ax[i,j].scatter(self.a[:,i],self.a[:,j], s=5, alpha=0.25)
                else:
                    ax[i,j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def plot_basis(self):
        
        plt.plot(self.x_fine.numpy(), self.phi_fine.numpy())
        plt.show()
                