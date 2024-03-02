# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:31:39 2023

@author: sebja
"""

import torch
import torch.nn as nn
import torch.optim as optim

import pdb
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import copy

import dill
from datetime import datetime

class net(nn.Module):
    
    def __init__(self, 
                 nIn, 
                 nOut, 
                 n_nodes = 36, 
                 n_layers=5,
                 device='cpu'):
        super(net, self).__init__()

        self.device = device
        
        self.in_to_hidden = nn.Linear(nIn, n_nodes).to(self.device)
        self.hidden_to_hidden = nn.ModuleList([nn.Linear(n_nodes, n_nodes).to(self.device) for i in range(n_layers-1)])
        self.hidden_to_out = nn.Linear(n_nodes, nOut).to(self.device)
        
        self.g = nn.SiLU()
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        
        h = self.g(self.in_to_hidden(x))
        for linear in self.hidden_to_hidden:
            h = self.g(linear(h))
            
        output = self.softplus(self.hidden_to_out(h))
        
        return output


class sde_barycentre():
    
    def __init__(self, X0, mu, sigma, rho, pi, d=2, T=1, Ndt = 501):
        
        
        self.X0 = X0
        self.mu = mu
        self.K = len(self.mu)
        self.d = d
        self.sigma = sigma
        self.rho = rho
        self.pi= pi
        
        self.mu_bar = lambda t,x : torch.sum(torch.cat([pi*mu(t,x).unsqueeze(2) 
                                                        for pi, mu in zip(self.pi, self.mu)], 
                                                       axis=2 ), axis=2)
        
        self.T = T
        self.Ndt = Ndt
        self.t = np.linspace(0,self.T, self.Ndt)
        self.dt = np.diff(self.t)
        
        self.omega = net(self.d+1, 1)
        self.optimizer = optim.AdamW(self.omega.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                              step_size=10,
                                              gamma=0.99)
        
        self.loss = []
        
        self.Z = torch.distributions.MultivariateNormal(torch.zeros(self.d), self.rho)
        self.sqrt_dt = np.sqrt(self.dt)
        
        
    def step(self, t, x, mu, sigma, dW, dt):
        
        s = sigma(t,x)
        xp = x + mu(t,x) * dt  + s * dW 
        
        # Milstein correction
        m = torch.zeros(x.shape[0],self.d)
        
        xc = x.detach().requires_grad_()
        for k in range(self.d):
            
            grad_s = torch.autograd.grad(torch.sum(sigma(t,xc)[:,k]), xc)[0]
            
            m[:,k] = 0.5* s[:,k] * grad_s[:,k] * (dW[:,k]**2-dt)
            
        xp += m
        
        return xp
        
        
    def simulate(self, batch_size = 256):
        """
        simulate paths under all measures and the mean measure

        Parameters
        ----------
        batch_size : TYPE, optional
            DESCRIPTION. The default is 256.

        Returns
        -------
        X : TYPE: tensor
            DESCRIPTION. tensor of dimension batch_size, Ndt, d, K containing all paths

        """
        
        X = torch.zeros(batch_size, self.Ndt, self.d, self.K+1)
        X[:,0,:,:] = self.X0.view(1,self.d,1).repeat(batch_size,1,1)
        
        for i, t in enumerate(self.t[:-1]):
            
            dW = self.sqrt_dt[i] * self.Z.sample((batch_size,))
            
            for k in range(self.K):
                
                X[:,i+1,:,k] = self.step(t, X[:,i,:,k], self.mu[k], self.sigma, dW, self.dt[i])
            
            X[:,i+1,:,-1] = self.step(t, X[:,i,:,-1], self.mu_bar, self.sigma, dW, self.dt[i])
        
        return X


    def simulate_pbar(self, batch_size = 256):
        """
        simulate paths under the mean measure

        Parameters
        ----------
        batch_size : TYPE, optional
            DESCRIPTION. The default is 256.

        Returns
        -------
        X : TYPE: tensor
            DESCRIPTION. tensor of dimension batch_size, Ndt, d, K containing all paths

        """
        
        X = torch.zeros(batch_size, self.Ndt, self.d)
        X[:,0,:] = self.X0.view(1,self.d).repeat(batch_size,1)
        
        var_sigma = torch.zeros(batch_size, self.Ndt)
        
        for i, t in enumerate(self.t[:-1]):
            
            dW = self.sqrt_dt[i] * self.Z.sample((batch_size,))
        
            sigma = self.sigma(t,X[:,i,:])
            mu_bar = self.mu_bar(t,X[:,i,:])
            
            Sigma = sigma.unsqueeze(axis=1) * self.rho.unsqueeze(axis=0) * sigma.unsqueeze(axis=2)
            inv_Sigma = torch.inverse(Sigma)
            
            for k in range(self.K):
                dmu = (self.mu[k](t,X[:,i,:]) - mu_bar)
                
                var_sigma[:,i] += self.pi[k]*torch.einsum("ij,ijk,ik->i", dmu, inv_Sigma, dmu)
            
            X[:,i+1,:] = self.step(t, X[:,i,:], self.mu_bar, self.sigma, dW, self.dt[i])
        
        return X, var_sigma
    
    def train(self, batch_size = 1024, n_iter=10_000, n_print=100):
        
        t = torch.tensor(self.t).float().view(1,-1,1).repeat(batch_size,1,1)
        
        for i in tqdm(range(n_iter)):
            
            X, var_sigma = self.simulate_pbar(batch_size)
            
            int_var_sigma = torch.cumsum(var_sigma.flip((1,))[:,1:]*torch.tensor(self.dt).float().unsqueeze(axis=0), axis=1).flip((1,))
            
            g = self.omega(torch.cat((t,X), axis=2))
            
            
            self.optimizer.zero_grad()
            
            loss = torch.mean( (g[:,:-1,0] - torch.exp(-0.5*int_var_sigma) )**2 )
            
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            
            self.loss.append(loss.item())
            
            if np.mod(i+1, n_print) ==0:
                self.plot_loss()
                self.plot_sample_qpaths(256)
            
            
        return 0
    
    def plot_sample_paths(self, batch_size = 4_096):
        """
        simulate paths and plot them

        Parameters
        ----------
        batch_size : TYPE, optional
            DESCRIPTION. The default is 4_096.

        Returns
        -------
        None.

        """
        
        print('start sim')
        X = self.simulate(batch_size).numpy()
        
        print('done sim')
        
        
        fig, axs = plt.subplots(self.d, self.K+1, figsize=(10, 5), sharex=True)
        
        for i in range(self.d):
            
            for k in range(self.K+1):
                
                qtl = np.quantile(X[:,:,i,k], [0.1, 0.5, 0.9], axis=0)
                axs[i,k].plot(self.t, X[:500,:,i,k].T, alpha=0.25, linewidth=1)
                axs[i,k].plot(self.t, qtl.T, color='k', linestyle='--',linewidth=1)
                axs[i,k].plot(self.t, X[0,:,i,k].T, color='b', linewidth=1)
               
        for k in range(self.K):
            axs[0,k].set_title('model $\mathbb{P}_' + str(k) + '$')
        
        axs[0,-1].set_title('model $\overline{\mathbb{P}}$')
        
        for i in range(self.d):
            axs[i,0].set_ylabel(r'$X_{' + str(i) +'}$')
        
        fig.add_subplot(111, frameon=False)      
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel(r'$t$')               
            
        plt.tight_layout()
        plt.show()
        
        
    def moving_average(self, x, n):
        
        y = np.zeros(len(x))
        y_err = np.zeros(len(x))
        y[0] = np.nan
        y_err[0] = np.nan
        
        for i in range(1,len(x)):
            
            if i < n:
                y[i] = np.mean(x[:i])
                y_err[i] = np.std(x[:i])
            else:
                y[i] = np.mean(x[i-n:i])
                y_err[i] = np.std(x[i-n:i])
                
        return y, y_err
        
    def plot_loss(self):

        mv, mv_err = self.moving_average(self.loss,100)
        
        plt.fill_between(np.arange(len(mv)), y1=mv-mv_err, y2=mv+mv_err, alpha=0.2)
        plt.plot(mv,  linewidth=1.5)
        plt.yscale('log')
        plt.show()
        
    def simulate_q(self, batch_size = 256):

        X = torch.zeros(batch_size, self.Ndt, self.d)
        X[:,0,:] = self.X0.view(1,self.d).repeat(batch_size,1)
        
        ones = torch.ones(batch_size, 1)
        
        
        def theta(t,x):
            sigma = self.sigma(t,x)
            mu_bar = self.mu_bar(t,x)
            
            Sigma = sigma.unsqueeze(axis=1) * self.rho.unsqueeze(axis=0) * sigma.unsqueeze(axis=2)
            
            grad_L = self.grad_L(t, x)
            
            return mu_bar - torch.einsum("ijk,ik->ij", Sigma, grad_L)
        
        for i, t in enumerate(self.t[:-1]):
            
            dW = self.sqrt_dt[i] * self.Z.sample((batch_size,))
        
            # sigma = self.sigma(t,X[:,i,:])
            # mu_bar = self.mu_bar(t,X[:,i,:])
            
            # Sigma = sigma.unsqueeze(axis=1) * self.rho.unsqueeze(axis=0) * sigma.unsqueeze(axis=2)
            
            # grad_L = self.grad_L(t*ones, X[:,i,:])
            
            # theta = mu_bar - torch.einsum("ijk,ik->ij", Sigma, grad_L)
            
            # X[:,i+1,:] = X[:,i,:] + self.dt[i] * theta + sigma * dW
            X[:,i+1,:] = self.step(t*ones, X[:,i,:], theta, self.sigma, dW, self.dt[i])
        
        return X        
        
    def plot_sample_qpaths(self, batch_size = 4_096):
        """
        simulate paths under the optimal measure and plot them

        Parameters
        ----------
        batch_size : TYPE, optional
            DESCRIPTION. The default is 4_096.

        Returns
        -------
        None.

        """
        
        print('start sim')
        X = self.simulate_q(batch_size).numpy()
        
        print('done sim')
        
        
        fig, axs = plt.subplots(self.d, 1,  figsize=(4,5), sharex=True)
        
        for i in range(self.d):
            
            qtl = np.quantile(X[:,:,i], [0.1, 0.5, 0.9], axis=0)
            axs[i].plot(self.t, X[:500,:,i].T, alpha=0.25, linewidth=1)
            axs[i].plot(self.t, qtl.T, color='k', linestyle='--',linewidth=1)
            axs[i].plot(self.t, X[0,:,i].T, color='b', linewidth=1)
               
        axs[0].set_title('model $\mathbb{Q}^*$')
        
        for i in range(self.d):
            axs[i].set_ylabel(r'$X_{' + str(i) +'}$')
        
        fig.add_subplot(111, frameon=False)      
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel(r'$t$')               
            
        plt.tight_layout()
        plt.show()
        
    def grad_L(self, t, X):
        
        X = X.detach().requires_grad_()
        L = - torch.sum( torch.log(  self.omega(torch.cat((t,X), axis=1)) ) )
        
        return torch.autograd.grad(L, X)[0]
        