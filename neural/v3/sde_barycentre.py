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

from scipy.optimize import fsolve
from tqdm import tqdm

class net(nn.Module):
    
    def __init__(self, 
                 nIn, 
                 nOut, 
                 n_nodes = 36, 
                 n_layers=5,
                 device='cpu',
                 output='none'):
        super(net, self).__init__()

        self.device = device
        
        self.in_to_hidden = nn.Linear(nIn, n_nodes).to(self.device)
        self.hidden_to_hidden = nn.ModuleList([nn.Linear(n_nodes, n_nodes).to(self.device) for i in range(n_layers-1)])
        self.hidden_to_out = nn.Linear(n_nodes, nOut).to(self.device)
        
        self.g = nn.SiLU()
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
        self.output=output
        
        
    def forward(self, x):
        
        h = self.g(self.in_to_hidden(x))
        for linear in self.hidden_to_hidden:
            h = self.g(linear(h))
            
        h = self.hidden_to_out(h)
        
        if self.output=='softplus':
            h = self.softplus(h)
        
        return h


class sde_barycentre():
    
    def __init__(self, X0, mu, sigma, rho, pi, f=[], g=[], T=1, Ndt = 501):
        
        self.f = f
        self.g = g
        self.X0 = X0
        self.mu = mu
        self.K = len(self.mu)
        self.d = X0.shape[0]
        self.sigma = sigma
        self.rho = rho
        self.pi= pi
        
        self.mu_bar = lambda t,x : torch.sum(torch.cat([pi*mu(t,x).unsqueeze(2) 
                                                        for pi, mu in zip(self.pi, self.mu)], 
                                                       axis=2 ), axis=2)
        
        self.T = T
        self.Ndt = Ndt
        self.t = np.linspace(0,self.T, self.Ndt)
        self.dt = self.t[1]-self.t[0]
        
        # features are t, x_1,..,x_d, eta_1,..,eta_n for constraints
        self.omega = net(nIn=self.d+1, 
                         nOut=1, output='softplus')
        self.omega_optimizer = optim.AdamW(self.omega.parameters(), lr=0.001)
        self.omega_scheduler = optim.lr_scheduler.StepLR(self.omega_optimizer,
                                                         step_size=10,
                                                         gamma=0.999)
        
        self.omega_loss = []
        
        self.Z = torch.distributions.MultivariateNormal(torch.zeros(self.d), self.rho)
        self.sqrt_dt = np.sqrt(self.dt)
        
    def step(self, t, x, mu, sigma, dW, dt):
        
        s = sigma(t,x)
        xp = x + mu(t,x) * dt  + s * dW 
        
        # # Milstein correction
        # m = torch.zeros(x.shape[0],self.d)
        
        # xc = x.detach().requires_grad_()
        # for k in range(self.d):
            
        #     grad_s = torch.autograd.grad(torch.sum(sigma(t,xc)[:,k]), xc)[0]
            
        #     m[:,k] = 0.5* s[:,k] * grad_s[:,k] * (dW[:,k]**2-dt)
            
        # xp += m
        
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
            
            dW = self.sqrt_dt * self.Z.sample((batch_size,))
            
            for k in range(self.K):
                
                X[:,i+1,:,k] = self.step(t, X[:,i,:,k], self.mu[k], self.sigma, dW, self.dt)
            
            X[:,i+1,:,-1] = self.step(t, X[:,i,:,-1], self.mu_bar, self.sigma, dW, self.dt)
        
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
            
            dW = self.sqrt_dt * self.Z.sample((batch_size,))
        
            sigma = self.sigma(t,X[:,i,:])
            mu_bar = self.mu_bar(t,X[:,i,:])
            
            Sigma = sigma.unsqueeze(axis=1) * self.rho.unsqueeze(axis=0) * sigma.unsqueeze(axis=2)
            inv_Sigma = torch.inverse(Sigma)
            
            for k in range(self.K):
                dmu = (self.mu[k](t,X[:,i,:]) - mu_bar)
                
                var_sigma[:,i] += self.pi[k]*torch.einsum("ij,ijk,ik->i", dmu, inv_Sigma, dmu)
            
            X[:,i+1,:] = self.step(t, X[:,i,:], self.mu_bar, self.sigma, dW, self.dt)
        
        return X, var_sigma
    
    def theta(self, t,x):
        
        pdb.set_trace()
        
        sigma = self.sigma(t,x)
        mu_bar = self.mu_bar(t,x)
        
        Sigma = sigma.unsqueeze(axis=1) * self.rho.unsqueeze(axis=0) * sigma.unsqueeze(axis=2)
        
        grad_L = self.grad_L(t, x)
        
        result = mu_bar - torch.einsum("...ijk,...ik->...ij", Sigma, grad_L)
                    
        return result  
    
    def simulate_q(self, batch_size = 256):

        X = torch.zeros(batch_size, self.Ndt, self.d)
        X[:,0,:] = self.X0.view(1,self.d).repeat(batch_size,1)
        
        ones = torch.ones(batch_size, 1)
        
        for i, t in enumerate(self.t[:-1]):
            
            dW = self.sqrt_dt * self.Z.sample((batch_size,))
        
            X[:,i+1,:] = self.step(t*ones, X[:,i,:], self.theta, self.sigma, dW, self.dt)
        
        return X  
    
    def int_tT(self, y):
        
        # result = torch.cumsum(y.flip((1,))[:,1:]*torch.tensor(self.dt).float().unsqueeze(axis=0), axis=1).flip((1,))
        
        y_flipped = torch.cat((torch.zeros(y.shape[0],1),y.flip((1,))[:,1:]),axis=1)
        result = torch.cumsum(y_flipped*self.dt, axis=1).flip((1,))
        
        return result
    
    def update_omega(self, eta, n_iter=10, batch_size=256):
        
        for i in tqdm(range(n_iter)):
            
            X, var_sigma = self.simulate_pbar(batch_size)
            
            # running constraints
            int_g = 0
            for k in range(len(self.g)):
                int_g += eta[k]*self.int_tT(self.g[k](self.t_train, X).squeeze())
            
            # terminal constraints
            F = 0
            for k in range(len(self.f)):
                F += eta[len(self.g)+k]*self.f[k](X).squeeze()
            
            int_var_sigma = self.int_tT(var_sigma)
            
            omega = self.omega(torch.cat((self.t_train, X), axis=2))
            
            loss = torch.mean( (omega[...,0] - torch.exp(F+int_g-0.5*int_var_sigma) )**2 )
            
            self.omega_optimizer.zero_grad()
            
            loss.backward()
            
            self.omega_optimizer.step()
            self.omega_scheduler.step()
            
            self.omega_loss.append(loss.item())        
        
    def constraint_loss(self, batch_size=256):
        
        
        X = self.simulate_q(batch_size=batch_size)
        
        # running constraints
        loss = np.zeros(len(self.f)+len(self.g))
        for k in range(len(self.g)):
            loss[k] = torch.mean((torch.sum(self.dt*self.g[k](self.t_train, X)[:,:-1],axis=1)))

        # terminal constraint
        for k in range(len(self.f)):
            loss[k+len(self.g)] = torch.mean((self.f[k](X[:,-1,:])))
            
        return loss
    #
    # add in Lagrange multiplier as feature to NN and added in terminal penalties
    # can compute using actor-critic methods.
    #
    def train(self, batch_size = 1024, n_iter=1_000, n_iter_omega=10, n_print=100):
        
        self.t_train = torch.tensor(self.t).float().view(1,-1,1).repeat(batch_size,1,1)        
        
        self.eta_hist = []
        
        self.count = 0
        
        self.plot_sample_qpaths(256)
        
        def error(a):
            
            self.update_omega(eta=a, n_iter=n_iter_omega, batch_size=batch_size)
            
            loss= self.constraint_loss(batch_size=batch_size)
            
            print(self.count, a, loss)
            
            self.count += 1 
            self.eta_hist.append(a)

            self.plot_loss(self.omega_loss, r'$\omega$')
            plt.plot(self.eta_hist)
            plt.show()
            self.plot_sample_qpaths(256)
            self.plot_mu()

            
            return loss
        
        self.eta_opt = fsolve(error, x0=np.zeros(len(self.f)+len(self.g)))
        
        return self.eta_opt
    
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
        
        if len(axs.shape)==1:
            axs = np.expand_dims(axs, axis=0)
        
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
        
    def plot_loss(self, x, title=""):

        mv, mv_err = self.moving_average(x,100)
        
        plt.fill_between(np.arange(len(mv)), y1=mv-mv_err, y2=mv+mv_err, alpha=0.2)
        plt.plot(mv,  linewidth=1.5)
        plt.yscale('log')
        plt.title(title)
        plt.show()
        
    def plot_sample_qpaths(self, eta, batch_size = 4_096):
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
        torch.manual_seed(12317874321)
        
        print('start sim')
        X = self.simulate_q(batch_size=batch_size).detach().numpy()
        
        print('done sim')
        
        
        fig, axs = plt.subplots(self.d, 1,  figsize=(6,4), sharex=True)
        
        if self.d == 1:
            axs = np.array([axs])
        
        for i in range(self.d):
            
            qtl = np.quantile(X[:,:,i], np.arange(1,10)/10, axis=0)
            
            plt.fill_between(self.t, qtl[0], qtl[-1], color='y', alpha=0.5)
            axs[i].plot(self.t, X[:500,:,i].T, alpha=0.1, linewidth=1)
            axs[i].plot(self.t, qtl.T, color='k', linestyle='--',linewidth=1)
            axs[i].plot(self.t, X[0,:,i].T, color='b', linewidth=1)
            
               
        axs[0].set_title('model $\mathbb{Q}^*$')
        
        for i in range(self.d):
            axs[i].set_ylabel(r'$X_{' + str(i) +'}$')
            axs[i].set_ylim(-1,2)
        
        fig.add_subplot(111, frameon=False)      
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel(r'$t$')               
            
        plt.tight_layout()
        plt.show()
        
    def grad_L(self, t, X):
        
        X = X.detach().requires_grad_()
        L = - torch.sum( torch.log(  self.omega(torch.cat((t,X), axis=-1)) ) )
        
        return torch.autograd.grad(L, X)[0]
    
    
    # def grad_L(self, t, X):
        
    #     eps=0.0001
    #     omega = self.omega(torch.cat((t,X), axis=1))
    #     omega_p = self.omega(torch.cat((t,X-eps), axis=1))
    #     omega_m = self.omega(torch.cat((t,X+eps), axis=1))

    #     grad_L = (omega_p-omega_m)/(2*eps)/omega
        
    #     return grad_L
    
    def plot_mu(self):
        
        tm, xm = torch.meshgrid(torch.linspace(0,self.T,51),
                                torch.linspace(-2,2,51))
        
        tm = tm.unsqueeze(axis=2)
        xm = xm.unsqueeze(axis=2)
        
        theta = self.theta(tm, xm)       
        
        plt.contourf(tm.numpy(), xm.numpy(), theta.detach().numpy())
        plt.show()