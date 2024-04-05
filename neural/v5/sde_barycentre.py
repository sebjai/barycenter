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
from scipy.optimize import minimize
from tqdm import tqdm
from numpy.random import choice

class net(nn.Module):
    
    def __init__(self, 
                 nIn, 
                 nOut, 
                 n_nodes=128, 
                 n_layers=5,
                 device='cpu',
                 output='none'):
        super(net, self).__init__()

        self.device = device
        
        self.in_to_hidden = nn.Linear(nIn, n_nodes).to(self.device)
        self.hidden_to_hidden = nn.ModuleList([nn.Linear(n_nodes+nIn, n_nodes).to(self.device) for i in range(n_layers-1)])
        self.hidden_to_out = nn.Linear(n_nodes+nIn, nOut).to(self.device)
        
        self.g = nn.SiLU()
        # self.g = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
        self.output=output
        
        
    def forward(self, x):
        
        h = self.g(self.in_to_hidden(x))
        for linear in self.hidden_to_hidden:
            h = torch.cat((h,x),axis=-1)
            h = self.g(linear(h))
            
        # concat orginal x to last hidden layer
        h = torch.cat((h,x),axis=-1)
        h = self.hidden_to_out(h)
        
        if self.output=='softplus':
            h = self.softplus(h)
        
        return h


class sde_barycentre():
    
    def __init__(self, X0, mu, sigma, rho, pi, f=[], g=[], T=1, Ndt = 501):
        
        if torch.cuda.is_available(): 
            dev = "cuda:0" 
        else: 
            dev = "cpu" 
        self.dev = torch.device(dev)
        
        self.f = f
        self.g = g
        self.X0 = X0
        self.mu = mu
        self.K = len(self.mu)
        self.d = X0.shape[0]
        self.sigma = sigma
        self.rho = torch.tensor(rho).float().to(self.dev)
        self.pi= pi
        
        self.T = T
        self.Ndt = Ndt
        self.t = torch.linspace(0,self.T, self.Ndt).to(self.dev)
        self.dt = self.t[1]-self.t[0]
        
        # features are t, x_1,..,x_d, eta_1,..,eta_n for constraints
        self.omega = {'net' : net(nIn=self.d+1, nOut=1, output='softplus', device=self.dev).to(self.dev) }
        self.omega['optimizer'] = optim.AdamW(self.omega['net'].parameters(), lr=0.001)
        self.omega['scheduler'] = optim.lr_scheduler.StepLR(self.omega['optimizer'],
                                                         step_size=1000,
                                                         gamma=0.999)
        
        self.omega['loss'] = []
        
        self.target = self.omega['net']
        self.tau = 0.001
        
        
        # self.eta = {'net' : net(nIn=1, nOut=len(self.f)+len(self.g))}
        # self.eta['optimizer']  = optim.AdamW(self.eta['net'].parameters(), lr=0.001)
        # self.eta['scheduler'] = optim.lr_scheduler.StepLR(self.eta['optimizer'],
        #                                                  step_size=10,
        #                                                  gamma=0.999)   
        self.eta = {'loss' : [],  'values': []}
        
        self.Z = torch.distributions.MultivariateNormal(torch.zeros(self.d).to(self.dev), self.rho)
        self.sqrt_dt = torch.sqrt(self.dt)
        
    def mu_bar(self,t,x):
        
        result = 0
        for k in range(len(self.pi)):
            result +=self.pi[k]*self.mu[k](t,x)
            
        return result
        
    def step(self, t, x, mu, sigma, dW, dt):
        
        s = sigma(t,x)
        xp = x + mu(t,x) * dt  + s * dW 
        
        # # Milstein correction
        # m = torch.zeros(x.shape[0],self.d).to(self.dev)
        
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
        
        X = torch.zeros(batch_size, self.Ndt, self.d, self.K+1).to(self.dev)
        X[:,0,:,:] = self.X0.view(1,self.d,1).repeat(batch_size,1,1).to(self.dev)
        
        for i, t in enumerate(self.t[:-1]):
            
            dW = self.sqrt_dt * self.Z.sample((batch_size,)).to(self.dev)
            
            for k in range(self.K):
                
                X[:,i+1,:,k] = self.step(t, X[:,i,:,k], self.mu[k], self.sigma, dW, self.dt)
            
            X[:,i+1,:,-1] = self.step(t, X[:,i,:,-1], self.mu_bar, self.sigma, dW, self.dt)
        
        return X


    def simulate_Qbar(self, batch_size = 256):
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
        
        X = torch.zeros(batch_size, self.Ndt, self.d).to(self.dev)
        X[:,0,:] = self.X0.view(1,self.d).repeat(batch_size,1).to(self.dev)
        
        var_sigma = torch.zeros(batch_size, self.Ndt).to(self.dev)
        
        for i, t in enumerate(self.t[:-1]):
            
            dW = self.sqrt_dt * self.Z.sample((batch_size,)).to(self.dev)
        
            sigma = self.sigma(t,X[:,i,:])
            mu_bar = self.mu_bar(t,X[:,i,:])
            
            Sigma = sigma.unsqueeze(axis=1) * self.rho.unsqueeze(axis=0) * sigma.unsqueeze(axis=2)
            inv_Sigma = torch.inverse(Sigma)
            
            for k in range(self.K):
                dmu = (self.mu[k](t,X[:,i,:]) - mu_bar)
                
                var_sigma[:,i] += 0.5*self.pi[k]*torch.einsum("ij,ijk,ik->i", dmu, inv_Sigma, dmu)
            
            X[:,i+1,:] = self.step(t, X[:,i,:], self.mu_bar, self.sigma, dW, self.dt)
            
        return X, var_sigma
    
    def theta(self, t,x):
        
        sigma = self.sigma(t,x)
        mu_bar = self.mu_bar(t,x)
        
        Sigma = sigma.unsqueeze(axis=-1) * self.rho.unsqueeze(axis=0) * sigma.unsqueeze(axis=-2)
        
        grad_L = self.grad_L(t, x)
        
        result = mu_bar - torch.einsum("...ijk,...ik->...ij", Sigma, grad_L)
                    
        return result  
    
    def simulate_q(self, batch_size = 256):

        X = torch.zeros(batch_size, self.Ndt, self.d).to(self.dev)
        X[:,0,:] = self.X0.view(1,self.d).repeat(batch_size,1).to(self.dev)
        
        ones = torch.ones(batch_size, 1).to(self.dev)
        
        for i, t in enumerate(self.t[:-1]):
            
            dW = self.sqrt_dt * self.Z.sample((batch_size,)).to(self.dev)
        
            X[:,i+1,:] = self.step(t*ones, X[:,i,:], self.theta, self.sigma, dW, self.dt)
        
        return X  
    
    def int_tT(self, y):
        
        y_flipped = torch.cat((torch.zeros(y.shape[0],1).to(self.dev), 
                               y.flip((1,))[:,1:]),axis=1)
        result = torch.cumsum(y_flipped*self.dt, axis=1).flip((1,))
        
        return result
    
    def estimate_omega(self, eta, n_iter=10, n_print=100, batch_size=256):
        
        self.t_train = torch.tensor(self.t).float().view(1,-1,1).to(self.dev).repeat(batch_size,1,1)  
        
        for i in tqdm(range(n_iter)):
            
            mask = torch.randint(self.X.shape[0], (batch_size,)).to(self.dev)
            
            # X, var_sigma = self.simulate_Qbar(batch_size=batch_size)
            X = self.X[mask]
            var_sigma = self.var_sigma[mask]
            
            # idx = torch.cat((torch.randint(X.shape[0],(batch_size,1)), 
            #                  torch.randint(X.shape[1],(batch_size,1))),axis=1)
            
            # idx[-int(0.1*batch_size):,1] = -1
            
            
            # # terminal constraints
            # F = 0
            # for k in range(len(self.f)):
            #     F += eta[len(self.g)+k]*self.f[k](X[idx[:,0],-1,0])
            
            
            # X = X[idx[:,0], idx[:,1],0].unsqueeze(axis=-1)
            # t = self.t_train[idx[:,0], idx[:,1],0].unsqueeze(axis=-1)
            
            # # # running constraints
            # int_g = 0
            # # for k in range(len(self.g)):
            # #     int_g += eta[k]*self.int_tT(self.g[k](self.t_train, X).squeeze() )
            
            t = self.t_train
            F = 0
            for k in range(len(self.f)):
                F += eta[len(self.g)+k]*self.f[k](X[:,-1,0])
                
            int_g = 0
            
            int_var_sigma = self.int_tT(var_sigma)
            
            omega = self.omega['net'](torch.cat((t, X), axis=-1))
            
            loss = torch.mean( ( omega - torch.exp(-F.reshape(-1,1)-int_var_sigma-int_g).unsqueeze(-1))**2 )
            
            self.omega['optimizer'].zero_grad()
            
            loss.backward()
            
            self.omega['optimizer'].step()
            self.omega['scheduler'].step()
            
            self.omega['loss'].append(np.sqrt(loss.item()))
            
            if np.mod(i+1,n_print) == 0:
                self.plot_loss(self.omega['loss'],r"$loss_\omega$")
                self.plot_mu()
                self.plot_sample_qpaths(eta, batch_size=1_000)
                f_err, g_err = self.estimate_errors(batch_size=1_000)
                print("\nerrors ", f_err, g_err)
            
    def estimate_errors(self, batch_size=1_000):
        
        X = self.simulate_q(batch_size=batch_size)
        g_err = []
        for k in range(len(self.g)):
            g_err.append(torch.mean(torch.sum(self.dt*self.g[k](self.t_train[:,:-1,:], X[:,:-1,:]), axis=1)).detach().cpu().numpy())
        
        f_err = []
        for k in range(len(self.f)):
            f_err.append(torch.mean(self.f[k](X[:,-1,:])).detach().cpu().numpy())
            
        return f_err, g_err
        
    def get_dQdQbar_T(self, eta, X, var_sigma):
        
        # running constraints
        int_g = 0
        for k in range(len(self.g)):
            int_g += eta[k]*torch.sum(self.dt*self.g[k](self.t_train[:,:-1,:], X[:,:-1,:]), axis=1)
        
        # terminal constraints
        F = 0
        for k in range(len(self.f)):
            F += eta[len(self.g)+k]*self.f[k](X[:,-1,:])
        
        int_var_sigma = 0.5*torch.sum(self.dt*var_sigma, axis=1).reshape(-1,1)
        
        dQ_dQbar = torch.exp(-F-int_g - int_var_sigma)
        mean = torch.mean(dQ_dQbar, axis=0)
        dQ_dQbar = dQ_dQbar / mean
        
        return dQ_dQbar
    
    def find_eta(self, batch_size):
        
        print("finding eta")
        
        self.t_train = torch.tensor(self.t).float().view(1,-1,1).to(self.dev).repeat(batch_size,1,1)  
        
        X, var_sigma = self.simulate_Qbar(batch_size=batch_size)
        
        def error(eta):
            
            dQ_dQbar = self.get_dQdQbar_T(eta, X, var_sigma)
            
            loss = np.zeros(len(eta))
            # running constraints
            for k in range(len(self.g)):
                loss[k] = torch.mean(dQ_dQbar * (torch.sum(self.dt*self.g[k](self.t_train, X)[:,:-1,:],axis=1)) ) 
                
            for k in range(len(self.f)):
                loss[k+len(self.g)] = torch.mean(dQ_dQbar * (self.f[k](X[:,-1,:])))
                
            self.eta['values'].append(1*eta)
            self.eta['loss'].append(1*loss)
            
            print(eta, loss)
                
            return np.sum(loss**2)
        
        # result = fsolve(lambda y : error(y), np.zeros(len(self.g)+len(self.f)))
        result = minimize(lambda y : error(y), 0.1*np.ones(len(self.g)+len(self.f)), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
        
        self.X = X
        self.var_sigma = var_sigma
        
        return result
        
    #
    # add in Lagrange multiplier as feature to NN and added in terminal penalties
    # can compute using actor-critic methods.
    #
    def train(self, batch_size = 1024, n_iter_omega=1_000, n_print=100):
        
        self.t_train = torch.tensor(self.t).float().view(1,-1,1).to(self.dev).repeat(batch_size,1,1)        
        
        self.count = 0
        
        self.plot_sample_qpaths(256)
        
        eta = self.find_eta(batch_size = 100_000)
        
        eta = [-3.85376584]
        
        self.estimate_omega(eta, n_iter=n_iter_omega, n_print=n_print, batch_size=batch_size)
        
    
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
        X = self.simulate(batch_size).cpu().numpy()
        t = self.t.cpu().numpy()
        
        print('done sim')
        
        
        fig, axs = plt.subplots(self.d, self.K+1, figsize=(10, 5), sharex=True)
        
        if len(axs.shape)==1:
            axs = np.expand_dims(axs, axis=0)
        
        for i in range(self.d):
            
            for k in range(self.K+1):
                
                qtl = np.quantile(X[:,:,i,k], [0.1, 0.5, 0.9], axis=0)
                axs[i,k].plot(t, X[:500,:,i,k].T, alpha=0.25, linewidth=1)
                axs[i,k].plot(t, qtl.T, color='k', linestyle='--',linewidth=1)
                axs[i,k].plot(t, X[0,:,i,k].T, color='b', linewidth=1)
               
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
        state = torch.get_rng_state()
        torch.manual_seed(12317874321)
        
        print('start sim')
        X = self.simulate_q(batch_size=batch_size).detach().cpu().numpy()
        t = self.t.cpu().numpy()
        
        print('done sim')
        
        
        fig, axs = plt.subplots(self.d, 1,  figsize=(6,4), sharex=True)
        
        if self.d == 1:
            axs = np.array([axs])
        
        for i in range(self.d):
            
            qtl = np.quantile(X[:,:,i], np.arange(1,10)/10, axis=0)
            
            plt.fill_between(t, qtl[0], qtl[-1], color='y', alpha=0.5)
            axs[i].plot(t, X[:500,:,i].T, alpha=0.1, linewidth=1)
            axs[i].plot(t, qtl.T, color='k', linestyle='--',linewidth=1)
            axs[i].plot(t, X[0,:,i].T, color='b', linewidth=1)
            
               
        axs[0].set_title('model $\mathbb{Q}^*$')
        
        for i in range(self.d):
            axs[i].set_ylabel(r'$X_{' + str(i) +'}$')
            axs[i].set_ylim(-1,2)
        
        fig.add_subplot(111, frameon=False)      
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel(r'$t$')               
            
        plt.tight_layout()
        plt.show()
        
        torch.set_rng_state(state)
        
        return X
        
    # def grad_L(self, t, X):
        
    #     eps = 1e-4
    #     self.omega['net'](torch.cat((t,X-eps), axis=-1)) 
    #     L = - torch.sum( torch.log(  ) )
        
    #     return torch.autograd.grad(L, X)[0]        
        
    def grad_L(self, t, X):
        
        X = X.detach().requires_grad_()
        L = - torch.sum( torch.log(  self.omega['net'](torch.cat((t,X), axis=-1)) ) )
        
        return torch.autograd.grad(L, X)[0]
    
    def plot_mu(self):
        
        fig = plt.figure(figsize=(5,4))
        
        tm, xm = torch.meshgrid(torch.linspace(0,self.T,101).to(self.dev),
                                torch.linspace(-2,2,101).to(self.dev))
        
        tm = tm.unsqueeze(axis=2)
        xm = xm.unsqueeze(axis=2)
        
        theta = self.theta(tm, xm).squeeze() 
        
        C = plt.contourf(tm.squeeze().cpu().numpy(),
                     xm.squeeze().cpu().numpy(),
                     theta.detach().cpu().numpy(), levels=np.linspace(-15,15,31), 
                     cmap='RdBu')
        
        cbar = plt.colorbar(C)
        
        plt.clim(-12,12)
        
        plt.show()
        
    def sample_dq_dqbar(self, eta, X, var_sigma):
        
        t = self.t.cpu().numpy()
        
        dQ_dQbar = self.get_dQdQbar_T(eta, X, var_sigma).detach().cpu().numpy()
        p = dQ_dQbar.reshape(-1)
        p /= np.sum(p)
        
        draw = choice(np.arange(X.shape[0]), X.shape[0], p=p)
        
        
        def makeplot(X_sample):
        
            fig, axs = plt.subplots(self.d, 1,  figsize=(6,4), sharex=True)
            
            if self.d == 1:
                axs = np.array([axs])
            
            for i in range(self.d):
                
                qtl = np.quantile(X_sample[:,:,i], np.arange(1,10)/10, axis=0)
                
                plt.fill_between(t, qtl[0], qtl[-1], color='y', alpha=0.5)
                axs[i].plot(t, X_sample[:500,:,i].T, alpha=0.1, linewidth=1)
                axs[i].plot(t, qtl.T, color='k', linestyle='--',linewidth=1)
                axs[i].plot(t, X_sample[0,:,i], color='b', linewidth=1)
                
                   
            axs[0].set_title('model $\mathbb{Q}^*$')
            
            for i in range(self.d):
                axs[i].set_ylabel(r'$X_{' + str(i) +'}$')
                axs[i].set_ylim(-1,2)
            
            fig.add_subplot(111, frameon=False)      
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel(r'$t$')               
                
            plt.tight_layout()
            plt.show()
        
        makeplot(X)
        makeplot(X[draw])
        
        return X[draw]
        