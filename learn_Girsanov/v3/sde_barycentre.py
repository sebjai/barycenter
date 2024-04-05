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
                 mu_bar,
                 n_nodes=128, 
                 n_layers=5,
                 device='cpu'):
        super(net, self).__init__()

        self.device = device
        
        self.in_to_hidden = nn.Linear(nIn, n_nodes).to(self.device)
        self.hidden_to_hidden = nn.ModuleList([nn.Linear(n_nodes, n_nodes).to(self.device) for i in range(n_layers-1)])
        self.hidden_to_out = nn.Linear(n_nodes, nOut).to(self.device)
        
        self.g = nn.SiLU()
        self.mu_bar = mu_bar
        
    def forward(self, x):
        
        h = self.g(self.in_to_hidden(x))
        for linear in self.hidden_to_hidden:
            h = self.g(linear(h))
            
        h = 10*torch.tanh(self.hidden_to_out(h))
        
        h = h + self.mu_bar(x[...,0].unsqueeze(-1), x[...,1:])
        
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
        self.rho = 1*rho.to(self.dev)
        self.rho_inv = torch.inverse(self.rho)
        self.U = torch.cholesky(self.rho.unsqueeze(axis=0))        
        
        self.pi= pi
        
        self.mu_bar = lambda t,x : torch.sum(torch.cat([pi*mu(t,x).unsqueeze(2) 
                                                        for pi, mu in zip(self.pi, self.mu)], 
                                                       axis=2 ), axis=2)
        
        self.T = T
        self.Ndt = Ndt
        self.t = np.linspace(0,self.T, self.Ndt)
        self.dt = self.t[1]-self.t[0]
        
        # features are t, x_1,..,x_dm 
        self.theta = self.get_net(nIn=1+self.d, nOut=self.d, mu_bar=self.mu_bar)
        
        self.Z = torch.distributions.MultivariateNormal(torch.zeros(self.d).to(self.dev),
                                                        self.rho)
        self.sqrt_dt = np.sqrt(self.dt)
        
    def get_net(self, nIn, nOut, mu_bar):

        obj = {'net' : net(nIn=nIn, nOut=nOut, mu_bar=mu_bar).to(self.dev) }
        obj['optimizer'] = optim.AdamW(obj['net'].parameters(), lr=0.001)
        obj['scheduler'] = optim.lr_scheduler.StepLR(obj['optimizer'],
                                                     step_size=1000,
                                                     gamma=0.999)
        obj['loss'] = []        

        return obj
        
        
        
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
        
        X = torch.zeros(batch_size, self.Ndt, self.d, self.K+1).to(self.dev)
        X[:,0,:,:] = self.X0.view(1,self.d,1).repeat(batch_size,1,1).to(self.dev)
        ones = torch.ones(batch_size,1).to(self.dev)
        for i, t in enumerate(self.t[:-1]):
            
            dW = self.sqrt_dt * self.Z.sample((batch_size,)).to(self.dev)
            
            for k in range(self.K):
                
                X[:,i+1,:,k] = self.step(t*ones, X[:,i,:,k], self.mu[k], self.sigma, dW, self.dt)
            
            X[:,i+1,:,-1] = self.step(t*ones, X[:,i,:,-1], self.mu_bar, self.sigma, dW, self.dt)
        
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
        dW = torch.zeros(batch_size, self.Ndt, self.d).to(self.dev)
        X[:,0,:] = self.X0.view(1,self.d).repeat(batch_size,1).to(self.dev)
        
        var_sigma = torch.zeros(batch_size, self.Ndt).to(self.dev)
        ones = torch.ones(batch_size, self.d).to(self.dev)
        for i, t in enumerate(self.t[:-1]):
            
            dW[:,i,:] = self.sqrt_dt * self.Z.sample((batch_size,)).to(self.dev)
        
            sigma = self.sigma(t*ones,X[:,i,:])
            mu_bar = self.mu_bar(t*ones,X[:,i,:])
            
            Sigma = sigma.unsqueeze(axis=1) * self.rho.unsqueeze(axis=0) * sigma.unsqueeze(axis=2)
            inv_Sigma = torch.inverse(Sigma)
            
            for k in range(self.K):
                dmu = (self.mu[k](t*ones,X[:,i,:]) - mu_bar)
                
                var_sigma[:,i] += 0.5*self.pi[k]*torch.einsum("ij,ijk,ik->i", dmu, inv_Sigma, dmu)
            
            X[:,i+1,:] = self.step(t*ones, X[:,i,:], self.mu_bar, self.sigma, dW[:,i,:], self.dt)
            
        return X, var_sigma, dW
    
    def simulate_Q(self, batch_size = 256):

        X = torch.zeros(batch_size, self.Ndt, self.d).to(self.dev)
        X[:,0,:] = self.X0.view(1,self.d).repeat(batch_size,1).to(self.dev)
        
        ones = torch.ones(batch_size, 1).to(self.dev)
        
        theta = lambda t,x : self.theta['net'](torch.cat((t,x),axis=-1))
        
        for i, t in enumerate(self.t[:-1]):
            
            dW = self.sqrt_dt * self.Z.sample((batch_size,)).to(self.dev)
        
            X[:,i+1,:] = self.step(t*ones, X[:,i,:], theta, self.sigma, dW, self.dt)
        
        return X  
    
    def int_tT(self, y):
        
        # pdb.set_trace()
        # y_flipped = torch.cat((torch.zeros(y.shape[0],1).to(self.dev), y.flip((1,))[:,1:]),axis=1)
        # result = torch.cumsum(y_flipped*self.dt, axis=1).flip((1,))
        result = torch.sum(y*self.dt, axis=1).squeeze()
        
        return result
    
    def cumsum(self, x):
        cx = torch.cumsum(x, axis=-1)
        zeros = torch.zeros(cx.shape[:-1]).unsqueeze(-1).to(self.dev)
        cx = torch.cat((zeros, cx[...,:-1]), axis=-1)
        
        return cx
    
    def get_stoch_exp(self, t, X, dW, theta):
        
        sigma = self.sigma(t,X)
        mu_bar = self.mu_bar(t,X)
        
        lam = (mu_bar-theta)/sigma
        
        a = torch.einsum("...ij,...jk,...ik->...i", lam, self.rho_inv, dW)
        A = torch.sum(a[...,:-1], axis=-1)
        
        b = torch.einsum("...ij,...jk,...ik->...i", lam, self.rho_inv, lam)
        B = torch.sum(b[...,:-1]* self.dt, axis=-1) 
        
        dQ_dQbar = torch.exp(-0.5*B - A)
        dQ_dQbar = dQ_dQbar / torch.mean(dQ_dQbar, axis=0)
        
        return dQ_dQbar
            
    def cost(self, t, X, theta):
        
        sigma = self.sigma(t,X)
        # Sigma = torch.einsum("...ij,...jk,...ik->...ijk", sigma, self.rho, sigma)
        # inv_Sigma = torch.linalg.inv(Sigma)
        
        cost = 0
        for k, mu in enumerate(self.mu):
            dtheta = (mu(t, X)-theta)/sigma
            A = torch.einsum("...ij,...jk,...ik->...i", dtheta, self.rho_inv, dtheta)
            cost += 0.5*self.pi[k]*torch.sum( A*self.dt, axis=-1)
            
        return cost
    
    def learn_theta(self, n_iter=10, n_print=100, batch_size=256):
        
        t = torch.tensor(self.t).float().view(1,-1,1).repeat(batch_size,1,1).to(self.dev)  
        self.F = []
        
        self.lam = [0.0*torch.tensor(self.eta).float().to(self.dev)]
        self.nu = [0.01*torch.ones(len(self.eta)).to(self.dev)]
        self.constr_err = []
        self.costs = []
        for i in tqdm(range(n_iter)):
            
            # X, var_sigma = self.simulate_Qbar(batch_size=batch_size)
            mask = torch.randint(self.X.shape[0], (batch_size,)).to(self.dev)
            X = self.X[mask]
            dW = self.dW[mask]
            
            theta = self.theta['net'](torch.cat((t,X), axis=-1))
            
            dQ_dQbar = self.get_stoch_exp(t, X, dW, theta)
            
            # if i == 0:
            #     pdb.set_trace()
            
            # constraint errors
            constr_err = torch.zeros(len(self.f)+len(self.g)).to(self.dev)
            
            # running constraints
            for k in range(len(self.g)):
                constr_err[k]=  (torch.mean(dQ_dQbar*self.int_tT(self.g[k](t, X) )))
            
            # terminal constraints
            for k in range(len(self.f)):
                constr_err[k+len(self.g)]= (torch.mean(dQ_dQbar * self.f[k](X[:,-1,0]) ))
                
            cost = self.cost(t, X, theta)
            
            loss = torch.mean( dQ_dQbar*cost ) + torch.sum(self.lam[-1]*constr_err + self.nu[-1]*(constr_err)**2)
            
            self.theta['optimizer'].zero_grad()
            
            loss.backward()
            
            self.theta['optimizer'].step()
            self.theta['scheduler'].step()
            
            self.theta['loss'].append(np.sqrt(loss.item()))
            
            self.lam.append((1*self.lam[-1] + self.nu[-1]*constr_err).detach())
            self.nu.append(torch.minimum(1.001*self.nu[-1],torch.tensor([1]).to(self.dev)))
            self.constr_err.append(constr_err.detach().cpu().numpy())
            self.costs.append(torch.mean( dQ_dQbar*cost ).detach().cpu().numpy())
            
            if np.mod(i+1,n_print) == 0:
                self.plot_loss(self.theta['loss'],r"$loss_\omega$")
                self.plot_mu()
                # self.plot_sample_qpaths(eta, batch_size=1_000)
                f_err, g_err = self.estimate_errors(batch_size=1_000)
                self.plot_sample_qpaths(1_000)
                print("\nerrors ", f_err, g_err)
            
    def estimate_errors(self, batch_size=1_000):
        
        t = torch.tensor(self.t).float().view(1,-1,1).repeat(batch_size,1,1).to(self.dev)  
        
        X = self.simulate_Q(batch_size=batch_size)
        g_err = []
        for k in range(len(self.g)):
            g_err.append(torch.mean(torch.sum(self.dt*self.g[k](t[:,:-1,:], X[:,:-1,:]), axis=1)).detach().cpu().numpy())
        
        f_err = []
        for k in range(len(self.f)):
            f_err.append(torch.mean(self.f[k](X[:,-1,:])).detach().cpu().numpy())
            
        return f_err, g_err
        
    def get_dQdQbar_T(self, eta, t, X, var_sigma):
        
        # running constraints
        int_g = 0
        for k in range(len(self.g)):
            int_g += eta[k]*torch.sum(self.dt*self.g[k](t[:,:-1,:], X[:,:-1,:]), axis=1)
        
        # terminal constraints
        F = 0
        for k in range(len(self.f)):
            F += eta[len(self.g)+k]*self.f[k](X[:,-1,:])
        
        int_var_sigma = 0.5*torch.sum(self.dt*var_sigma, axis=1).reshape(-1,1)
        
        dQ_dQbar = torch.exp(-F-int_g - int_var_sigma)
        mean = torch.mean(dQ_dQbar, axis=0)
        dQ_dQbar = dQ_dQbar / mean
        
        return dQ_dQbar
    
    def find_eta(self):
        
        print("finding eta")
        
        t = torch.tensor(self.t).float().view(1,-1,1).repeat(self.X.shape[0],1,1).to(self.dev)
        
        def error(eta):
            
            dQ_dQbar = self.get_dQdQbar_T(eta, t, self.X, self.var_sigma)
            
            loss = np.zeros(len(eta))
            # running constraints
            for k in range(len(self.g)):
                loss[k] = torch.mean(dQ_dQbar * (torch.sum(self.dt*self.g[k](t, self.X)[:,:-1,:],axis=1)) ) 
                
            for k in range(len(self.f)):
                loss[k+len(self.g)] = torch.mean(dQ_dQbar * (self.f[k](self.X[:,-1,:])))
                
            print(eta, loss)
                
            return loss
        
        result = fsolve(lambda y : error(y), 0.1*np.ones(len(self.g)+len(self.f)))
        # result = minimize(lambda y : error(y), 0.1*np.ones(len(self.g)+len(self.f)), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
        
        self.eta = result
        
    #
    def train(self, batch_size = 1024, n_iter_eta=1_000,  
              n_iter_omega=1_000, n_print=100):
        
        self.X, self.var_sigma, self.dW = self.simulate_Qbar(batch_size=100_000)
        self.eta = []
        
        self.find_eta()
        
        self.learn_theta(n_iter=n_iter_omega, n_print=n_print, batch_size=batch_size)
        
    
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
        plt.yscale('symlog')
        plt.title(title)
        plt.show()
        
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
        state = torch.get_rng_state()
        torch.manual_seed(12317874321)
        
        print('start sim')
        X = self.simulate_Q(batch_size=batch_size).cpu().detach().numpy()
        
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
        
        torch.set_rng_state(state)
        
        return X
        
    def plot_mu(self):
        
        tm, xm = torch.meshgrid(torch.linspace(0,self.T,51).to(self.dev),
                                torch.linspace(-2,2,51).to(self.dev))
        
        tm = tm.unsqueeze(axis=2)
        xm = xm.unsqueeze(axis=2)
        
        def plot(mu):
            plt.contourf(tm.squeeze().cpu().numpy(),
                         xm.squeeze().cpu().numpy(),
                         mu.detach().cpu().numpy(),
                         cmap='RdBu',
                         levels=np.linspace(-10,10,21))
            plt.colorbar()
            plt.show()
        
        plot(self.theta['net'](torch.cat((tm, xm),axis=-1)).squeeze() )
        for mu in self.mu:
            plot(mu(tm, xm).squeeze() )
            
        plot(self.mu_bar(tm,xm).squeeze())
        
    def sample_dq_dqbar(self, eta, t, X, var_sigma):
        
        
        dQ_dQbar = self.get_dQdQbar_T(eta, t, X, var_sigma).detach().numpy()
        p = dQ_dQbar.reshape(-1)
        p /= np.sum(p)
        
        draw = choice(np.arange(X.shape[0]), X.shape[0], p=p)
        
        def makeplot(X_sample):
        
            fig, axs = plt.subplots(self.d, 1,  figsize=(6,4), sharex=True)
            
            if self.d == 1:
                axs = np.array([axs])
            
            for i in range(self.d):
                
                qtl = np.quantile(X_sample[:,:,i], np.arange(1,10)/10, axis=0)
                
                plt.fill_between(self.t, qtl[0], qtl[-1], color='y', alpha=0.5)
                axs[i].plot(self.t, X_sample[:500,:,i].T, alpha=0.1, linewidth=1)
                axs[i].plot(self.t, qtl.T, color='k', linestyle='--',linewidth=1)
                axs[i].plot(self.t, X_sample[0,:,i], color='b', linewidth=1)
                
                   
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
