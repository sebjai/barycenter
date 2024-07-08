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
                 n_nodes=36, 
                 n_layers=5,
                 device='cpu',
                 output='none'):
        super(net, self).__init__()

        self.device = device
        
        self.in_to_hidden = nn.Linear(nIn, n_nodes).to(self.device)
        
        # self.hidden_to_hidden = nn.ModuleList([nn.Linear(n_nodes+nIn, n_nodes).to(self.device) for i in range(n_layers-1)])
        self.hidden_to_hidden = nn.ModuleList([nn.Linear(n_nodes, n_nodes).to(self.device) for i in range(n_layers-1)])
        
        self.hidden_to_out1 = nn.Linear(n_nodes+nIn, n_nodes).to(self.device)
        self.out1_to_out = nn.Linear(n_nodes, nOut).to(self.device)
        
        # self.hidden_to_out = nn.Linear(n_nodes, nOut).to(self.device)
        
        self.g = nn.SiLU()
        # self.g = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
        self.output=output
        
        
    def forward(self, x):
        
        h = self.g(self.in_to_hidden(x))
        for linear in self.hidden_to_hidden:
            # h = torch.cat((h,x),axis=-1)
            h = self.g(linear(h))
            
        # concat orginal x to last hidden layer
        h = torch.cat((h,x),axis=-1)
        h = self.hidden_to_out1(h)
        h = self.out1_to_out(self.g(h))
        
        # h = self.hidden_to_out(h)
        
        if self.output=='softplus':
            h = self.softplus(h)
        
        return h


class sde_barycentre():
    
    def __init__(self, X0, mu, sigma, pi, f=[], g=[], T=1, Ndt = 501):
        
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
        self.pi= pi
        
        self.T = T
        self.Ndt = Ndt
        self.t = torch.linspace(0,self.T, self.Ndt).to(self.dev)
        self.dt = self.t[1]-self.t[0]
        
        # features are t, x_1,..,x_d, eta_1,..,eta_n for constraints
        self.omega = {'net' : net(nIn=self.d+1, nOut=1, output='softplus', device=self.dev).to(self.dev) }
        self.omega['optimizer'] = optim.Adam(self.omega['net'].parameters(), lr=0.005)
        self.omega['scheduler'] = optim.lr_scheduler.StepLR(self.omega['optimizer'],
                                                         step_size=5,
                                                         gamma=0.9997)
        
        self.omega['loss'] = []
        self.f_err = []
        self.g_err = []
        self.i = []
        
        self.target = self.omega['net']
        self.tau = 0.001
        
        self.eta = {'loss' : [],  'values': []}
        
        self.sqrt_dt = torch.sqrt(self.dt)
        
    def mu_bar(self,t,x):
        
        result = 0
        for k in range(len(self.pi)):
            result +=self.pi[k]*self.mu[k](t,x)
            
        return result
        
    def step(self, t, x, mu, sigma, dB, dt):
        
        s = torch.cholesky(sigma(t,x))
        xp = x + mu(t,x) * dt  + torch.einsum("...jk,...k->...j",s,dB) 
        
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
            
            dB = self.sqrt_dt * torch.randn(batch_size,self.d).to(self.dev)
            
            for k in range(self.K):
                
                X[:,i+1,:,k] = self.step(t, X[:,i,:,k], self.mu[k], self.sigma, dB, self.dt)
            
            X[:,i+1,:,-1] = self.step(t, X[:,i,:,-1], self.mu_bar, self.sigma, dB, self.dt)
        
        return X


    def simulate_Qbar(self, batch_size = 256, init_randomize=False):
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
        dB = torch.zeros(batch_size, self.Ndt, self.d).to(self.dev)
        
        if init_randomize:
            X[:,0,:] = self.X0.view(1,self.d).repeat(batch_size,1).to(self.dev) \
                + 0.5*torch.randn(batch_size, self.d).to(self.dev)
        else:
            X[:,0,:] = self.X0.view(1,self.d).repeat(batch_size,1).to(self.dev)
        
        var_sigma = torch.zeros(batch_size, self.Ndt).to(self.dev)
        ones = torch.ones(batch_size, self.d).to(self.dev)
        for i, t in enumerate(self.t[:-1]):
            
            dB[:,i,:] = self.sqrt_dt * torch.randn(batch_size,self.d).to(self.dev)
        
            sigma = self.sigma(t*ones,X[:,i,:])
            sigma_inv = torch.linalg.inv(sigma)
            
            mu_bar = self.mu_bar(t*ones,X[:,i,:])
            
            for k in range(self.K):
                dmu = (self.mu[k](t*ones,X[:,i,:]) - mu_bar)
                
                var_sigma[:,i] += 0.5*self.pi[k]*torch.einsum("...j,...jk,...k->...", dmu, sigma_inv, dmu)
            
            X[:,i+1,:] = self.step(t*ones, X[:,i,:], self.mu_bar, self.sigma, dB[:,i,:], self.dt)
            
        return X, var_sigma, dB
    
    def theta(self, t,x):
        
        Sigma = self.sigma(t,x)
        mu_bar = self.mu_bar(t,x)
        
        grad_L = self.grad_L(t, x)
        
        try:
            result = mu_bar - torch.einsum("...jk,...k->...j", Sigma, grad_L)
        except:
            print("\n\n***")
            pdb.set_trace()
                    
        return result  
    
    def simulate_Q(self, batch_size = 256):

        X = torch.zeros(batch_size, self.Ndt, self.d).to(self.dev)
        X[:,0,:] = self.X0.view(1,self.d).repeat(batch_size,1).to(self.dev)
        
        ones = torch.ones(batch_size, 1).to(self.dev)
        
        for i, t in enumerate(self.t[:-1]):
            
            dB = self.sqrt_dt * torch.randn(batch_size,self.d).to(self.dev)
        
            X[:,i+1,:] = self.step(t*ones, X[:,i,:], self.theta, self.sigma, dB, self.dt)
        
        return X
    
    def int_tT(self, y):
        
        y_flipped = torch.cat((torch.zeros(y.shape[0],1).to(self.dev), 
                               y.flip((1,))[:,1:]),axis=1)
        result = torch.cumsum(y_flipped*self.dt, axis=1).flip((1,))
        
        return result
    
    def estimate_omega(self, n_iter=10, print_at_list=[10], batch_size=256, rule="L2"):
        
        self.t_train = self.t.view(1,-1,1).to(self.dev).repeat(batch_size,1,1)  
        
        print(print_at_list)
        
        print_at = print_at_list.pop()
        print("*** first at: ", print_at)
        
        if len(self.f+self.g) > 0:
            eta = self.eta['values'][-1]
        
        for i in tqdm(range(n_iter)):
            
            mask = torch.randint(self.X.shape[0], (batch_size,)).to(self.dev)
            
            X = self.X[mask]
            var_sigma = self.var_sigma[mask]
            t = self.t_train
            
            # # # running constraints
            int_g = 0
            F = 0
                
            if len(self.f+self.g) > 0:
                
                for k in range(len(self.g)):
                    int_g += eta[k]*self.int_tT(self.g[k](self.t_train, X).squeeze() )
                for k in range(len(self.f)):
                    F += eta[len(self.g)+k]*self.f[k](X[:,-1,:])
                
            
            int_var_sigma = self.int_tT(var_sigma)
            
            omega = self.omega['net'](torch.cat((t, X), axis=-1))
            
            y = torch.exp(-F-int_var_sigma-int_g)
            z = omega.squeeze()
            
            if rule=="L2":            
                score = (z-y)**2
            elif rule =="Poisson":
                score = 2.0*(y*torch.log(y/z)+z - y)
            elif rule == "Gamma":
                score = 2*(torch.log(z/y)+(y/z)-1.0)
            
            loss = torch.nanmean( score )
            
            self.omega['optimizer'].zero_grad()
            
            loss.backward()
            
            self.omega['optimizer'].step()
            self.omega['scheduler'].step()
            
            self.omega['loss'].append(loss.item())
            
            
            if i >= print_at:
                
                self.i.append(i+1)
                
                self.plot_loss(self.omega['loss'],r"$loss_\omega$")
                # self.plot_mu()
                self.plot_sample_qpaths(batch_size=1_000)
                f_err, g_err = self.estimate_errors(batch_size=batch_size)
                self.f_err.append(f_err)
                self.g_err.append(g_err)
                
                print_at = print_at_list.pop()
                
                print("\n\n**** i = ", i, " next print at ", print_at)
                print("errors ", f_err, g_err)            
            
    def estimate_errors(self, batch_size=1_000):
        
        X = self.simulate_Q(batch_size=batch_size)
        g_err = []
        est = lambda Z : np.array([torch.nanmean(Z).detach().cpu().numpy(), 
                                   torch.std(Z).detach().cpu().numpy()/np.sqrt(batch_size)])
        
        for k in range(len(self.g)):
            Z = torch.sum(self.dt*self.g[k](self.t_train[:,:-1,:], X[:,:-1,:]), axis=1)
            g_err.append(est(Z))
        
        f_err = []
        for k in range(len(self.f)):
            Z = self.f[k](X[:,-1,:])
            f_err.append(est(Z))
            
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
        
        int_var_sigma = torch.sum(self.dt*var_sigma, axis=1).reshape(-1,1)
        
        dQ_dQbar = torch.exp(-F-int_g - int_var_sigma)
        mean = torch.mean(dQ_dQbar, axis=0)
        dQ_dQbar = dQ_dQbar / mean
        
        return dQ_dQbar
    
    def find_eta(self, batch_size):
        
        print("finding eta")
        
        self.t_train = self.t.view(1,-1,1).to(self.dev).repeat(batch_size,1,1)  
        
        X, var_sigma, _ = self.simulate_Qbar(batch_size=batch_size)
        
        if len(self.f) + len(self.g) == 0:
            return 0
        
        def error(eta):
            
            dQ_dQbar = self.get_dQdQbar_T(eta, X, var_sigma)
            
            loss = np.zeros(len(eta))
            # running constraints
            for k in range(len(self.g)):
                loss[k] = torch.nanmean(dQ_dQbar * (torch.sum(self.dt*self.g[k](self.t_train, X)[:,:-1,:],axis=1)) ) 
                
            for k in range(len(self.f)):
                loss[k+len(self.g)] = torch.nanmean(dQ_dQbar * (self.f[k](X[:,-1,:])))
                
            self.eta['values'].append(1*eta)
            self.eta['loss'].append(1*loss)
            
            print(eta, loss)
                
            return loss
        
        result = fsolve(lambda y : error(y), 0.01*np.ones(len(self.g)+len(self.f)))
        # result = minimize(lambda y : error(y), 0.1*np.ones(len(self.g)+len(self.f)), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
        
        self.eta['values'].append(result)
        self.eta['loss'].append(error(self.eta['values'][-1]))
        
    def generate_training_batch(self, batch_size):
        X, var_sigma, _ = self.simulate_Qbar(batch_size=batch_size, init_randomize=True)
        self.X = X
        self.var_sigma = var_sigma          
    
    
    #
    # add in Lagrange multiplier as feature to NN and added in terminal penalties
    # can compute using actor-critic methods.
    #
    def train(self, batch_size = 1024, n_iter=1_000, print_at_list=[1_000], rule="L2", rng_state=None):
        
        if rng_state is not None:
            torch.set_rng_state(rng_state['torch'])
            np.random.set_state(rng_state['np'])
            # torch.use_deterministic_algorithms(True)
        
        self.t_train = self.t.view(1,-1,1).to(self.dev).repeat(batch_size,1,1)        
        
        self.count = 0
        
        self.plot_sample_qpaths(256)
        
        self.estimate_omega(n_iter=n_iter, print_at_list=print_at_list, batch_size=batch_size, rule=rule)
        
    
    def plot_sample_paths(self, batch_size = 4_096, filename=""):
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
        state = torch.get_rng_state()
        
        
        print('start sim')
        torch.manual_seed(12317874323)
        X = self.simulate(batch_size)
        
        torch.manual_seed(12317874323)
        Y = self.simulate_Q(batch_size=batch_size).unsqueeze(-1)
        
        X = torch.cat((X,Y),axis=-1).cpu().detach().numpy()
        
        print('done sim')
        
        torch.set_rng_state(state)
        
        
        fig, axs = plt.subplots(self.d, X.shape[-1], figsize=(10, 1.5*self.d), sharex='col', sharey='row')
        
        if len(axs.shape)==1:
            axs = np.expand_dims(axs, axis=0)
        
        for i in range(self.d):
            
            for k in range(X.shape[-1]):
                
                qtl = np.nanquantile(X[:,:,i,k], [0.1, 0.3, 0.5, 0.7, 0.9], axis=0)
                # axs[i,k].plot(self.t.cpu(), X[:100,:,i,k].T, alpha=0.1, linewidth=1)
                # axs[i,k].plot(self.t.cpu(), qtl.T, color='k', linestyle='--',linewidth=1)
                axs[i,k].fill_between(self.t.cpu(), qtl[0], qtl[-1], color='r',alpha=0.5)
                axs[i,k].plot(self.t.cpu(), X[0,:,i,k].T, color='b', linewidth=1)
                # axs[i,k].set_ylim(-2,2)
               
        for k in range(self.K):
            axs[0,k].set_title('model $\mathbb{P}_' + str(k+1) + '$')
        
        axs[0,-2].set_title('model $\overline{\mathbb{Q}}$')
        axs[0,-1].set_title('model $\mathbb{Q}^{\!\!*}$')
        
        for i in range(self.d):
            axs[i,0].set_ylabel(r'$X_{' + str(i+1) +'}$')
        
        fig.add_subplot(111, frameon=False)      
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel(r'$t$')               
            
        plt.tight_layout()
        
        if not(filename == ""):
            plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.show()
        
        return self.t, X
        
        
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

        mv, mv_err = self.moving_average(x,500)
        
        plt.fill_between(np.arange(len(mv)), y1=mv-mv_err, y2=mv+mv_err, alpha=0.2)
        plt.plot(mv,  linewidth=1.5)
        plt.plot(x,  alpha=0.1)
        plt.yscale('log')
        plt.title(title)
        # plt.yticks([0.01,0.1,1])
        plt.show()
        
    def plot_sample_qpaths(self, batch_size = 4_096, filename=None):
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
        X = self.simulate_Q(batch_size=batch_size).detach().cpu().numpy()
        t = self.t.cpu().numpy()
        
        print('done sim')
        
        
        fig, axs = plt.subplots(self.d, 1,  figsize=(6,2*self.d), sharex=True)
        
        if self.d == 1:
            axs = np.array([axs])
        
        for i in range(self.d):
            
            qtl = np.nanquantile(X[:,:,i], np.arange(1,5)/5, axis=0)
            
            # plt.fill_between(t, qtl[0], qtl[-1], color='y', alpha=0.5)
            # axs[i].plot(t, X[:500,:,i].T, alpha=0.1, linewidth=1)
            # axs[i].plot(t, qtl.T, color='k', linestyle='--',linewidth=1)
            axs[i].fill_between(t, qtl[0], qtl[-1], color='r', alpha=0.2)
            axs[i].plot(t, X[0,:,i].T, color='b', linewidth=1)
            
               
        axs[0].set_title('model $\mathbb{Q}^*$')
        
        for i in range(self.d):
            axs[i].set_ylabel(r'$X_{' + str(i+1) +'}$')
            # axs[i].set_ylim(-1,2)
        
        fig.add_subplot(111, frameon=False)      
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel(r'$t$')               
            
        plt.tight_layout()
        
        if filename is not None:
            plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.show()
        
        torch.set_rng_state(state)
        
        return X
        
    # def grad_L(self, t, X):
        
    #     eps = 1e-4
    #     self.omega['net'](torch.cat((t,X-eps), axis=-1)) 
    #     L = - torch.sum( torch.log(  ) )
        
    #     return torch.autograd.grad(L, X)[0]        
        
    def grad_L(self, t, X):
        
        
        try:
            
            X = X.detach().requires_grad_()
            L = - torch.sum( torch.log(  self.omega['net'](torch.cat((t,X), axis=-1)) ) )
            
            result = torch.autograd.grad(L, X)[0]
        
        except:
            print("could not compute grad")
            pdb.set_trace()
        
        return result
    
    # def plot_mu(self):
        
    #     fig = plt.figure(figsize=(5,4))
        
    #     tm, xm = torch.meshgrid(torch.linspace(0,self.T,101).to(self.dev),
    #                             torch.linspace(-2,2,101).to(self.dev))
        
    #     tm = tm.unsqueeze(axis=2)
    #     xm = xm.unsqueeze(axis=2)
        
    #     theta = self.theta(tm, xm).squeeze() 
        
    #     C = plt.contourf(tm.squeeze().cpu().numpy(),
    #                  xm.squeeze().cpu().numpy(),
    #                  theta.detach().cpu().numpy(), levels=np.linspace(-15,15,31), 
    #                  cmap='RdBu')
        
    #     cbar = plt.colorbar(C)
        
    #     plt.clim(-12,12)
        
    #     plt.show()
    
    
    def plot_mu_1d(self):
        
        tm, xm = torch.meshgrid(torch.linspace(0,self.T,51).to(self.dev),
                                torch.linspace(-2,2,51).to(self.dev))
        
        tm = tm.unsqueeze(axis=2)
        xm = xm.unsqueeze(axis=2)
        
        def plot(mu):
            
            fig, ax = plt.subplots(nrows=1, ncols=1)
            
            im = ax.contourf(tm.squeeze().cpu().numpy(),
                        xm.squeeze().cpu().numpy(),
                        mu[...,0].detach().cpu().numpy(),
                        cmap='RdBu',
                        levels=np.linspace(-10,10,21))
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            plt.show()
            
        plot(self.theta(tm, xm) )
        for mu in self.mu:
            plot(mu(tm, xm)  )
            
        plot(self.mu_bar(tm,xm) )        
        
    def plot_mu_2d(self):
        
        tm, xm = torch.meshgrid(torch.linspace(0,self.T,51).to(self.dev),
                                torch.linspace(0,0.2,51).to(self.dev))
        
        ys = [-0.5, 0, 0.5]
        
        tm = tm.unsqueeze(axis=2)
        xm = xm.unsqueeze(axis=2)
        ones = torch.ones(xm.shape).to(self.dev)
        
        def plot(mu, title):
            
            fig, axs = plt.subplots(nrows=len(ys), ncols=self.d)
            
            for i in range(self.d):
                
                for j, y in enumerate(ys):
                    
                    drift = mu(y)
                    
                    im = axs[j,i].contourf(tm.squeeze().cpu().numpy(),
                                          xm.squeeze().cpu().numpy(),
                                          drift[...,i].detach().cpu().numpy(),
                                          cmap='RdBu',
                                          levels=np.linspace(-1,1,21))
            fig.suptitle(title)
            plt.tight_layout()
            
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            
            plt.show()
            
        theta = lambda z : self.theta(tm, torch.cat((z *ones, xm),axis=-1))
        plot(theta, r'$\theta$' )
        
        for i, mu_ in enumerate(self.mu):
            mu = lambda z: mu_(tm, torch.cat((z *ones, xm), axis=-1))
            plot(mu, r'$\mu^{(' + str(i) +')}$')
            
        mu = lambda z : self.mu_bar(tm, torch.cat((z *ones, xm), axis=-1))
        plot(mu, r'$\bar\mu$') 
    
    def plot_mu(self):
        
        if self.d == 1:
            self.plot_mu_1d()
        elif self.d==2:
            self.plot_mu_2d()    
        
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
                
                qtl = np.quantile(X_sample[:,:,i], np.arange(1,5)/5, axis=0)
                
                plt.fill_between(t, qtl[0], qtl[-1], color='r', alpha=0.2)
                # axs[i].plot(t, X_sample[:500,:,i].T, alpha=0.1, linewidth=1)
                # axs[i].plot(t, qtl.T, color='k', linestyle='--',linewidth=1)
                axs[i].plot(t, X_sample[0,:,i], color='b', linewidth=1)
                
                   
            axs[0].set_title('model $\mathbb{Q}^*$')
            
            for i in range(self.d):
                axs[i].set_ylabel(r'$X_{' + str(i) +'}$')
                # axs[i].set_ylim(-1,2)
            
            fig.add_subplot(111, frameon=False)      
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel(r'$t$')               
                
            plt.tight_layout()
            plt.show()
        
        makeplot(X)
        makeplot(X[draw])
        
        return X[draw]
        