# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:33:44 2024

@author: jaimunga
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from tqdm import tqdm


# neural network to model the drift of the SDE
class drift_net(nn.Module):

    def __init__(self, nIn, nHidden, nLayers, nOut, dev):
        super(drift_net, self).__init__()

        self.nIn = nIn
        self.nHidden = nHidden
        self.nLayers = nLayers
        self.nOut = nOut
        
        self.dev = dev

        self.in_to_hidden = nn.Linear(nIn, nHidden).to(self.dev)
        self.hidden_to_hidden = nn.ModuleList([ nn.Linear(nHidden, nHidden).to(self.dev) 
                                               for i in range(nLayers)])
        self.hidden_to_out = nn.Linear(nHidden, nOut).to(self.dev)
            
        self.g = nn.SiLU()

    def forward(self, x):

        h = self.g(self.in_to_hidden(x))
        for prop in self.hidden_to_hidden:
            h = self.g(prop(h))
        
        nu = 5*torch.tanh(self.hidden_to_out(h))
        
        return nu
 
#%%
# neural network to model the diffusion of the SDE
class sigma_net(nn.Module):

    def __init__(self, nIn, nHidden, nLayers, nOut, dev):
        super(sigma_net, self).__init__()

        self.nIn = nIn
        self.nHidden = nHidden
        self.nLayers = nLayers
        self.nOut = nOut

        self.dev = dev

        # GRU layers of the diffusion neural network
        # feed-forward neural network for mapping hidden states of the GRU at the last time step 
        # into Cholesky decomposition of diffusion
        
        self.in_to_hidden = nn.Linear(nIn, nHidden).to(self.dev)
        self.hidden_to_hidden = nn.ModuleList([ nn.Linear(nHidden, nHidden).to(self.dev) 
                                               for i in range(nLayers)])
        self.hidden_to_L = nn.Linear(nHidden, int(nOut * (nOut + 1) / 2)).to(self.dev)
        
        
        self.hidden_to_L.bias.data.fill_(0.01)
        self.hidden_to_L.weight.data.uniform_(-0.001,0.001)
        
        self.tril_indices = torch.tril_indices(row=self.nOut, col=self.nOut, offset=0).to(self.dev)
        
        self.g = nn.SiLU()

    def forward(self, x):

        h = self.g(self.in_to_hidden(x))
        for prop in self.hidden_to_hidden:
            h = self.g(prop(h))
            
        h = 5*torch.tanh(self.hidden_to_L(h))
        
        # following steps generate the diffusion by getting its Cholesky decomposition L first
        L = torch.zeros((h.shape[0], self.nOut, self.nOut)).to(self.dev)

        L[:, self.tril_indices[0], self.tril_indices[1]] = h

        I = torch.zeros(L.shape).to(self.dev)
        rng = range(L.shape[1])
        I[:, rng, rng] = 1

        # L L' + eps**2 * I
        Sigma = torch.matmul(L, torch.transpose(L, 1, 2)) + 1e-2 * I

        return Sigma

class neural_sde():
    
    def __init__(self, nDim=1, nHidden=36, nLayers=5, dt=1.0/365.0):
        
        self.nDim=nDim
        self.nHidden=nHidden
        self.nLayers=nLayers
        
        if torch.cuda.is_available(): 
            dev = "cuda:0" 
        else: 
            dev = "cpu" 
        self.dev = torch.device(dev)
        
        self.initialize_nets()
        
        self.dt = dt
        
        self.nLL = []
        self.ploss = []
        self.loss = []
        
        self.Z = torch.distributions.Normal(0, 1)
        
    
    def get_optim_sched(self, net):
        
        optimizer = optim.Adam(net.parameters(), lr=0.005)
        sched = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=5,
                                          gamma=0.9999)
        return optimizer, sched
        
    def initialize_nets(self):
        
        self.nu = {'net' : drift_net(nIn=self.nDim, nHidden=self.nHidden, nLayers=self.nLayers, nOut=self.nDim, 
                                   dev=self.dev),
                   'optimizer' : [],
                   'scheduler' : []}
        
        self.nu['optimizer'], self.nu['scheduler'] = self.get_optim_sched(self.nu['net'])

        self.sigma = {'net' : sigma_net(nIn=self.nDim, nHidden=self.nHidden, nLayers=self.nLayers, nOut=self.nDim, 
                                      dev=self.dev),
                      'optimizer' : [],
                      'scheduler' : []}
        self.sigma['optimizer'], self.sigma['scheduler'] = self.get_optim_sched(self.sigma['net'])
        
    def reset_optim_sched(self):
        
        self.nu['optimizer'], self.nu['scheduler'] = self.get_optim_sched(self.nu['net'])
        
    def log_likelihood(self, x, xp):
        
        
        nu = self.nu['net'](x)*self.dt
        sigma = self.sigma['net'](x) * self.dt
    
        sigma_inv = torch.linalg.inv(sigma) 
        det = torch.linalg.det(sigma)
        
        z = ((xp-x) - nu)
        
        L = torch.mean(-0.5*torch.einsum("...ij,...ijk,...ik->...i", z, sigma_inv, z)
                       -0.5*torch.log(det) )
        
        return L
    
    def grab_batch(self, batch_size=256):
        
        # idx = torch.randint(0, self.N-1, (batch_size,))

        x = self.data[:-1,...]
        xp = self.data[1:,...]
        
        
        # idx = torch.randint(0,self.N-1-batch_size,(1,))
        
        # x = self.data[idx:idx+batch_size,...]
        # xp = self.data[idx+1:idx+batch_size+1,...]
        
        return x, xp
    
    
    def train_nu_ll(self, n_iter=1_000, batch_size=256, progress=False, sched_step=True):
        
        if progress == False:
            idx = range(n_iter)
        else:
            idx = tqdm(range(n_iter))

        for i in idx:
            
            x, xp = self.grab_batch(batch_size)
            
            nLL = -self.log_likelihood(x, xp)
            pit_loss = self.pit_loss(x, xp)
            
            loss = nLL + self.eta*pit_loss
            
            self.nu['optimizer'].zero_grad()
            
            loss.backward()
            
            self.nu['optimizer'].step()
            self.nu['scheduler'].step()
            
            self.nLL.append(nLL.item())
            self.ploss.append(pit_loss.item())
            

            
    def train_nu(self, n_iter=1_000, batch_size=256, progress=False, sched_step=False):
        
        if progress == False:
            idx = range(n_iter)
        else:
            idx = tqdm(range(n_iter))
        for i in idx:
            
            x, xp = self.grab_batch(batch_size)
            
            # loss = - self.log_likelihood(x, xp)
            nu = self.nu['net'](x)
            loss = torch.mean(((xp-x) - nu*self.dt)**2)
            
            self.sigma['optimizer'].zero_grad()
            self.nu['optimizer'].zero_grad()
            
            loss.backward()
            
            self.nu['optimizer'].step()
            if sched_step:
                self.nu['scheduler'].step()
            
            # self.nll.append(L.item())
        

    def train_sigma(self, n_iter=1_000, batch_size=256):
        
        for i in range(n_iter):
            
            x, xp = self.grab_batch(batch_size)
            
            L = -self.log_likelihood(x, xp)
            
            self.sigma['optimizer'].zero_grad()
            
            L.backward()
            
            self.sigma['optimizer'].step()
            self.sigma['scheduler'].step()
            
            self.nLL.append(L.item())
            
    def train_all(self, n_iter=1_000, batch_size=256):

        for i in range(n_iter):
            
            x, xp = self.grab_batch(batch_size)
            
            nLL = -self.log_likelihood(x, xp)
            pit_loss = self.pit_loss(x, xp)
            
            loss = nLL + self.eta * pit_loss
            
            self.sigma['optimizer'].zero_grad()
            self.nu['optimizer'].zero_grad()
            
            loss.backward()
            
            self.sigma['optimizer'].step()
            self.sigma['scheduler'].step()
            
            self.nu['optimizer'].step()
            self.nu['scheduler'].step()
            
            self.nLL.append(nLL.item())
            self.ploss.append(pit_loss.item())
            self.loss.append(loss.item())
        
    def train(self, data, batch_size=512, 
              n_iter=10_000, n_print=10, 
              train_nu_only=False):
        
        self.data = data.to(self.dev)
        self.N = self.data.shape[0]
        self.d = self.data.shape[1]
        
        self.plot_loss()
        self.plot_predictions(-200,50)
        self.plot_pits()
        
        # if not train_nu_only:
        #     print("initial burn in of nu...")
        #     self.train_nu(n_iter=20_000, batch_size=batch_size, progress=True, sched_step=False)
        #     print("...done burn in")
        
        #     self.plot_loss()
        #     self.plot_predictions(-200,50)
        #     self.plot_pits()
        
        self.eta  = 1
        self.etas = [self.eta]
        
        for i in tqdm(range(n_iter)):
            
            if train_nu_only:
                self.train_nu_ll(n_iter=1, batch_size=batch_size)
            else:
                self.train_all(n_iter=1, batch_size=batch_size)
            
            ploss = np.mean(self.ploss[-100:])
            nll = np.mean(self.nLL[-100:])
            
            self.eta = self.eta * 0.995 + 0.005 * np.abs(nll/ploss)
            
            self.etas.append(self.eta)
            
            if np.mod(i,n_print) == 0:
                print(self.nu['net'].hidden_to_out.weight[0][:5])
                self.plot_loss()
                self.plot_predictions(-200,50)
                self.plot_pits()
                
                print("\n\n*******")
                for p in self.nu['net'].parameters():
                    print(p.grad)
                # print("\n\n>>>")
                # for p in self.sigma['net'].parameters():
                #     print(p.grad)
                
    def plot_pits(self):
        
        u = self.pit(self.data[:-1,:], self.data[1:,:]).detach()
        for k in range(5):
            plt.hist(u[:,k].cpu().numpy(), bins=np.linspace(0,1,51), alpha=0.2)
        plt.show()
        
    def pit_loss(self, x, xp):
        
        u = self.pit(x, xp)
        eps = 0.01
        x = torch.linspace(0,1,101).to(self.dev)
        
        f = torch.sum( torch.exp(self.Z.log_prob( (x.reshape(1,1,-1)-u.unsqueeze(axis=-1)) /eps ))/eps/u.shape[0], axis=0 )
        
        loss =  torch.mean((f - 1)**2)        
        
        return loss

    def pit(self, x, xp):
        
        u = torch.zeros(x.shape).to(self.dev)
        
        nu = self.nu['net'](x).squeeze()
        sigma = self.sigma['net'](x).squeeze()
        
        z = ( (xp-x) - nu*self.dt)/(torch.diagonal(sigma, dim1 = -2, dim2 = -1)*np.sqrt(self.dt))
        
        u = self.Z.cdf(z)        
            
        return u
                
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

        def make_plot(x, ax):
            x = np.array(x)
            ma, ma_err = self.moving_average(x, 100)
            
            ax.fill_between(np.arange(len(ma)), ma-2*ma_err, ma+2*ma_err, color='r', alpha=0.2)
            ax.plot(x, alpha=0.5)
            ax.plot(ma)
            
            if np.min(x)<0:
                ax.set_yscale('symlog')        
            else:
                ax.set_yscale('log')        
        
        if len(self.nLL) > 0:
            fig, ax = plt.subplots(1,4)
            make_plot(self.nLL, ax[0])
            make_plot(self.ploss, ax[1])
            make_plot(self.loss, ax[2])
            make_plot(self.etas, ax[3])
            plt.tight_layout()
            plt.show()
            
    def plot_predictions(self, n_start=-10, n_steps=10):

        x_data = torch.zeros(self.d, n_steps).to(self.dev)
        
        x_pred = torch.zeros(self.d, n_steps).to(self.dev)
        x_pred[:,0] = torch.nan
        x_pred_std = torch.zeros(self.d, n_steps).to(self.dev)
        x_pred_std[:,0] = torch.nan
        
        for i in range(n_steps-1):
            
            x_data[:,i] = self.data[n_start+i,:]
            
            nu=self.nu['net'](x_data[:,i].unsqueeze(axis=0)).detach()
            sigma = self.sigma['net'](x_data[:,i].unsqueeze(axis=0)).detach()
            
            dB = torch.randn(1, self.d).to(self.dev)
            U = torch.cholesky(sigma)
            
            std = np.sqrt(self.dt)*torch.matmul(dB,U)
            
            x_pred[:,i+1] = x_data[:,i]  + nu*self.dt
            x_pred_std[:,i+1] = std
        
        x_data = x_data.cpu().numpy()
        x_pred = x_pred.cpu().numpy()
        x_pred_std = x_pred_std.cpu().numpy()

        fig, ax= plt.subplots(1,self.d, figsize=(10,3), sharex=True, sharey=True)
        
        xd = x_pred-2*x_pred_std
        xu = x_pred+2*x_pred_std
        
        for k in range(self.d):
            
            ax[k].fill_between(np.arange(n_steps), xd[k,:], xu[k,:], color='r', alpha=0.5)
            ax[k].plot(x_data[k,:], color='k')
            ax[k].plot(x_pred[k,:], color='r')
        
        plt.tight_layout()
        plt.show()
        
        
    def plot_sim(self, x0, batch_size=512, T=1):
        
        t, x = self.simulate(x0, batch_size, T)
        
        t = t.cpu().numpy()
        x = x.cpu().numpy()
        
        fig, ax= plt.subplots(1,self.d, figsize=(10,3), sharex=True, sharey=True)
        
        for k in range(self.d):
            
            qtl = np.quantile(x, [0.1, 0.5, 0.9], axis=0)
            ax[k].fill_between(t, qtl[0,k,:], qtl[-1,k,:], color='r', alpha=0.5)
            ax[k].plot(t, x[0,k,:], color='k')
            ax[k].set_ylim(-2,2)
            # ax[k].plot(x_pred[k,:], color='r')
        
        plt.tight_layout()
        plt.show()    
        
        return t, x
        
    def simulate(self, x0, batch_size=512, T=1):
        
        t = torch.linspace(0,T, int(T/self.dt)+1)
    
        x = torch.zeros(batch_size, self.d, len(t)).to(self.dev)
        x[:,:,0] = x0
        
        for i in range(len(t)-1):
            
            nu=self.nu['net'](x[:,:,i]).detach()
            sigma = self.sigma['net'](x[:,:,i]).detach()
            
            dB = torch.randn(batch_size, self.d).to(self.dev)
            
            try:
                U = torch.cholesky(sigma)
            except:
                pdb.set_trace()
                print(torch.linalg.eigvals(sigma))
            
            x[:,:,i+1] = x[:,:,i] + nu*self.dt + np.sqrt(self.dt)*torch.matmul(dB.unsqueeze(axis=1),U).squeeze()
            
        return t, x            