# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 00:16:49 2024

@author: sebja
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sde_barycentre import sde_barycentre 
import pdb
import torch

#%%
SMALL_SIZE = 12
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#%%
mu = []
# mu.append(lambda t,x : torch.cat(( (-0.5*x[...,1]**2).unsqueeze(-1),
#                                     5*(0.2**2-x[...,1]).unsqueeze(-1)), axis=-1) )

# mu.append(lambda t,x : torch.cat(( (-0.5*x[...,1]**2).unsqueeze(-1),
#                                     4*(0.3**2-x[...,1]).unsqueeze(-1)), axis=-1) )

# mu.append(lambda t,x : torch.cat(( (-0.5*x[...,1]**2).unsqueeze(-1),
#                                     3*(0.4**2-x[...,1]).unsqueeze(-1)), axis=-1) )

mu.append(lambda t,x : torch.cat(( 2*((0.1*torch.sin(2*torch.pi*t[...,0])-x[...,0])).unsqueeze(-1),
                                    5*(0.2-x[...,1]).unsqueeze(-1)), axis=-1) )

mu.append(lambda t,x : torch.cat(( 3*((0.2*torch.sin(3*torch.pi*t[...,0])-x[...,0])).unsqueeze(-1),
                                    4*(0.3-x[...,1]).unsqueeze(-1)), axis=-1) )

mu.append(lambda t,x : torch.cat(( 4*((0.3*torch.sin(4*torch.pi*t[...,0])-x[...,0])).unsqueeze(-1),
                                    3*(0.4-x[...,1]).unsqueeze(-1)), axis=-1) )



sigma = lambda t,x : torch.cat((torch.abs(x[...,1]).unsqueeze(-1),
                                0.5*torch.abs(x[...,1]).unsqueeze(-1)), 
                               axis=-1)

# sigma = lambda t, x : 0.2*torch.ones(x.shape).to(model.dev) + 1e-20*x


f= []
# f.append(lambda x : x[...,0].unsqueeze(-1)-0.165 )
# # f.append(lambda x : (x[...,0]*x[...,1]).unsqueeze(-1)-0.1 )
# f.append(lambda x : torch.sqrt(x[...,1].unsqueeze(-1))-0.3)
g = []


X0 = torch.tensor([0, 0.3])

rho = torch.ones(2,2)
rho[0,1] = -0.5
rho[1,0] = -0.5

pi = [0.6, 0.2, 0.2]

model = sde_barycentre(X0, mu, sigma, rho, pi, 
                       f=f, g=g, T=1, Ndt=101)
model.plot_sample_paths()

# %%
model.train(batch_size=512, n_print=1000, n_iter= 20_000)