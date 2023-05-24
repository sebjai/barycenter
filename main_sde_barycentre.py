# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:47:12 2023

@author: sebja
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

from sde_barycentre import sde_barycentre 

#%%
mu = []
mu.append(lambda t,x : torch.cat((-2*x[:,0].reshape(-1,1),
                                    5*(1-x[:,1]).reshape(-1,1)), axis=1) )

mu.append(lambda t,x : torch.cat((2*(x[:,1]-x[:,0]).reshape(-1,1),
                                    5*(1-x[:,1]).reshape(-1,1)), axis=1) )

mu.append(lambda t,x : torch.cat((2*(x[:,1]-x[:,0]).reshape(-1,1),
                                    5*(-1-x[:,1]).reshape(-1,1)), axis=1) )

pi = [0.6, 0.2, 0.2]

sigma = lambda t,x : torch.cat( (0.25*torch.ones(x.shape[0],1),
                                 1.0*torch.ones(x.shape[0],1)), axis=1)

rho = torch.ones(2,2)
rho[0,1] = -0.5
rho[1,0] = -0.5

model = sde_barycentre(mu, sigma, rho, pi, Ndt=101)

X = model.simulate(256)
model.plot_sample_paths()
model.__estimate_L__()