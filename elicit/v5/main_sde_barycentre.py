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

# #%%
# mu = []
# mu.append(lambda t,x : torch.cat((-2*x[:,0].reshape(-1,1),
#                                     5*(1-x[:,1]).reshape(-1,1)), axis=1) )

# mu.append(lambda t,x : torch.cat((2*(x[:,1]-x[:,0]).reshape(-1,1),
#                                     5*(1-x[:,1]).reshape(-1,1)), axis=1) )

# mu.append(lambda t,x : torch.cat((2*(x[:,1]-x[:,0]).reshape(-1,1),
#                                     5*(-1-x[:,1]).reshape(-1,1)), axis=1) )

# pi = [0.25, 0.5, 0.25]

# sigma = lambda t,x : torch.cat( (0.25*torch.ones(x.shape[0],1),
#                                   1.0*torch.ones(x.shape[0],1)), axis=1)

# rho = torch.ones(2,2)
# rho[0,1] = -0.5
# rho[1,0] = -0.5

# model = sde_barycentre(mu, sigma, rho, pi, Ndt=101)

# X = model.simulate(256)
# model.plot_sample_paths()
# model.train(batch_size=256, n_print=100, n_iter =1_000)


#%%
mu = []
# mu.append(lambda t,x : torch.cat(( (-0.5*x[:,1]**2).reshape(-1,1),
#                                     5*(0.2**2-x[:,1]).reshape(-1,1)), axis=1) )

# mu.append(lambda t,x : torch.cat(( (-0.5*x[:,1]**2).reshape(-1,1),
#                                     4*(0.3**2-x[:,1]).reshape(-1,1)), axis=1) )

# mu.append(lambda t,x : torch.cat(( (-0.5*x[:,1]**2).reshape(-1,1),
#                                     3*(0.4**2-x[:,1]).reshape(-1,1)), axis=1) )


mu.append(lambda t,x: torch.cat( ( 3*(x[...,0]-x[...,1]).unsqueeze(-1),
                                  5*(1-x[...,1]).unsqueeze(-1)  ), axis=-1 ) )

mu.append(lambda t,x: torch.cat( ( 3*(x[...,0]-x[...,1]).unsqueeze(-1),
                                5*(0.5-x[...,1]).unsqueeze(-1)  ), axis=-1 ) )

pi = [0.2, 0.6, 0.2]

# the sigma is not outputting the correct dimesions when x is a nxnxd -- only works with nxd
print("the sigma is not outputting the correct dimesions when x is a nxnxd -- only works with nxd")
# sigma = lambda t,x : torch.cat((torch.sqrt(torch.abs(x[:,1])+1e-6).reshape(-1,1),
#                                 0.5*torch.sqrt(torch.abs(x[:,1])+1e-6).reshape(-1,1)), axis=1)

sigma = lambda t,x: 0.1*torch.ones(x.shape)


X0 = torch.tensor([0, 0.3**2])

rho = torch.ones(2,2)
rho[0,1] = -0.5
rho[1,0] = -0.5

model = sde_barycentre(X0, mu, sigma, rho, pi, Ndt=101)

X = model.simulate(256)
model.plot_sample_paths()
model.train(batch_size=256, n_print=100, n_iter =1_000)
