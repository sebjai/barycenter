# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 00:16:49 2024

@author: sebja
"""

import numpy as np
import matplotlib.pyplot as plt
from sde_barycentre import sde_barycentre 
import pdb
import torch
import dill

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
# mu.append(lambda t, x: -2*x)
mu.append(lambda t, x: (4*t-0.7*x))
mu.append(lambda t, x: 3*(t+torch.sin(4*torch.pi*t+torch.pi/12)-x))

sigma = lambda t, x : 1*torch.ones(x.shape).to(model.dev) + 1e-20*x

f = []
g = []
f.append(lambda x : 1*(x>0.8)*(x<1.2) - 0.9)
# f.append(lambda x : 0*x)


g.append(lambda t, x: 1*(x<t)-0.2)
# I = lambda x, a : torch.sigmoid((x-a)/0.001)
# f.append(lambda x : (1-I(x,1.2))*I(x,0.8) - 0.95)
# g.append(lambda t, x: (1-I(x,t))-0.2)

X0 = torch.tensor([0])
rho = torch.ones(1,1)

# pi_all = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

pi = [0.25, 0.75]

model = sde_barycentre(X0, mu, sigma, rho, pi, 
                       f=f, g=g, T=1, Ndt=101)
# save initial state
# model.theta = dill.load(open('theta0.pkl','rb'))

model.plot_sample_paths()
model.train(batch_size=1024, n_print=5_000, n_iter= 20_000)