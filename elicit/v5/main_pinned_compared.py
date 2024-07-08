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
import copy
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
if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
dev = torch.device(dev)

mu = []
# mu.append(lambda t, x: -2*x)
mu.append(lambda t, x: (4*t-0.7*x))
mu.append(lambda t, x: 3*(t+torch.sin(4*torch.pi*t+torch.pi/12)-x))

sigma = lambda t, x : 1*torch.ones(x.shape).to(dev) + 1e-20*x

f = []
g = []

f.append(lambda x : (x-0.5))
f.append(lambda x : (x**2-(0.05+0.5**2)))

# f.append(lambda x : 1*(x>0.8)*(x<1.2) - 0.95)
# g.append(lambda t, x: 1*(x<t)-0.2)
# I = lambda x, a : torch.sigmoid((x-a)/0.001)
# f.append(lambda x : (1-I(x,1.2))*I(x,0.8) - 0.95)
# g.append(lambda t, x: (1-I(x,t))-0.2)

X0 = torch.tensor([0])
rho = torch.ones(1,1)

# pi_all = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

pi = [0.5, 0.5]

model = sde_barycentre(X0, mu, sigma, rho, pi, 
                       f=f, g=g, T=1, Ndt=501)
# X = model.simulate(256)

# model.plot_sample_paths()
#%%
n_print=250

model.generate_training_batch(100_000)
model.find_eta(100_000)

rng_state= {'np': np.random.get_state(), 
            'torch': torch.get_rng_state()}


models = [copy.deepcopy(model)]
models.append(copy.deepcopy(model))
models.append(copy.deepcopy(model))

model = []
torch.cuda.empty_cache()


# models[0].train(batch_size=1024, n_print=n_print, n_iter= 29542, rule="L2" )
# models[1].train(batch_size=1024, n_print=n_print, n_iter= 29542, rule="Poisson" )


models[0].train(batch_size=512, n_print=n_print, n_iter= 30_000, rule="L2", rng_state=rng_state)
models[1].train(batch_size=512, n_print=n_print, n_iter= 30_000, rule="Poisson", rng_state=rng_state )
models[2].train(batch_size=512, n_print=n_print, n_iter= 30_000, rule="Gamma", rng_state=rng_state )


dill.dump(models, open("learnt_models_101.pkl","wb"))

#%% plot model error estimators
def plot_err(err, symb):
    for k in range(err.shape[1]):
        # plt.errorbar(model.i, err[:,k,0], 3*err[:,k,1], label=r'$\mathbb{E}[' + symb + '_' + str(k) + ']$')
        plt.plot(model.i, err[:,k,0], label=r'$\mathbb{E}[' + symb + '_' + str(k) + ']$')
        plt.fill_between(model.i, err[:,k,0]-3*err[:,k,1], err[:,k,0]+3*err[:,k,1], alpha=0.2)
    plt.axhline(0,linestyle='--', color='k')
    plt.legend()
    plt.xlabel('iteration')
    plt.xscale('log')
    plt.xticks([1,10,100,1000,10000,20000])
    plt.xlim(model.i[0],model.i[-1])

for j, model in enumerate(models):
    plot_err(np.array(model.f_err), 'F^'+str(j))
    if len(g) > 0:
        plot_err(np.array(model.g_err), 'G')
        
plt.show()