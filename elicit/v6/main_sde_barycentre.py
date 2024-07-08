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

mu.append(lambda t,x: torch.cat( ( 5*(x[...,1]-x[...,0]).unsqueeze(-1), 
                                10*(2.0-x[...,1]).unsqueeze(-1) ), axis=-1   ) )

def mu2(t,x):
    
    if len(t.shape) ==0:
        t = t*torch.ones(x.shape).to(model.dev)
    out = torch.cat( ( 5*(x[...,1]-x[...,0]).unsqueeze(-1),
                      5*(0.5*torch.sin(torch.pi+2*torch.pi*t[...,0])+0.5-x[...,1]).unsqueeze(-1) ), axis=-1   )

    return out
    
mu.append(lambda t,x: mu2(t,x) )

pi = [0.4, 0.6]

sigma = lambda t,x: torch.cat((0.8*torch.ones(x[...,0].shape).to(model.dev).unsqueeze(-1),
                               2*torch.ones(x[...,1].shape).to(model.dev).unsqueeze(-1)), axis=-1 )


X0 = torch.tensor([0.75, 0.75])

rho = torch.ones(2,2)
rho[0,1] = -0.5
rho[1,0] = -0.5

f = []
f.append(lambda x : 0.5*(x[...,0]+x[...,1]).unsqueeze(-1) - 1.2)
f.append(lambda x : (0.5*(x[...,0]+x[...,1]).unsqueeze(-1) )**2 - (0.05+1.2**2))

g=[]

model = sde_barycentre(X0, mu, sigma, rho, pi, f=f, g=g, Ndt=501)

model.plot_sample_paths()

#%%
model.generate_training_batch(100_000)
model.find_eta(100_000)

print_at_list = list(np.unique(np.array(np.ceil(1.21**(np.arange(0,58))-1), int)))
# print_at_list = list(np.unique(np.arange(0,100_000,5000)))
print_at_list.sort(reverse=True)

model.train(batch_size=1024, print_at_list=print_at_list, n_iter =40_000)

#%% plot model error estimators
def plot_err(err, symb):
    for k in range(err.shape[1]):
        # plt.errorbar(model.i, err[:,k,0], 3*err[:,k,1], label=r'$\mathbb{E}[' + symb + '_' + str(k) + ']$')
        plt.plot(model.i, err[:,k,0], label=r'$\mathbb{E}^{\mathbb{Q}}[' + symb + '_' + str(k) + ']$')
        plt.fill_between(model.i, err[:,k,0]-3*err[:,k,1], err[:,k,0]+3*err[:,k,1], alpha=0.2)
    plt.axhline(0,linestyle='--', color='k')
    plt.legend()
    plt.xlabel('iteration')
    plt.xscale('log')
    # plt.xticks([1,10,100,1000,10000,20e3,40e3])
    plt.savefig('constraint.pdf',format='pdf',bbox_inches='tight')
    plt.show()

plot_err(np.array(model.f_err), 'F')

if len(g) > 0:
    plot_err(np.array(model.g_err), 'G')
    
model.plot_sample_paths(batch_size=10_000, filename='optimal.sim')