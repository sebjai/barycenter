# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:47:44 2024

@author: jaimunga
"""

from neural_sde import neural_sde
from sde_barycentre import sde_barycentre

import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

import dill

from typing import List
import pdb



#%% load FDA model
fda_model_full = dill.load(open('fda_model_full.pkl','rb'))

#%% 
fig, axs = plt.subplots(fda_model_full['z'].shape[1], 1,
                        figsize=(6,5), sharex=True)

for i in range(fda_model_full['z'].shape[1]):
    axs[i].plot(fda_model_full['z'][:,i], linewidth=0.5)
    axs[i].set_ylabel(r'$x_{t,' + str(i+1) +'}$', fontsize=18)

axs[-1].set_xlabel(r'$t$', fontsize=18)
plt.tight_layout()
plt.savefig('x.pdf',format='pdf',bbox_inches='tight')
plt.show()
#%%
# vector: List[float] = list()
sde_model : List[neural_sde] = list()
for k in range(3):
    filename = 'model_' +str(k+1) +'.pkl'
    print(filename)
    sde_model.append(dill.load(open(filename,'rb')))
    sde_model[-1].plot_pits()
    
#%%
# rng_state = torch.get_rng_state()
# for k in range(len(sde_model)):
#     torch.set_rng_state(rng_state)
#     t, x = sde_model[k].plot_sim(x0 = fda_model_full['z'][-1,:].to(sde_model[k].dev),
#                                   batch_size=1024,
#                                   T=1/12)
#     print(x[:2,:,:5], "***\n\n")

#%%
mu = []
for k in range(len(sde_model)):
    mu.append(lambda t, x, k=k: sde_model[k].nu['net'](x).detach())
sigma = lambda t, x: sde_model[0].sigma['net'](x).detach()

#%% impose constraint of 2nd derivative is 0.1


x = 0.5*(fda_model_full['fda'].x_fine+1)
dx = torch.diff(fda_model_full['fda'].x_fine)[0]

idx = np.where(np.abs(x-0.5)<0.5*dx)[0][0]

# find d_delta \phi(delta)
d_phi = (fda_model_full['fda'].phi_fine[2:,:]
          -fda_model_full['fda'].phi_fine[:-2,:])/(2*dx)


d_phi = d_phi[idx].to(sde_model[0].dev)

# find d2_delta \phi(delta)
d2_phi = (fda_model_full['fda'].phi_fine[2:,:]
          -2*fda_model_full['fda'].phi_fine[1:-1,:]
          +fda_model_full['fda'].phi_fine[:-2,:])/(dx**2)


d2_phi = d2_phi[idx].to(sde_model[0].dev)


fda_model_full['mean'] = fda_model_full['mean'].to(sde_model[0].dev)
fda_model_full['std'] = fda_model_full['std'].to(sde_model[0].dev)
fda_model_full['iv_std'] = fda_model_full['iv_std'].to(sde_model[0].dev)
fda_model_full['iv_mean'] = fda_model_full['iv_mean'].to(sde_model[0].dev)

def test1(x):
    
    F = (torch.sum(d_phi*(fda_model_full['mean'] 
                          +fda_model_full['std']*x),axis=1)
         *fda_model_full['iv_std'])
    
    result = (F-0.05).unsqueeze(-1)    
    
    return result

def test2(x):
    
    F = (torch.sum(d_phi*(fda_model_full['mean'] 
                          +fda_model_full['std']*x),axis=1)
         *fda_model_full['iv_std'])
    
    result = (F**2-(0.01+0.05**2)).unsqueeze(-1)    
    
    return result

# def test2(x):
    
#     result = ((torch.sum(d2_phi*(fda_model_full['mean']
#                                        +fda_model_full['std']*x),axis=1)
#                      *fda_model_full['iv_std'])-0.0).unsqueeze(-1)
    
#     return result


# def test3(x):
#     result = ( 1*((torch.sum(d_phi*(fda_model_full['mean']
#                                        +fda_model_full['std']*x),axis=1)
#                      *fda_model_full['iv_std'])<0)-0.8 ).unsqueeze(-1)    
    
#   return result

# f = [ lambda x : test3(x) ]

# f = [lambda x : test1(x), 
#      lambda x : test2(x) ]

f = [lambda x : test1(x)]


#%%
X0 = fda_model_full['z'][-1,:].to(sde_model[0].dev)
Ndt=31
dt = sde_model[0].dt

bary_model = sde_barycentre(X0=X0,
                            mu = mu,
                            sigma = sigma,
                            pi=[0.2, 0.2, 0.6], 
                            f=f, g=[], 
                            T=Ndt*dt, Ndt=Ndt)

t, X = bary_model.plot_sample_paths()

bary_model.find_eta(10_000)
bary_model.generate_training_batch(10_000)

#%%
print_at_list = list(np.arange(0,51)*1000)
print_at_list.sort(reverse=True)

bary_model.train(batch_size=512, n_iter=50_001, print_at_list=print_at_list)



#%%
t, X = bary_model.plot_sample_paths(batch_size=10_000, 
                                    filename='smile_all_models.pdf')

#%%
def get_iv(j):
    
    iv = []

    fig = plt.figure()
    ax = plt.subplot(111)            
    
    for k in range(X.shape[-1]):
        a = fda_model_full['mean']\
            + torch.tensor(X[plt_idx,j,:,k]).float().to(sde_model[0].dev)*fda_model_full['std']
        
        iv.append( fda_model_full['fda'].fit(a.cpu()).to(sde_model[0].dev) )
        
        iv[-1] = fda_model_full['iv_mean'] + iv[-1]*fda_model_full['iv_std']
        
        if k < X.shape[-1]-2:
            label = '$\mathbb{P}_{'+ str(k+1) + '}$'
        elif k == X.shape[-1]-2:
            label = '$\overline{\mathbb{Q}}$'
        elif k == X.shape[-1]-1:
            label = '${\mathbb{Q}}^*$'
            
        ax.plot(x, iv[-1].cpu().numpy(), label=label)
        
        if k == X.shape[-1]-1:
        
            a = fda_model_full['mean']\
                + torch.tensor(X[:,j,:,k]).float().to(sde_model[0].dev)*fda_model_full['std']
                
            iv_all = fda_model_full['fda'].fit(a.cpu()).to(sde_model[0].dev)
            iv_all = fda_model_full['iv_mean'] + iv_all*fda_model_full['iv_std']
            
            qtl = torch.nanquantile(iv_all.cpu(),torch.tensor([0.1,0.9]), axis= 0).squeeze().numpy()
            ax.fill_between(x, qtl[0], qtl[1], alpha=0.2)

    ax.set_title('$t={0:0.3f}$'.format(t[j]))     
    ax.set_xlabel(r'$\Delta$', fontsize=18)
    ax.set_ylim(0.2, 0.45)
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig('smile_path_' + str(j+1) +'.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    return iv

plt_idx = 0

for j in range(X.shape[1]):
    get_iv(j)
