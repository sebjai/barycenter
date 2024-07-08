# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:47:12 2023

@author: sebja
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')


from neural_sde import neural_sde
from fda import fda
import pdb
import dill
import copy

#%% load data
# Reload the array
loaded_flattened_array = np.loadtxt('AMZN.csv', delimiter=',')

# Reshape the 2D array back to the original 3D shape
tau = torch.tensor([0.02739726, 0.08219178, 0.16438356, 0.24931507, 0.33424658, 0.41643836, 0.49863014, 0.74794521, 1.        , 1.49863014, 2.        ]).float()
delta = torch.linspace(0.1,0.9,17).float()
iv = torch.tensor(loaded_flattened_array.reshape((2893, 11, 17))).float()
iv = iv[:,1:,:]

iv_mean = torch.mean(iv)
iv_std = torch.std(iv)

iv -= torch.mean(iv)
iv /= torch.std(iv)

#%%
for i in range(10):
    plt.plot(iv[-i,2,:].numpy())
    
plt.show()

#%%
fda_model = fda(nbasis=5, data=iv[:,2,:],delta=delta)
fda_model.plot_basis()
fda_model.plot_fits()

#%%
mean_a = torch.mean(fda_model.a, axis=0)
std_a = torch.std(fda_model.a, axis=0)

z = (fda_model.a - mean_a)/std_a

#%%
fda_model_full = {'fda' : fda_model, 
                  'delta' : delta,
                  'tau' : tau,
                  'mean' : mean_a, 
                  'std' : std_a, 
                  'iv_mean' : iv_mean,
                  'iv_std' : iv_std,
                  'z' : z}


#%%
sde_model = neural_sde(z.shape[1])
sde_model.train(z, batch_size=128, n_print=500, n_iter=2_000)
dill.dump(sde_model,open('model_all.pkl','wb'))


#%%
idx = [0,964,1928,-1]

sde_models = []
for i in range(len(idx)-1):

    sde_models.append(copy.deepcopy(sde_model))
    
    sde_models[-1].reset_optim_sched()
    sde_models[-1].train(z[idx[i]:idx[i+1],:],
                         batch_size=128,
                         n_print=500, n_iter=2_000, 
                         train_nu_only=True)
    dill.dump(sde_models[-1],open('model_' +str(i+1) + '.pkl','wb'))