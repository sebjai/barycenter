# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:25:58 2024

@author: jaimunga
"""

import dill
import torch
import matplotlib.pyplot as plt
import numpy as np
#%%

model = dill.load(open('learnt_1001.pkl','rb'))


#%%
state = torch.get_rng_state()
torch.manual_seed(12317874321)

X = model.simulate(10_000)

#%%
def estimate_errors(X):
    
    t = torch.tensor(model.t).float().view(1,-1,1).repeat(X.shape[0],1,1).to(model.dev)  
    
    G = []
    F = []
    for i in range(X.shape[-1]):
            
        Y = X[...,i]
        
        g_err = []
        for k in range(len(model.g)):
            g_err.append(torch.mean(torch.sum(model.dt*model.g[k](t[:,:-1,:], Y[:,:-1,:]), axis=1)).detach().cpu().numpy())
        
        f_err = []
        for k in range(len(model.f)):
            f_err.append(torch.mean(model.f[k](Y[:,-1,:])).detach().cpu().numpy())
     
        G.append(g_err)
        F.append(f_err)
        
    G = np.array(G)
    F = np.array(F)
        
    return G, F

G, F = estimate_errors(X)