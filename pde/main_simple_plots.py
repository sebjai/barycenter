# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:08:25 2023

@author: sebja
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

t = np.linspace(0,1,1001)

def Sim(t, mu, nsims):
    
    x = np.zeros((nsims, len(t)))
    dt = np.diff(t)
    
    for i in range(len(t)-1):
        dW = np.sqrt(dt[i]) * np.random.normal(0, 1, nsims)
        x[:,i+1] = x[:,i] + mu * dt[i] + dW
        
    return x

def plot_expert(mu, name):
    
    x_sim = Sim(t, mu, 10_000)
    plt.plot(t, x_sim[:500,:].T, alpha=0.1, linewidth=1)
    qtl = np.quantile(x_sim, [0.1, 0.5, 0.9], axis=0)
    plt.plot(t, qtl.T, color='k')
    plt.plot(t, x_sim[0,:], linewidth=1, color='b')
    plt.ylim(-3,3)
    plt.savefig( name +'.pdf', format='pdf')
    plt.show()
    
plot_expert(0, 'Bmtn-expert-1')
plot_expert(1, 'Bmtn-expert-2')