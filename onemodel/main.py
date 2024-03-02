# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:47:12 2023

@author: sebja
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

from KL_model import KL_model
from generator import generator


#%%
gen = generator(Ndt = 100, T=1)
x = gen.simulate(10_000)

qtl = np.quantile(x, [0.1, 0.5, 0.9], axis=0)
plt.fill_between(gen.t, qtl[0,:], qtl[2,:], color='blue', alpha=0.2)
plt.plot(gen.t, x[:100,:].T, alpha = 0.2)
plt.plot(gen.t, qtl[0,:], color='k')
plt.plot(gen.t, qtl[1,:], color='k')
plt.plot(gen.t, qtl[2,:], color='k')
plt.show()

#%%
x0 = 0
f = lambda x : (x**2 - 2)

model = KL_model(f, x)

X = model.simulate(256)
model.plot_sample_paths()
model.train(batch_size=256, n_print=100, n_iter =1_000)
