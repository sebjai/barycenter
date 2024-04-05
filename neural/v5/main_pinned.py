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
def plot_mu(model, filename):
    tm, xm = np.meshgrid(model.t,model.x)
    plt.contour(tm, xm, mu[0](tm,xm), cmap='Blues', levels=10)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.show()
    
    
    tm, xm = np.meshgrid(model.t,model.x)
    plt.contour(tm, xm, mu[1](tm,xm), cmap='Reds', levels=10)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.show()
    
    
    tm, xm = np.meshgrid(model.t,model.x)
    plt.contour(model.t, model.x, model.mu_opt.T, levels=10)
    plt.contour(tm, xm, mu[0](tm,xm), cmap='Blues', alpha=0.5, linestyles='dashed', levels=10)
    plt.contour(tm, xm, mu[1](tm,xm), cmap='Reds', alpha=0.5, linestyles='dashed', levels=10)
    plt.xlabel(r'$t$', fontsize=14)
    plt.ylabel(r'$x$', fontsize=14)
    
    plt.savefig(filename +'_mu.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    def plot_mu_3d(tm, xm, mu_m, title=None):
        
        fig = plt.figure()
        
        ax = plt.axes(projection ='3d')
        ax.plot_surface(tm, xm, mu_m, cmap='jet')
        ax.view_init(elev=10., azim=60)
        ax.set_box_aspect(aspect=None, zoom=0.8)
        
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(r"$\mu^*(t,x)$", rotation=90)
    
        plt.xlim(0,1)
        plt.ylim(-1,3)
        
        if title is not None:
            plt.savefig(title +'_mu3d.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
    tm, xm = np.meshgrid(model.t,np.linspace(-1,3,101))
    
    plot_mu_3d(tm, xm, mu[0](tm,xm), title='ref_0')
    plot_mu_3d(tm, xm, mu[1](tm,xm), title='ref_1')
    
    f = interpolate.interp2d(model.t, model.x, model.mu_opt.T, kind='cubic')
    mu_opt = np.zeros(tm.shape)
    for i in range(tm.shape[0]):
        mu_opt[i,:] = f(tm[i,:],xm[i,0])
        
    plot_mu_3d(tm, xm, mu_opt, title=filename)
    

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
f.append(lambda x : 1*(x>0.8)*(x<1.2) - 0.95)
# g.append(lambda t, x: 1*(x<t)-0.2)
# I = lambda x, a : torch.sigmoid((x-a)/0.001)
# f.append(lambda x : (1-I(x,1.2))*I(x,0.8) - 0.95)
# g.append(lambda t, x: (1-I(x,t))-0.2)

X0 = torch.tensor([0])
rho = torch.ones(1,1)

# pi_all = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

pi = [0, 1]

model = sde_barycentre(X0, mu, sigma, rho, pi, 
                       f=f, g=g, T=1, Ndt=501)
X = model.simulate(256)

model.plot_sample_paths()
model.train(batch_size=512, n_print=10_000, n_iter_omega= 50_000 )