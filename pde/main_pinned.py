# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 08:55:01 2023

@author: sebja
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from BCSP import BCSP
import pdb

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
mu = []
# mu.append(lambda t, x: -2*x)
mu.append(lambda t, x: (4*t-0.7*x))
mu.append(lambda t, x: 3*(t+np.sin(4*np.pi*t+np.pi/12)-x))

sigma = lambda x : 1

f = lambda x : (x<1.2)*(x>0.8) - 0.95
# g = lambda t, x: 0
g = lambda t, x: (x<t)-0.2




# pi_all = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

pi_all = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

pi_all = [0.5]

state = np.random.get_state()
n_sims = 10_000
Ndt = 501

Z = np.random.randn(Ndt, n_sims)

for a in pi_all:
    
    pi = [a, 0]
    pi[1] = 1-pi[0]
    model = BCSP(mu, sigma, pi, f=f, g=g, Z=Z, Ndt=Ndt)
    
    eta = model.FindOptimalEta()
    
    print("\n\n******")
    print('eta=', eta)
    
    f_err = interpolate.interp1d(model.x, model.QExpectation(eta)[0][0,:])
    g_err = interpolate.interp1d(model.x, model.QExpectation(eta)[1][0,:])
    print('E[F]=',f_err(0) )
    print('$E[\int_0^T g_s\,ds]$=', g_err(0) )
    
    paths = model.Simulate()
    
    print("******\n\n")
    
    if f is not None:
        print(np.nanmean(f(paths[2][-1,:])))
    
    def PlotPaths(paths, title=""):
        plt.plot(model.t,paths[:,:25], alpha=0.25, linewidth=1)
        plt.plot(model.t,paths[:,0], color='b', linewidth=1)
        plt.axhline(1.2, linestyle='--',color='r')
        plt.axhline(0.8, linestyle='--',color='r')
        plt.plot(model.t, model.t, linewidth=2, linestyle='--', color='maroon', alpha=1)
        
        q = (1+np.arange(9))/10
        qtl = np.nanquantile(paths, q, axis=1)
        for i in range(len(qtl)):
            plt.plot(model.t, qtl[i], color='gray', linewidth=1)
        plt.fill_between(model.t, qtl[0], qtl[-1], color='y', alpha=0.2)
        
        m = np.mean(paths, axis=1)
        plt.plot(model.t, m, color='k')
        # qtl = np.nanquantile(paths, q_bar, axis=1)
        # plt.plot(model.t, qtl, color='r', linewidth=1)            
        
        # lvl = np.mean(paths[-1,:]<x_bar)
        # qtl = np.nanquantile(paths, lvl, axis=1)
        # plt.plot(model.t, qtl, color='g', linewidth=1)
        plt.ylim(-1,3)
        plt.xlim(0,1)
        plt.savefig(title, format='pdf')
        plt.show()
        
    filename = 'pinned_combined_{0:2.0f}'.format(100*pi[0])
    PlotPaths(paths[-1], filename + '.pdf')
    plot_mu(model, filename)

