# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 00:16:49 2024

@author: sebja
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pdb
import torch
import dill


method = "elicit" 
# method = "girsanov"
# method = "test"

if method == "elicit":
    from sde_barycentre_elicit import sde_barycentre 
elif method == "girsanov":
    from sde_barycentre_girsanov import sde_barycentre 
else:
    raise ValueError('not a valid method.')

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

# f.append(lambda x : 1*(x>0.8)*(x<1.2) - 0.9)
# g.append(lambda t, x: 1*(x<0.5*t)-0.2)

# f.append(lambda x : (x-0.5))
# f.append(lambda x : (x**2-(0.05+0.5**2)))

#I = lambda x, a : 1*(x>a) 

I = lambda x, a : torch.sigmoid((x-a)/0.01)

# f.append(lambda x : (1-I(x,1.2))*I(x,0.8) - 0.8)
g.append(lambda t, x: (1-I(x,1-(0.5-t)**2))-0.8)


f.append(lambda x : (x- 1))
f.append(lambda x : (x- 1)**2 - 0.05)
# g.append(lambda t, x: ((x-t)-0.1) )

X0 = torch.tensor([0])
rho = torch.ones(1,1)

# pi_all = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

pi = [0.5, 0.5]

model = sde_barycentre(X0, mu, sigma, rho, pi, 
                       f=f, g=g, T=1, Ndt=1001)
# X = model.simulate(256)

# model.plot_sample_paths()
#%%
print_at_list = list(np.unique(np.array(np.ceil(1.21**(np.arange(0,63))-1), int)))
print_at_list.sort(reverse=True)

# print_at_list = list(np.arange(0,20_000,1_000))
# print_at_list.sort(reverse=True)

model.generate_training_batch(100_000, init_randomize=False)
model.find_eta()

# model.generate_training_batch(100_000, init_randomize=True)
model.train(batch_size=1024, 
            print_at_list=print_at_list, 
            n_iter= 40_000 )

dill.dump(model, open("learnt_" + method +".pkl","wb"))

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
    # plt.xticks([1,10,100,1000,10000,20e3,40e3])
    plt.savefig('constraint' +method + '.pdf',format='pdf',bbox_inches='tight')
    plt.show()

plot_err(np.array(model.f_err), 'F')

if len(g) > 0:
    plot_err(np.array(model.g_err), 'G')
    
#%%
f_err = np.array(model.f_err)
g_err = np.array(model.g_err)

def plot(err, label):
    plt.plot(model.i, err[:,0], label=label)
    plt.fill_between(model.i, 
                     err[:,0]-3*err[:,1], 
                     err[:,0]+3*err[:,1], 
                     alpha=0.2)
for k in range(f_err.shape[1]):
    plot(f_err[:,k,:], r'$\mathbb{E}[f_{' + str(k+1) + ',T}]$')

for k in range(g_err.shape[1]):
    plot(g_err[:,k,:], r'$\mathbb{E}[\int_0^T g_{'+str(k+1)+',u}]$')


plt.axhline(0,linestyle='--', color='k')
plt.legend()
plt.xlabel('iteration')
plt.xscale('log')
plt.title('Learning the Value Func.')
# plt.xticks([1,10,100,1000,10000,20e3,40e3])
plt.savefig('constraint' + method + '.pdf',format='pdf',bbox_inches='tight')
plt.show()


#%%
log_dQ_dQbar, log_dQeta_dQbar = model.plot_hist_measure_change()

#%%
zz = np.load(open("girsanov_diff.npy","rb"))
plt.hist(np.exp(log_dQeta_dQbar)-np.exp(log_dQ_dQbar),np.linspace(-0.6,0.6,101), alpha=0.5,density=True, label="Value Func.")
plt.hist(zz,np.linspace(-0.6,0.6,101), alpha=0.5,density=True, label="Drift")
plt.axvline(0,linestyle='--',color='k')
plt.xlabel(r"$\frac{d\mathbb{Q}[\cdot]}{d\mathbb{Q}[\bar{\mu}]}-\frac{d\mathbb{Q}[\theta_{\eta^*}]}{d\mathbb{Q}[\bar{\mu}]}$")
plt.legend()
plt.savefig("dq_dqbar_error" + method + ".pdf",format='pdf',bbox_inches='tight')
plt.show()

#%%
# import pandas as pd
# import numpy as np
from scipy import stats

group1 = np.exp(log_dQeta_dQbar.numpy())-np.exp(log_dQ_dQbar.numpy())
group2 = np.load(open("girsanov_diff.npy","rb"))
# data ={'value_func' : np.exp(log_dQeta_dQbar)-np.exp(log_dQ_dQbar), 'girsanov' : zz}
# pd = pd.DataFrame(data)

t_stat_welch, p_value_unequal_var = stats.ttest_ind(group1, group2, equal_var=False)
print("\nWelch's T-Test assuming unequal variances:")
print(f"T-statistic: {t_stat_welch}")
print(f"P-value: {p_value_unequal_var}")


def do_test(data):
    t_statistic, p_value = stats.ttest_1samp(data, 0)
    print(f"t-statistic: {t_statistic}")
    print(f"P-value: {p_value}")
    
do_test(group1)
do_test(group2)

#%%
model.omega['net'](torch.cat((torch.zeros(1,1).to(model.dev),
                              torch.zeros(1,1).to(model.dev)),axis=-1))
