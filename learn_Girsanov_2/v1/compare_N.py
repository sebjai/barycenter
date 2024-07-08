# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 05:56:12 2024

@author: jaimunga
"""

import dill
import torch
import matplotlib.pyplot as plt
#%%

model = []
model.append(dill.load(open('learnt_11.pkl','rb')))
model.append(dill.load(open('learnt_101.pkl','rb')))
model.append(dill.load(open('learnt_1001.pkl','rb')))

def sim(mdl):
    
    state = torch.get_rng_state()
    torch.manual_seed(12317874321)
    
    X, var_sigma, dW = mdl.simulate_Qbar(5_000)
    t = torch.tensor(mdl.t).float().view(1,-1,1).repeat(X.shape[0],1,1).to(mdl.dev) 
    
    theta = mdl.theta['net'](torch.cat((t,X), axis=-1))
    
    log_dQ_dQbar = torch.log(mdl.get_stoch_exp(t, X, dW, theta)).detach().cpu()
    log_dQeta_dQbar = torch.log(mdl.dQeta_dQbar(mdl.eta, t, X, var_sigma).reshape(-1)).detach().cpu()
    
    torch.cuda.empty_cache()
    
    torch.set_rng_state(state)
    
    return log_dQ_dQbar, log_dQeta_dQbar

#%%
myc = ['tab:green', 'tab:blue', 'tab:red']
for j in range(1,4):
    for i, mdl in enumerate(model[:j]):
        
        log_dQ_dQbar, log_dQeta_dQbar = sim(mdl)
        
        qtl = torch.quantile(torch.cat((log_dQ_dQbar,log_dQeta_dQbar)),
                             torch.tensor([0.005,0.995]))
        
        pl = plt.scatter(log_dQeta_dQbar, log_dQ_dQbar,  s=5, alpha=0.1, color=myc[i],label=str(len(mdl.t)-1))
        # plt.scatter(log_dQ_dQbar[0], log_dQeta_dQbar[0],label=str(len(mdl.t)-1), color=myc[i])
        
        torch.cuda.empty_cache()
    
    leg = plt.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    
    z=[-6,2]
    plt.plot(z,z,color='k', linewidth=1)    
    plt.xlim(z)
    plt.ylim(z)
    plt.ylabel(r"$\log\frac{d\mathbb{Q}^*}{d\overline{\mathbb{Q}}}$",fontsize=20)
    plt.xlabel(r"$\log\frac{d\mathbb{Q}_{\eta^*}}{d\overline{\mathbb{Q}}}$",fontsize=20)
    
    plt.savefig('scatter_'+str(j) +'.pdf', format='pdf', bbox_inches='tight')
    plt.show()    
        
#%%
for j, mdl in enumerate(model):
    mv, mv_err = mdl.moving_average(mdl.theta['loss'],500)
    plt.plot(mv, color=myc[j], label=str(len(mdl.t)-1))
    plt.plot(mdl.theta['loss'], color=myc[j], alpha=0.2)
plt.legend(loc='upper right')
plt.yscale('log')
plt.savefig('loss.pdf', format='pdf', bbox_inches='tight')
plt.show()