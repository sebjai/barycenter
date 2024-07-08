# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:06:53 2024

@author: jaimunga
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pdb
import copy

#%%

S = lambda y,z : 2*(np.log(z/y)+(y/z) - 1)

z = np.linspace(0.01,20,1001)

y = np.exp(2-0.5+np.random.randn(10_000))

plt.hist(y, bins=np.linspace(0,20,51))
plt.show()

print(np.mean(y), np.std(y)/np.sqrt(len(y)))

#%%
loss = np.mean( S(y.reshape(-1,1),
                  z.reshape(1,-1)), axis=0 )

mloss = np.min(loss)
mz = z[np.argmin(loss)]
plt.plot(z, loss)
plt.scatter(mz, mloss, s=10, color='r')
plt.yscale('log')
plt.show()

print(np.mean(y), np.std(y)/np.sqrt(len(y)))
print(mz)


#%%
class net(nn.Module):
    
    def __init__(self, 
                 nIn, 
                 nOut, 
                 n_nodes=128, 
                 n_layers=5,
                 device='cpu',
                 output='none'):
        super(net, self).__init__()

        self.device = device
        
        self.in_to_hidden = nn.Linear(nIn, n_nodes).to(self.device)
        # self.hidden_to_hidden = nn.ModuleList([nn.Linear(n_nodes+nIn, n_nodes).to(self.device) for i in range(n_layers-1)])
        self.hidden_to_hidden = nn.ModuleList([nn.Linear(n_nodes, n_nodes).to(self.device) for i in range(n_layers-1)])
        # self.hidden_to_out = nn.Linear(n_nodes, nOut).to(self.device)
        self.hidden_to_out1 = nn.Linear(n_nodes+nIn, n_nodes).to(self.device)
        self.out1_to_out = nn.Linear(n_nodes, nOut).to(self.device)
        
        self.g = nn.SiLU()
        # self.g = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
        self.output=output
        
        
    def forward(self, x):
        
        h = self.g(self.in_to_hidden(x))
        for linear in self.hidden_to_hidden:
            # h = torch.cat((h,x),axis=-1)
            h = self.g(linear(h))
            
        # # concat orginal x to last hidden layer
        h = torch.cat((h,x),axis=-1)
        h = self.hidden_to_out1(h)
        h = self.out1_to_out(self.g(h))
        
        # h = self.hidden_to_out(h)
        
        if self.output=='softplus':
            h = self.softplus(h)
        
        return h

#%%
class elicit():
    
    def __init__(self):
        
        if torch.cuda.is_available(): 
            dev = "cuda:0" 
        else: 
            dev = "cpu" 
        self.dev = torch.device(dev)
        
        self.omega = net(2, 1, device=self.dev, output='softplus')
        self.optimizer = optim.Adam(self.omega.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=2,
                                                   gamma=0.999)
        
        self.T = 1
        self.N = 101
        self.t = torch.linspace(0,self.T, self.N).to(self.dev)
        self.dt = self.t[1]-self.t[0]
        
        self.loss = []
        
    def grab_batch(self, batch_size=256):
        
        W = torch.zeros(batch_size, self.N).to(self.dev)
        W[:,0] = torch.randn(batch_size).to(self.dev)
        
        for i in range(self.N-1):
            W[:,i+1] = W[:,i] + torch.sqrt(self.dt)*torch.randn(batch_size).to(self.dev)
            
        return W
    
    def store_batch(self, batch_size=10_000):
        self.X  = self.grab_batch(100_000)

    def train(self, n_iter=10_000, batch_size=256, n_print=100, rule="L2"):
                
        t = self.t.view(1,-1).repeat(batch_size,1)  
        
        j=1
        
        for i in tqdm(range(n_iter)):
            
            idx = torch.randint(0,self.X.shape[0], (batch_size,))
            X = self.X[idx]
            
            Y = (1.0*(X[:,-1]>0)).reshape(-1,1)+1e-3
            
            # pdb.set_trace()
            
            z = self.omega( torch.cat((t.unsqueeze(-1), X.unsqueeze(-1)), axis=-1)).squeeze()
            
            if rule =="L2": 
                score = (z-Y)**2
            elif rule == "Gamma":
                score = 2*(torch.log(z/Y)+(Y/z)-1.0)
            elif rule == "Poisson":
                score = 2*(Y*torch.log(Y/z)+z - Y)
            
            # score = score[:,:-1]
            
            self.optimizer.zero_grad()
            
            loss = torch.mean(score)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            
            self.loss.append(loss.item())
            
            if np.mod(i,n_print)==0:
                # plt.plot(np.array(self.loss))
                # plt.show()
                self.plot(rule + "_iter = " + str(i), rule +'_'+ str(j))
                j+=1
                
        self.plot(rule+ "_iter = " + str(i), rule +'_'+ str(j))
        
    def plot(self, title, filename):
        
        t_all = torch.linspace(0,self.T*0.99,4)
        
        x = torch.linspace(-3,3,51).to(self.dev)
        ones = torch.ones(x.shape).to(self.dev)
        Z = torch.distributions.Normal(0,1)
        f = lambda t, x : Z.cdf( x/torch.sqrt(self.T-t))
        for t in t_all:
            omega = self.omega( torch.cat((t*ones.unsqueeze(-1), x.unsqueeze(-1)), axis=-1) ).squeeze()
            plt.plot(x.cpu().numpy(), omega.detach().cpu().numpy(), label='$t='+str(t.cpu().numpy())+'$')
            plt.plot(x.cpu().numpy(), f(t, x).cpu().numpy(), linestyle='--', color='k')
            
        plt.ylim(-0.05,1.05)
        plt.legend()    
        plt.title(title)
        plt.savefig(filename+'.pdf', format='pdf', bbox_inches='tight')
        plt.show()
#%%        
rules = ["L2", "Gamma", "Poisson"]
model = elicit()
model.store_batch(100_000)

models = {'L2': copy.deepcopy(model), 
          'Gamma' : copy.deepcopy(model),
          'Poisson' : copy.deepcopy(model)}
#%%
for rule, model in models.items():
    model.train(batch_size = 1024, n_print=250, n_iter=5_000, rule=rule)
            
            
