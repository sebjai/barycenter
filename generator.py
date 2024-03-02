# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:17:32 2023

@author: sebja
"""

import numpy as np
import pdb

class generator():
    
    def __init__(self, Ndt = 100, T=1):
        
        self.Ndt = Ndt
        self.t = np.linspace(0, T, Ndt+1)
        self.dt = self.t[1]-self.t[0]
        self.T = T
        
    def simulate(self, nsims = 100):
        
        x = np.zeros((nsims, len(self.t)))
        
        for i in range(len(self.t)-1):
            
            x[:,i+1] = x[:,i] + np.sqrt(self.dt)*np.random.randn(nsims)
            
        return x