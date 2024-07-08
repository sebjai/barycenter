# -*- coding: utf-8 -*-
"""
Created on Wed May  1 06:24:40 2024

@author: sebja
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from math import factorial

lam = 10

t = np.linspace(0,1,1001)

VaR_L = []
VaR_U = []
LTE = []
UTE = []

for i in range(len(t)):
    
    VaR_L.append(poisson.ppf(0.1, t[i]*lam))
    
    numer = np.sum([k*(lam*t[i])**k/factorial(k) for k in range(0,int(VaR_L[-1]))])
    denom = np.sum([(lam*t[i])**k/factorial(k) for k in range(0,int(VaR_L[-1]))])
    
    LTE.append(numer/denom)
    
    VaR_U.append(poisson.ppf(0.9, t[i]*lam))
    
    numer = np.sum([k*(lam*t[i])**k/factorial(k) for k in range(int(VaR_U[-1])+1,100)])
    denom = np.sum([(lam*t[i])**k/factorial(k) for k in range(int(VaR_U[-1])+1,100)])    

    UTE.append(numer/denom)
    
    