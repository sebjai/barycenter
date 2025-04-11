# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 09:14:29 2025

@author: Sebastian
"""
import numpy as np
import matplotlib.pyplot as plt

x=[3.675,3.54,3.9,4,3.92,2.88,4,3.9,3.86,3.83,4,4,3.7,3.8,3.9,4,3.664,3.87990762124711,3.8,4,3.95,3.94,3.78,3.44,3.2,2.59,4,3.84,3.73,3.49,3.83,2.06222222222222,3.93,3.7,3.88,2.91,3.85666666666667]


qtl = np.quantile(x,1-24/38)
plt.hist(x, bins= np.linspace(2.5,4,15))
plt.xlim(2.5,4.0)
plt.xlabel("CGPA",fontsize=16)
plt.ylabel("Frequency",fontsize=16)
plt.xticks(fontsize=12)
plt.axvline(qtl, linestyle='--', color='k')