#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 09:56:40 2018

@author: sannebjartmar
"""

import numpy as np

def behaviour_policy(dist, behaviour_p, quant = 0.95, const = 1):
    out = np.array([])
    
    if behaviour_p == "mean":
        out = dist.mean(2)[0]
    
    if behaviour_p == "sharp ratio":
        for actions in range(dist.shape[1]):
            d = dist[0][actions]
            out = np.append(out, d.mean()/np.abs(d.var()))
   
    elif behaviour_p == "sortino sharp ratio":
        for actions in range(dist.shape[1]):
            d = dist[0][actions]
            neg_var = d[d < np.median(d)].var()
            out = np.append(out, d.mean()/neg_var)#d.mean()/neg_var)
    
    elif behaviour_p == "weigthed VaR":
        for actions in range(dist.shape[1]):
            d = dist[0][actions]
            d = np.sort(d)
            idx = len(d) - int(quant * len(d))
            VaR = d[idx]
            out = np.append(out, d.mean() + const * VaR)
    
    elif behaviour_p == "weigthed cVaR":
        out = np.array([])
        for actions in range(dist.shape[1]):
            d = dist[0][actions]
            d = np.sort(d)
            idx = len(d) - int(quant * len(d))
            w = (np.array(range(idx)) + 1)/len(d) - 0.5/len(d)
            cVaR = (w * d[0:idx]).mean()
            out = np.append(out, d.mean() + const * cVaR)
    
    return out