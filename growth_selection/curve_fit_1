#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 14:16:47 2019

@author: andour
"""

import pickle
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


os.chdir("/Users/andour/Google Drive/projects/Dissertation")
synthetic_dataset = pickle.load( open( "simulated_data_classification", "rb" ))
synthetic_dataset = synthetic_dataset.set_index(pd.Index(
        range(0,len(synthetic_dataset))
        ))

x = synthetic_dataset.loc[0, "dataset"][:,1]
y = synthetic_dataset.loc[0, "dataset"][:,0]

# Need to fit each dataset with a sigmoid and a linear. The method will be
# OLS and non-linear LS



#https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

#example = synthetic_dataset.loc[0, "dataset"]

p0 = [max(y), np.median(x),1,min(y)] # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, x, y, p0, method='dogbox')


def linear(a,x,b):
    y = a * x + b
    return(y)

p0_linear = [0, min(y)]
curve_fit(linear, x, y, p0_linear)


