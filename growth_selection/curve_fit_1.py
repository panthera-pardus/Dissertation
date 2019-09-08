
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


os.chdir("/Users/andour/Google Drive/projects/Dissertation")
synthetic_dataset = pickle.load( open( "simulated_data_classification", "rb" ))
synthetic_dataset = synthetic_dataset.set_index(pd.Index(
        range(0,len(synthetic_dataset))
        ))
synthetic_dataset.head()

x = synthetic_dataset.loc[16009, "dataset"][:,1]
y = synthetic_dataset.loc[16009, "dataset"][:,0]

#%%
# Need to fit each dataset with a sigmoid and a linear. The method will be
# OLS and non-liπnear LS

#https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
def sigmoid(x, L ,x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return (y)

#example = synthetic_dataset.loc[0, "dataset"]

p0 = [max(y), np.median(x),1] # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, x, y, p0, method='dogbox')
y_fit = sigmoid(x, *popt)

plt.plot(x, y, label = 'data')
plt.plot(x, y_fit, label = 'fit')

#%%
mean_squared_error(y_true = y, y_pred = y_fit)









#%%
def linear(a,x,b):
    y = a * x + b
    return(y)

p0_linear = [0, min(y)]
popt, pcov = curve_fit(linear, x, y, p0_linear)
y_fit_linear = linear(x, *popt)

mean_squared_error(y_true = y,
y_pred = y_fit_linear)
# plt.plot(x, y, label = 'data')
# plt.plot(x, y_fit_linear, label = 'fit')
