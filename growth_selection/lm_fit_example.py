#%%
import emcee
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner
import lmfit

sns.set_style("darkgrid")


#%% Read Data
os.chdir("/Users/andour/Google Drive/projects/Dissertation")
synthetic_dataset = pickle.load(open( "simulated_data_classification_2", "rb" ))
synthetic_dataset = synthetic_dataset.set_index(pd.Index(
        range(0,len(synthetic_dataset))
        ))
synthetic_dataset["x_array"] = synthetic_dataset.apply(lambda x : x["dataset"][:,1], axis = 1)
synthetic_dataset["y_array"] = synthetic_dataset.apply(lambda x : x["dataset"][:,0], axis = 1)
synthetic_dataset
#%% Take an example
x_trial = synthetic_dataset["x_array"][39970]
x = x_trial
y_trial = synthetic_dataset["y_array"][39970]
y = y_trial
true_values = synthetic_dataset["parameters"][39970]
true_values

p = lmfit.Parameters()
p.add_many(('alpha', 0), ('beta', 0.3))

def residual(p):
    v = p.valuesdict()
    return (v['alpha'] + v['beta'] * x) - y

mi = lmfit.minimize(residual, p, method='nelder', nan_policy='omit')
lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)
#%% Prepare parameters and define the residual for MCMC for linear
params_linear = lmfit.Parameters()
params_linear.add('alpha',value =  0)
params_linear.add('beta', value = 0)
params_linear.add('__lnsigma', value=np.log(0.1), min=np.log(0.1), max=np.log(1))

def residual_linear(parameters = params_linear):
    v = parameters.valuesdict()
    alpha = v['alpha']
    beta = v['beta']
    y_model = alpha + beta * x
    return y_model - y


#%% Sample MCMC for linear_df
mi.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))

res = lmfit.minimize(residual, method='emcee', nan_policy='omit', burn=300, steps=1000, thin=20,
                     params=mi.params, is_weighted=False)
res
