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

#%% Take an example
x_trial = synthetic_dataset["x_array"][10000]
x = x_trial
y_trial = synthetic_dataset["y_array"][10000]
y = y_trial
true_values = synthetic_dataset["parameters"][10000]
true_values

#%% Prepare parameters and define the residual for MCMC for logistic
params_logistic = lmfit.Parameters()
params_logistic.add('x0', value = 100, min = 0, max = 350)
params_logistic.add('k', value = 1,  min = 0.1, max = 2)
params_logistic.add('__lnsigma', value=np.log(0.1), min=np.log(0.1), max=np.log(1))

def residual_logistic(parameters = params_logistic):
    v = parameters.valuesdict()
    x0 = v['x0']
    k = v['k']
    y_model = 1/(1 + np.exp(-k * (x - x0)))
    return y_model - y

#%% Prepare parameters and define the residual for MCMC for linear
params_linear = lmfit.Parameters()
params_linear.add('alpha',value =  0, min = 0)
params_linear.add('beta', value = 0, min = 0)
params_linear.add('__lnsigma', value=np.log(0.1), min=np.log(0.1), max=np.log(1))

def residual_linear(parameters = params_linear):
    v = parameters.valuesdict()
    alpha = v['alpha']
    beta = v['beta']
    y_model = alpha + beta * x
    return y_model - y

#%% Sample using MCMC logistic
res_logistic = lmfit.minimize(residual_logistic,
                    method = 'emcee',
                    nan_policy = 'omit',
                    burn = 300,
                    steps = 1000,
                    thin = 20,
                    params = params_logistic,
                    is_weighted = False,
                    float_behavior = 'posterior')

res_logistic


#%% Sample MCMC for linear_df
res_linear = lmfit.minimize(residual_linear,
                    method = 'emcee',
                    nan_policy = 'omit',
                    burn = 300,
                    steps = 1000,
                    thin = 20,
                    params = params_linear,
                    is_weighted = False,
                    float_behavior = 'posterior')
res_linear
#%%
# def integrate_posterior_logistic(log_posterior, xlim, ylim, data=data):
#     func = lambda theta1, theta0: np.exp(log_posterior([theta0, theta1], data))
#     return integrate.dblquad(func, xlim[0], xlim[1],
#                              lambda x: ylim[0], lambda x: ylim
# res.
# lmfit.printfuncs.report_fit(res.params,
#                             min_correl=0.5)


def logistic(parameters, x):
    k = parameters.params["k"].value
    x0 = parameters.params["x0"].value
    sigma = parameters.params["__lnsigma"].value
    return 1/(1 + np.exp(-k * (x - x0)))

def linear(parameters, x):
    alpha = parameters.params["alpha"].value
    beta = parameters.params["beta"].value
    return alpha + beta * x

plt.plot(x, y, 'b')
plt.plot(x,
        logistic(parameters = res_logistic, x = x),
        'r')
plt.plot(x,
        linear(parameters = res_linear, x = x),
        'g')
plt.show()
#%%

corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))
np.exp(res.lnprob)


res.t


res_logistic.params

x_trial[0]






y_trial













#%%
res.params
res = lmfit.minimize(residual, method='emcee', nan_policy='omit', burn=300, steps=1000, thin=20, params=mi.params, is_weighted=False, float_behavior = 'posterior')
res.chain.shape
corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))

res

#%%

print("median of posterior probability distribution")
print('--------------------------------------------')
lmfit.report_fit(res.params)


import numpy as np
import lmfit
import matplotlib.pyplot as plt
x = np.linspace(1, 10, 250)
np.random.seed(0)
y = 3.0 * np.exp(-x / 2) - 5.0 * np.exp(-(x - 0.1) / 10.) + 0.1 * np.random.randn(len(x))
plt.plot(x, y, 'b')
plt.show()
p = lmfit.Parameters()
p.add_many(('a1', 4.), ('a2', 4.), ('t1', 3.), ('t2', 3., True))
def residual(p):
    v = p.valuesdict()
    return v['a1'] * np.exp(-x / v['t1']) + v['a2'] * np.exp(-(x - 0.1) / v['t2']) - y
mi = lmfit.minimize(residual, p, method='Nelder', nan_policy='omit')
lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)
