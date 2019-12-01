#%%
import emcee
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner


sns.set_style("darkgrid")


#%% Read Data
os.chdir("/Users/andour/Google Drive/projects/Dissertation/Final data")
synthetic_dataset = pickle.load(open( "simulated_data_raw", "rb" ))
print(emcee.__version__)
#%%
x_trial = synthetic_dataset["x_array"][10]
y_trial = synthetic_dataset["y_array"][10]

#%%
def log_likelihood(theta, x, y):
    k, x0, sigma = theta
    y_model = 1/(1 + np.exp(-k * (x - x0)))
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def log_prior(theta):
    k, x0, sigma = theta
    if sigma < 0:
        return -np.inf  # log(0)
    else:
        return 0.0

def log_posterior(theta, x, y):
    return log_prior(theta) + log_likelihood(theta, x, y)

ndim = 3  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nburn = 1000  # "burn-in" period to let chains stabilize
nsteps = 2000  # number of MCMC steps to take

np.random.seed(0)
starting_guesses = np.random.random((nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x_trial, y_trial])
%time sampler.run_mcmc(starting_guesses, nsteps)
print("done")


#%%
