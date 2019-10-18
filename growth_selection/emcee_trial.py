#%%
import emcee
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


#%% Read Data
os.chdir("/Users/andour/Google Drive/projects/Dissertation")
synthetic_dataset = pickle.load(open( "simulated_data_classification_2", "rb" ))
synthetic_dataset = synthetic_dataset.set_index(pd.Index(
        range(0,len(synthetic_dataset))
        ))
synthetic_dataset["x_array"] = synthetic_dataset.apply(lambda x : x["dataset"][:,1], axis = 1)
synthetic_dataset["y_array"] = synthetic_dataset.apply(lambda x : x["dataset"][:,0], axis = 1)

#%%

x_trial = synthetic_dataset["x_array"][10]
y_trial = synthetic_dataset["y_array"][10]

#%% Functions preparing the equations for analysis
def log_prior_log(theta):
    k, x0, sigma = theta
    if sigma < 0:
        return -np.inf  # log(0)
    else:
        return - np.log(sigma)
        # -1.5 * np.log(1 + beta ** 2) - np.log(sigma)

def log_likelihood_log(theta, x, y):
    k, x0, sigma = theta
    y_model = 1/(1 + np.exp(-k * (x - x0)))
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def log_posterior_log(theta, x, y):
    return log_prior_log(theta) + log_likelihood_log(theta, x, y)

def compute_mcmc(num_param,
                x,
                y,
                log_posterior,
                nwalkers=50,
                nburn=1000,
                nsteps=2000):

    ndim = num_param  # this determines the model
    rng = np.random.RandomState(0)
    starting_guesses = rng.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x,y])
    sampler.run_mcmc(starting_guesses, nsteps)
    trace = sampler.get_chain()
    print("complete")
    return trace

#%%

x_trial = synthetic_dataset["x_array"][10]
y_trial = synthetic_dataset["y_array"][10]


#%%

trace = compute_mcmc(3, x_trial, y_trial, log_posterior_log)


plt.plot(x_trial, y_trial)
