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

    sigma_min = 0.0
    sigma_max = 1.0

    x0_min = 25.0
    x0_max = 750.0

    k_min = 0.5
    k_max = 2.0


    if 0.0 < sigma < 1.0 and 250.0 < x0 < 300.0 and 0.5 < k < 2.0:
        return 0.0  # log(1)

    return -np.inf
        # -1.5 * np.log(1 + beta ** 2) - np.log(sigma)

def log_likelihood_log(theta, x, y):
    k, x0, sigma = theta
    y_model = 1/(1 + np.exp(-k * (x - x0)))
    return -0.5*np.sum(((y_model - y)/sigma)**2)

def log_posterior_log(theta, x, y):
    lp = log_prior_log(theta)
    if not np.isfinite(lp) :
        return -np.inf
    return lp + log_likelihood_log(theta, x, y)

def compute_mcmc(num_param,
                x,
                y,
                log_posterior,
                nwalkers=100,
                nburn=1000,
                nsteps=2000):

    ndim = num_param  # this determines the model
    rng = np.random.RandomState()
    starting_guesses = rng.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x,y])
    sampler.run_mcmc(starting_guesses, nsteps)
    # trace = sampler.get_chain()
    print("complete")
    return sampler


#%%

sampler = compute_mcmc(3, x_trial, y_trial, log_posterior_log)
samples = sampler.chain.reshape((-1, 3))

np.min(samples[:,2])
# corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"])
k, x0, sigma

samples

plt.plot(x_trial, y_trial)
plt.plot(x_trial, y_trial)


trace.shape
samples[:,0]
x_trial.shape
plt.plot(x_trial, 1/(1 + np.exp(-samples[1000][0] * (x_trial - samples[1000][1]))), color="k", alpha=0.1)
np.max(samples[:,1])


samples[1000][1]
