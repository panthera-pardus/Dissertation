#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 09:16:20 2019

@author: andour
"""

import numpy as np
import pickle
import pymc3 as pm
import matplotlib.pyplot as plt
import os
#%%
os.chdir("/Users/andour/Google Drive/projects/Dissertation")
file = open("simulated_data_1000", "rb")
simulated_data_1000 = pickle.load(file)

data_example = simulated_data_1000[1]

X, y = data_example[:,1], data_example[:,0]

#%%
# =============================================================================
# # Context for the model
# with pm.Model() as normal_model:
#
#     # The prior for the data likelihood is a Normal Distribution
#     family = pm.glm.families.Normal()
#
#     # Creating the model requires a formula and data (and optionally a family)
#     pm.GLM.from_formula(formula, data = X_train, family = family)
#
#     # Perform Markov Chain Monte Carlo sampling letting PyMC3 choose the algorithm
#     normal_trace = pm.sample(draws=2000, chains = 2, tune = 500, njobs=-1)
# =============================================================================




#%%

# =============================================================================
# with pm.Model() as linear_model:
#     weights = pm.Normal('weights', mu=0, sigma=1)
#     #noise = pm.Gamma('noise', alpha=2, beta=1)
#     mu = X * weights
#     y_observed = pm.Normal('y_observed',
#                 mu=mu,
#                 #sigma=noise,
#                 observed=y)
#
#     prior = pm.sample_prior_predictive()
#     posterior = pm.sample()
#     posterior_pred = pm.sample_posterior_predictive(posterior)
#
# =============================================================================
#%%
x, y = data_example[:,1], data_example[:,0]
with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = pm.Normal('Intercept', 0, sigma=1)
    beta_1 = pm.Normal('x', 0, sigma=1)

    # Define likelihood
    likelihood = pm.Normal('y', mu=intercept + beta_1 * x,
                        sigma=sigma, observed=y)

    # Inference!
    trace = pm.sample(3000 , cores=2) # draw 3000 posterior samples using NUTS sampling
    posterior_pred = pm.sample_posterior_predictive(trace)


#%%


plt.figure(figsize=(7, 7))
pm.traceplot(trace[100:])
plt.tight_layout();

#%%


plt.figure(figsize=(7, 7))
plt.plot(x, y, 'x', label='data')
pm.plot_posterior_predictive_glm(trace, samples=3000,
                              label='posterior predictive regression lines', eval = np.linspace(0, 250, 50))

plt.title('Posterior predictive regression lines')
plt.legend(loc=0)
plt.xlabel('x')
plt.ylabel('y');
