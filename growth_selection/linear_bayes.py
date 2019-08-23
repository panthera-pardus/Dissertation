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

x, y = data_example[:,1], data_example[:,0]
#%%

def linear_bayesian(x,y, number_samples):
    
    '''Generates a Bayesian linear regression with uninformative priors
        sigma Half Cauchy, intercept and coefficient are std normal
        Returns a dict with the model that can be used to compare, the trace
        and main plots'''
    
    # Set container
    result = dict.fromkeys(["model", 
                            "trace",
                            "posterior_pred",
                            "plot_param",
                            "plot_data",
                            "plot_uncertainty"], None)

    
    with pm.Model() as model:
    
        # Define priors
        sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
        intercept = pm.Normal('Intercept', 0, sigma=1)
        beta_1 = pm.Normal('x', 0, sigma=1)

        # Define likelihood
        likelihood = pm.Normal('y', mu=intercept + beta_1 * x,
                               sigma=sigma, observed=y)
    
        # Inference!
        trace = pm.sample(number_samples , cores=2) # draw 3000 posterior samples using NUTS sampling
        result["trace"] = trace
        
        posterior_pred = pm.sample_posterior_predictive(trace)
        result["posterior_pred"] = posterior_pred
        result["model"] = model
        
        
        plot_param = plt.figure(figsize=(7, 7))
        pm.traceplot(trace[100:])
        plt.tight_layout()
        result["plot_param"] = plot_param
        
        plot_data = plt.figure(figsize=(7, 7))
        plt.plot(x, y, 'x', label='data')
        pm.plot_posterior_predictive_glm(trace, samples=3000,
                                      label='posterior predictive regression lines', eval = np.linspace(0, 250, 50))
        
        plt.title('Posterior predictive regression lines')
        plt.legend(loc=0)
        plt.xlabel('x')
        plt.ylabel('y');
        result["plot_data"] = plot_data
        
        return(result)
        
        
        
#%%
def logistic_bayesian(x,y, number_samples):
    
    '''Generates a Bayesian linear regression with uninformative priors
        sigma Half Cauchy, intercept and coefficient are std normal
        Returns a dict with the model that can be used to compare, the trace
        and main plots'''
    
    # Set container
    result = dict.fromkeys(["model", 
                            "trace",
                            "posterior_pred",
                            "plot_param",
                            "plot_data",
                            "plot_uncertainty"], None)
    

    
    with pm.Model() as model:
    
        # Define priors
        sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
        intercept = pm.Normal('Intercept', 0, sigma=1)
        beta_1 = pm.Normal('x', 0, sigma=1)

        # Define likelihood
        likelihood = pm.Normal('y', intercept + beta_1 * x,
                               sigma=sigma, observed=y)
    
        # Inference!
        trace = pm.sample(number_samples , cores=2) # draw 3000 posterior samples using NUTS sampling
        result["trace"] = trace
        
        posterior_pred = pm.sample_posterior_predictive(trace)
        result["posterior_pred"] = posterior_pred
        result["model"] = model
        
        
        plot_param = plt.figure(figsize=(7, 7))
        pm.traceplot(trace[100:])
        plt.tight_layout()
        result["plot_param"] = plot_param
        
        plot_data = plt.figure(figsize=(7, 7))
        plt.plot(x, y, 'x', label='data')
        pm.plot_posterior_predictive_glm(trace, samples=3000,
                                      label='posterior predictive regression lines', eval = np.linspace(0, 250, 50))
        
        plt.title('Posterior predictive regression lines')
        plt.legend(loc=0)
        plt.xlabel('x')
        plt.ylabel('y');
        result["plot_data"] = plot_data
        
        return(result)
        
    


#%%


