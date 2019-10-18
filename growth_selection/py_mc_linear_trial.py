#%%
import pymc3 as pm
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

x_trial = synthetic_dataset["x_array"][1000]
y_trial = synthetic_dataset["y_array"][1000]

#%%

def linear_bayesian_flat_prior(x, y, number_samples = 3000):

    '''Generates a Bayesian linear regression with half-flat priors gaussian
    likelihood
    Returns a dict with the model that can be used to compare, the trace
    and main plots'''

    # Set container
    result = dict.fromkeys(["model",
                            "trace",
                            "posterior_pred",
                            "plot_param",
                            "plot_data",
                            "plot_uncertainty"], None)

    try :


        with pm.Model() as model:

            # Define priors
            # sigma = pm.HalfFlat('sigma')
            # intercept = pm.distributions.continuous.HalfFlat('Intercept')
            # beta_1 = pm.distributions.continuous.HalfFlat('x')

            sigma = pm.HalfNormal('sigma', sigma = 0.4)
            intercept = pm.Bound(pm.HalfFlat,lower = 100, upper = 1000)('Intercept')
            beta_1 = pm.Bound(pm.HalfFlat, upper = 100)('beta_1')

            # Define likelihood
            likelihood = pm.Normal('y', mu = intercept + beta_1 * x,
                                   sigma = sigma, observed=y)

            # Inference!
            trace = pm.sample(number_samples , cores=2) # draw 3000 posterior samples using NUTS sampling
            result["trace"] = trace

            posterior_pred = pm.sample_posterior_predictive(trace)
            result["posterior_pred"] = posterior_pred
            result["model"] = model


            plot_param = plt.figure(figsize=(7, 7))
            pm.traceplot(trace)
            plt.tight_layout()
            result["plot_param"] = plot_param

            plot_data = plt.figure(figsize=(7, 7))
            plt.plot(x, y , label = 'data')
            pm.plot_posterior_predictive_glm(trace, samples=number_samples,
                                              label='posterior predictive regression lines',
                                              eval = np.linspace(0, np.max(x), 50),
                                              lm = lambda x, trace : trace['Intercept'] + trace['beta_1'] * x)

            plt.title('Posterior predictive regression lines')
            plt.legend(loc=0)
            plt.xlabel('x')
            plt.ylabel('y');
            result["plot_data"] = plot_data

            return(result)

    except RuntimeError:

            return(np.NaN)


#%%
a = linear_bayesian_flat_prior(x = x_trial, y = y_trial)

#%%

trace = a['trace']



plt.plot(x_trial, y_trial)
pm.plot_posterior_predictive_glm(trace, samples=6000,
                                  label='posterior predictive regression lines',
                                  eval = np.linspace(0, np.max(x_trial), 50),
                                  lm = lambda x, trace : trace['Intercept'] + trace['beta_1'] * x)
