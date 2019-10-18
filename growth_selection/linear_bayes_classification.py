# %% Import cell
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

#%% Read Data
os.chdir("/Users/andour/Google Drive/projects/Dissertation")
synthetic_dataset = pickle.load(open( "simulated_data_classification_2", "rb" ))
synthetic_dataset = synthetic_dataset.set_index(pd.Index(
        range(0,len(synthetic_dataset))
        ))

synthetic_dataset["x_array"] = synthetic_dataset.apply(lambda x : x["dataset"][:,1], axis = 1)
synthetic_dataset["y_array"] = synthetic_dataset.apply(lambda x : x["dataset"][:,0], axis = 1)

#%% Define function for linear

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

            sigma = pm.HalfFlat('sigma')
            intercept = pm.HalfFlat('Intercept')
            beta_1 = pm.HalfFlat('x')

            # Define likelihood
            likelihood = pm.Normal('y', mu = intercept + beta_1 * x,
                                   sigma=sigma, observed=y)

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
            plt.plot(x, y, 'x', label = 'data')
            pm.plot_posterior_predictive_glm(trace, samples=number_samples,
                                          label='posterior predictive regression lines',
                                          eval = np.linspace(0, np.max(x), 50))

            plt.title('Posterior predictive regression lines')
            plt.legend(loc=0)
            plt.xlabel('x')
            plt.ylabel('y');
            result["plot_data"] = plot_data

            return(result)

    except RuntimeError:

            return(np.NaN)


#%% Define function for logistic
def logistic_bayesian(x, y, number_samples):

    '''Generates a Bayesian logistic function with flat priors and gaussian
    likelihood.
    Returns a dict with the model that can be used to compare, the trace
     main plots'''

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





#%% call linear function

synthetic_dataset["linear_bayes"] = synthetic_dataset.apply(lambda x :
linear_bayesian_flat_prior(x = x.x_array, y = x.y_array, number_samples = 500), axis = 1)

#%%
y_trial = synthetic_dataset["y_array"][1000]
x_trial = synthetic_dataset["x_array"][1000]

list(synthetic_dataset.columns)



#%% linear df run
linear_df = synthetic_dataset[ synthetic_dataset.label == "linear"]

linear_df["linear_bayes"] = linear_df.apply(lambda x :
linear_bayesian_flat_prior(x = x.x_array, y = x.y_array, number_samples = 500), axis = 1)

#%%

trial = linear_bayesian_flat_prio(x_trial, y_trial, 10000)
trial
trial["trace"][3]



#%%

len(synthetic_dataset)

trial = linear_df.sample(10)

trial["linear_bayes"] = trial.apply(lambda x :
linear_bayesian_flat_prior(x = x.x_array, y = x.y_array, number_samples = 500), axis = 1)



trial["linear_bayes"]
