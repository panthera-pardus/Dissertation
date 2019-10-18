
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
def log_prior_linear(theta):
    alpha, beta, sigma = theta
    if sigma < 0:
        return -np.inf  # log(0)
    else:
        return - np.log(sigma)
        # -1.5 * np.log(1 + beta ** 2) - np.log(sigma)

def log_likelihood_linear(theta, x, y):
    alpha, beta, sigma = theta
    y_model = alpha + beta * x
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def log_posterior_linear(theta, x, y):
    return log_prior_linear(theta) + log_likelihood_linear(theta, x, y)

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
    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
    print("complete")
    return trace



#%% Functions for plotting parameters
def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]

    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)


def plot_MCMC_trace(ax, xdata, ydata, trace, scatter=False, **kwargs):
    """Plot traces and contours"""
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        ax.plot(trace[0], trace[1], ',k', alpha=0.1)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')


def plot_MCMC_model(ax, xdata, ydata, trace):
    """Plot the linear model and 2sigma contours"""
    ax.plot(xdata, ydata, 'ok')

    alpha, beta = trace[:2]
    xfit = np.linspace(-20, 120, 10)
    yfit = alpha[:, None] + beta[:, None] * xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)

    ax.plot(xfit, mu, '-k')
    ax.fill_between(xfit, mu - sig, mu + sig, color='lightgray')

    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_MCMC_results(xdata, ydata, trace, colors='k'):
    """Plot both the trace and the model together"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_MCMC_trace(ax[0], xdata, ydata, trace, True, colors=colors)
    plot_MCMC_model(ax[1], xdata, ydata, trace)

#%% Trial on linear part of the samples

linear_df = synthetic_dataset[ synthetic_dataset.label == "linear"]

linear_df["linear_bayes"] = linear_df.apply(lambda x :
compute_mcmc(x = x.x_array, y = x.y_array, log_posterior = log_posterior_linear, num_param = 3), axis = 1)


plot_MCMC_results(x_trial, y_trial, compute_mcmc(3, x_trial, y_trial, log_posterior_linear))


file = open("linear_bayes", "wb")
pickle.dump(linear_df, file)
file.close()


#%% Same thing for logistic

logistic_df = synthetic_dataset[synthetic_dataset.label == "logistic"]

logistic_df["linear_bayes"] = logistic_df.apply(lambda x :
compute_mcmc(x = x.x_array, y = x.y_array, log_posterior = log_posterior_linear, num_param = 3), axis = 1)

file = open("logistic_bayes", "wb")
pickle.dump(logistic_df, file)
file.close()







print(2)






#%% Parameter prep for sampling
ndim = 3  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nburn = 1000  # "burn-in" period to let chains stabilize
nsteps = 2000  # number of MCMC steps to take

# set theta near the maximum likelihood, with
np.random.seed(0)
starting_guesses = np.random.random((nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x_trial, y_trial])
%time sampler.run_mcmc(starting_guesses, nsteps)
print("done")

compute_mcmc(3, x_trial, y_trial, log_posterior_linear)

emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T

plot_MCMC_results(x_trial, y_trial, emcee_trace)

plt.plot(sampler.chain[:,nburn:,0].T, '-', color='k', alpha=0.3)
plt.axhline(alpha_true, color='blue')


plt.plot(x_trial,y_trial)
