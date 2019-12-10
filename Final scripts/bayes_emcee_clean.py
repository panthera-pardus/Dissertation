#%% Imports
import emcee
import pickle
import os
import pandas as pd
import numpy as np
import lmfit
import time

print(emcee.__version__)

#%% Read Data
os.chdir("/Users/andour/Google Drive/projects/Dissertation/Final data")
synthetic_dataset = pickle.load(open( "simulated_data_raw", "rb" ))

#%% Prepare parameters and define the residual for MCMC for logistic
params_logistic = lmfit.Parameters()
params_logistic.add('x0', value = 100, min = 0)
params_logistic.add('k', value = 1,  min = 0.1, max = 2)
params_logistic.add('__lnsigma', value=np.log(0.1), min=np.log(0.1), max=np.log(1))

def residual_logistic(parameters, x, y):
    v = parameters.valuesdict()
    x0 = v['x0']
    k = v['k']
    y_model = 1/(1 + np.exp(-k * (x - x0)))
    return y_model - y

#%% Prepare parameters and define the residual for MCMC for linear
params_linear = lmfit.Parameters()
params_linear.add('alpha',value =  0.1, min = 0, max = 1)
params_linear.add('beta', value = 0.1, min = 0, max = 10)
params_linear.add('__lnsigma', value=np.log(0.1), min=np.log(0.1), max=np.log(1))

def residual_linear(parameters, x, y):
    v = parameters.valuesdict()
    alpha = v['alpha']
    beta = v['beta']
    y_model = alpha + beta * x
    return y_model - y

def callable_logistic_posterior(k, x0, x, y):

    '''Logistic Function that needs to be integrated for Bayes factor computation'''

    sigma = 1
    y_model = 1/(1 + np.exp(-k * (x - x0)))
    if sigma < 0 or k < 0 or x0 < 0:
        return -np.inf
    else:
        return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def callable_linear_posterior(alpha, beta, x, y):

    '''Linear Function that needs to be integrated for Bayes factor computation'''

    sigma = 1
    y_model = alpha + beta * x

    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)


# Need to create callable functions of the posteriors to compute the posterior
def f_posterior(x_array, y_array, function_type):
    '''function to return a callable function of the posterior given the data used'''
    if function_type == "logistic":
        return lambda k, x0 : np.exp(callable_logistic_posterior(k, x0, x = x_array, y = x_array))

    elif function_type == "linear":
        return lambda alpha, beta : np.exp(callable_linear_posterior(alpha, beta, x = x_array, y = y_array))

def waic_classification(model_draws_linear, model_draws_logistic):

    waic_linear = -2 * np.sum(np.log(np.mean(np.exp(model_draws_linear), axis = 0))) - np.var(model_draws_linear)
    waic_logistic = -2 * np.sum(np.log(np.mean(np.exp(model_draws_logistic), axis = 0))) - np.var(model_draws_logistic)

    if waic_linear < waic_logistic :
        return "linear"

    else :
        return "logistic"

#%% Sample using MCMC logistic
time_dict = {}
synthetic_dataset["logistic_evalutaion"] = synthetic_dataset.apply(lambda x :lmfit.minimize(residual_logistic,
                                                                                        method = 'emcee',
                                                                                        nan_policy = 'omit',
                                                                                        burn = 300,
                                                                                        steps = 1000,
                                                                                        thin = 20,
                                                                                        params = params_logistic,
                                                                                        is_weighted = False,
                                                                                        float_behavior = 'posterior', args=[x.x_array, x.y_array]), axis = 1)


synthetic_dataset["logistic_param_estimation"] = synthetic_dataset.apply(lambda x : {"x0" : [x.logistic_evalutaion.params["x0"].value + x.logistic_evalutaion.params["x0"].stderr,
                                                                                            x.logistic_evalutaion.params["x0"].value - x.logistic_evalutaion.params["x0"].stderr],

                                                                                     "k" : [x.logistic_evalutaion.params["k"].value + x.logistic_evalutaion.params["k"].stderr,
                                                                                            x.logistic_evalutaion.params["k"].value - x.logistic_evalutaion.params["k"].stderr]}, axis = 1)
start = time.time()
synthetic_dataset["logistic_posterior_integral"] = synthetic_dataset.apply(lambda df : integrate.dblquad(f_posterior(x_array=df.x_array, y_array=df.y_array, function_type = "logistic"), np.min(df.logistic_evalutaion.flatchain["k"]), np.max(df.logistic_evalutaion.flatchain["k"]),
                                                                                                         lambda x0: np.min(df.logistic_evalutaion.flatchain["x0"]),
                                                                                                         lambda x0: np.max(df.logistic_evalutaion.flatchain["x0"])), axis = 1)
end = time.time()
time_dict["logist_integral"] = [end - start]


#%% Sample MCMC for linear_df
synthetic_dataset["linear_evalutaion"] = synthetic_dataset.apply(lambda x :lmfit.minimize(residual_linear,
                                                                                        method = 'emcee',
                                                                                        nan_policy = 'omit',
                                                                                        burn = 300,
                                                                                        steps = 1000,
                                                                                        thin = 20,
                                                                                        params = params_linear,
                                                                                        is_weighted = False,
                                                                                        float_behavior = 'posterior', args=[x.x_array, x.y_array]), axis = 1)



synthetic_dataset["linear_param_estimation"] = synthetic_dataset.apply(lambda x : {"alpha" : [x.linear_evalutaion.params["alpha"].value + x.linear_evalutaion.params["alpha"].stderr,
                                                                                            x.linear_evalutaion.params["alpha"].value - x.linear_evalutaion.params["alpha"].stderr],

                                                                                     "beta" : [x.linear_evalutaion.params["beta"].value + x.linear_evalutaion.params["beta"].stderr,
                                                                                            x.linear_evalutaion.params["beta"].value - x.linear_evalutaion.params["beta"].stderr]}, axis = 1)
start = time.time()
synthetic_dataset["linear_posterior_integral"] = synthetic_dataset.apply(lambda df : integrate.dblquad(f_posterior(x_array=df.x_array, y_array=df.y_array, function_type = "linear"), np.min(df.linear_evalutaion.flatchain["alpha"]), np.max(df.linear_evalutaion.flatchain["alpha"]),
                                                                                                         lambda x0: np.min(df.linear_evalutaion.flatchain["beta"]),
                                                                                                         lambda x0: np.max(df.linear_evalutaion.flatchain["beta"])), axis = 1)
end = time.time()
time_dict["linear_integral"] = [end - start]

synthetic_dataset["bayes_classification"] = synthetic_dataset.apply(lambda x : "logistic" if 0 >= np.any(x.linear_posterior_integral) else "linear", axis = 1)
time_dict["BF_time/waic"] = time_dict["logist_integral"][0] + time_dict["linear_integral"][0]

synthetic_dataset["waic_classification"] = synthetic_dataset.apply(lambda df : waic_classification(model_draws_linear = df.linear_evalutaion.lnprob, model_draws_logistic = df.logistic_evalutaion.lnprob), axis = 1)

time_df = pd.DataFrame.from_dict(time_dict)

#%% Save dataset
file = open("simulated_data_bayes", "wb")
pickle.dump(synthetic_dataset, file)
file.close()

file = open("simulated_data_bayes_time", "wb")
pickle.dump(time_df, file)
file.close()













# time_df
#
#
# res_linear = lmfit.minimize(residual_linear,
#                     method = 'emcee',
#                     nan_policy = 'omit',
#                     burn = 300,
#                     steps = 1000,
#                     thin = 20,
#                     params = params_linear,
#                     is_weighted = False,
#                     float_behavior = 'posterior')
#
# alpha_pred = res_linear.params["alpha"].value
# beta_pred = res_linear.params["beta"].value
#
#
# res_linear.chain
# #%%
# # def integrate_posterior_logistic(log_posterior, xlim, ylim, data=data):
# #     func = lambda theta1, theta0: np.exp(log_posterior([theta0, theta1], data))
# #     return integrate.dblquad(func, xlim[0], xlim[1],
# #                              lambda x: ylim[0], lambda x: ylim
# # res.
# # lmfit.printfuncs.report_fit(res.params,
# #                             min_correl=0.5)
#
#
#
#
# -2 * np.sum(np.log(np.mean(np.exp(synthetic_dataset["trace_logistic"][1347].lnprob), axis = 0))) - np.var(synthetic_dataset["trace_logistic"][1347].lnprob)
#
# res_logistic.params["x0"].stderr
# k_pred = res_logistic.params["k"]
# def logistic(parameters, x):
#     k = parameters.params["k"].value
#     x0 = parameters.params["x0"].value
#     sigma = parameters.params["__lnsigma"].value
#     return 1/(1 + np.exp(-k * (x - x0)))
#
# def linear(parameters, x):
#     alpha = parameters.params["alpha"].value
#     beta = parameters.params["beta"].value
#     return alpha + beta * x
#
# plt.plot(x, y, 'b')
# plt.plot(x,
#         logistic(parameters = res_logistic, x = x),
#         'r')
# plt.plot(x,
#         linear(parameters = res_linear, x = x),
#         'g')
# plt.show()
# #%%
#
# corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))
# np.exp(res.lnprob)
#
#
# res.t
#
#
# res_logistic.params
#
# x_trial[0]
#
#
#
#
#
#
# y_trial
#
#
#
#
#
#
#
#
#
#
#
#
#
# #%%
# res.params
# res = lmfit.minimize(residual, method='emcee', nan_policy='omit', burn=300, steps=1000, thin=20, params=mi.params, is_weighted=False, float_behavior = 'posterior')
# res.chain.shape
# corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))
#
# res
#
# #%%
#
# print("median of posterior probability distribution")
# print('--------------------------------------------')
# lmfit.report_fit(res.params)
#
#
# import numpy as np
# import lmfit
# import matplotlib.pyplot as plt
# x = np.linspace(1, 10, 250)
# np.random.seed(0)
# y = 3.0 * np.exp(-x / 2) - 5.0 * np.exp(-(x - 0.1) / 10.) + 0.1 * np.random.randn(len(x))
# plt.plot(x, y, 'b')
# plt.show()
# p = lmfit.Parameters()
# p.add_many(('a1', 4.), ('a2', 4.), ('t1', 3.), ('t2', 3., True))
# def residual(p):
#     v = p.valuesdict()
#     return v['a1'] * np.exp(-x / v['t1']) + v['a2'] * np.exp(-(x - 0.1) / v['t2']) - y
# mi = lmfit.minimize(residual, p, method='Nelder', nan_policy='omit')
# lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)
