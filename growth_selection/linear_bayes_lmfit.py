'''
Linear Bayesian estimation for dataset
'''

#%% Imports and settings
import emcee
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner
import lmfit
import pickle
from sklearn_doc_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


os.chdir("/Users/andour/Google Drive/projects/Dissertation")
sns.set_style("darkgrid")

#%% Function needed and parameter settings:

def define_logistic_param():
    params_logistic = lmfit.Parameters()
    params_logistic.add('x0', value = 100)
    params_logistic.add('k', value = 1)
    params_logistic.add('__lnsigma', value=np.log(0.1), min=np.log(0.1), max=np.log(1))

    return params_logistic


def residual_linear(parameters):
    v = parameters.valuesdict()
    alpha = v['alpha']
    beta = v['beta']
    y_model = alpha + beta * x

    return y_model - y

# def linear_bayes_call(x, y, parameters = define_linear_param()):
#
#
#     res = lmfit.minimize(residual_linear,
#                         method='emcee',
#                         nan_policy='omit',
#                         burn=300,
#                         steps=1000,
#                         thin=20,
#                         params=define_linear_param(),
#                         is_weighted=False)

def define_linear_param():
    params_linear = lmfit.Parameters()
    params_linear.add('alpha',value =  0, min = 0)
    params_linear.add('beta', value = 0, min = 0)
    params_linear.add('__lnsigma', value=np.log(0.1), min=np.log(0.1), max=np.log(1))

    return params_linear


def residual_logistic(parameters):
    v = parameters.valuesdict()
    x0 = v['x0']
    k = v['k']
    y_model = 1/(1 + np.exp(-k * (x - x0)))

    return y_model - y


    # res = lmfit.minimize(residual_linear, params=parameters, method='nelder', nan_policy='omit')
    return res
#%% Read the data

synthetic_dataset = pickle.load(open( "simulated_data_classification_2", "rb" ))
synthetic_dataset = synthetic_dataset.set_index(pd.Index(
        range(0,len(synthetic_dataset))
        ))
synthetic_dataset["x_array"] = synthetic_dataset.apply(lambda x : x["dataset"][:,1], axis = 1)
synthetic_dataset["y_array"] = synthetic_dataset.apply(lambda x : x["dataset"][:,0], axis = 1)

# Let us sample 100 examples for each noise bucket
sample_data = synthetic_dataset.groupby(["noise_bucket", "label"])\
.apply(lambda x: x.sample(100, random_state = 120))
sample_data.size

#%% Run the MCMC estimate
# Because fucking lmfit.minimize needs a callable function lambda becomes impossible to implement
x_arrays = sample_data["x_array"]
y_arrays = sample_data["y_array"]
container_linear = {}
container_logistic = {}

for index, value in x_arrays.iteritems():

    x = value
    y = y_arrays[index]


    p = define_linear_param()
    mi = lmfit.minimize(residual_linear,
                        p,
                        method='nelder',
                        nan_policy='omit')

    container_linear[index[2]] = lmfit.minimize(residual_linear,
                        method='emcee',
                        nan_policy='omit',
                        burn=300,
                        steps=1000,
                        thin=20,
                        params=mi.params,
                        is_weighted=False)

    container_logistic[index[2]] = lmfit.minimize(residual_logistic,
                        method='emcee',
                        nan_policy='omit',
                        burn=300,
                        steps=1000,
                        thin=20,
                        params=define_logistic_param(),
                        is_weighted=False)
    print(index)


sample_data
file = open("bayes_linear_containers", "wb")
pickle.dump(container_linear, file)
file.close()


file = open("bayes_logistic_containers", "wb")
pickle.dump(container_logistic, file)
file.close()

linear_bayes_df = pd.DataFrame.from_dict(container_linear, orient='index')\
                                    .rename(columns = {0: "linear_bayes_result"})

logistic_bayes_df = pd.DataFrame.from_dict(container_logistic, orient='index')\
                                    .rename(columns = {0: "logistic_bayes_result"})

bayes_master_df = logistic_bayes_df.merge(linear_bayes_df,
                                        left_index = True,
                                        right_index = True)




sample_data = synthetic_dataset.merge(bayes_master_df,
                                    left_index = True,
                                    right_index = True,
                                    how = "inner")

sample_data.head()
sample_data["BIC_logistic"] = sample_data.apply(lambda x : x.logistic_bayes_result.bic, axis = 1)
sample_data["BIC_linear"] = sample_data.apply(lambda x : x.linear_bayes_result.bic, axis = 1)

sample_data["bayes_preditcion"] = sample_data.apply(lambda x : "linear" if x.BIC_linear > x.BIC_logistic else "logistic" , axis = 1)

confusion_mat_by_bucket = sample_data.groupby('noise_bucket').\
apply(lambda x : confusion_matrix(x.label, x.bayes_preditcion))


plot_confusion_matrix(
cm = confusion_mat_by_bucket[1],
classes = ('logistic', 'linear'),
title = 'Noise bucket 1',
normalize = False)













# sample_data['linear_bayes_result'] = sample_data.apply(lambda x : lmfit.minimize(residual_linear,
#                                                                                 method='emcee',
#                                                                                 nan_policy='omit',
#                                                                                 burn=300,
#                                                                                 steps=1000,
#                                                                                 thin=20,
#                                                                                 params=define_linear_param(),
#                                                                                 is_weighted=False),
#                                                                                     axis = 1)
container_logistic[1422]
container_linear[1422]
