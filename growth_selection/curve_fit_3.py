# %% Import cell
import pickle
import os
os.chdir("/Users/andour/Google Drive/projects/Dissertation/growth_selection/")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import random
from sklearn.metrics import confusion_matrix
from sklearn_doc_confusion_matrix import plot_confusion_matrix

#%% Read Data
os.chdir("/Users/andour/Google Drive/projects/Dissertation/data")
synthetic_dataset = pickle.load(open( "simulated_data_classification_2", "rb" ))
synthetic_dataset = synthetic_dataset.set_index(pd.Index(
        range(0,len(synthetic_dataset))
        ))

synthetic_dataset["x_array"] = synthetic_dataset.apply(lambda x : x["dataset"][:,1], axis = 1)
synthetic_dataset["y_array"] = synthetic_dataset.apply(lambda x : x["dataset"][:,0], axis = 1)
#%% Define functions

def linear(x,a,b):
    y = a * x + b
    return(y)

def sigmoid(x, L ,x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return (y)

# %% Goal : Get a column with y_pred_linear, y_pred_sigmoid to calculate MSE
def curve_fit_prediction(functional_form, dataset_x, dataset_y):

    if functional_form == 'linear':
        # p0 = [0, min(x_dataset)] # corresponds to the initial guess
        popt, pcov = curve_fit(linear, dataset_x, dataset_y, method = 'lm')
        y_fit = linear(dataset_x, *popt)
        dev = np.sqrt(np.diag(pcov))

    elif functional_form == 'logistic':

        p0 = [1, random.choice(dataset_x), random.uniform(0, 1)] # initial guess
        popt, pcov = curve_fit(f = sigmoid,
        xdata = dataset_x,
        ydata = dataset_y,
        method='trf',
        maxfev=100000, bounds=(0, [1, max(dataset_x), 1]))
        y_fit = sigmoid(dataset_x, *popt)
        dev = np.sqrt(np.diag(pcov))

    return(y_fit, dev)


synthetic_dataset['y_pred_linear'], synthetic_dataset['std_dev'] = zip(*synthetic_dataset.apply(lambda x : curve_fit_prediction(
dataset_x = x.x_array, dataset_y = x.y_array, functional_form = 'linear'), axis = 1))

synthetic_dataset['y_pred_logistic'], synthetic_dataset['std_dev'] = zip(*synthetic_dataset.apply(lambda x : curve_fit_prediction(
dataset_x = x.x_array, dataset_y = x.y_array, functional_form = 'logistic'), axis = 1))

synthetic_dataset['linear_mse'] = synthetic_dataset.apply(lambda x : mean_squared_error(
y_true = x.y_array,
y_pred = x.y_pred_linear), axis = 1)

synthetic_dataset['logistic_mse'] = synthetic_dataset.apply(lambda x : mean_squared_error(
y_true = x.y_array,
y_pred = x.y_pred_logistic), axis = 1)

#%% Classification and analysis
synthetic_dataset['curve_fit_classification'] = synthetic_dataset.apply(lambda x :
'logistic' if x.logistic_mse < x.linear_mse
else 'linear', axis = 1)

confusion_mat_by_bucket = synthetic_dataset.groupby('noise_bucket').\
apply(lambda x : confusion_matrix(x.label, x.curve_fit_classification))


plot_confusion_matrix(
cm = confusion_mat_by_bucket[0.1],
classes = ('logistic', 'linear'),
title = 'Noise bucket 0.1',
normalize = False)
plt.savefig("Consusion matrix example 1_2")


plot_confusion_matrix(
cm = confusion_mat_by_bucket[1],
classes = ('logistic', 'linear'),
title = 'Noise bucket 1.0',
normalize = False)
plt.savefig("Consusion matrix example 2_2")

#%% Save dataset

file = open("simulated_data_curve_fit_2", "wb")
pickle.dump(synthetic_dataset, file)
file.close()
