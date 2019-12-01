# Script to curve fit and calculate the MSE, Mean Absolute Error and R-squared adjusted
# and Chi square classification

#%% Import cell
import pickle
import os
os.chdir("/Users/andour/Google Drive/projects/Dissertation/growth_selection/")
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
from scipy.stats import chi2
from RegscorePy import aic, bic
import time

#%% Read dataset and define functions
os.chdir("/Users/andour/Google Drive/projects/Dissertation/Final data")
synthetic_dataset = pickle.load(open( "simulated_data_raw", "rb" ))

def linear(x,a,b):
    y = a * x + b
    return(y)

def sigmoid(x, L ,x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return (y)

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

def compute_rss(y_true, y_predicted):
    residual = (y_true - y_predicted)

    return (np.sum(residual ** 2))

def compute_dof(y_predicted, logistic_bool):
    if logistic_bool == True:
        dof = len(y_predicted) - 3
    else :
        dof = len(y_predicted) - 2

    return(dof)

def chi2_likelihood(y_true, y_predicted, logistic_bool):
    chi = compute_rss(y_true, y_predicted)
    dof = compute_dof(y_predicted, logistic_bool)

    return (chi2(dof).pdf(10))

def residual_entropy(std_dev_error):
    return (0.5 + np.log10(np.sqrt(2* np.pi * std_dev_error)))

def shanon_bic(std_dev_error, bic, sample_size):
    return(bic - sample_size * np.log10(residual_entropy(std_dev_error)))

def shanon_aic(std_dev_error, aic, sample_size):
    return(aic - sample_size * np.log10(residual_entropy(std_dev_error)))

#%% Call functions and calculate the MSE, MAE and adjusted R^2
synthetic_dataset['y_pred_linear'], synthetic_dataset['std_dev'] = zip(*synthetic_dataset.apply(lambda x : curve_fit_prediction(
dataset_x = x.x_array, dataset_y = x.y_array, functional_form = 'linear'), axis = 1))

synthetic_dataset['y_pred_logistic'], synthetic_dataset['std_dev'] = zip(*synthetic_dataset.apply(lambda x : curve_fit_prediction(
dataset_x = x.x_array, dataset_y = x.y_array, functional_form = 'logistic'), axis = 1))

file = open("simulated_data_freq_time_stat", "wb") # Open file to measure the time taken for each estimation
time_dict = {}

# Calculate MSE
start = time.time()

synthetic_dataset['linear_mse'] = synthetic_dataset.apply(lambda x : mean_squared_error(
y_true = x.y_array,
y_pred = x.y_pred_linear), axis = 1)

synthetic_dataset['logistic_mse'] = synthetic_dataset.apply(lambda x : mean_squared_error(
y_true = x.y_array,
y_pred = x.y_pred_logistic), axis = 1)

end = time.time()
time_dict["mse"] = [end - start]

# Calculate MAE
start = time.time()

synthetic_dataset['linear_mae'] = synthetic_dataset.apply(lambda x : mean_absolute_error(
y_true = x.y_array,
y_pred = x.y_pred_linear), axis = 1)

synthetic_dataset['logistic_mae'] = synthetic_dataset.apply(lambda x : mean_absolute_error(
y_true = x.y_array,
y_pred = x.y_pred_logistic), axis = 1)

end = time.time()
time_dict["mae"] = [end - start]

# Calculate adjusted R^2
start = time.time()

synthetic_dataset['linear_r2'] = synthetic_dataset.apply(lambda x : r2_score(
y_true = x.y_array,
y_pred = x.y_pred_linear), axis = 1)

synthetic_dataset['logistic_r2'] = synthetic_dataset.apply(lambda x : r2_score(
y_true = x.y_array,
y_pred = x.y_pred_logistic), axis = 1)

end = time.time()
time_dict["r2"] = [end - start]

# Calculating chi2 likelihood
start = time.time()

synthetic_dataset['likelihood_linear'] = synthetic_dataset.apply(lambda x : chi2_likelihood(
y_true = x.y_array,
y_predicted= x.y_pred_linear,
logistic_bool = False), axis = 1)

synthetic_dataset['likelihood_logistic'] = synthetic_dataset.apply(lambda x : chi2_likelihood(
y_true = x.y_array,
y_predicted= x.y_pred_logistic,
logistic_bool = True), axis = 1)

end = time.time()
time_dict["chi2"] = [end - start]

# Calculating aic
start = time.time()

synthetic_dataset['aic_linear'] = synthetic_dataset.apply(lambda x : aic.aic(
y = x.y_array,
y_pred= x.y_pred_linear,
p = 2), axis = 1)

synthetic_dataset['aic_logistic'] = synthetic_dataset.apply(lambda x : aic.aic(
y = x.y_array,
y_pred = x.y_pred_logistic,
p = 3), axis = 1)

end = time.time()
time_dict["aic"] = [end - start]

# Calculating bic
start = time.time()

synthetic_dataset['bic_linear'] = synthetic_dataset.apply(lambda x : bic.bic(
y = x.y_array,
y_pred = x.y_pred_linear,
p = 2), axis = 1)

synthetic_dataset['bic_logistic'] = synthetic_dataset.apply(lambda x : bic.bic(
y = x.y_array,
y_pred = x.y_pred_logistic,
p = 3), axis = 1)

end = time.time()
time_dict["bic"] = [end - start]

time_df = pd.DataFrame.from_dict(time_dict)
pickle.dump(time_df, file)
file.close()

# Calculating Shanon standardized BIC and AIC
synthetic_dataset['shanon_bic_logistic'] = synthetic_dataset.apply(lambda x : shanon_bic(std_dev_error = np.std(x.y_array - x.y_pred_logistic),
bic = x.bic_logistic,
sample_size = len(x.y_pred_logistic)), axis = 1)

synthetic_dataset["shanon_bic_linear"] = synthetic_dataset.apply(lambda x : shanon_bic(std_dev_error = np.std(x.y_array - x.y_pred_linear),
bic = x.bic_linear,
sample_size = len(x.y_pred_linear)), axis = 1)


synthetic_dataset['shanon_aic_logistic'] = synthetic_dataset.apply(lambda x : shanon_aic(std_dev_error = np.std(x.y_array - x.y_pred_logistic),
aic = x.aic_logistic,
sample_size = len(x.y_pred_logistic)), axis = 1)

synthetic_dataset["shanon_aic_linear"] = synthetic_dataset.apply(lambda x : shanon_aic(std_dev_error = np.std(x.y_array - x.y_pred_linear),
aic = x.aic_linear,
sample_size = len(x.y_pred_linear)), axis = 1)

#%% Classification for mse, mae, r^2
synthetic_dataset['mse_classification'] = synthetic_dataset.apply(lambda x : 'logistic' if x.logistic_mse < x.linear_mse
                                                                                       else 'linear', axis = 1)

synthetic_dataset['mae_classification'] = synthetic_dataset.apply(lambda x : 'logistic' if x.logistic_mae < x.linear_mae
                                                                                       else 'linear', axis = 1)

synthetic_dataset['r2_classification'] = synthetic_dataset.apply(lambda x : 'logistic' if x.logistic_r2 > x.linear_r2
                                                                                       else 'linear', axis = 1)

synthetic_dataset['chi2_classification'] = synthetic_dataset.apply(lambda x : 'logistic' if x.likelihood_logistic > x.likelihood_linear
                                                                                       else 'linear', axis = 1)

synthetic_dataset['aic_classification'] = synthetic_dataset.apply(lambda x : 'logistic' if x.aic_logistic < x.aic_linear
                                                                                       else 'linear', axis = 1)

synthetic_dataset['bic_classification'] = synthetic_dataset.apply(lambda x : 'logistic' if x.bic_logistic < x.bic_linear
                                                                                       else 'linear', axis = 1)

synthetic_dataset['shanon_bic_classification'] = synthetic_dataset.apply(lambda x : 'logistic' if x.shanon_bic_logistic < x.shanon_bic_linear
                                                                                       else 'linear', axis = 1)

synthetic_dataset['shanon_aic_classification'] = synthetic_dataset.apply(lambda x : 'logistic' if x.shanon_aic_logistic < x.shanon_aic_linear
                                                                                       else 'linear', axis = 1)
#%% Save dataset
file = open("simulated_data_freq", "wb")
pickle.dump(synthetic_dataset, file)
file.close()
