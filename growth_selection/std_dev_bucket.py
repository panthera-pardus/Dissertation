# Parameter estimation analysis

#%% Import cell
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
os.chdir("/Users/andour/Google Drive/projects/Dissertation")
synthetic_dataset = pickle.load(open( "simulated_data_curve_fit_2", "rb" ))
synthetic_dataset = synthetic_dataset.set_index(pd.Index(
        range(0,len(synthetic_dataset))
        ))

synthetic_dataset["x_array"] = synthetic_dataset.apply(lambda x : x["dataset"][:,1], axis = 1)
synthetic_dataset["y_array"] = synthetic_dataset.apply(lambda x : x["dataset"][:,0], axis = 1)


#%% Take only the logistic true positives
synthetic_dataset.head()
synthtic_logistic = synthetic_dataset[(synthetic_dataset.curve_fit_classification == 'logistic')\
 & (synthetic_dataset.curve_fit_classification == 'logistic')] # logistic example

#%% Get pooled standard deviation L ,x0, k
synthtic_logistic["std_dev_weighted"] = synthtic_logistic.apply(lambda x : np.asarray(x["std_dev"]), axis = 1) * synthtic_logistic.apply(lambda x : len(x["x_array"]), axis = 1)

synthtic_logistic["L_std_dev_weighted"] = synthtic_logistic.apply(lambda x : x["std_dev_weighted"][0], axis = 1)
synthtic_logistic["x0_std_dev_weighted"] = synthtic_logistic.apply(lambda x : x["std_dev_weighted"][1], axis = 1)
synthtic_logistic["k_std_dev_weighted"] = synthtic_logistic.apply(lambda x : x["std_dev_weighted"][2], axis = 1)
std_dev_by_bucket = synthtic_logistic.groupby("noise_bucket")[["L_std_dev_weighted",
                                            "x0_std_dev_weighted",
                                            "k_std_dev_weighted"]].\
                                            agg("sum").\
                                            apply(lambda x : x / len(x))

# re-scaled standard deviation
std_dev_by_bucket_scaled = (std_dev_by_bucket - std_dev_by_bucket.iloc[0,:])/std_dev_by_bucket.iloc[0,:]
std_dev_by_bucket_scaled.plot(y = "L_std_dev_weighted")
plt.title("weighted average L standard deviation (index noise_bucket = 0.1)")
plt.savefig("std_dev_L")
std_dev_by_bucket_scaled.plot(y = "x0_std_dev_weighted")
plt.title("weighted average x0 standard deviation (index noise_bucket = 0.1)")
plt.savefig("std_dev_x0")
std_dev_by_bucket_scaled.plot(y = "k_std_dev_weighted")
plt.title("weighted average k standard deviation (index noise_bucket = 0.1)")
plt.savefig("std_dev_k")



#%%
def sigmoid(x, L ,x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return (y)

popt, pcov = curve_fit(f = sigmoid,
xdata = x,
ydata = y,
method='trf',
maxfev=100000, bounds=(0, [1, max(x), 1]))

#%% Compute the weighted standard deviation for true positive logistic
pcov
perr = np.sqrt(np.diag(pcov))
perr

perr_weighted =
