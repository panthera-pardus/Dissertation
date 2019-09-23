# Accuracy vs noise and F1 vs noise and ROC curve

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
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, roc_auc_score
from sklearn_doc_confusion_matrix import plot_confusion_matrix

#%% Read Data
os.chdir("/Users/andour/Google Drive/projects/Dissertation")
synthetic_dataset = pickle.load(open( "simulated_data_curve_fit_2", "rb" ))
synthetic_dataset = synthetic_dataset.set_index(pd.Index(
        range(0,len(synthetic_dataset))
        ))

synthetic_dataset.head()


#%% Create a pdf for linear and logistic using (𝑥−𝑥∗)/σ

def dummy_creator(x):
    if x == "logistic":
        return(1)
    else:
        return(0)


synthetic_dataset["dummy_label"] = synthetic_dataset["label"].apply(dummy_creator)
synthetic_dataset["dummy_classification"] = synthetic_dataset["curve_fit_classification"].apply(dummy_creator)

F1_by_bucket = synthetic_dataset.groupby("noise_bucket").\
apply(lambda x : f1_score(x.dummy_label, x.dummy_classification))

accuracy_by_bucket = synthetic_dataset.groupby("noise_bucket").\
apply(lambda x : accuracy_score(x.dummy_label, x.dummy_classification))

ROC_by_bucket = synthetic_dataset.groupby("noise_bucket").\
apply(lambda x : roc_curve(x.dummy_label, x.dummy_classification))


F1_by_bucket.plot()
accuracy_by_bucket.plot()

for noise_bucket in np.linspace(0.1,1, 10):
    plt.plot(ROC_by_bucket[noise_bucket][0], ROC_by_bucket[noise_bucket][1], label = "bucket {0}".format(noise_bucket))
    plt.title('ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
