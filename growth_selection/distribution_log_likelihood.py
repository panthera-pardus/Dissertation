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
from sklearn.metrics import confusion_matrix
from sklearn_doc_confusion_matrix import plot_confusion_matrix

#%% Read Data
os.chdir("/Users/andour/Google Drive/projects/Dissertation")
synthetic_dataset = pickle.load(open( "simulated_data_curve_fit_2", "rb" ))
synthetic_dataset = synthetic_dataset.set_index(pd.Index(
        range(0,len(synthetic_dataset))
        ))

synthetic_dataset.head()


#%% Create a pdf for linear and logistic using (𝑥−𝑥∗)/σ
np.linspace(-4, -1, 19)
