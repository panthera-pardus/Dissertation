# Script to create the histogram to demonstrate normality of error term
#%% Import cell
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("/Users/andour/Google Drive/projects/Dissertation/Final data")
synthetic_dataset = pickle.load(open( "simulated_data_freq", "rb" ))
sns.set_palette("Dark2")
sns.set_style("darkgrid")

#%% Sample a row and plot
synthetic_dataset.iloc[3999]
# plt.plot(synthetic_dataset.iloc[3999]["x_array"],synthetic_dataset.iloc[3999]["y_array"])
error_eg = synthetic_dataset.iloc[3999]["y_array"] - synthetic_dataset.iloc[3999]["y_pred_linear"]
plt.hist(error_eg, color= sns.xkcd_rgb["cerulean blue"])
plt.title("Example of the distribution of the error of linear model where noise_bucket = 1")
plt.savefig("/Users/andour/Google Drive/projects/Dissertation/Final figures/freq_norm_distrib")
