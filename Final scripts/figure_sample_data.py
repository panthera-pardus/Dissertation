#%% Import necessary modules
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

#%% Import data and set directory
os.chdir("/Users/andour/Google Drive/projects/Dissertation/Final data")
synthetic_dataset = pickle.load(open( "simulated_data_raw", "rb"))
synthetic_dataset.head()

os.chdir("/Users/andour/Google Drive/projects/Dissertation/Final figures")

#%% Create plots for a sample dataset of low buckets
X_logistic_low_example = synthetic_dataset.loc[(synthetic_dataset["noise_bucket"] == 0.1) &
                     (synthetic_dataset["label"] == "logistic") &
                     (synthetic_dataset["drift"] == False)].\
                     sample(1, random_state = 1)["x_array"].\
                     to_numpy()[0]


Y_logistic_low_example = synthetic_dataset.loc[(synthetic_dataset["noise_bucket"] == 0.1) &
                     (synthetic_dataset["label"] == "logistic") &
                     (synthetic_dataset["drift"] == False)].\
                     sample(1, random_state = 1)["y_array"].\
                     to_numpy()[0]


X_linear_low_example = synthetic_dataset.loc[(synthetic_dataset["noise_bucket"] == 0.1) &
                     (synthetic_dataset["label"] == "linear") &
                     (synthetic_dataset["drift"] == False)].\
                     sample(1, random_state = 3)["x_array"].\
                     to_numpy()[0]


Y_linear_low_example = synthetic_dataset.loc[(synthetic_dataset["noise_bucket"] == 0.1) &
                     (synthetic_dataset["label"] == "linear") &
                     (synthetic_dataset["drift"] == False)].\
                     sample(1, random_state = 3)["y_array"].\
                     to_numpy()[0]


sns.set_palette("Dark2")
sns.lineplot(X_logistic_low_example, Y_logistic_low_example, color = sns.xkcd_rgb["cerulean blue"], label = "Logistic")
sns.lineplot(X_linear_low_example, Y_linear_low_example, color = sns.xkcd_rgb["bright orange"] ,label = "Linear").set(xlim=(0,120))
plt.title("Example of linear and logistic datasets with \u03C3 = 0.1")
plt.savefig("linear_v_logistic_low.png")


#%% Create plots for a sample dataset of high buckets

X_logistic_high_example = synthetic_dataset.loc[(synthetic_dataset["noise_bucket"] == 1) &
                     (synthetic_dataset["label"] == "logistic") &
                     (synthetic_dataset["drift"] == False)].\
                     sample(1, random_state = 1)["x_array"].\
                     to_numpy()[0]


Y_logistic_high_example = synthetic_dataset.loc[(synthetic_dataset["noise_bucket"] == 1) &
                     (synthetic_dataset["label"] == "logistic") &
                     (synthetic_dataset["drift"] == False)].\
                     sample(1, random_state = 1)["y_array"].\
                     to_numpy()[0]


X_linear_high_example = synthetic_dataset.loc[(synthetic_dataset["noise_bucket"] == 1) &
                     (synthetic_dataset["label"] == "linear") &
                     (synthetic_dataset["drift"] == False)].\
                     sample(1, random_state = 1)["x_array"].\
                     to_numpy()[0]


Y_linear_high_example = synthetic_dataset.loc[(synthetic_dataset["noise_bucket"] == 1) &
                     (synthetic_dataset["label"] == "linear") &
                     (synthetic_dataset["drift"] == False)].\
                     sample(1, random_state = 1)["y_array"].\
                     to_numpy()[0]


sns.set_style("darkgrid")
sns.set_palette("Dark2")
sns.lineplot(X_logistic_high_example, Y_logistic_high_example, color = sns.xkcd_rgb["cerulean blue"] ,label = "Logistic")
sns.lineplot(X_linear_high_example, Y_linear_high_example, color = sns.xkcd_rgb["bright orange"] ,label = "Linear").set(xlim=(0,500))
plt.title("Example of linear and logistic datasets with \u03C3 = 1")
plt.savefig("linear_v_logistic_high.png")
