#%% import necessary modules
import os
os.chdir("/Users/andour/Google Drive/projects/Dissertation/Final scripts")
import data_simulation_fn_clean as simulate
import numpy as np
import pandas as pd
import pickle


#%% Set directory to save data and prepare the dataframe to hole the data
# Create 1000 examples of each functional form - each functional form will have
# 10 sets of 1000 examples (with 10 buckets of variance noise)

os.chdir("/Users/andour/Google Drive/projects/Dissertation/Final data")
synthetic_data = pd.DataFrame(columns = ["dataset",
                                         "parameters",
                                         "noise_bucket",
                                         "label",
                                         "drift"])

#%% Loop to create the datasets
for variance_bucket in np.linspace(0.1,1,10):

    logistic_dataset = simulate.Noisy_logistic_generator_2(100, variance_bucket)

    logistic_df = pd.DataFrame.from_dict(logistic_dataset).\
    assign(noise_bucket = np.array([variance_bucket] * 100)).\
    assign(label = np.array(["logistic"] * 100)).\
    assign(drift = np.array([False] * 100))

    logistic_dataset_drift = simulate.Noisy_logistic_generator_2(100, variance_bucket, True)

    logistic_df_drift = pd.DataFrame.from_dict(logistic_dataset_drift).\
    assign(noise_bucket = np.array([variance_bucket] * 100)).\
    assign(label = np.array(["logistic"] * 100)).\
    assign(drift = np.array([True] * 100))


    linear_dataset = simulate.Noisy_linear_generator_2(100, variance_bucket)

    linear_df = pd.DataFrame.from_dict(linear_dataset).\
    assign(noise_bucket = np.array([variance_bucket] * 100)).\
    assign(label = np.array(["linear"] * 100)).\
    assign(drift = np.array([False] * 100))

    linear_dataset_drift = simulate.Noisy_linear_generator_2(100, variance_bucket, True)

    linear_df_drift = pd.DataFrame.from_dict(linear_dataset).\
    assign(noise_bucket = np.array([variance_bucket] * 100)).\
    assign(label = np.array(["linear"] * 100)).\
    assign(drift = np.array([True] * 100))

    dataset = pd.concat([logistic_df,linear_df,
                        logistic_df_drift, linear_df_drift], sort=False)

    synthetic_data = pd.concat([synthetic_data, dataset], sort=False)

# Unnesting the columns that can be useful and check the data
synthetic_data["x_array"] = synthetic_data.apply(lambda x : x["dataset"][:,1], axis = 1)
synthetic_data["y_array"] = synthetic_data.apply(lambda x : x["dataset"][:,0], axis = 1)
synthetic_data.head()

#%% Save the data
file = open("simulated_data_raw", "wb")
pickle.dump(synthetic_data, file)
file.close()
