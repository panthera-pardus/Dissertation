#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:13:32 2019

@author: andour
"""

import data_simulation as simulate
import numpy as np
import pandas as pd
import os
import pickle

os.chdir("/Users/andour/Google Drive/projects/Dissertation")

#%%
# Create 10 000 examples of each functional form - each functional form will have
# 10 sets of 1000 examples with 10 buckets of variance noise

synthetic_data = pd.DataFrame(columns = ["dataset",
                                         "noise_bucket",
                                         "label",
                                         "trend"])

for variance_bucket in np.linspace(0.1,1,10):
        
    logistic_dataset = simulate.Noisy_logistic_generator(1000, var = variance_bucket)
    
    logistic_df = pd.DataFrame.from_dict(logistic_dataset, orient = "index").\
    rename(columns = {0 : "dataset"}).\
    assign(noise_bucket = np.array([variance_bucket] * 1000)).\
    assign(label = np.array(["logistic"] * 1000)).\
    assign(trend = np.array([False] * 1000))
    
    
    linear_dataset = simulate.Noisy_linear_generator(1000, var = variance_bucket)
    
    linear_df = pd.DataFrame.from_dict(linear_dataset, orient = "index").\
    rename(columns = {0 : 'dataset'}).\
    assign(noise_bucket = np.array([variance_bucket] * 1000)).\
    assign(label = np.array(["linear"] * 1000)).\
    assign(trend = np.array([False] * 1000))
    
    dataset = pd.concat([logistic_df,linear_df])
    
    synthetic_data = pd.concat([synthetic_data, dataset])
    
#%%
    
file = open("simulated_data_classification", "wb")
pickle.dump(synthetic_data, file)
file.close()

    


    

    
    
    
        
        
#        simulate.Noisy_linear_generator(250, var = variance_example)
        
