#%%
os.chdir("/Users/andour/Google Drive/projects/Dissertation/growth_selection")
import data_simulation_2 as simulate
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt


#%%
os.chdir("/Users/andour/Google Drive/projects/Dissertation")

a = simulate.Noisy_logistic_generator_2(1, 1)
ax = a["dataset"][0][:,1]
ay = a["dataset"][0][:,0]

a_plt = plt.plot(ax,ay)

#%%
b = simulate.Noisy_linear_generator_2(1, 1)
bx = b["dataset"][0][:,1]
by = b["dataset"][0][:,0]

b_plt = plt.plot(bx,by)
