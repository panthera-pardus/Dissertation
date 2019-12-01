# Create plots for the classifications
#%% Import required functions

import pickle
import os
os.chdir("/Users/andour/Google Drive/projects/Dissertation/Final scripts/")
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn_doc_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt

sns.set_palette("Dark2")
sns.set_style("whitegrid")
