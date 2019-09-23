# %% Import cell
import numpy as np
import random
import collections as collec
import math

#%% Define basic functions used
def sigmoid_sim(x, L ,x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return (y)

def linear_sim(a, b, x):
    y = a * x + b
    return(y)

#%%
def Noisy_logistic_generator_2(number_samples, variance_error, trend = False):
    # y = L/(1 + exp(-k(x - x0)))

    # Get collectors
    output = {
    'parameters' : collec.Counter(),
     'dataset' : collec.Counter()
     }

    # Run loop for numbers of smaples required
    for sample in range(number_samples):
        #Define parameters
        x = np.array(range(int(random.uniform(100, 1000)) + 1))

        L = random.uniform(0.9, 1.1)

        x0 = int(random.uniform(
        np.amax(x) * 0.25,
        np.amax(x) * 0.75
        ))

        k = random.uniform(0.5, 2)

        # Create the array and the corresponding sifmoid function and save it
        simres = np.zeros((np.amax(x)+1,2),np.float)
        simres[:,1] = x
        simres[:,0] = sigmoid_sim(x, L, x0, k) + np.random.normal(0,
                                                      variance_error,
                                                      size = np.shape(simres[:,1]))
        output['parameters'][sample] = {
        'x0': x0,
        'L' : L,
        'k' : k
        }

        output['dataset'][sample] = simres

    return(output)


#%%
def Noisy_linear_generator_2(number_samples, variance_error, trend = False):
    # y = a * x + b

    # Get collectors
    output = {
    'parameters' : collec.Counter(),
     'dataset' : collec.Counter()
     }

    # Run loop for numbers of smaples required
    for sample in range(number_samples):
        #Define parameters
        a = random.uniform(100, 1000)
        b = random.uniform(0,100)
        x = np.array(range(int(random.uniform(100, 1000)) + 1))

        # Create the array and the corresponding sifmoid function and save it
        simres = np.zeros((np.amax(x)+1,2),np.float)
        simres[:,1] = x
        simres[:,0] = linear_sim(a, b, x)
        simres[:,0] = (simres[:,0] - np.amin(simres[:,0]))/ (np.amax(simres[:,0]) - np.amin(simres[:,0]))
        simres[:,0] += np.random.normal(0, variance_error, size = np.shape(simres[:,1]))
        simres[:,0] = (simres[:,0] - np.amin(simres[:,0]))/ (np.amax(simres[:,0]) - np.amin(simres[:,0]))

        output['parameters'][sample] = {
        'a' : a,
        'b' : b
        }

        output['dataset'][sample] = simres

    return(output)







#%%
Noisy_logistic_generator_2(2, 0)
#%%
Noisy_linear_generator_2(2, 0)

# %%
# Noisy_linear_generator_2(2, 0)
