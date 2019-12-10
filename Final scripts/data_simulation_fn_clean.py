# %% Import cell
import numpy as np
import random
import collections as collec
import math
from lmfit import Model, Parameter, report_fit
from scipy.optimize import curve_fit

#%% Define basic functions used - a sigmoid and a linear function
def sigmoid_sim(x, L ,x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return (y)

def linear_sim(a, b, x):
    y = a * x + b
    return(y)

#%% Define logistic data generation function
def Noisy_logistic_generator_2(number_samples, variance_error, drift_bool = False):
    '''Function to create collection of datasets that follows the
    functional for y = L/(1 + exp(-k(x - x0))) with a additive noise paramater
    and a drift when called upon'''

    # Get collectors
    output = {
    'parameters' : collec.Counter(),
     'dataset' : collec.Counter()
     }

    # Run loop for numbers of smaples required
    for sample in range(number_samples):
        # Define parameters
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
        simres[:,0] = sigmoid_sim(x, L, x0, k)

        if drift_bool == False :
            simres[:,0] = (simres[:,0] - np.amin(simres[:,0]))/ (np.amax(simres[:,0]) - np.amin(simres[:,0]))
            y = simres[:,0]
            p0 = [1, random.choice(x), random.uniform(0, 1)] # initial guess
            popt, pcov = curve_fit(f = sigmoid_sim,
            xdata = x,
            ydata = y,
            method='trf',
            maxfev=100000, bounds=(0, [1, np.amax(x), 1]))
            L ,x0, k = popt
            simres[:,0] += np.random.normal(0, variance_error, size = np.shape(simres[:,1]))

            output['parameters'][sample] = {
            'x0': x0,
            'L' : L,
            'k' : k
            }

        elif drift_bool == True :
            drift = random.uniform(100, 500)
            random_index = int(random.uniform(len(simres[:,0])/2,
                                              len(simres[:,0])))
            simres[:,0][random_index:len(simres[:,0])] = simres[:,0][random_index:len(simres[:,0])] + drift * simres[:,1][random_index:len(simres[:,1])]
            simres[:,0] = (simres[:,0] - np.amin(simres[:,0]))/ (np.amax(simres[:,0]) - np.amin(simres[:,0]))
            y = simres[:,0]
            p0 = [1, random.choice(x), random.uniform(0, 1)] # initial guess
            popt, pcov = curve_fit(f = sigmoid_sim,
            xdata = x,
            ydata = y,
            method='trf',
            maxfev=100000, bounds=(0, [1, np.amax(x), 1]))
            L ,x0, k = popt
            simres[:,0] += np.random.normal(0, variance_error, size = np.shape(simres[:,1]))

            output['parameters'][sample] = {
            'x0': x0,
            'L' : L,
            'k' : k,
            'drift' : drift
            }

        output['dataset'][sample] = simres


    return(output)


#%% Define linear data generation function
def Noisy_linear_generator_2(number_samples, variance_error, drift_bool = False):
    '''Function to create collection of datasets that follows the
    functional for y = a * x + b with a additive noise paramater
    and a drift when called upon'''


    # Get collectors
    output = {
    'parameters' : collec.Counter(),
     'dataset' : collec.Counter()
     }

    # Run loop for numbers of smaples required
    for sample in range(number_samples):
        #Define parameters
        a = random.uniform(0, 0.05)
        b = random.uniform(0,0.2)
        x = np.array(range(int(random.uniform(100, 1000)) + 1)) # adding 1 to avoid having arrays with only 0

        # Create the array and the corresponding linear function and save it
        # Note that with the linear a rescaling is necssary for values to be [0,1]
        simres = np.zeros((np.amax(x)+1,2),np.float)
        simres[:,1] = x
        simres[:,0] = linear_sim(a, b, x)

        if drift_bool == False :
            simres[:,0] = (simres[:,0] - np.amin(simres[:,0]))/ (np.amax(simres[:,0]) - np.amin(simres[:,0]))
            y = simres[:,0]
            popt, pcov = curve_fit(linear_sim, x, y, method = 'lm')
            a,b = popt
            simres[:,0] += np.random.normal(0, variance_error, size = np.shape(simres[:,1]))

            output['parameters'][sample] = {
            'a' : a,
            'b' : b
            }

        elif drift_bool == True :
            drift = random.uniform(100, 500)
            random_index = int(random.uniform(len(simres[:,0])/2,
                                              len(simres[:,0])))
            simres[:,0][random_index:len(simres[:,0])] = simres[:,0][random_index:len(simres[:,0])] + drift * simres[:,1][random_index:len(simres[:,1])]
            simres[:,0] = (simres[:,0] - np.amin(simres[:,0]))/ (np.amax(simres[:,0]) - np.amin(simres[:,0]))
            y = simres[:,0]
            popt, pcov = curve_fit(linear_sim, x, y, method = 'lm')
            a,b = popt
            simres[:,0] += np.random.normal(0, variance_error, size = np.shape(simres[:,1]))

            output['parameters'][sample] = {
            'a' : a,
            'b' : b,
            'drift' : drift
            }

        output['dataset'][sample] = simres

    return(output)



#%% Test the function
# import matplotlib.pyplot as plt
# a = Noisy_logistic_generator_2(2, 1, drift_bool=True)
# plt.plot(a['dataset'][0][:,1], a['dataset'][0][:,0])
# #%%
# b = Noisy_linear_generator_2(2, 1, True)
# plt.plot(b['dataset'][0][:,1], b['dataset'][0][:,0])
