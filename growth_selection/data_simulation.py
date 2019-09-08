import numpy as np
import random
import collections as collec
import math
#%%
def Noisy_linear_generator(number_samples, var, drift = False):
     output = collec.Counter()
     for sample_i in range(number_samples):

         # Prepare data points
        B0 = random.uniform(0,10)
        upper_bound_x = int(random.uniform(100, 1000))
        upper_bound_y = random.uniform(100, 1000)
        r = (upper_bound_y - B0)/ (upper_bound_x)

        # Trace linear data and wrap in array
        simres = np.zeros((upper_bound_x+1,2),np.float)
        simres[:,1] = np.array(range(upper_bound_x + 1))
        simres[:,0] = np.array([t*r + B0 for t in range(upper_bound_x + 1)])

        # Rescale between 0 and 1
        simres[:,0] = (simres[:,0] - np.amin(simres[:,0]))/ (np.amax(simres[:,0]) - np.amin(simres[:,0]))

        #  Add the error
        simres[:, 0] = simres[:, 0] + np.random.normal(0,
                                                      var,
                                                      size = np.shape(simres[:, 1]))

        if drift == True :
            simres[:, 0] = simres[:, 0] + B0 + int(random.uniform(1,10))


        output[sample_i] = simres


     return(output)


#%%
def Noisy_logistic_generator(number_samples, var, drift = False):
     # Logistinc is given by y = L/(1 + exp(-r(x - x0)))
    output = collec.Counter()

    for sample_i in range(number_samples):

        # Prepare data points
        L = random.uniform(100, 1000)
        r = random.uniform(0, 1)
        upper_bound_x = int(random.uniform(100, 1000))
        x0 = int(random.uniform(0, upper_bound_x))

        # Trace sigmoid and wrap in array
        simres = np.zeros((upper_bound_x+1,2),np.float)
        simres[:,1] = np.array(range(upper_bound_x + 1))
        midpoint_difference = simres[:,1] - np.array([x0] * len(simres[:,1]))
        simres[:,0] = L/(1 + np.exp(-r * midpoint_difference))

        # Rescale between 0 and 1
        simres[:,0] = (simres[:,0] - np.amin(simres[:,0]))/ (np.amax(simres[:,0]) - np.amin(simres[:,0]))

        #  Add the error
        simres[:, 0] = simres[:, 0] + np.random.normal(0,
                                                      var,
                                                      size = np.shape(simres[:, 1]))

        if drift == True :
            simres[:, 0] = simres[:, 0] + B0 + int(random.uniform(1,10))


        output[sample_i] = simres
    return(output)


# Could redefine Noisy_logistic_generator in terms of sigmoid function
# Will do for linear and sigmoid for clarity
# def sigmoid_sim(x, L ,x0, k):
#     y = L / (1 + np.exp(-k*(x-x0)))
#     return (y)
#
# x = np.array(range(-100, 100))
# y = sigmoid_sim(x = x, L = 1, x0 = np.mean(x), k = 0.08)
# plt.plot(x,y)
