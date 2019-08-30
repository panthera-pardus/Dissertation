import numpy as np
import random
import collections as collec
import math
#%%
def Noisy_logistic_generator(number_samples, var, drift = False):
    '''Discrete stochastic logistic model simulation.  Carrying capacity K,
intrinsic growth rate r, initial population size (inoculum) N0.  Returns an
array with a (discrete) time column and a (continuous) population size column.'''

# Based on function : https://gist.github.com/CnrLwlss/4431230
    
#    random.seed(31)
#    np.random.seed(31)
    output = collec.Counter()

    for sample_i in range(number_samples):
        
        N0 = int(random.uniform(1,10))
        K = int(random.uniform(100, 1000))
        r = random.uniform(0,1)
        
        # Event size and prep
        eventNo=K-N0
        unifs=np.random.uniform(size=eventNo)
        simres=np.zeros((eventNo+1,2),np.float)
        simres[:,1]=range(N0,K+1)
        
        # exponential random numbers using the inversion method
        dts=-np.log(1-unifs)/(r*simres[1:,1]*(1-simres[1:,1]/K))
        simres[1:,0]=np.cumsum(dts)
        simres = simres[:-1]
        
        # rescale between 0  and 1
        simres[:,1] = (simres[:,1] - np.amin(simres[:,1]))/ (np.amax(simres[:,1]) - np.amin(simres[:,1]))
        
        # Add the error 
        simres[:, 1] = simres[:, 1] + np.random.normal(0,
                                                      var,
                                                      size = np.shape(simres[:, 1]))
        
        if drift == True :
            simres[:, 1] = simres[:, 1] + N0 = int(random.uniform(1,10))
                    
        
        output[sample_i] = simres
    return(output)
    

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
        simres[:,0] = np.array(range(upper_bound_x + 1))
        simres[:,1] = np.array([t*r + B0 for t in range(upper_bound_x + 1)])
        
        # Rescale between 0 and 1
        simres[:,1] = (simres[:,1] - np.amin(simres[:,1]))/ (np.amax(simres[:,1]) - np.amin(simres[:,1]))
        
        #  Add the error 
        simres[:, 1] = simres[:, 1] + np.random.normal(0,
                                                      var,
                                                      size = np.shape(simres[:, 1]))
        
        if drift == True :
            simres[:, 1] = simres[:, 1] + N0 = int(random.uniform(1,10))
        
        
        output[sample_i] = simres

     
     return(output)
        
     
        
        
        
    

