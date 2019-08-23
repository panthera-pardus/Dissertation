import numpy as np
import random
import collections as collec

def Noisy_logistic_generator(number_samples):
    '''Discrete stochastic logistic model simulation.  Carrying capacity K,
intrinsic growth rate r, initial population size (inoculum) N0.  Returns an
array with a (continuous) time column and a (discrete) population size column.'''

# Based on function : https://gist.github.com/CnrLwlss/4431230
    
    random.seed(31)
    np.random.seed(31)
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
        
        # Let us add the error 
        simres[:, 0] = simres[:, 0] + np.random.normal(np.average(simres[:, 0]),
                                                      np.std(simres[:, 0]),
                                                      size = np.shape(simres[:, 0]))
        
        output[sample_i] = simres
    return(output)
