# import necessary packages
import random
import numpy as np
from scipy.integrate import odeint
from numba import jit  # significant speed-up for scipy's odeint
import os

def constructNetwork(N = 50, k = 20, gamma = 0.4):
    """
    @param N: the number of oscillators in the system (int)
    @param k: degree of the network (int)
    @param gamma: minimal frequency gap (float)
    """
    
    # the number of links in the network
    L = N*k//2

    # the natural frequencies follow a uniform distribution
    omega = np.random.uniform(low = 0, high = 1, size = N)

    # initialize the adjacency matrix
    A = np.zeros((N,N), dtype = "int")

    # construct FGC random network
    counter = 0
    while counter < L:
        i, j = np.random.choice(range(N), size = 2, replace = False)
        if abs(omega[i]-omega[j]) > gamma and A[i][j] == 0:
            A[i][j] = 1
            counter += 1 

    return A, omega

def kuramoto(d, A, omega, phi0, noise = 0, T = 100):
    """
    @param d: the coupling strength (float)
    @param A: adjacency matrix of the FGC network (2D numpy array)
    @param omega: natural frequencies (list or 1d numpy array)
    @param phi0: initial condition for the phases (list or 1d numpy array)
    @param noise: standard deviation of the Gaussian noise (float)
    @param T: length of the integration (int)
    @param M: order of the GC model (int)
    @param save: whether or not to save the result to file (boolean)
    """

    #### 1. INITIALIZATION
            
    # number of oscillators in the network
    N = len(A)   
    
    #### 2. INTEGRATION

    # time intervals on which to perform the integration:
    ini = 100
    T = T + ini
    t = np.linspace(0,T,T+1)
    
    # function to integrate
    @jit
    def f(phi,t):
        dphidt = np.array([omega[i] for i in range(N)])
        for i in range(N):
            for j in range(N):
                dphidt[i] += d*A[j][i]*np.sin(phi[j]-phi[i])
        return dphidt
    
    # perform integration
    sol = odeint(f,phi0,t)
    # sol is the theta matrix in PTE calculation, and the x matrix in GC

    # we assume the first ini time steps are used to relax the system
    theta = sol[ini:,:]

    # add random noise to the solution
    theta = theta + np.random.normal(0, np.sqrt(2*noise), np.shape(theta))

    #### 3. CALCULATION OF THE PHASE SYNCHRONIZATION

    # evolution of order parameter
    r = np.zeros(len(theta))
    for t in range(len(theta)):
        r[t] = abs(sum(np.exp(1j*theta[t])))/N
    
    # compute phase synchronization S as the time average of the order parameter
    S = np.average(r)

    #### 4. CALCULATE Xr MATRIX FOR GC CALCULATIONS

    X = theta.T[:,:]
    Xr = X[:,:-1]
    x = np.array(X[:,1:]) - np.array(X[:,:-1])
    x = x.T[:,:]

    # theta[-1,:] : final state, used as initial condition for next run
    # theta : time series, used in PTE
    # x: phase increments, used in GCE
    #return S, theta[-1,:]%(2*np.pi), theta%(2*np.pi), x, Xr
    return S, (theta[-1,:])%(2*np.pi), theta%(2*np.pi), x, Xr
