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

def kuramoto(d, A, omega, phi0, T = 100, m = 1, save = False):
    """
    @param d: the coupling strength (float)
    @param A: adjacency matrix of the FGC network (2D numpy array)
    @param omega: natural frequencies (list or 1d numpy array)
    @param T: length of the integration (int)
    @param M: order of the GC model (int)
    @param save: whether or not to save the result to file (boolean)
    """

    #### 1. INITIALIZATION
            
    # number of oscillators in the network
    N = len(A)   
    
    #### 2. INTEGRATION

    # time intervals on which to perform the integration:
    t = np.linspace(0,T+20,T+21) 
    
    # function to integrate
    @jit
    def f(phi,t):
        dphidt = np.array([omega[i] for i in range(N)])
        for i in range(N):
            for j in range(N):
                dphidt[i] += d*A[i][j]*np.sin(phi[j]-phi[i])
        return dphidt
    
    # perform integration
    sol = odeint(f,phi0,t)%(2*np.pi) # ensure phases are in the interval [0,2pi)
    # sol is the theta matrix in PTE calculation, and the x matrix in GC

    # we assume the first 20 time steps are used to relax the system
    ini = 20
    theta = sol[ini:,:]

    #### 3. CALCULATION OF THE PHASE SYNCHRONIZATION

    # evolution of order parameter
    r = np.zeros(len(theta))
    for t in range(len(theta)):
        r[t] = abs(sum(np.exp(1j*theta[t])))/N
    
    # compute phase synchronization S as the time average of the order parameter
    S = np.average(r)

    #### 4. CALCULATE Xr MATRIX FOR GC CALCULATIONS

    X = theta.T[:,:]
    n = len(X[0])
    Xr = np.zeros((N*m, n))
    for t in range(m,n):
        for i in range(m):
            Xr[i*N:(i+1)*N,t] = X[:,t-(i+1)]
    Xr = Xr[:,m:]
    nl = len(Xr[0])
    x = theta[-nl:,:]

    return S, x[-1,:], x, Xr

    #### 5. SAVE THE MATRICES Xr AND x
    if save:
        dstr = "%.3f"%d
        dir = f"data/N={N}/d={dstr}"
        os.makedirs(dir) # yields error if directory exists

        for m in range(1,6):
            X = theta.T[:]
            n = len(X[0])
            Xr = np.zeros((N*M, n))
            for t in range(m,n):
                for i in range(m):
                    Xr[i*N:(i+1)*N,t] = X[:,t-(i+1)]
            Xr = Xr[:,m:]
            nl = len(Xr[0])
            x = theta[-nl:,:]

            np.savetxt(f"{dir}/x{m}.txt", x)
            np.savetxt(f"{dir}/Xr{m}.txt", Xr)

    