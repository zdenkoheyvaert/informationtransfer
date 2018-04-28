#!python
#cython: boundscheck=False, wraparound=False

from scipy.linalg import eigh
import numpy as np
cimport cython
from cython.view cimport array as cvarray

cdef extern from "math.h":
    double sqrt(double m)
    double cos(double m)
    double sin(double m)

cdef double[:] normalize(double[:] x):
    cdef double norm = 0
    cdef Py_ssize_t length = len(x)
    cdef Py_ssize_t i
    for i in range(length):
        norm += x[i]*x[i]
    norm = sqrt(norm)

    cdef double[:] result = cvarray((length,),sizeof(double),'d')
    for i in range(length):
        result[i] = x[i]/norm
    return result

cdef double[:] subtract(double[:] x, double[:] y):
    cdef Py_ssize_t length = len(x)
    cdef double[:] result = cvarray((length,),sizeof(double),'d')
    cdef Py_ssize_t i
    for i in range(length):
        result[i] = x[i] - y[i]
    return result

cdef double[:] add(double[:] x, double[:] y):
    cdef Py_ssize_t length = len(x)
    cdef double[:] result = cvarray((length,),sizeof(double),'d')
    cdef Py_ssize_t i
    for i in range(length):
        result[i] = x[i] + y[i]
    return result

cdef double mean(double[:] x):
    cdef double result = 0
    cdef Py_ssize_t length = len(x)
    cdef Py_ssize_t i
    for i in range(length):
        result += x[i]
    return result/length

cdef double[:] meana(double[:] x):
    cdef double mu = mean(x)
    cdef Py_ssize_t length = len(x)
    cdef double[:] result = cvarray((length,),sizeof(double),'d')
    cdef Py_ssize_t i
    for i in range(length):
        result[i] = mu
    return result

cdef double[:] cosa(double[:] x):
    cdef Py_ssize_t length = len(x)
    cdef double[:] result = cvarray((length,),sizeof(double),'d')
    cdef Py_ssize_t i
    for i in range(length):
        result[i] = cos(x[i])
    return result

cdef double[:] sina(double[:] x):
    cdef Py_ssize_t length = len(x)
    cdef double[:] result = cvarray((length,),sizeof(double),'d')
    cdef Py_ssize_t i
    for i in range(length):
        result[i] = sin(x[i])
    return result

cdef double[:] center(double[:] Li):
    cdef double[:] tmp = subtract(Li, meana(Li))
    return normalize(tmp)

def fourier(XX, int par, float f):
    """
    @param X: 2d numpy array
    @param par: number of parameters (int)
    @param f: threshold (float)
    """

    cdef double[:,:] X = XX

    cdef Py_ssize_t m, n
    m, n = np.shape(X)
    cdef Py_ssize_t nn = m*(m-1)//2
    cdef Py_ssize_t i, j, k

    cdef double[:,:] L

    if par == 1:
        L = cvarray((n,2*m),sizeof(double),'d')
        for i in range(m):
            L[:,i] = center(cosa(X[i,:]))
            L[:,i+m] = center(sina(X[i,:]))

    elif par == 2:
        L = np.empty((n, 2*nn))
        k = 0
        for i in range(m):
            for j in range(i+1,m):
                L[:,k] = center(cosa(subtract(X[i,:],X[j,:])))
                L[:,k+nn] = center(sina(subtract(X[i,:],X[j,:])))
    
    elif par == 3:
        L = cvarray((n,2*(m+nn)),sizeof(double),'d')
        for i in range(m):
            L[:,i] = center(cosa(X[i,:]))
            L[:,i+m] = center(cosa(X[i,:]))
        k = 2*m - 1
        for i in range(m):
            for j in range(i+1,m):
                k += 1
                L[:,k] = center(cosa(subtract(X[i,:],X[j,:])))
                L[:,k+nn] = center(sina(subtract(X[i,:],X[j,:])))

    elif par == 4:
        L = cvarray((n,4*(m+nn)),sizeof(double),'d')
        for i in range(m):
            L[:,i] = center(cosa(X[i,:]))
            L[:,i+m] = center(sina(X[i,:]))
            L[:,i+2*m] = center(cosa(add(X[i,:],X[i,:])))
            L[:,i+3*m] = center(sina(add(X[i,:],X[i,:])))
        k = 4*m - 1
        for i in range(m):
            for j in range(i+1,m):
                k += 1
                L[:,k] = center(cosa(subtract(X[i,:],X[j,:])))
                L[:,k+nn] = center(sina(subtract(X[i,:],X[j,:])))
                L[:,k+2*nn] = center(cosa(add(X[i,:],X[j,:])))
                L[:,k+3*nn] = center(sina(add(X[i,:],X[j,:])))

    # The number of columns of L grows quadratically with N*m, so
    # it quickly becomes a large vector. Calculating the eigenvalues
    # of L.T@L is thus slow. However,    cdef  Lii = Li through the necessary algebra
    # we can find the eigenvalues and -vectors of L.T@L from
    # L@L.T, which is generally MUCH smaller (only exception is
    # integration over very long time    cdef  Lii = Li of a very small number of oscillators)

    # Another observation is dat L.T@L and L@L.T are symmetric matrices,
    # which allows us to use scipy's eigh function instead of eig, which 
    # is much faster.

    L_ = np.asarray(L)

    def calcV(L):
        nrow, ncol = np.shape(L)

        if nrow < ncol:
            D, VN_ = eigh(L @ L.T)
            D = np.flip(D, axis = 0)
            VN_ = np.flip(VN_, axis = 1)
            r = np.sum(D > f*D[0])
            D = D[:r]
            VN_ = VN_[:,:r]
            VN = L.T @ VN_
            
        else:
            D, VN = eigh(L.T @ L)
            D = np.flip(D, axis = 0)
            VN = np.flip(VN, axis = 1)
            idx = D > f*D[0]
            D = D[idx]
            VN = VN[:,idx]
        
        return D, VN

    D, VN = calcV(L_)
    V = L_ @ VN
    xnorm = np.sqrt(np.diag(V.T @ V))
    V /= xnorm

    return V, D