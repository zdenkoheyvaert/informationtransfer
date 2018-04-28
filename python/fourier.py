import numpy as np
from scipy.linalg import eigh

def fourier(X, par, f):
    """
    @param X: 2d numpy array / pandas dataframe
    @param par: number of parameters (int)
    @param f: threshold (float)
    """

    def center(Li):
        """
        @param Li: 1d numpy array
        """
        tmp = Li - np.mean(Li)
        return tmp/np.sqrt(np.dot(tmp,tmp))
    

    X = np.array(X)
    m, n = np.shape(X)
    nn = m*(m-1)//2

    if par == 1:
        L = np.zeros((n, 2*m))
        for i in range(m):
            L[:,i] = center(np.cos(X[i,:]))
            L[:,i+m] = center(np.sin(X[i,:]))

    elif par == 2:
        L = np.zeros((n, 2*nn))
        k = 0
        for i in range(m):
            for j in range(i+1,m):
                L[:,k] = center(np.cos(X[i,:]-X[j,:]))
                L[:,k+nn] = center(np.sin(X[i,:]-X[j,:]))
    
    elif par == 3:
        L = np.zeros((n,2*(m+nn)))
        for i in range(m):
            L[:,i] = center(np.cos(X[i,:]))
            L[:,i+m] = center(np.sin(X[i,:]))
        k = 2*m - 1
        for i in range(m):
            for j in range(i+1,m):
                k += 1
                L[:,k] = center(np.cos(X[i,:]-X[j,:]))
                L[:,k+nn] = center(np.sin(X[i,:]-X[j,:]))

    elif par == 4:
        L = np.zeros((n,4*(m+nn)))
        for i in range(m):
            L[:,i] = center(np.cos(X[i,:]))
            L[:,i+m] = center(np.sin(X[i,:]))
            L[:,i+2*m] = center(np.cos(2*X[i,:]))
            L[:,i+3*m] = center(np.sin(2*X[i,:]))
        k = 4*m - 1
        for i in range(m):
            for j in range(i+1,m):
                k += 1
                L[:,k] = center(np.cos(X[i,:]-X[j,:]))
                L[:,k+nn] = center(np.sin(X[i,:]-X[j,:]))
                L[:,k+2*nn] = center(np.cos(X[i,:]+X[j,:]))
                L[:,k+3*nn] = center(np.sin(X[i,:]+X[j,:]))

    # The number of columns of L grows quadratically with N*m, so
    # it quickly becomes a large vector. Calculating the eigenvalues
    # of L.T@L is thus slow. However, through the necessary algebra
    # we can find the eigenvalues and -vectors of L.T@L from
    # L@L.T, which is generally MUCH smaller (only exception is
    # integration over very long time of a very small number of oscillators)

    # Another observation is dat L.T@L and L@L.T are symmetric matrices,
    # which allows us to use scipy's eigh function instead of eig, which 
    # is much faster.

    def calcV(L):
        nrow, ncol = np.shape(L)

        if nrow < ncol: # usually the case
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

    D, VN = calcV(L)
    V = L @ VN
    xnorm = np.sqrt(np.diag(V.T @ V))
    V /= xnorm

    return V, D