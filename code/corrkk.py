import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import eigh

def corrkk(VT, V, VV, x):

    f = 1e-8

    n = len(V)-1
    fast = len(VV[0]) < n-2

    if fast:
        A = VV.T @ VT
        B = (VV.T @ V) @ (V.T @ VT)
        KKN = A.T @ A - B @ A - A.T @ B.T + B @ B.T

        D, VVN = eigh(KKN)
        idx = np.abs(D) > f*np.max(np.abs(D))
        VN = VV @ VVN[:,idx]

    else:
        K = VT @ VT.T
        P = V @ V.T
        KT = K - P @ K - K @ P + P @ K @ P

        D, VN = eigh(KT)
        idx = np.abs(D) > f*np.max(np.abs(D))
        VN = VN[:,idx]

    l = sum(idx)

    # initialize Pearson correlation and p-value vectors
    r, p = np.zeros(l), np.zeros(l)

    # calculate correlations between x and each column in VN
    for i in range(l):
        r[i], p[i] = pearsonr(x, VN[:,i])

    return r, p