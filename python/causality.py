# import necessary modules
import numpy as np

# import functions
from fourier import fourier
from corrkk import corrkk

def causality(idx, XXX, xxx, par = 4, th = 0.05, f = 1e-6):

    """
    @param par: number of parameters in Fourier kernel (int)
    """
    x, X = np.array(xxx[idx]), np.array(XXX[idx])
    n, nvar = np.shape(x)
    m = len(X)//nvar
    ngr = nvar

    for j in range(nvar):
        x[:,j] /= np.sqrt(np.dot(x[:,j],x[:,j]))
    
    VV, DD = fourier(X, par, f)
    VT = VV @ np.diag(np.sqrt(DD))

    rr = np.zeros((ngr, ngr, n))
    pp = np.ones((ngr, ngr, n))

    kk = 0
    for i in range(ngr):
        # tbr: indices to be removed
        tbr = np.arange(i, nvar*m, step = nvar)
        Xr = np.delete(X, tbr, axis = 0)
        for j in range(ngr):
            if i != j:
                V, _ = fourier(Xr, par, f)
                xv = V @ V.T @ x[:,j]
                rrp, ppp = corrkk(VT, V, VV, x[:,j]-xv)
                nrr = len(rrp)
                rr[i,j,:nrr] = rrp
                pp[i,j,:nrr] = ppp
                kk += nrr
    
    # Bonferroni
    thb = th/kk
    indpr = pp > thb
    rn = rr[:]
    rn[indpr] = 0

    GC = np.sum(rn**2)

    return GC
