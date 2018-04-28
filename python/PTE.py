def PTE(theta):

    import numpy as np

    #### 1. INITIALIZATION

    # number of samples
    L = len(theta[:,0])
    # number of signals
    N = len(theta[0,:])

    #### 2. COMPUTE DELAY

    counter1, counter2 = 0, 0
    for j in range(N):
        for i in range(1,L-1):
            counter1 += 1
            if (theta[i-1,j]-np.pi)*(theta[i+1,j]-np.pi) < 0:
                counter2 += 1
    delay = counter1//counter2

    ##### 3. COMPUTE BINSIZE
    # Defined according to Scott's choice

    # the binsize
    binsize = 3.49*np.mean(np.std(theta,axis=0))*np.power(N,-1/3)

    # the bins
    bins_w = np.arange(start = 0, stop = 2*np.pi, step = binsize)

    # number of bins
    Nbins = len(bins_w)

    #### 4. COMPUTE PTE

    # PTE matrix of pairwise transfer entropy
    PTE = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if i != j:
                # initialize
                Py = np.zeros(Nbins)
                Pypr_y = np.zeros((Nbins,Nbins))
                Py_x = np.zeros((Nbins,Nbins))
                Pypr_y_x = np.zeros((Nbins,Nbins,Nbins))
                
                # fill the bins of the phase histograms: re-write phase data as binned data
                rn_ypr = np.array(np.ceil(theta[delay:L,j]/binsize), dtype = int) - 1
                rn_y = np.array(np.ceil(theta[:(L-delay),j]/binsize), dtype = int) - 1
                rn_x = np.array(np.ceil(theta[:(L-delay),i]/binsize), dtype = int) - 1
                
                for kk in range(L-delay):
                    Py[rn_y[kk]] += 1
                    Pypr_y[rn_ypr[kk],rn_y[kk]] += 1
                    Py_x[rn_y[kk],rn_x[kk]] += 1
                    Pypr_y_x[rn_ypr[kk],rn_y[kk],rn_x[kk]] += 1
                    
                # normalize probabilities
                Py /= (L-delay)
                Pypr_y /= (L-delay)
                Py_x /= (L-delay)
                Pypr_y_x /= (L-delay)
                
                # we will take logarithm: therefore replace 0 with missing values
                Py[Py==0] = np.nan
                Pypr_y[Pypr_y==0] = np.nan
                Py_x[Py_x==0] = np.nan
                Pypr_y_x[Pypr_y_x==0] = np.nan
                
                # calculate entropies
                Hy = -1*np.nansum(Py*np.log2(Py))
                Hypr_y = -1*np.nansum(Pypr_y*np.log2(Pypr_y))
                Hy_x = -1*np.nansum(Py_x*np.log2(Py_x))
                Hypr_y_x = -1*np.nansum(Pypr_y_x*np.log2(Pypr_y_x))
                
                # compute PTE
                PTE[i,j] = Hypr_y + Hy_x - Hy - Hypr_y_x

    return np.sum(PTE)