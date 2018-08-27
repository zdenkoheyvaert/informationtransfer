# import necessary modules

import numpy as np
import matplotlib.pyplot as plt
import time

# parallel processing to speed up the program

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count() # number of cores (4 on pc, 12 on Helios)

# import functions

from Kuramoto import constructNetwork, kuramoto
from PTE import PTE
from causality import causality

# initialize

N = 15
A, omega = constructNetwork(N, k = 10, gamma = 0.4)
phi0 = [1.0 for _ in range(N)]
coupling = np.linspace(0,0.2,40)

S = []
x = []
theta = []
Xr = []
for d in coupling:
    S_, phi, theta_, x_, Xr_ = kuramoto(d, A, omega, phi0, T = 200, m = 5)
    theta.append(theta_)
    S.append(S_)
    x.append(x_)
    Xr.append(Xr_)
    phi0 = phi[:]

pte = Parallel(n_jobs=num_cores)(delayed(PTE)(theta_) for theta_ in theta)

start = time.time()

idx = np.arange(len(coupling))
gc = Parallel(n_jobs=num_cores)(delayed(causality)(idx_, Xr, x, 2) for idx_ in idx)

print(f"{(time.time()-start)} s")