# import numpy as np

# # import functions

# from Kuramoto import constructNetwork, kuramoto
# from PTE import PTE
# from causality import causality
# import fourier as f
# import fourier_py as fp

# N = 15
# A, omega = constructNetwork(N, k=5, gamma = 0.0)
# phi0 = [1.0 for _ in range(N)]
# d = 0.5
# S, phi, x, X = kuramoto(d, A, omega, phi0, T = 200, m = 5)

# import time

# runs = 20
# ordinary = [0 for _ in range(runs)]
# cythonized = [0 for _ in range(runs)]

# for i in range(runs):
#     start = time.time()
#     V, D = f.fourier(X, 3, 1e-6)
#     cythonized[i] = (time.time()-start)*1000

# print(f"{np.mean(cythonized)} ms")

# for i in range(runs):
#     start = time.time()
#     Vp, Dp = fp.fourier(X, 3, 1e-6)
#     ordinary[i] = (time.time()-start)*1000

# print(f"{np.mean(ordinary)} ms")

# import matplotlib.pyplot as plt
# plt.boxplot([cythonized, ordinary])
# plt.show()



# import necessary modules

import numpy as np
import matplotlib.pyplot as plt

# parallel processing to speed up the program

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count() # number of cores (4 on pc, 12 on Helios)

# import functions

from Kuramoto import constructNetwork, kuramoto
from PTE import PTE
from causality import causality

import time

start = time.time()

N = 15
A, omega = constructNetwork(N, k = 10, gamma = 0.4)
phi0 = [1.0 for _ in range(N)]
coupling = np.linspace(0,0.15,40)

print(f"Initializing: {time.time()-start} s")
start = time.time()

S = []
x = []
Xr = []
for d in coupling:
    S_, phi, x_, Xr_ = kuramoto(d, A, omega, phi0, T = 300, m = 2)
    S.append(S_)
    x.append(x_)
    Xr.append(Xr_)
    phi0 = phi[:]

print(f"Kuramoto: {time.time()-start} s")
start = time.time()

idx = np.arange(len(coupling))
gc = Parallel(n_jobs=num_cores)(delayed(causality)(idx_, Xr, x, 4) for idx_ in idx)

print(f"GC: {(time.time()-start)/60} min")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(coupling, S, color = 'firebrick')
ax.plot(coupling, S, color = 'firebrick')
ax.set_ylabel(r'phase synchronization $S$', color='firebrick')
ax.tick_params('y', colors='firebrick')
ax.set_ylim(0,1)
ax2 = ax.twinx()
ax2.plot(coupling, gc, color = 'blue')
ax2.scatter(coupling, gc, color = 'blue')
ax2.set_ylabel(r'GC', color='blue')
ax2.tick_params('y', colors='blue')
ax.set_xlabel(r'coupling $\lambda$')

plt.title(r'$\gamma=0.4, N=15, <k>=10, m=2, T=300$, par=4')
plt.tight_layout()
plt.show()
