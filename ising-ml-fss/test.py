
""" Time test for fast_model.py """

import time
from src.ising.fast_model import FastIsing2D
import numpy as np

L = 81
T = np.linspace(2.0, 2.6, 25)
n_sweeps = 1000

model = FastIsing2D(L=L, J=1.0, seed=123, init_state="random")

start = time.perf_counter()

for _ in range(n_sweeps):
    model.heat_bath_sweep(T)

end = time.perf_counter()

print(f"Elapsed time: {end - start:.3f} seconds")





""" Time test for model.py """

import time
from src.ising.model import Ising2D

L = 81
T = 2.269
n_sweeps = 1000

model = Ising2D(L=L, J=1.0, seed=123)

start = time.perf_counter()

for _ in range(n_sweeps):
    model.metropolis_sweep(T)

end = time.perf_counter()

print(f"Elapsed time: {end - start:.3f} seconds")