"""
Plot Interpolation (Gunn/Kinzer) and Rogers/Yau approximations of 
terminal velocity as a function of droplet radius.
"""

import numpy as np
import matplotlib.pyplot as plt
from PySDM.physics import si
from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics.terminal_velocity import Interpolation, RogersYau
from PySDM.formulae import Formulae
from PySDM.particulator import Particulator

# radius values of 10**BOUND m
LOG_LOWER_BOUND = -5
LOG_UPPER_BOUND = -2.5

N = 1000

particulator = Particulator(0, CPU(Formulae()))
idx = particulator.Index.identity_index(N)

interpolation = Interpolation(particulator)
rogers_yau = RogersYau(particulator=particulator)

radii_arr = np.logspace(LOG_LOWER_BOUND, LOG_UPPER_BOUND, N)
radii = particulator.IndexedStorage.from_ndarray(
    idx, radii_arr)

interpolation_output = particulator.IndexedStorage.empty(
    idx,
    (N,),
    dtype=float,
)
rogers_yau_output = particulator.IndexedStorage.empty(
    idx,
    (N,),
    dtype=float,
)

interpolation(output=interpolation_output, radius=radii)
rogers_yau(output=rogers_yau_output, radius=radii)

interpolation_arr = interpolation_output.to_ndarray()
rogers_yau_arr = rogers_yau_output.to_ndarray()

plt.plot(radii_arr*si.metres/si.micrometres, rogers_yau_arr, "r-", label="Rogers Yau", alpha=0.5, linewidth=3)
plt.plot(radii_arr*si.metres/si.micrometres, interpolation_arr, "b-", label="Interpolation", alpha=0.5, linewidth=3)

plt.xscale("log")
plt.xlabel("Radius ($\\mu$m)")
plt.ylabel("Terminal Velocity (m/s)")

plt.legend()

plt.show()
