"""
Plot Interpolation (Gunn/Kinzer) and Rogers/Yau approximations of 
terminal velocity as a function of droplet radius.
"""

from typing import Type, Union
import numpy as np
import matplotlib.pyplot as plt
from PySDM.physics import si
from PySDM.backends import CPU
from PySDM.dynamics.terminal_velocity import Interpolation, RogersYau
from PySDM.formulae import Formulae
from PySDM.particulator import Particulator

# radius values of 10**BOUND m
LOG_LOWER_BOUND = -8
LOG_UPPER_BOUND = -2.5

N = 1000

def get_approx(radii_arr, particulator=None, approx: Union[Type[RogersYau], Type[Interpolation]]=RogersYau):
    """
    Get Rogers Yau approximation of terminal velocity as a function of 
    droplet radius.
    """
    if particulator is None:
        particulator = Particulator(0, CPU(Formulae()))

    radii = particulator.Storage.from_ndarray(radii_arr)

    rogers_yau_output = particulator.Storage.empty(radii_arr.shape, dtype=float)
    approx(particulator=particulator)(output=rogers_yau_output, radius=radii)

    return rogers_yau_output.to_ndarray()


if __name__ == "__main__":
    particulator = Particulator(0, CPU(Formulae()))

    radii_arr = np.logspace(LOG_LOWER_BOUND, LOG_UPPER_BOUND, N)

    interpolation_arr = get_approx(radii_arr, particulator=particulator, approx=Interpolation)
    rogers_yau_arr = get_approx(radii_arr, particulator=particulator, approx=RogersYau)


    plt.plot(radii_arr*si.metres/si.micrometres, rogers_yau_arr,
             "r-", label="Rogers Yau", alpha=0.5, linewidth=3)
    plt.plot(radii_arr*si.metres/si.micrometres, interpolation_arr,
             "b-", label="Interpolation", alpha=0.5, linewidth=3)

    plt.xscale("log")
    plt.xlabel("Radius ($\\mu$m)")
    plt.ylabel("Terminal Velocity (m/s)")

    plt.legend()

plt.show()