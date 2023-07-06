import numpy as np
from PySDM.backends import CPU
from PySDM.dynamics.terminal_velocity import Interpolation, RogersYau
from PySDM.formulae import Formulae
from PySDM.particulator import Particulator


def init_fall_momenta(volume: np.ndarray, rho_w: float, zero: bool = False, terminal_velocity_approx = RogersYau):
    """
    Calculate default values of the FallMomentum attribute
    (needed when using FallVelocity attribute)

    Parameters:
        - volume: a numpy array of superdroplet volumes
        - rho_w: the density of water (generally found in formulae.constants)

    Returns:
        - a numpy array of initial momentum values
    """
    if zero:
        return np.zeros_like(volume)

    particulator = Particulator(0, CPU(Formulae()))

    approximation = terminal_velocity_approx(particulator=particulator)

    radii_arr = particulator.formulae.trivia.radius(volume)
    radii = particulator.Storage.from_ndarray(radii_arr)

    output = particulator.Storage.empty((len(volume),), dtype=float)

    approximation(output=output, radius=radii)

    return output.to_ndarray() * volume * rho_w 
