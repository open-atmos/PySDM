import numpy as np
from PySDM.backends import CPU
from PySDM.dynamics.terminal_velocity import Interpolation
from PySDM.formulae import Formulae
from PySDM.particulator import Particulator


def init_fall_momenta(volume: np.ndarray, rho_w: float):
    """
    Calculate default values of the FallMomentum attribute
    (needed when using FallVelocity attribute)

    Parameters:
        - volume: a numpy array of superdroplet volumes
        - rho_w: the density of water (generally found in formulae.constants)

    Returns:
        - a numpy array of initial momentum values
    """
    # TODO: good luck
    # for now it just makes velocity 4
    return volume*rho_w*4
