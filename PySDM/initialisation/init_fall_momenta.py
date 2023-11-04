"""
Initialize the `PySDM.attributes.physics.relative_fall_velocity.RelativeFallMomentum`
of droplets
"""

import numpy as np

from PySDM.backends import CPU
from PySDM.dynamics.terminal_velocity import GunnKinzer1949
from PySDM.formulae import Formulae
from PySDM.particulator import Particulator


def init_fall_momenta(
    volume: np.ndarray,
    rho_w: float,  # TODO #798 - we plan to use masses instead of volumes soon
    zero: bool = False,
    terminal_velocity_approx=GunnKinzer1949,  # TODO #1155
):
    """
    Calculate default values of the
    `PySDM.attributes.physics.relative_fall_velocity.RelativeFallMomentum` attribute
    (needed when using
    `PySDM.attributes.physics.relative_fall_velocity.RelativeFallVelocity` attribute)

    Parameters:
        - volume: a numpy array of superdroplet volumes
        - rho_w: the density of water (generally found in formulae.constants)

    Returns:
        - a numpy array of initial momentum values
    """
    if zero:
        return np.zeros_like(volume)

    particulator = Particulator(0, CPU(Formulae()))  # TODO #1155

    approximation = terminal_velocity_approx(particulator=particulator)

    radii_arr = particulator.formulae.trivia.radius(volume)
    radii = particulator.Storage.from_ndarray(radii_arr)

    output = particulator.Storage.empty((len(volume),), dtype=float)

    approximation(output=output, radius=radii)

    return output.to_ndarray() * volume * rho_w  # TODO #798 this assumes no ice
