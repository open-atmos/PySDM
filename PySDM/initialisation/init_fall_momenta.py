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
    water_mass: np.ndarray,
    zero: bool = False,
    terminal_velocity_approx=GunnKinzer1949,  # TODO #1155
):
    """
    Calculate default values of the
    `PySDM.attributes.physics.relative_fall_velocity.RelativeFallMomentum` attribute
    (needed when using
    `PySDM.attributes.physics.relative_fall_velocity.RelativeFallVelocity` attribute)

    Parameters:
        - water_mass: a numpy array of superdroplet water masses

    Returns:
        - a numpy array of initial momentum values
    """
    if zero:
        return np.zeros_like(water_mass)

    particulator = Particulator(0, CPU(Formulae()))  # TODO #1155

    approximation = terminal_velocity_approx(particulator=particulator)

    volume_arr = particulator.formulae.particle_shape_and_density.mass_to_volume(
        water_mass
    )
    radii_arr = particulator.formulae.trivia.radius(volume=volume_arr)
    radii = particulator.Storage.from_ndarray(radii_arr)

    output = particulator.Storage.empty((len(water_mass),), dtype=float)

    approximation(output=output, radius=radii)

    return output.to_ndarray() * water_mass
