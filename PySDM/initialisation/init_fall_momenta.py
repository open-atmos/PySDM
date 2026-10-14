"""
Initialize the `PySDM.attributes.physics.relative_fall_velocity.RelativeFallMomentum`
of droplets
"""

import numpy as np


def init_fall_momenta(
    *,
    terminal_velocity_approx,
    backend,
    water_mass: np.ndarray,
    zero: bool = False,
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

    approximation = terminal_velocity_approx(backend=backend)

    volume_arr = backend.formulae.particle_shape_and_density.mass_to_volume(water_mass)
    radii_arr = backend.formulae.trivia.radius(volume=volume_arr)
    radii = backend.Storage.from_ndarray(radii_arr)

    output = backend.Storage.empty((len(water_mass),), dtype=float)

    approximation(output=output, radius=radii)

    return output.to_ndarray() * water_mass
