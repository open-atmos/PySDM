"""
Attributes for tracking droplet velocity
"""

import warnings

from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute


class RelativeFallMomentum(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="relative fall momentum", dtype=float)


# could eventually make an attribute that calculates momentum
# from terminal velocity instead when no RelaxedVelocity
# dynamic is present
def get_relative_fall_momentum(dynamics):
    """
    Returns fall momentum and throws warning if
    there is no RelaxedVelocity dynamic.
    """
    if "RelaxedVelocity" not in dynamics:
        warnings.warn(
            "Relative fall momentum attribute requested but no RelaxedVelocity dynamic exists."
        )

    return RelativeFallMomentum


class RelativeFallVelocity(DerivedAttribute):
    def __init__(self, builder):
        self.momentum = builder.get_attribute("relative fall momentum")
        self.volume = builder.get_attribute("volume")
        self.rho_w = builder.formulae.constants.rho_w

        super().__init__(
            builder,
            name="relative fall velocity",
            dependencies=(self.momentum, self.volume),
        )

    def recalculate(self):
        self.data.ratio(self.momentum.get(), self.volume.get())
        self.data[:] *= 1 / self.rho_w
