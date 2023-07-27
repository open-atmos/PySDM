"""
Attributes for tracking droplet velocity
"""

from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute


class RelativeFallMomentum(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="relative fall momentum", dtype=float)


class RelativeFallVelocity(DerivedAttribute):
    def __init__(self, builder):
        self.momentum = builder.get_attribute("relative fall momentum")
        self.volume = builder.get_attribute("volume")
        self.rho_w = builder.formulae.constants.rho_w  # TODO #798

        super().__init__(
            builder,
            name="relative fall velocity",
            dependencies=(self.momentum, self.volume),
        )

    def recalculate(self):
        self.data.ratio(self.momentum.get(), self.volume.get())
        self.data[:] *= 1 / self.rho_w  # TODO #798
