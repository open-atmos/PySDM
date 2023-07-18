from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute


class FallMomentum(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="fall momentum", dtype=float)


class FallVelocity(DerivedAttribute):
    def __init__(self, builder):
        self.momentum = builder.get_attribute("fall momentum")
        self.volume = builder.get_attribute("volume")
        self.rho_w = builder.formulae.constants.rho_w

        super().__init__(
            builder, name="fall velocity", dependencies=(self.momentum, self.volume)
        )

    def recalculate(self):
        self.data.ratio(self.momentum.get(), self.volume.get())
        self.data[:] *= 1 / self.rho_w
