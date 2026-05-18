"""diagnosed diffusional mass change within a timestep
(temporarily without support for collisional processes), developed to enable
clean coupling between condensation and other dynamics (e.g., isotopic
fractionation)"""

from PySDM.attributes.impl import register_attribute, DerivedAttribute


@register_attribute()
class DiffusionalGrowthMassChange(DerivedAttribute):
    def __init__(self, builder):
        self.water_mass = builder.get_attribute("signed water mass")
        super().__init__(
            builder,
            name="diffusional growth mass change",
            dependencies=(self.water_mass,),
        )
        self.old = None
        assert "Collision" not in builder.particulator.dynamics
        for triggers in (
            builder.particulator.observers,
            builder.particulator.initialisers,
        ):
            triggers.append(self)

    def setup(self):
        self.old = self.water_mass.data.data.copy()
        self.data.data[:] = 0

    def notify(self):
        new = self.water_mass.data.data
        self.data.data[:] = new - self.old
        self.old[:] = new

    def recalculate(self):
        pass
