from PySDM.attributes.impl import register_attribute, DerivedAttribute


@register_attribute()
class DiffusionalGrowthMassChange(DerivedAttribute):
    def __init__(self, builder):
        self.water_mass = builder.get_attribute("water mass")
        super().__init__(
            builder,
            name="diffusional growth mass change",
            dependencies=(self.water_mass,),
        )
        builder.particulator.observers.append(self)
        assert "Collision" not in builder.particulator.dynamics
        self.notify()

    def notify(self):
        self.data[:] = -self.water_mass.data

    def recalculate(self):
        self.data[:] += self.water_mass.data
