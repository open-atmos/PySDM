"""
particle wet radius (calculated from the volume)
"""

from PySDM.attributes.impl import DerivedAttribute, register_attribute


@register_attribute()
class Radius(DerivedAttribute):
    def __init__(self, builder):
        self.volume = builder.get_attribute("volume")
        dependencies = [self.volume]
        super().__init__(builder, name="radius", dependencies=dependencies)

    def recalculate(self):
        self.data.product(self.volume.get(), 1 / self.formulae.constants.PI_4_3)
        self.data **= 1 / 3


@register_attribute()
class SquareRootOfRadius(DerivedAttribute):
    def __init__(self, builder):
        self.radius = builder.get_attribute("radius")

        super().__init__(
            builder,
            name="square root of radius",
            dependencies=(self.radius,),
        )

    def recalculate(self):
        self.data.fill(self.radius.data)
        self.data **= 0.5
