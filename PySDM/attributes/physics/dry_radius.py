"""
particle dry radius computed from dry volume
"""

from PySDM.attributes.impl import DerivedAttribute, register_attribute


@register_attribute()
class DryRadius(DerivedAttribute):
    def __init__(self, builder):
        self.volume_dry = builder.get_attribute("dry volume")
        dependencies = [self.volume_dry]
        super().__init__(builder, name="dry radius", dependencies=dependencies)

    def recalculate(self):
        self.data.product(self.volume_dry.get(), 1 / self.formulae.constants.PI_4_3)
        self.data **= 1 / 3
