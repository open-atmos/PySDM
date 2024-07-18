"""
particle mass, derived attribute for coalescence
in simulation involving mixed-phase clouds, positive values correspond to
liquid water and negative values to ice
"""

import PySDM
from PySDM.attributes.impl import ExtensiveAttribute


@PySDM.register_attribute()
class WaterMass(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="water mass")
