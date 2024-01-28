"""
particle mass, derived attribute for coalescence
in simulation involving mixed-phase clouds, positive values correspond to
liquid water and negative values to ice
"""

from PySDM.attributes.impl import ExtensiveAttribute


class WaterMass(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="water mass")
