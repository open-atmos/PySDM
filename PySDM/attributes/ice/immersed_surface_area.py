"""
immersed INP surface area (assigned at initialisation, modified through collisions only,
 used in time-dependent regime)
"""

import PySDM
from ..impl import ExtensiveAttribute


@PySDM.register_attribute()
class ImmersedSurfaceArea(ExtensiveAttribute):
    def __init__(self, particles_builder):
        super().__init__(particles_builder, name="immersed surface area")
