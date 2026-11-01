"""
immersed INP surface area (assigned at initialisation, modified through collisions only,
 used in time-dependent regime)
"""

from ..impl import ExtensiveAttribute, register_attribute


@register_attribute()
class ImmersedSurfaceArea(ExtensiveAttribute):
    def __init__(self, particulator):
        super().__init__(particulator, name="immersed surface area")
