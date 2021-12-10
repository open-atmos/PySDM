from ..impl import ExtensiveAttribute

class ImmersedSurfaceArea(ExtensiveAttribute):
    def __init__(self, particles_builder):
        super().__init__(particles_builder, name='immersed surface area')
