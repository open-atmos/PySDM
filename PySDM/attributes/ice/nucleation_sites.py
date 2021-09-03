from ..impl import ExtensiveAttribute

class NucleationSites(ExtensiveAttribute):
    def __init__(self, particles_builder):
        super().__init__(particles_builder, name='nucleation sites')
