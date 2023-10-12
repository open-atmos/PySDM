"""
Base class for volume-based fragmentation functions
"""


class VolumeBasedFragmentationFunction:
    def __init__(self):
        self.particulator = None

    def __call__(self, nf, frag_mass, u01, is_first_in_pair):
        frag_volume_aliased_to_mass = frag_mass
        self.compute_fragment_number_and_volumes(
            nf, frag_volume_aliased_to_mass, u01, is_first_in_pair
        )
        frag_volume_aliased_to_mass *= self.particulator.formulae.constants.rho_w

    def compute_fragment_number_and_volumes(
        self, nf, frag_volumes, u01, is_first_in_pair
    ):
        raise NotImplementedError()

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute("volume")
