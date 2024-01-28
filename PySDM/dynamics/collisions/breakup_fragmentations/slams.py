"""
Based on [Jokulsdottir & Archer 2016 (GMD)](https://doi.org/10.5194/gmd-9-1455-2016)
for ocean particles
"""

from .impl import VolumeBasedFragmentationFunction


class SLAMS(VolumeBasedFragmentationFunction):
    def __init__(self, vmin=0.0, nfmax=None):
        super().__init__()
        self.p_vec = None
        self.sum_of_volumes = None
        self.vmin = vmin
        self.nfmax = nfmax

    def compute_fragment_number_and_volumes(
        self, nf, frag_volume, u01, is_first_in_pair
    ):
        self.sum_of_volumes.sum(
            self.particulator.attributes["volume"], is_first_in_pair
        )
        self.particulator.backend.slams_fragmentation(
            n_fragment=nf,
            frag_volume=frag_volume,
            x_plus_y=self.sum_of_volumes,
            probs=self.p_vec,
            rand=u01,
            vmin=self.vmin,
            nfmax=self.nfmax,
        )

    def register(self, builder):
        super().register(builder)
        self.p_vec = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.sum_of_volumes = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
