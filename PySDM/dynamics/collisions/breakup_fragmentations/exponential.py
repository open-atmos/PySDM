"""
P(x) = exp(-x / lambda); lambda specified in volume units
"""

# TODO #796: introduce common code with Feingold fragmentation, including possible limiter
from .impl import VolumeBasedFragmentationFunction


class Exponential(VolumeBasedFragmentationFunction):
    def __init__(self, scale, vmin=0.0, nfmax=None):
        super().__init__()
        self.scale = scale
        self.vmin = vmin
        self.nfmax = nfmax
        self.sum_of_volumes = None

    def register(self, builder):
        super().register(builder)
        self.sum_of_volumes = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )

    def compute_fragment_number_and_volumes(
        self, nf, frag_volume, u01, is_first_in_pair
    ):
        self.sum_of_volumes.sum(
            self.particulator.attributes["volume"], is_first_in_pair
        )
        self.particulator.backend.exp_fragmentation(
            n_fragment=nf,
            scale=self.scale,
            frag_volume=frag_volume,
            x_plus_y=self.sum_of_volumes,
            rand=u01,
            vmin=self.vmin,
            nfmax=self.nfmax,
        )
