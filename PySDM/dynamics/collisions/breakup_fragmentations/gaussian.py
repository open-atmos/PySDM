"""
P(x) = exp(-(x-mu)^2 / 2 sigma^2); mu and sigma are volumes
"""

from .impl import VolumeBasedFragmentationFunction


class Gaussian(
    VolumeBasedFragmentationFunction
):  # pylint: disable=too-many-instance-attributes
    def __init__(self, mu, sigma, vmin=0.0, nfmax=None):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
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
        self.particulator.backend.gauss_fragmentation(
            n_fragment=nf,
            mu=self.mu,
            sigma=self.sigma,
            frag_volume=frag_volume,
            x_plus_y=self.sum_of_volumes,
            rand=u01,
            vmin=self.vmin,
            nfmax=self.nfmax,
        )
