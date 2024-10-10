"""
P(m; x, y) = nu^2 * (x+y) exp(-m * nu)
nu = 1/m* where m* is a scaling factor for fragment volume dist.
see [Feingold et al. 1999](https://doi.org/10.1175/1520-0469(1999)056<4100:TIOGCC>2.0.CO;2)
"""

from .impl import VolumeBasedFragmentationFunction


class Feingold1988(
    VolumeBasedFragmentationFunction
):  # pylint: disable=too-many-instance-attributes
    def __init__(self, scale, fragtol=1e-3, vmin=0.0, nfmax=None):
        super().__init__()
        self.scale = scale
        self.fragtol = fragtol
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
        self.particulator.backend.feingold1988_fragmentation(
            n_fragment=nf,
            scale=self.scale,
            frag_volume=frag_volume,
            x_plus_y=self.sum_of_volumes,
            rand=u01,
            fragtol=self.fragtol,
            vmin=self.vmin,
            nfmax=self.nfmax,
        )
