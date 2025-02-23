"""
See [Straub et al. 2010](https://doi.org/10.1175/2009JAS3175.1)
"""

from PySDM.physics.constants import si

from .impl import VolumeBasedFragmentationFunction


class Straub2010Nf(VolumeBasedFragmentationFunction):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, vmin=0.0, nfmax=None):
        super().__init__()
        self.vmin = vmin
        self.nfmax = nfmax
        self.arrays = {}
        self.straub_tmp = {}
        self.max_size = None
        self.sum_of_volumes = None
        self.const = None

    def register(self, builder):
        super().register(builder)
        self.max_size = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.sum_of_volumes = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.const = self.particulator.formulae.constants
        builder.request_attribute("radius")
        builder.request_attribute("relative fall velocity")
        for key in ("Sc", "tmp", "tmp2", "CKE", "We", "gam", "CW", "ds"):
            self.arrays[key] = self.particulator.PairwiseStorage.empty(
                self.particulator.n_sd // 2, dtype=float
            )
        for key in ("Nr1", "Nr2", "Nr3", "Nr4", "Nrt", "d34"):
            self.straub_tmp[key] = self.particulator.PairwiseStorage.empty(
                self.particulator.n_sd // 2, dtype=float
            )

    def compute_fragment_number_and_volumes(
        self, nf, frag_volume, u01, is_first_in_pair
    ):
        self.max_size.max(self.particulator.attributes["volume"], is_first_in_pair)
        self.sum_of_volumes.sum(
            self.particulator.attributes["volume"], is_first_in_pair
        )
        self.arrays["ds"].min(self.particulator.attributes["radius"], is_first_in_pair)
        self.arrays["ds"] *= 2

        # compute the dimensionless numbers and CW=CKE * We
        self.arrays["tmp"].sum(self.particulator.attributes["volume"], is_first_in_pair)
        self.arrays["Sc"].fill(self.arrays["tmp"])
        self.arrays["Sc"] **= 2 / 3
        self.arrays["Sc"] *= (
            self.const.PI * self.const.sgm_w * (6 / self.const.PI) ** (2 / 3)
        )
        self.arrays["tmp2"].distance(
            self.particulator.attributes["relative fall velocity"], is_first_in_pair
        )
        self.arrays["tmp2"] **= 2
        self.arrays["CKE"].multiply(
            self.particulator.attributes["volume"], is_first_in_pair
        )
        self.arrays["CKE"].divide_if_not_zero(self.arrays["tmp"])
        self.arrays["CKE"] *= self.arrays["tmp2"]
        self.arrays["CKE"] *= self.const.rho_w / 2

        self.arrays["We"].fill(self.arrays["CKE"])
        self.arrays["We"].divide_if_not_zero(self.arrays["Sc"])

        self.arrays["CW"].fill(self.arrays["We"])
        self.arrays["CW"] *= self.arrays["CKE"]
        self.arrays["CW"] /= si.uJ

        self.arrays["gam"].max(self.particulator.attributes["radius"], is_first_in_pair)
        self.arrays["tmp"].min(self.particulator.attributes["radius"], is_first_in_pair)
        self.arrays["gam"].divide_if_not_zero(self.arrays["tmp"])

        for key in ("Nr1", "Nr2", "Nr3", "Nr4", "Nrt"):
            self.straub_tmp[key].fill(0)

        self.particulator.backend.straub_fragmentation(
            n_fragment=nf,
            CW=self.arrays["CW"],
            gam=self.arrays["gam"],
            ds=self.arrays["ds"],
            frag_volume=frag_volume,
            v_max=self.max_size,
            x_plus_y=self.sum_of_volumes,
            rand=u01,
            vmin=self.vmin,
            nfmax=self.nfmax,
            Nr1=self.straub_tmp["Nr1"],
            Nr2=self.straub_tmp["Nr2"],
            Nr3=self.straub_tmp["Nr3"],
            Nr4=self.straub_tmp["Nr4"],
            Nrt=self.straub_tmp["Nrt"],
            d34=self.straub_tmp["d34"],
        )
