"""
See [Low & List 1982](https://doi.org/10.1175/1520-0469(1982)039<1607:CCABOR>2.0.CO;2)
"""

from .impl import VolumeBasedFragmentationFunction


class LowList1982Nf(VolumeBasedFragmentationFunction):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, vmin=0.0, nfmax=None):
        super().__init__()
        self.vmin = vmin
        self.nfmax = nfmax
        self.arrays = {}
        self.ll82_tmp = {}
        self.sum_of_volumes = None
        self.const = None

    def register(self, builder):
        super().register(builder)
        self.sum_of_volumes = self.particulator.PairwiseStorage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.const = self.particulator.formulae.constants
        builder.request_attribute("radius")
        builder.request_attribute("relative fall velocity")
        for key in ("Sc", "St", "tmp", "tmp2", "CKE", "We", "W2", "ds", "dl", "dcoal"):
            self.arrays[key] = self.particulator.PairwiseStorage.empty(
                self.particulator.n_sd // 2, dtype=float
            )
        for key in ("Rf", "Rs", "Rd"):
            self.ll82_tmp[key] = self.particulator.PairwiseStorage.empty(
                self.particulator.n_sd // 2, dtype=float
            )

    def compute_fragment_number_and_volumes(
        self, nf, frag_volume, u01, is_first_in_pair
    ):
        self.sum_of_volumes.sum(
            self.particulator.attributes["volume"], is_first_in_pair
        )
        self.arrays["ds"].min(self.particulator.attributes["radius"], is_first_in_pair)
        self.arrays["ds"] *= 2
        self.arrays["dl"].max(self.particulator.attributes["radius"], is_first_in_pair)
        self.arrays["dl"] *= 2
        self.arrays["dcoal"].sum(
            self.particulator.attributes["volume"], is_first_in_pair
        )

        self.arrays["dcoal"] /= self.const.PI / 6
        self.arrays["dcoal"] **= 1 / 3

        # compute the surface energy, CKE, & dimensionless numbers
        self.arrays["Sc"].sum(self.particulator.attributes["volume"], is_first_in_pair)
        self.arrays["Sc"] **= 2 / 3
        self.arrays["Sc"] *= (
            self.const.PI * self.const.sgm_w * (6 / self.const.PI) ** (2 / 3)
        )
        self.arrays["St"].min(self.particulator.attributes["radius"], is_first_in_pair)
        self.arrays["St"] *= 2
        self.arrays["St"] **= 2
        self.arrays["tmp"].max(self.particulator.attributes["radius"], is_first_in_pair)
        self.arrays["tmp"] *= 2
        self.arrays["tmp"] **= 2
        self.arrays["St"] += self.arrays["tmp"]
        self.arrays["St"] *= self.const.PI * self.const.sgm_w

        self.arrays["tmp"].sum(self.particulator.attributes["volume"], is_first_in_pair)
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
        self.arrays["W2"].fill(self.arrays["CKE"])
        self.arrays["We"].divide_if_not_zero(self.arrays["Sc"])
        self.arrays["W2"].divide_if_not_zero(self.arrays["St"])

        for key in ("Rf", "Rs", "Rd"):
            self.ll82_tmp[key] *= 0.0

        self.particulator.backend.ll82_fragmentation(
            n_fragment=nf,
            CKE=self.arrays["CKE"],
            W=self.arrays["We"],
            W2=self.arrays["W2"],
            St=self.arrays["St"],
            ds=self.arrays["ds"],
            dl=self.arrays["dl"],
            dcoal=self.arrays["dcoal"],
            frag_volume=frag_volume,
            x_plus_y=self.sum_of_volumes,
            rand=u01,
            vmin=self.vmin,
            nfmax=self.nfmax,
            Rf=self.ll82_tmp["Rf"],
            Rs=self.ll82_tmp["Rs"],
            Rd=self.ll82_tmp["Rd"],
        )
