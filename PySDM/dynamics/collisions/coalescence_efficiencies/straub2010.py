"""
TODO #744
"""

import numpy as np

# TODO #744: TEST


class Straub2010Ec:
    def __init__(self):
        self.particulator = None
        self.pair_tmp = None
        self.arrays = {}
        self.const = None

    def register(self, builder):
        self.particulator = builder.particulator
        self.const = self.particulator.formulae.constants
        builder.request_attribute("volume")
        builder.request_attribute("relative fall velocity")
        for key in ("Sc", "tmp", "tmp2", "We"):
            self.arrays[key] = self.particulator.PairwiseStorage.empty(
                self.particulator.n_sd // 2, dtype=float
            )

    def __call__(self, output, is_first_in_pair):
        self.arrays["tmp"].sum(self.particulator.attributes["volume"], is_first_in_pair)
        self.arrays["Sc"].fill(self.arrays["tmp"])
        self.arrays["Sc"] *= 6 / self.const.PI
        self.arrays["tmp"] *= 2

        self.arrays["tmp2"].distance(
            self.particulator.attributes["relative fall velocity"], is_first_in_pair
        )
        self.arrays["tmp2"] **= 2
        self.arrays["We"].multiply(
            self.particulator.attributes["volume"], is_first_in_pair
        )
        self.arrays["We"].divide_if_not_zero(self.arrays["tmp"])
        self.arrays["We"] *= self.arrays["tmp2"]
        self.arrays["We"] *= self.const.rho_w

        self.arrays["Sc"] **= 2 / 3
        self.arrays["Sc"] *= self.const.PI * self.const.sgm_w

        self.arrays["We"].divide_if_not_zero(self.arrays["Sc"])
        self.arrays["We"] *= -1.15

        output.fill(np.exp(self.arrays["We"]))
