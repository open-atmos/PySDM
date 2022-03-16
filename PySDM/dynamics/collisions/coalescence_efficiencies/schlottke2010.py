"""
TODO #744
"""
import numpy as np

# TODO #744: TEST


class Schlottke2010:
    def __init__(self):
        self.particulator = None
        self.pair_tmp = None
        self.arrays = {}
        self.const = None

    def register(self, builder):
        self.particulator = builder.particulator
        self.const = self.particulator.formulae.constants
        builder.request_attribute("volume")
        builder.request_attribute("terminal velocity")
        for key in ("Sc", "tmp", "tmp2", "We"):
            self.arrays[key] = self.particulator.PairwiseStorage.empty(
                self.particulator.n_sd // 2, dtype=float
            )

    def __call__(self, output, is_first_in_pair):
        self.arrays["tmp"].sum(self.particulator.attributes["volume"], is_first_in_pair)
        self.arrays["tmp"] /= self.const.pi / 6

        self.arrays["tmp2"].distance(
            self.particulator.attributes["terminal velocity"], is_first_in_pair
        )
        self.arrays["tmp2"] **= 2
        self.arrays["We"].multiply(
            self.particulator.attributes["volume"], is_first_in_pair
        )
        self.arrays["We"] /= self.arrays["tmp"]
        self.arrays["We"] *= self.arrays["tmp2"]
        self.arrays["We"] *= self.const.pi / 12 * self.const.rho_w

        self.arrays["Sc"][:] = self.arrays["tmp"][:]
        self.arrays["Sc"] **= 2 / 3
        self.arrays["Sc"] *= self.const.pi * self.const.sgm_w

        self.arrays["We"] /= self.arrays["Sc"]
        self.arrays["We"] *= -1.15

        output[:] = np.exp(self.arrays["We"])
