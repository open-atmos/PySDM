"""
piecewise kernel from Eq (11) in
[Long 1974](https://doi.org/10.1175/1520-0469%281974%29031%3C1040%3ASTTDCE%3E2.0.CO%3B2)

Default parameters are:
    lin_coeff = 5.78e3 / s
    sq_coeff = 9.44e15 / m^3 / s
    r_thresh = 5e-5 m
"""


class Long1974:
    def __init__(self, lin_coeff=5.78e3, sq_coeff=9.44e15, r_thres=5e-5):
        self.lc = lin_coeff
        self.sc = sq_coeff
        self.rt = r_thres
        self.particulator = None
        self.largeR = None
        self.arrays = {}

    def register(self, builder):
        self.particulator = builder.particulator
        builder.request_attribute("volume")
        builder.request_attribute("radius")
        for key in (
            "r_lg",
            "v_lg",
            "v_sm",
            "v_ratio",
            "tmp",
            "tmp1",
        ):
            self.arrays[key] = self.particulator.PairwiseStorage.empty(
                self.particulator.n_sd // 2, dtype=float
            )
            self.arrays["condition"] = self.particulator.PairwiseStorage.empty(
                self.particulator.n_sd // 2, dtype=bool
            )

    def __call__(self, output, is_first_in_pair):
        # get smaller and larger radii, volume
        self.arrays["r_lg"].max(
            self.particulator.attributes["radius"], is_first_in_pair
        )
        self.arrays["v_lg"].max(
            self.particulator.attributes["volume"], is_first_in_pair
        )
        self.arrays["v_sm"].min(
            self.particulator.attributes["volume"], is_first_in_pair
        )

        # compute volume ratio
        self.arrays["v_ratio"].fill(self.arrays["v_sm"])
        self.arrays["v_ratio"].divide_if_not_zero(self.arrays["v_lg"])

        # compute small radius limit
        self.arrays["tmp1"].fill(self.arrays["v_ratio"])
        self.arrays["tmp1"] **= 2.0
        self.arrays["tmp1"] += 1.0
        self.arrays["tmp"].fill(self.arrays["v_lg"])
        self.arrays["tmp"] **= 2.0
        self.arrays["tmp1"] *= self.arrays["tmp"]
        self.arrays["tmp1"] *= self.sc

        # compute large radius (linear) limit
        self.arrays["v_ratio"] += 1.0
        self.arrays["v_ratio"] *= self.arrays["v_lg"]
        self.arrays["v_ratio"] *= self.lc

        # apply piecewise
        self.arrays["condition"].isless(self.arrays["r_lg"], self.rt)
        output.where(
            self.arrays["condition"], self.arrays["tmp1"], self.arrays["v_ratio"]
        )
