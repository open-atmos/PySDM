"""
Koop homogeneous nucleation rate parameterization for solution droplets [Koop et al. 2000] corrected 
such that it coincides with homogeneous nucleation rate parameterization for pure water droplets 
[Koop and Murray 2016] at water saturation between 235K < T < 240K
 ([Spichtinger et al. 2023](https://doi.org/10.5194/acp-23-2035-2023))
"""


class KOOP_CORR:  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        pass

    @staticmethod
    def j_hom(const, T, da_w_ice):
        return 10**(const.KOOP_2000_C1 + const.KOOP_2000_C2 * da_w_ice - const.KOOP_2000_C3 * da_w_ice**2. + const.KOOP_2000_C4 * da_w_ice**3. + const.KOOP_CORR) * const.KOOP_UNIT
