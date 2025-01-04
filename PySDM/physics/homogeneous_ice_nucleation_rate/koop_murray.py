"""
Koop and Murray homogeneous nucleation rate parameterization for pure water droplets 
at water saturation
 ([Koop and Murray 2016](https://doi.org/10.1063/1.4962355))
"""


class KOOP_MURRAY:  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        pass

    @staticmethod
    def j_hom(const, T, da_w_ice):
        T_diff = T - const.T_tri
        return 10**(const.KOOP_MURRAY_C0 + const.KOOP_MURRAY_C1*T_diff + const.KOOP_MURRAY_C2*T_diff**2. + const.KOOP_MURRAY_C3*T_diff**3. + const.KOOP_MURRAY_C4*T_diff**4. + const.KOOP_MURRAY_C5*T_diff**5. + const.KOOP_MURRAY_C6*T_diff**6.) * const.KOOP_UNIT
