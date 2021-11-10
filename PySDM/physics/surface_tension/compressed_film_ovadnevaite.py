import numpy as np
from PySDM.physics import constants as const

sgm_org = np.nan
delta_min = np.nan


class CompressedFilmOvadnevaite:
    """
    Compressed film model of surface-partitioning of organics from Ovadnevaite et al. (2017)
    and as described in Lowe et al. (2019).

    Assumes a surface tension of the organic material `sgm_org` and minimum film thickness,
    the thickness of a monolayer of the organic molecules, `delta_min`. Assumes all organic
    partitions to the surface of the droplet and that the surface tension is a weighted
    average of the surface tension of water and the organic material.
    """
    @staticmethod
    def _check():
        assert np.isfinite(sgm_org)
        assert np.isfinite(delta_min)

    @staticmethod
    def sigma(T, v_wet, v_dry, f_org):  # pylint: disable=unused-argument
        # convert wet volume to wet radius
        r_wet = ((3 * v_wet) / (4 * np.pi)) ** (1 / 3)

        # calculate the minimum shell volume, v_delta
        v_delta = v_wet - ((4 * np.pi) / 3 * (r_wet - delta_min) ** 3)

        # calculate the total volume of organic, v_beta
        v_beta = f_org * v_dry

        # calculate the coverage parameter
        c_beta = np.minimum(v_beta / v_delta, 1)

        # calculate sigma
        sgm = (1-c_beta) * const.sgm_w + c_beta * sgm_org
        return sgm
