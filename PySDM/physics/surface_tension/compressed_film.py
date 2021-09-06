from PySDM.physics import constants as const
import numpy as np


class CompressedFilm:
    @staticmethod
    def sigma(T, v_wet, v_dry, f_org):
        # convert wet volume to wet radius
        r_wet = ((3 * v_wet) / (4 * np.pi)) ** (1 / 3)

        # calculate the minimum shell volume, v_delta
        v_delta = v_wet - ((4 * np.pi) / 3 * (r_wet - const.delta_min) ** 3)

        # calculate the total volume of organic, v_beta
        v_beta = f_org * v_dry

        # calculate the coverage parameter
        c_beta = np.minimum(v_beta / v_delta, 1)
        
        # calculate sigma
        sgm = (1-c_beta) * const.sgm_w + c_beta * const.sgm_org
        return sgm
