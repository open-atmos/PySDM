import numpy as np


class RogersYau:
    def __init__(self, _):
        pass

    @staticmethod
    def v_term(const, radius):
        return np.where(
            radius < const.ROGERS_YAU_TERM_VEL_SMALL_R_LIMIT,
            const.ROGERS_YAU_TERM_VEL_SMALL_K * radius**const.TWO,
            np.where(
                radius < const.ROGERS_YAU_TERM_VEL_MEDIUM_R_LIMIT,
                const.ROGERS_YAU_TERM_VEL_MEDIUM_K * radius,
                const.ROGERS_YAU_TERM_VEL_LARGE_K * radius**const.ONE_HALF,
            ),
        )
