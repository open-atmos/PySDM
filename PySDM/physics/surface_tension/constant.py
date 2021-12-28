"""
constant surface tension coefficient
"""
import PySDM.physics.constants as const


class Constant:
    """
    Assumes aerosol is dissolved in the bulk of the droplet, and the
    droplet surface is composed of pure water with constant surface
    tension `sgm_w`.
    """

    def __init__(self, const):
        pass

    @staticmethod
    def sigma(const, T, v_wet, v_dry, f_org):  # pylint: disable=unused-argument
        return const.sgm_w
