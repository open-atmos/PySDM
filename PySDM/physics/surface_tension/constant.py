import PySDM.physics.constants as const

class Constant:
    """
    Assumes aerosol is dissolved in the bulk of the droplet, and the 
    droplet surface is composed of pure water with constant surface
    tension `sgm_w`.
    """
    @staticmethod
    def sigma(T, v_wet, v_dry, f_org):
        return const.sgm_w
