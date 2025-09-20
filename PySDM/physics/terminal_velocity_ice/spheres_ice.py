"""
terminal velocity of smooth ice spheres
"""


class IceSphere:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def v_base_term(const, radius, prefactor):
        return prefactor * const.TWO * radius

    @staticmethod
    def stokes_regime(const, radius, dynamic_viscosity):
        return const.g_std * const.rho_i * radius / const.NINE / dynamic_viscosity

    # TODO #1602 general functional relationship between reynolds number and drag coefficient
    @staticmethod
    def general_flow_regime(const):
        pass
