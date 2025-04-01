"""
terminal velocity of smooth ice spheres
"""

class IceSphere:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def v_base_term(radius, prefactor):
        return( prefactor * 2 * radius)

    @staticmethod
    def stokes_regime(const, radius, dynamic_viscosity):
        return( const.g_std * const.rho_i * radius / 9 / dynamic_viscosity )

    # TODO: find parametrisation for general functional relationship between reynolds number and drag coefficient
    @staticmethod
    def general_flow_regime(const):
        pass