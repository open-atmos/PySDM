"""
columnar particles with constant density of ice
"""

class Columnar:
    def __init__(self, const, _):
        self.mass_density = const.rho_i

    @staticmethod
    def supports_mixed_phase(_=None):
        return True

