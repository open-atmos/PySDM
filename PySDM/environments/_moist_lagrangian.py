"""
Created at 2019
"""

from ._moist import _Moist


class _MoistLagrangian(_Moist):

    def __init__(self, core, dt, mesh, variables, mass_of_dry_air):
        super().__init__(core, dt, mesh, variables)
        self.mass_of_dry_air = mass_of_dry_air

    @property
    def dv(self):
        rhod_mean = (self.get_predicted("rhod")[0] + self["rhod"][0]) / 2
        return self.mass_of_dry_air / rhod_mean
