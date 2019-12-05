from ._moist import _Moist


class _MoistLagrangian(_Moist):
    def __init__(self, particles, variables, mass_of_dry_air):
        super().__init__(particles, variables)
        self.mass_of_dry_air = mass_of_dry_air

    @property
    def dv(self):
        return self.mass_of_dry_air / self.get_predicted("rhod")[0]
