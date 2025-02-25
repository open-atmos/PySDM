"""
Zero-dimensional (Lagrangian) moist environment commons
"""

from PySDM.environments.impl.moist import Moist


class MoistLagrangian(Moist):
    """base class for moist Lagrangian environments (parcel, chamber, ...)"""

    def get_thd(self):
        return self["thd"]

    def get_water_vapour_mixing_ratio(self):
        return self["water_vapour_mixing_ratio"]

    def sync(self):
        self.sync_moist_vars()
        self.advance_moist_vars()
        super().sync()

    def post_register(self):
        self.sync_moist_vars()
        super().sync()
        self.notify()

    def sync_moist_vars(self):
        raise NotImplementedError()

    def advance_moist_vars(self):
        raise NotImplementedError()
