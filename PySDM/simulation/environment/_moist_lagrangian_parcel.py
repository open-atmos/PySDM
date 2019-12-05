from ._moist_lagrangian import _MoistLagrangian


class _MoistLagrangianParcel(_MoistLagrangian):
    def __init__(self, particles, variables, mass_of_dry_air):
        super().__init__(particles, variables, mass_of_dry_air)
        self.parcel_vars = variables

    def _get_thd(self):
        return self['thd']

    def _get_qv(self):
        return self['qv']

    def sync_parcel_vars(self):
        for var in self.parcel_vars:
            self._tmp[var][:] = self[var][:]

    def sync(self):
        self.sync_parcel_vars()
        self.advance_parcel_vars()
        super().sync()

    def advance_parcel_vars(self): raise NotImplemented()
