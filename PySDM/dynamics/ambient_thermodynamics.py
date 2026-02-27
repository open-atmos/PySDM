"""
environment-sync triggering class
"""

from PySDM.dynamics.impl import register_dynamic


@register_dynamic()
class AmbientThermodynamics:
    def __init__(self, relaxed: bool = False):
        self.particulator = None
        self.relaxed = relaxed

    def register(self, builder):
        self.particulator = builder.particulator
        builder.particulator.observers.append(self)

        for var in ("water vapour mixing ratio", "dry air potential temperature"):
            builder.request_attribute(f"dropwise {var}")
            builder.request_attribute(f"dropwise {var} tendency")

    def __call__(self):
        """beginning of timestep logic: fill the tendency arrays with environment-forcings"""
        self.particulator.environment.sync()

        # TODO - after condensation would be merged into depositional growth, this call should happen here!
        # self.particulator.drop_local_thermodynamics(self.relaxed)

    def notify(self):
        """end of timestep logic: apply the tendencies and zero the tendency arrays"""
        if self.relaxed:
            self.particulator.apply_dropwise_thermodynamic_tendency()
        self.particulator.update_ambient_thermodynamics_wrt_ice_growth()
