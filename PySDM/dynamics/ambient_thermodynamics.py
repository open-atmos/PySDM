"""
environment-sync triggering class
"""

from PySDM.dynamics.impl import register_dynamic


# TODO
VAR2KEY = {
    "water vapour mixing ratio": "water_vapour_mixing_ratio",
    "dry air potential temperature": "thd",
}


@register_dynamic()
class AmbientThermodynamics:
    def __init__(self):
        self.particulator = None

    def register(self, builder):
        self.particulator = builder.particulator
        builder.particulator.observers.append(self)
        self.cellwise_tendencies = {
            key: self.particulator.Storage.empty(
                self.particulator.mesh.n_cell, dtype=float
            )
            for key in ("water vapour mixing ratio", "dry air potential temperature")
        }
        self.dropwise_tendencies = {
            key: builder.get_attribute(f"dropwise {key} tendency")
            for key in self.cellwise_tendencies
        }

    def __call__(self):
        """beginning of timestep logic: fill the tendency arrays with environment-forcings"""
        self.particulator.environment.sync()
        for var, tendency in self.cellwise_tendencies.items():

            tendency.data[:] = (
                self.particulator.environment.get_predicted(VAR2KEY[var]).data
                - self.particulator.environment[VAR2KEY[var]].data
            ) / self.particulator.dt

    def notify(self):
        """end of timestep logic: apply the tendencies and zero the tendency arrays"""
        for var in self.cellwise_tendencies:
            self.particulator.apply_drop_and_cell_wise_tendencies_to_the_environment_and_zero_input_arrays(
                dropwise_tendency=self.dropwise_tendencies[var],
                cellwise_tendency=self.cellwise_tendencies[var],
                environment_state=self.particulator.environment.get_predicted(
                    VAR2KEY[var]
                ),
            )
