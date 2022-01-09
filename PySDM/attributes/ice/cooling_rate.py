import numpy as np
from PySDM.attributes.impl import DerivedAttribute


class CoolingRate(DerivedAttribute):
    def __init__(self, builder):
        self.cell_id = builder.get_attribute('cell id')
        super().__init__(
            builder=builder,
            name='cooling rate',
            dependencies=(self.cell_id,)
        )
        self.prev_T = builder.particulator.backend.Storage.from_ndarray(
            np.full(builder.particulator.n_sd, np.nan)
        )

    def recalculate(self):
        cell_id = self.particulator.attributes['cell id']
        env_T = self.particulator.environment['T']
        self.data[:] = env_T[cell_id]
        self.data -= self.prev_T
        self.data /= self.particulator.environment.dt
        self.prev_T[:] = env_T[cell_id]
