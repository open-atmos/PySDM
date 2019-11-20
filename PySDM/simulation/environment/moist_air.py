"""
Created at 06.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np


class MoistAir:
    def __init__(self, simulation, thd_xzt_lambda, qv_xzt_lambda, rhod_z_lambda):
        self.simulation = simulation

        self.thd_lambda = thd_xzt_lambda
        self.qv_lambda = qv_xzt_lambda

        grid = simulation.grid
        n_cell = simulation.n_cell

        self.qv = simulation.backend.array((n_cell,), float)
        self.thd = simulation.backend.array((n_cell,), float)

        self.RH = simulation.backend.array((n_cell,), float)
        self.p = simulation.backend.array((n_cell,), float)
        self.T = simulation.backend.array((n_cell,), float)

        rhod = np.repeat(rhod_z_lambda((np.arange(grid[1]) + 1/2)/grid[1]).reshape((1, grid[1])), grid[0], axis=0)
        self.rhod = simulation.backend.from_ndarray(rhod.ravel())

        self.sync()

    def sync(self):
        self.simulation.backend.upload(self.qv_lambda().ravel(), self.qv)
        self.simulation.backend.upload(self.thd_lambda().ravel(), self.thd)

        self.simulation.backend.apply(
            function=self.simulation.backend.temperature_pressure_RH,
            args=(self.rhod, self.thd, self.qv),
            output=(self.T, self.p, self.RH)
        )

