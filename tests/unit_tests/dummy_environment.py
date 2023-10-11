# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM.impl.mesh import Mesh


class DummyEnvironment:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        *,
        timestep=None,
        grid=None,
        size=None,
        volume=None,
        courant_field_data=None,
        halo=None,
    ):
        self.particulator = None
        self.dt = timestep
        if grid is None:
            self.mesh = Mesh.mesh_0d(volume)
        else:
            if size is None:
                size = tuple(1 for _ in range(len(grid)))
            self.mesh = Mesh(grid, size)
            if halo is not None:
                self.halo = halo
                self.water_vapour_mixing_ratio = np.empty(
                    (grid[0] + 2 * halo, grid[1] + 2 * halo)
                )
                self.thd = np.empty((grid[0] + 2 * halo, grid[1] + 2 * halo))
                self.pred = {}
                self.step_counter = 0
        self.courant_field_data = courant_field_data

    def register(self, particulator):
        self.particulator = particulator
        if hasattr(self, "halo"):
            self.pred["water_vapour_mixing_ratio"] = particulator.backend.Storage.empty(
                self.mesh.n_cell, dtype=float
            )
            self.pred["thd"] = particulator.backend.Storage.empty(
                self.mesh.n_cell, dtype=float
            )

    def get_courant_field_data(self):
        return self.courant_field_data

    def get_predicted(self, key):
        return self.pred[key]

    def get_water_vapour_mixing_ratio(self):
        if self.halo is not None:
            halo = int(self.halo)
            return self.water_vapour_mixing_ratio[halo:-halo, halo:-halo]
        raise ValueError()

    def get_thd(self):
        if self.halo is not None:
            halo = int(self.halo)
            return self.thd[halo:-halo, halo:-halo]
        raise ValueError()

    def sync(self):
        pass
