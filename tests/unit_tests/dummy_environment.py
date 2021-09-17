from PySDM.state.mesh import Mesh
import numpy as np


class DummyEnvironment:

    def __init__(self, dt=None, grid=None, size=None, dv=None, courant_field_data=None, halo=None):
        self.core = None
        self.dt = dt
        if grid is None:
            self.mesh = Mesh.mesh_0d(dv)
        else:
            if size is None:
                size = tuple(1 for _ in range(len(grid)))
            self.mesh = Mesh(grid, size)
            if halo is not None:
                self.halo = halo
                self.qv = np.empty((grid[0] + 2*halo, grid[1] + 2*halo))
                self.thd = np.empty((grid[0] + 2*halo, grid[1] + 2*halo))
                self.pred = {}
                self.step_counter = 0
        self.courant_field_data = courant_field_data

    def register(self, core):
        self.core = core
        if hasattr(self, 'halo'):
            self.pred['qv'] = core.bck.Storage.empty(self.mesh.n_cell, dtype=float)
            self.pred['thd'] = core.bck.Storage.empty(self.mesh.n_cell, dtype=float)

    def get_courant_field_data(self):
        return self.courant_field_data

    def get_predicted(self, key):
        return self.pred[key]

    def get_qv(self):
        return self.qv[self.halo:-self.halo, self.halo:-self.halo]

    def get_thd(self):
        return self.thd[self.halo:-self.halo, self.halo:-self.halo]

    def sync(self):
        pass
