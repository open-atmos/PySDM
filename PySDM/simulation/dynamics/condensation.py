"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state.state import State
from PySDM.simulation.ambient_air.moist_air import MoistAir
from PySDM.utils import Physics


class Condensation:
    def __init__(self, ambient_air: MoistAir, dt, kappa):
        self.ambient_air = ambient_air

        self.dt = dt
        self.kappa = kappa

        self.rd = None

    def __call__(self, state: State):
        self.ambient_air.sync()

        x = state.get_backend_storage("x")
        rdry = state.get_backend_storage("dry radius")

        for i in state.idx[:state.SD_num]:
            cid = state.cell_id[i]
            r = Physics.x2r(x[i])
            T = self.ambient_air.T[cid]
            p = self.ambient_air.p[cid]
            S = (self.ambient_air.RH[cid] - 1)
            kp = self.kappa
            rd = rdry[i]

            # explicit Euler
            r_new = r + self.dt * state.backend.dr_dt_MM(r, T, p, S, kp, rd)

            x[i] = Physics.r2x(r_new)

        # TODO: update drop radii
        #       update fields due to condensation/evaporation
        #       ensure the above does include droplets that precipitated out of the domain


