"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state.state import State
from PySDM.simulation import phys
from PySDM.simulation.phys import si
from PySDM.simulation.ambient_air.moist_air import MoistAir
from PySDM.utils import Physics

import numpy as np
from scipy import optimize as root


class Condensation:
    def __init__(self, ambient_air: MoistAir, dt, kappa):
        self.ambient_air = ambient_air

        self.dt = dt * si.seconds
        self.kappa = kappa

        self.rd = None

    @staticmethod
    def r_wet_init(r_dry: np.ndarray, ambient_air: MoistAir, cell_id: np.ndarray, kappa):
        r_wet = np.empty_like(r_dry)

        for i, r_d in enumerate(r_dry):
            cid = cell_id[i]
            # root-finding initial guess
            a = r_d
            b = phys.mgn(phys.r_cr(kappa, r_d * si.metres, ambient_air.T[cid] * si.kelvins))
            # minimisation
            f = lambda r_w: phys.mgn(phys.dr_dt_MM(
                r_w * si.metres,
                ambient_air.T[cid] * si.kelvin,
                ambient_air.p[cid] * si.pascal,
                np.minimum(0, ambient_air.RH[cid] - 1) * si.dimensionless,
                kappa,
                r_d * si.metres
            ))
            r_wet[i] = root.brentq(f, a, b)

        return r_wet

    def __call__(self, state: State):
        self.ambient_air.sync()

        x = state.get_backend_storage("x")
        rdry = state.get_backend_storage("dry radius")

        for i in state.idx[:state.SD_num]:
            cid = state.cell_id[i]
            r = Physics.x2r(x[i]) * si.metres
            T = self.ambient_air.T[cid] * si.kelvin
            p = self.ambient_air.p[cid] * si.pascal
            S = (self.ambient_air.RH[cid] - 1) * si.dimensionless
            kp = self.kappa
            rd = rdry[i] * si.metres

            # explicit Euler
            r_new = r + self.dt * phys.dr_dt_MM(r, T, p, S, kp, rd)

            x[i] = Physics.r2x(phys.mgn(r_new))

        # TODO: update drop radii
        #       update fields due to condensation/evaporation
        #       ensure the above does include droplets that precipitated out of the domain


