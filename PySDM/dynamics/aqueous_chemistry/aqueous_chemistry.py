from chempy import Substance
import numba
import numpy as np
from PySDM.backends.numba.numba_helpers import temperature_pressure_RH
from .support import EqConst
from PySDM.physics.constants import H_u, dT_u, ROOM_TEMP


HENRY_CONST = {
    "HNO3": EqConst((2.10 * 10 ** 5) * H_u, 0 * dT_u),
    "H2O2": EqConst((7.45 * 10 ** 4) * H_u, 7300 * dT_u),
    "NH3":  EqConst(62 * H_u, 4110 * dT_u),
    "SO2":  EqConst(1.23 * H_u, 3150 * dT_u),
    "CO2":  EqConst((3.4 * 10 ** -2) * H_u, 2440 * dT_u),
    "O3":   EqConst((1.13 * 10 ** -2) * H_u, 2540 * dT_u),
}


def dissolve_env_gases(super_droplet_ids, amounts_SO2, env_SO2, env_T): #, amounts, V_w, n, *, dt, steps=1, T=ROOM_TEMP):
    todo = HENRY_CONST["SO2"].at(env_T)

    for i in super_droplet_ids:
        pass
    #     hconc = amounts[self.hindex]
    #     for henry, gasi, envi in zip(self.henry_processes,
    #                                  self.gas_indices,
    #                                  self.environment_indices.values()):
    #         A = amounts[gasi]
    #         c = self.environment[envi]
    #         for i in range(steps):
    #             A = henry(A, c, T=T, V_w=V_w, H=hconc, dt=dt)
    #
    #             # Don't accumulate to avoid numerical errors
    #             # We multiply by n, since all the droplets suck in the gases
    #             c = (self.environment[envi]
    #                  - (A - amounts[gasi]) * n * V_w / self.particles.environment.dv)
    #
    #             # Ensure we do not take too much
    #             if A < 0:
    #                 print(f"Borrowed gas: {self._indices[gasi]}")
    #                 A = 0
    #             if c < 0:
    #                 print(f"Borrowed gas: {self._indices[gasi]}")
    #                 c = 0
    #         amounts[gasi] = A
    #         self.environment[envi] = c
    #
    # return amounts


class AqueousChemistry:
    def __init__(self, environment_amount):
        self.environment_amount = environment_amount
        self.mesh = None
        self.core = None
        self.env = None

    def register(self, builder):
        self.mesh = builder.core.mesh
        self.core = builder.core
        self.env = builder.core.env
        if self.mesh.dimension != 0:
            raise NotImplementedError()

    def __call__(self):
        n_cell = self.mesh.n_cell
        n_threads = 1  # TODO
        cell_order = np.arange(n_cell)  # TODO
        cell_start_arg = self.core.particles.cell_start.data
        idx = self.core.particles._Particles__idx

        rhod = self.env["rhod"]
        thd = self.env["thd"]
        qv = self.env["qv"]
        prhod = self.env.get_predicted("rhod")

        # TODO: same code in condensation
        for thread_id in numba.prange(n_threads):
            for i in range(thread_id, n_cell, n_threads):
                cell_id = cell_order[i]

                cell_start = cell_start_arg[cell_id]
                cell_end = cell_start_arg[cell_id + 1]
                n_sd_in_cell = cell_end - cell_start
                if n_sd_in_cell == 0:
                    continue

                rhod_mean = (prhod[cell_id] + rhod[cell_id]) / 2
                T, p, RH = temperature_pressure_RH(rhod_mean, thd[cell_id], qv[cell_id])
                dissolve_env_gases(
                    super_droplet_ids=idx[cell_start:cell_end],
                    amounts_SO2=self.core.particles["SO2"],
                    env_SO2=self.environment_amount["SO2"],
                    env_T=T
                )
