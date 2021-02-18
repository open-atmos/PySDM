from chempy import Substance
import numba
import numpy as np
from PySDM.backends.numba.numba_helpers import temperature_pressure_RH
from .support import EqConst
from PySDM.physics.constants import H_u, dT_u, ROOM_TEMP, M_SO2, Md
from PySDM.physics.formulae import mole_fraction_2_mixing_ratio, mixing_ratio_2_partial_pressure


HENRY_CONST = {
    "HNO3": EqConst((2.10 * 10 ** 5) * H_u, 0 * dT_u),
    "H2O2": EqConst((7.45 * 10 ** 4) * H_u, 7300 * dT_u),
    "NH3":  EqConst(62 * H_u, 4110 * dT_u),
    "SO2":  EqConst(1.23 * H_u, 3150 * dT_u),
    "CO2":  EqConst((3.4 * 10 ** -2) * H_u, 2440 * dT_u),
    "O3":   EqConst((1.13 * 10 ** -2) * H_u, 2540 * dT_u),
}

eps_SO2 = M_SO2 / Md


def delta_mixing_ratio(delta_mole_amount, dv, rho_d):
    return delta_mole_amount * M_SO2 / (rho_d * dv)


def dissolve_env_gases(super_droplet_ids, mole_amounts_SO2, env_mixing_ratio_SO2, env_T, env_p, env_rho_d, dv, droplet_volume, multiplicity, system_type):
    henrysConstant = HENRY_CONST["SO2"].at(env_T)  # mol m−3 Pa−1
    # TODO: effective H (dissociation) ... as option for tests
    # TODO: diffusion law formulation using mass accommodation coefficient
    mole_amount_taken = 0
    for i in super_droplet_ids:
        p_SO2 = mixing_ratio_2_partial_pressure(mixing_ratio=env_mixing_ratio_SO2, specific_gravity=eps_SO2, pressure=env_p)  # TODO: p vs. pd ?
        c_SO2 = henrysConstant * p_SO2  # mol / m3
        new_mole_amount_per_real_droplet = c_SO2 * droplet_volume[i]
        mole_amount_taken += multiplicity[i] * (new_mole_amount_per_real_droplet - mole_amounts_SO2[i])
        mole_amounts_SO2.data[i] = new_mole_amount_per_real_droplet
        assert mole_amounts_SO2[i] >= 0
    delta_mr = delta_mixing_ratio(mole_amount_taken, dv, env_rho_d)
    assert delta_mr <= env_mixing_ratio_SO2
    if system_type == 'closed':
        env_mixing_ratio_SO2 -= delta_mr


class AqueousChemistry:
    def __init__(self, environment_mole_fractions, system_type):
        self.environment_mixing_ratios = {}
        for compound in environment_mole_fractions:
            if compound != 'SO2':
                continue  # TODO
            self.environment_mixing_ratios[compound] = np.full((1,), mole_fraction_2_mixing_ratio(environment_mole_fractions[compound], eps_SO2))
        self.mesh = None
        self.core = None
        self.env = None
        assert system_type in ('open', 'closed')
        self.system_type = system_type

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
                    mole_amounts_SO2=self.core.particles["SO2"],
                    env_mixing_ratio_SO2=self.environment_mixing_ratios["SO2"][cell_id:cell_id+1],
                    env_T=T,
                    env_p=p,
                    env_rho_d=rhod_mean,
                    dv=self.mesh.dv,
                    droplet_volume=self.core.particles["volume"],
                    multiplicity=self.core.particles["n"],
                    system_type=self.system_type
                )
