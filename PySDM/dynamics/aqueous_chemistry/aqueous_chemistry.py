import numba
import numpy as np
from PySDM.backends.numba.numba_helpers import temperature_pressure_RH
from .support import EqConst, COMPOUNDS
from PySDM.physics.constants import H_u, dT_u, _weight, Md, M
from PySDM.physics.formulae import mole_fraction_2_mixing_ratio, mixing_ratio_2_partial_pressure
from scipy import optimize


HENRY_CONST = {
    "HNO3": EqConst((2.10 * 10 ** 5) * H_u, 0 * dT_u),
    "H2O2": EqConst((7.45 * 10 ** 4) * H_u, 7300 * dT_u),
    "NH3":  EqConst(62 * H_u, 4110 * dT_u),
    "SO2":  EqConst(1.23 * H_u, 3150 * dT_u),
    "CO2":  EqConst((3.4 * 10 ** -2) * H_u, 2440 * dT_u),
    "O3":   EqConst((1.13 * 10 ** -2) * H_u, 2540 * dT_u),
}

EQUILIBRIUM_CONST = {  # Reaction Specific units, K
    # ("HNO3(aq) = H+ + NO3-", 15.4, 0),
    "K_HNO3": EqConst(15.4 * M, 0 * dT_u),
    # ("H2SO3(aq) = H+ + HSO3-", 1.54*10**-2 * KU, 1960),
    "K_SO2":  EqConst((1.3 * 10 ** -2) * M, 1960 * dT_u),
    # ("NH4+ = NH3(aq) + H+", 10**-9.25 * M, 0),
    "K_NH3":  EqConst((1.7 * 10 ** -5) * M, -450 * dT_u),
    # ("H2CO3(aq) = H+ + HCO3-", 4.3*10**-7 * KU, -1000),
    "K_CO2":  EqConst((4.3 * 10 ** -7) * M, -1000 * dT_u),
    # ("HSO3- = H+ + SO3-2", 6.6*10**-8 * KU, 1500),
    "K_HSO3": EqConst((6.6 * 10 ** -8) * M, 1500 * dT_u),
    # ("HCO3- = H+ + CO3-2", 4.68*10**-11 * KU, -1760),
    "K_HCO3": EqConst((4.68 * 10 ** -11) * M, -1760 * dT_u),
    # ("HSO4- = H+ + SO4-2", 1.2*10**-2 * KU, 2720),
    "K_HSO4": EqConst((1.2 * 10 ** -2) * M, 2720 * dT_u)
}

K_H2O = 1e-14 * M * M

SPECIFIC_GRAVITY = {compound: _weight(compound) / Md for compound in COMPOUNDS}


def dissolve_env_gases(super_droplet_ids, mole_amounts, env_mixing_ratio, henrysConstant, env_p, env_rho_d, dv, droplet_volume,
                       multiplicity, system_type, specific_gravity, compound):
    # TODO: effective H (dissociation) ... as option for tests
    # TODO: diffusion law formulation using mass accommodation coefficient
    mole_amount_taken = 0
    for i in super_droplet_ids:
        partial_pressure = mixing_ratio_2_partial_pressure(mixing_ratio=env_mixing_ratio, specific_gravity=specific_gravity, pressure=env_p)  # TODO: p vs. pd ?
        concentration = henrysConstant * partial_pressure  # mol / m3
        new_mole_amount_per_real_droplet = concentration * droplet_volume[i]
        mole_amount_taken += multiplicity[i] * (new_mole_amount_per_real_droplet - mole_amounts[i])
        mole_amounts.data[i] = new_mole_amount_per_real_droplet
        assert mole_amounts[i] >= 0
    delta_mr = mole_amount_taken * specific_gravity * Md / (dv * env_rho_d)
    assert delta_mr <= env_mixing_ratio
    if system_type == 'closed':
        env_mixing_ratio -= delta_mr


def equilibrate_pH(super_droplet_ids, particles, env_T):
    N_III = particles["conc_NH3"].data
    N_V = particles["conc_HNO3"].data
    C_IV = particles["conc_CO2"].data
    S_IV = particles["conc_SO2"].data
    S_VI = particles["conc_HSO4"].data
    H = particles["conc_H"].data
    volume = particles["volume"].data

    K_NH3 = EQUILIBRIUM_CONST["K_NH3"].at(env_T)
    K_SO2 = EQUILIBRIUM_CONST["K_SO2"].at(env_T)
    K_HSO3 = EQUILIBRIUM_CONST["K_HSO3"].at(env_T)
    K_HSO4 = EQUILIBRIUM_CONST["K_HSO4"].at(env_T)
    K_HCO3 = EQUILIBRIUM_CONST["K_HCO3"].at(env_T)
    K_CO2 = EQUILIBRIUM_CONST["K_CO2"].at(env_T)
    K_HNO3 = EQUILIBRIUM_CONST["K_HNO3"].at(env_T)

    def concentration(H_i, args):
        i = args
        ammonia = (N_III[i] * H_i * K_NH3) / (K_H2O + K_NH3 * H_i)
        nitric = N_V[i] * K_HNO3 / (H_i + K_HNO3)
        sulfous = S_IV[i] * K_SO2 * (H_i + 2 * K_HSO3) / (H_i * H_i + H_i * K_SO2 + K_SO2 * K_HSO3)
        water = K_H2O / H_i
        sulfuric = S_VI[i] * (H_i + 2 * K_HSO4) / (H_i + K_HSO4)
        carbonic = C_IV[i] * K_CO2 * (H_i + 2 * K_HCO3) / (H_i * H_i + H_i * K_CO2 + K_CO2 * K_HCO3)
        zero = H_i + ammonia - (nitric + sulfous + water + sulfuric + carbonic)
        return zero

    for i in super_droplet_ids:
        args = (i,)
        result = optimize.root_scalar(concentration, x0=1e-10, x1=1, args=args)
        assert result.converged
        H[i] = result.root * volume[i]


class AqueousChemistry:
    def __init__(self, environment_mole_fractions, system_type):
        self.environment_mixing_ratios = {}
        for compound in COMPOUNDS:
            shape = (1,)  # TODO
            if compound in environment_mole_fractions:
                self.environment_mixing_ratios[compound] = np.full(
                    shape,
                    mole_fraction_2_mixing_ratio(environment_mole_fractions[compound], SPECIFIC_GRAVITY[compound])
                )
        self.mesh = None
        self.core = None
        self.env = None
        assert system_type in ('open', 'closed')
        self.system_type = system_type

    def register(self, builder):
        self.mesh = builder.core.mesh
        self.core = builder.core
        self.env = builder.core.env
        for compound in COMPOUNDS:
            builder.request_attribute("conc_" + compound)

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
                T, p, RH = temperature_pressure_RH(rhod_mean, thd[cell_id], qv[cell_id])  # TODO: this is surely already computed elsewhere!

                for compound in self.environment_mixing_ratios:
                    dissolve_env_gases(
                        super_droplet_ids=idx[cell_start:cell_end],
                        mole_amounts=self.core.particles['moles_'+compound],
                        env_mixing_ratio=self.environment_mixing_ratios[compound][cell_id:cell_id+1],
                        henrysConstant=HENRY_CONST[compound].at(T),  # mol m−3 Pa−1
                        env_p=p,
                        env_rho_d=rhod_mean,
                        dv=self.mesh.dv,
                        droplet_volume=self.core.particles["volume"],
                        multiplicity=self.core.particles["n"],
                        system_type=self.system_type,
                        specific_gravity=SPECIFIC_GRAVITY[compound],
                        compound=compound
                    )

                equilibrate_pH(
                    super_droplet_ids=idx[cell_start:cell_end],
                    particles=self.core.particles,
                    env_T=T
                )
