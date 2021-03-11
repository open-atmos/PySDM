import numba
import numpy as np
from PySDM.backends.numba.numba_helpers import temperature_pressure_RH
from .support import HENRY_CONST, EQUILIBRIUM_CONST, DIFFUSION_CONST, \
    MASS_ACCOMMODATION_COEFFICIENTS, AQUEOUS_COMPOUNDS, GASEOUS_COMPOUNDS, aqq_SO2, MEMBER, KINETIC_CONST, \
    SPECIFIC_GRAVITY, K_H2O
from PySDM.physics.constants import Md, M, si, R_str, Rd
from PySDM.physics.formulae import mole_fraction_2_mixing_ratio, radius
from scipy import optimize

from ...attributes.chemistry.ionic_strength import calc_ionic_strength


def dissolve_env_gases(super_droplet_ids, mole_amounts, env_mixing_ratio, henrysConstant, env_p, env_T,
                       env_rho_d, dt, dv, droplet_volume,
                       multiplicity, system_type, specific_gravity, alpha, diffusion_constant,
                       hconcdep, H_conc):
    mole_amount_taken = 0
    for i in super_droplet_ids:
        Mc = specific_gravity * Md
        Rc = R_str / Mc
        cinf = env_p / env_T / (Rd/env_mixing_ratio + Rc) / Mc
        ksi = hconcdep(H_conc[i])
        r_w = radius(volume=droplet_volume[i])
        v_avg = np.sqrt(8 * R_str * env_T / (np.pi * Mc))
        scale = (4 * r_w / (3 * v_avg * alpha) + r_w ** 2 / (3 * diffusion_constant))
        A_old = mole_amounts.data[i] / droplet_volume[i]
        # TODO: multiply cinf by ksi ???
        A_new = (A_old + dt * ksi * cinf / scale) / (1 + dt / (scale * ksi * henrysConstant * R_str * env_T))

        new_mole_amount_per_real_droplet = A_new * droplet_volume[i]
        mole_amount_taken += multiplicity[i] * (new_mole_amount_per_real_droplet - mole_amounts[i])
        assert new_mole_amount_per_real_droplet >= 0
        mole_amounts.data[i] = new_mole_amount_per_real_droplet
        assert mole_amounts[i] >= 0
    delta_mr = mole_amount_taken * specific_gravity * Md / (dv * env_rho_d)
    assert delta_mr <= env_mixing_ratio
    if system_type == 'closed':
        env_mixing_ratio -= delta_mr


def equilibrate_pH(super_droplet_ids, particles, env_T):
    N_III = particles["conc_N_mIII"].data
    N_V = particles["conc_N_V"].data
    C_IV = particles["conc_C_IV"].data
    S_IV = particles["conc_S_IV"].data
    S_VI = particles["conc_S_VI"].data
    volume = particles["volume"].data
    moles_H = particles["moles_H"].data

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

    pH_min=-1
    pH_max=9
    H_min = 10**(-pH_max) * (si.m**3 / si.litre)
    H_max = 10**(-pH_min) * (si.m**3 / si.litre)
    for i in super_droplet_ids:
        args = (i,)
        result = optimize.root_scalar(concentration, x0=H_min, x1=H_max, args=args)
        assert result.converged
        moles_H[i] = result.root * volume[i]


# NB: magic_const in the paper is k4.
# The value is fixed at 13 M^-1 (from Ania's Thesis)
magic_const = 13 / M


def oxidize(super_droplet_ids, particles, env_T, dt, droplet_volume):
    k0 = KINETIC_CONST["k0"].at(env_T)
    k1 = KINETIC_CONST["k1"].at(env_T)
    k2 = KINETIC_CONST["k2"].at(env_T)
    k3 = KINETIC_CONST["k3"].at(env_T)
    K_SO2 = EQUILIBRIUM_CONST["K_SO2"].at(env_T)
    K_HSO3 = EQUILIBRIUM_CONST["K_HSO3"].at(env_T)

    H = particles.attributes["conc_H"].data
    O3 = particles.attributes["conc_O3"].data
    H2O2 = particles.attributes["conc_H2O2"].data
    S_IV = particles.attributes["conc_S_IV"].data

    moles_O3 = particles.attributes["moles_O3"].data
    moles_H2O2 = particles.attributes["moles_H2O2"].data
    moles_S_IV = particles.attributes["moles_S_IV"].data
    moles_S_VI = particles.attributes["moles_S_VI"].data

    for i in super_droplet_ids:
        SO2aq = S_IV[i] / aqq_SO2(H[i])

        # NB: This might not be entirely correct
        # https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JD092iD04p04171
        # https://www.atmos-chem-phys.net/16/1693/2016/acp-16-1693-2016.pdf

        # NB: There is also slight error due to "borrowing" compounds when
        # the concentration is close to 0. That way, if the rate is big enough,
        # it will consume more compound than there is.

        ozone = (k0 + (k1 * K_SO2 / H[i]) + (k2 * K_SO2 * K_HSO3 / H[i]**2)) * O3[i] * SO2aq
        peroxide = k3 * K_SO2 / (1 + magic_const * H[i]) * H2O2[i] * SO2aq

        dconc_dt_O3 = -ozone
        dconc_dt_S_IV = -(ozone + peroxide)
        dconc_dt_H2O2 = -peroxide
        dconc_dt_S_VI = ozone + peroxide

        a = dt * droplet_volume.data[i]

        if (
            moles_O3.data[i] + dconc_dt_O3 * a < 0 or
            moles_S_IV.data[i] + dconc_dt_S_IV * a < 0 or
            moles_S_VI.data[i] + dconc_dt_S_VI * a < 0 or
            moles_H2O2.data[i] + dconc_dt_H2O2 * a < 0
        ):
            continue

        moles_O3.data[i] += dconc_dt_O3 * a
        moles_S_IV.data[i] += dconc_dt_S_IV * a
        moles_S_VI.data[i] += dconc_dt_S_VI * a
        moles_H2O2.data[i] += dconc_dt_H2O2 * a


class AqueousChemistry():
    def __init__(self, environment_mole_fractions, system_type):
        self.environment_mixing_ratios = {}
        for key, compound in GASEOUS_COMPOUNDS.items():
            shape = (1,)  # TODO #157
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
        for key in AQUEOUS_COMPOUNDS.keys():
            builder.request_attribute("conc_" + key)

    def __call__(self):
        n_cell = self.mesh.n_cell
        n_threads = 1  # TODO #157
        cell_order = np.arange(n_cell)  # TODO #157
        cell_start_arg = self.core.particles.cell_start.data
        idx = self.core.particles._Particles__idx

        rhod = self.env["rhod"]
        thd = self.env["thd"]
        qv = self.env["qv"]
        prhod = self.env.get_predicted("rhod")

        # TODO #157: same code in condensation
        n_substep = 50
        for _ in range(n_substep):
            for thread_id in numba.prange(n_threads):
                for i in range(thread_id, n_cell, n_threads):
                    cell_id = cell_order[i]

                    cell_start = cell_start_arg[cell_id]
                    cell_end = cell_start_arg[cell_id + 1]
                    n_sd_in_cell = cell_end - cell_start
                    if n_sd_in_cell == 0:
                        continue

                    rhod_mean = (prhod[cell_id] + rhod[cell_id]) / 2
                    T, p, RH = temperature_pressure_RH(rhod_mean, thd[cell_id], qv[cell_id])  # TODO #157: this is surely already computed elsewhere!

                    super_droplet_ids = []
                    for sd_id in idx[cell_start:cell_end]:
                        if calc_ionic_strength(
                            Hp=self.core.particles['conc_H'][sd_id],
                            N_III=self.core.particles['conc_N_mIII'][sd_id],
                            N_V=self.core.particles['conc_N_V'][sd_id],
                            C_IV=self.core.particles['conc_C_IV'][sd_id],
                            S_IV=self.core.particles['conc_S_IV'][sd_id],
                            S_VI=self.core.particles['conc_S_VI'][sd_id],
                            env_T=T
                        ) < 0.02 * M:
                            super_droplet_ids.append(sd_id)

                    for key, compound in GASEOUS_COMPOUNDS.items():
                        dissolve_env_gases(
                            super_droplet_ids=super_droplet_ids,
                            mole_amounts=self.core.particles['moles_'+key],
                            env_mixing_ratio=self.environment_mixing_ratios[compound][cell_id:cell_id+1],
                            henrysConstant=HENRY_CONST[compound].at(T),  # mol m−3 Pa−1
                            env_p=p,
                            env_T=T,
                            env_rho_d=rhod_mean,
                            dt=self.core.dt/n_substep,
                            dv=self.mesh.dv,
                            droplet_volume=self.core.particles["volume"],
                            multiplicity=self.core.particles["n"],
                            system_type=self.system_type,
                            specific_gravity=SPECIFIC_GRAVITY[compound],
                            alpha=MASS_ACCOMMODATION_COEFFICIENTS[compound],
                            diffusion_constant=DIFFUSION_CONST[compound],
                            hconcdep=MEMBER[compound],
                            H_conc=self.core.particles["conc_H"]
                        )
                        self.core.particles.attributes[f'moles_{key}'].mark_updated()  # TODO: not within threads loop!!!

                    equilibrate_pH(
                        super_droplet_ids=super_droplet_ids,
                        particles=self.core.particles,
                        env_T=T
                    )
                    self.core.particles.attributes['moles_H'].mark_updated()  # TODO: not qithin threads loop !!!

                    oxidize(
                        super_droplet_ids=super_droplet_ids,
                        particles=self.core.particles,
                        env_T=T,
                        dt=self.core.dt / n_substep,
                        droplet_volume=self.core.particles["volume"]
                    )
                    self.core.particles.attributes['moles_S_IV'].mark_updated()
                    self.core.particles.attributes['moles_S_VI'].mark_updated()
                    self.core.particles.attributes['moles_H2O2'].mark_updated()
                    self.core.particles.attributes['moles_O3'].mark_updated()


