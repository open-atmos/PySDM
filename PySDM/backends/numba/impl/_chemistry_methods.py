import numba
import numpy as np

from PySDM.backends.numba import conf
from PySDM.backends.numba.numba_helpers import pH2H, bisec, H2pH
from PySDM.physics.constants import Md, R_str, Rd, M, K_H2O
from PySDM.physics.formulae import radius
from PySDM.dynamics.aqueous_chemistry.support import HENRY_CONST, SPECIFIC_GRAVITY, \
    MASS_ACCOMMODATION_COEFFICIENTS, DIFFUSION_CONST, GASEOUS_COMPOUNDS, DISSOCIATION_FACTORS, \
    KINETIC_CONST, EQUILIBRIUM_CONST


class ChemistryMethods:
    @staticmethod
    def dissolution(*, n_cell, n_threads, cell_order, cell_start_arg, idx, do_chemistry_flag, mole_amounts,
                    env_mixing_ratio, env_T, env_p, env_rho_d, ksi, dt, dv, system_type, droplet_volume,
                    multiplicity):
        for thread_id in numba.prange(n_threads):
            for i in range(thread_id, n_cell, n_threads):
                cell_id = cell_order[i]

                cell_start = cell_start_arg[cell_id]
                cell_end = cell_start_arg[cell_id + 1]
                n_sd_in_cell = cell_end - cell_start
                if n_sd_in_cell == 0:
                    continue

                super_droplet_ids = numba.typed.List()
                for sd_id in idx[cell_start:cell_end]:
                    if do_chemistry_flag.data[sd_id]:
                        super_droplet_ids.append(sd_id)

                if len(super_droplet_ids) == 0:
                    return

                for key, compound in GASEOUS_COMPOUNDS.items():
                    ChemistryMethods.dissolution_body(
                        super_droplet_ids=super_droplet_ids,
                        mole_amounts=mole_amounts[key].data,
                        env_mixing_ratio=env_mixing_ratio[compound][cell_id:cell_id + 1],
                        henrysConstant=HENRY_CONST[compound].at(env_T[cell_id]),
                        env_p=env_p[cell_id],
                        env_T=env_T[cell_id],
                        env_rho_d=env_rho_d[cell_id],
                        dt=dt,
                        dv=dv,
                        droplet_volume=droplet_volume.data,
                        multiplicity=multiplicity.data,
                        system_type=system_type,
                        specific_gravity=SPECIFIC_GRAVITY[compound],
                        alpha=MASS_ACCOMMODATION_COEFFICIENTS[compound],
                        diffusion_constant=DIFFUSION_CONST[compound],
                        ksi=ksi[compound].data
                    )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def dissolution_body(super_droplet_ids, mole_amounts, env_mixing_ratio, henrysConstant, env_p, env_T, env_rho_d, dt, dv,
                    droplet_volume, multiplicity, system_type, specific_gravity, alpha, diffusion_constant, ksi):
        mole_amount_taken = 0
        for i in super_droplet_ids:
            Mc = specific_gravity * Md
            Rc = R_str / Mc
            cinf = env_p / env_T / (Rd/env_mixing_ratio[0] + Rc) / Mc
            r_w = radius(volume=droplet_volume[i])
            v_avg = np.sqrt(8 * R_str * env_T / (np.pi * Mc))
            scale = (4 * r_w / (3 * v_avg * alpha) + r_w ** 2 / (3 * diffusion_constant))
            A_old = mole_amounts[i] / droplet_volume[i]
            A_new = (A_old + dt * cinf / scale) / (1 + dt / (scale * ksi[i] * henrysConstant * R_str * env_T))
            # TODO #442 !!!!!!!!! A_new = (A_old + dt * ksi[i] * cinf / scale) / (1 + dt / (scale * ksi[i] * henrysConstant * R_str * env_T))
            new_mole_amount_per_real_droplet = A_new * droplet_volume[i]
            assert new_mole_amount_per_real_droplet >= 0

            mole_amount_taken += multiplicity[i] * (new_mole_amount_per_real_droplet - mole_amounts[i])
            mole_amounts[i] = new_mole_amount_per_real_droplet
        delta_mr = mole_amount_taken * specific_gravity * Md / (dv * env_rho_d)
        assert delta_mr <= env_mixing_ratio
        if system_type == 'closed':
            env_mixing_ratio -= delta_mr

    @staticmethod
    def oxidation(n_sd, cell_ids, do_chemistry_flag,
                  k0, k1, k2, k3, K_SO2, K_HSO3, dt, droplet_volume, pH, O3, H2O2, S_IV, dissociation_factor_SO2,
                  # output
                  moles_O3, moles_H2O2, moles_S_IV, moles_S_VI):
        ChemistryMethods.oxidation_body(
            n_sd, cell_ids.data, do_chemistry_flag.data, k0.data, k1.data, k2.data, k3.data, K_SO2.data, K_HSO3.data,
            dt, droplet_volume.data, pH.data, O3.data, H2O2.data, S_IV.data, dissociation_factor_SO2.data,
            # output
            moles_O3.data, moles_H2O2.data, moles_S_IV.data, moles_S_VI.data
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def oxidation_body(n_sd, cell_ids, do_chemistry_flag,
                  k0, k1, k2, k3, K_SO2, K_HSO3, dt, droplet_volume, pH, O3, H2O2, S_IV, dissociation_factor_SO2,
                  # output
                  moles_O3, moles_H2O2, moles_S_IV, moles_S_VI):
        # NB: magic_const in the paper is k4.
        # The value is fixed at 13 M^-1 (from Ania's Thesis)
        magic_const = 13 / M

        for i in range(n_sd):
            if not do_chemistry_flag[i]:
                continue

            cid = cell_ids[i]
            H = pH2H(pH[i])
            SO2aq = S_IV[i] / dissociation_factor_SO2[i]

            # NB: This might not be entirely correct
            # https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JD092iD04p04171
            # https://www.atmos-chem-phys.net/16/1693/2016/acp-16-1693-2016.pdf

            # NB: There is also slight error due to "borrowing" compounds when
            # the concentration is close to 0. That way, if the rate is big enough,
            # it will consume more compound than there is.

            ozone = (k0[cid] + (k1[cid] * K_SO2[cid] / H) + (k2[cid] * K_SO2[cid] * K_HSO3[cid] / H**2)) * O3[i] * SO2aq
            peroxide = k3[cid] * K_SO2[cid] / (1 + magic_const * H) * H2O2[i] * SO2aq

            dconc_dt_O3 = -ozone
            dconc_dt_S_IV = -(ozone + peroxide)
            dconc_dt_H2O2 = -peroxide
            dconc_dt_S_VI = ozone + peroxide

            a = dt * droplet_volume[i]
            if (
                moles_O3[i] + dconc_dt_O3 * a < 0 or
                moles_S_IV[i] + dconc_dt_S_IV * a < 0 or
                moles_S_VI[i] + dconc_dt_S_VI * a < 0 or
                moles_H2O2[i] + dconc_dt_H2O2 * a < 0
            ):
                continue

            moles_O3[i] += dconc_dt_O3 * a
            moles_S_IV[i] += dconc_dt_S_IV * a
            moles_S_VI[i] += dconc_dt_S_VI * a
            moles_H2O2[i] += dconc_dt_H2O2 * a


    @staticmethod
    # @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})  # TODO #440
    def chem_recalculate_drop_data(dissociation_factors, equilibrium_consts, cell_id, pH):
        for i in range(len(pH)):
            H = pH2H(pH.data[i])
            for key in DIFFUSION_CONST.keys():
                dissociation_factors[key].data[i] = DISSOCIATION_FACTORS[key](H, equilibrium_consts, cell_id.data[i])

    @staticmethod
    # @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})  # TODO #440
    def chem_recalculate_cell_data(equilibrium_consts, kinetic_consts, T):
        for i in range(len(T)):
            for key in equilibrium_consts.keys():
                equilibrium_consts[key].data[i] = EQUILIBRIUM_CONST[key].at(T.data[i])
            for key in kinetic_consts.keys():
                kinetic_consts[key].data[i] = KINETIC_CONST[key].at(T.data[i])

    @staticmethod
    def equilibrate_H(equilibrium_consts, cell_id, N_mIII, N_V, C_IV, S_IV, S_VI, do_chemistry_flag, pH,
                      H_min, H_max, ionic_strength_threshold, rtol):
        ChemistryMethods.equilibrate_H_body(cell_id.data,
                                            N_mIII.data, N_V.data, C_IV.data, S_IV.data, S_VI.data,
                                            K_NH3=equilibrium_consts["K_NH3"].data,
                                            K_SO2=equilibrium_consts["K_SO2"].data,
                                            K_HSO3=equilibrium_consts["K_HSO3"].data,
                                            K_HSO4=equilibrium_consts["K_HSO4"].data,
                                            K_HCO3=equilibrium_consts["K_HCO3"].data,
                                            K_CO2=equilibrium_consts["K_CO2"].data,
                                            K_HNO3=equilibrium_consts["K_HNO3"].data,
                                            # output
                                            do_chemistry_flag=do_chemistry_flag.data,
                                            pH=pH.data,
                                            # params
                                            H_min=H_min,
                                            H_max=H_max,
                                            ionic_strength_threshold=ionic_strength_threshold,
                                            rtol=rtol
                                            )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})  # TODO #440
    def equilibrate_H_body(cell_id, N_mIII, N_V, C_IV, S_IV, S_VI,
                           K_NH3, K_SO2, K_HSO3, K_HSO4, K_HCO3, K_CO2, K_HNO3,
                           do_chemistry_flag, pH,
                           # params
                           H_min, H_max, ionic_strength_threshold, rtol
    ):
        # TODO #439 (iterate in logarithm?)
        for i in range(len(pH)):
            cid = cell_id[i]
            args = (
                N_mIII[i], N_V[i], C_IV[i], S_IV[i], S_VI[i],
                K_NH3[cid], K_SO2[cid], K_HSO3[cid], K_HSO4[cid], K_HCO3[cid], K_CO2[cid], K_HNO3[cid]
            )
            H = bisec(concentration, H_min, H_max - H_min, args, rtol=rtol)
            flag = calc_ionic_strength(H, *args) <= ionic_strength_threshold
            pH[i] = H2pH(H)
            do_chemistry_flag[i] = flag


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def calc_ionic_strength(H, N_mIII, N_V, C_IV, S_IV, S_VI, K_NH3, K_SO2, K_HSO3, K_HSO4, K_HCO3, K_CO2, K_HNO3):
    # Directly adapted
    # https://github.com/igfuw/libcloudphxx/blob/0b4e2455fba4f95c7387623fc21481a85e7b151f/src/impl/particles_impl_chem_strength.ipp#L50
    # https://en.wikipedia.org/wiki/Ionic_strength

    # H+ and OH-
    water = H + K_H2O / H

    # HSO4- and SO4 2-
    czS_VI = H * S_VI / (H + K_HSO4) + 4 * K_HSO4 * S_VI / (H + K_HSO4)

    # HCO3- and CO3 2-
    cz_CO2 = K_CO2 * H * C_IV / (H * H + K_CO2 * H + K_CO2 * K_HCO3) + \
        4 * K_CO2 * K_HCO3 * C_IV / (H * H + K_CO2 * H + K_CO2 * K_HCO3)

    # HSO3- and HSO4 2-
    cz_SO2 = K_SO2 * H * S_IV / (H * H + K_SO2 * H + K_SO2 * K_HSO3) + \
        4 * K_SO2 * K_HSO3 * S_IV / (H * H + K_SO2 * H + K_SO2 * K_HSO3)

    # NO3-
    cz_HNO3 = K_HNO3 * N_V / (H + K_HNO3)

    # NH4+
    cz_NH3 = K_NH3 * H * N_mIII / (K_H2O + K_NH3 * H)

    return 0.5 * (water + czS_VI + cz_CO2 + cz_SO2 + cz_HNO3 + cz_NH3)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def concentration(H, N_mIII, N_V, C_IV, S_IV, S_VI, K_NH3, K_SO2, K_HSO3, K_HSO4, K_HCO3, K_CO2, K_HNO3):
    ammonia = (N_mIII * H * K_NH3) / (K_H2O + K_NH3 * H)
    nitric = N_V * K_HNO3 / (H + K_HNO3)
    sulfous = S_IV * K_SO2 * (H + 2 * K_HSO3) / (H * H + H * K_SO2 + K_SO2 * K_HSO3)
    water = K_H2O / H
    sulfuric = S_VI * (H + 2 * K_HSO4) / (H + K_HSO4)
    carbonic = C_IV * K_CO2 * (H + 2 * K_HCO3) / (H * H + H * K_CO2 + K_CO2 * K_HCO3)
    zero = H + ammonia - (nitric + sulfous + water + sulfuric + carbonic)
    return zero


