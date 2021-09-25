import numba
import numpy as np

from PySDM.backends.numba import conf
from PySDM.backends.numba.toms748 import toms748_solve
from PySDM.physics.constants import Md, R_str, Rd, K_H2O
from PySDM.physics.aqueous_chemistry.support import HenryConsts, SPECIFIC_GRAVITY, \
    MASS_ACCOMMODATION_COEFFICIENTS, DIFFUSION_CONST, GASEOUS_COMPOUNDS, DISSOCIATION_FACTORS, \
    KineticConsts, EquilibriumConsts, k4

_max_iter_quite_close = 8
_max_iter_default = 32
_realy_close_threshold = 1e-6
_quite_close_threshold = 1
_quite_close_multiplier = 2


class ChemistryMethods:
    def __init__(self):
        self.HENRY_CONST = HenryConsts(self.formulae)
        self.KINETIC_CONST = KineticConsts(self.formulae)
        self.EQUILIBRIUM_CONST = EquilibriumConsts(self.formulae)

    def dissolution(self, *, n_cell, n_threads, cell_order, cell_start_arg, idx, do_chemistry_flag, mole_amounts,
                    env_mixing_ratio, env_T, env_p, env_rho_d, dissociation_factors, dt, dv, system_type, droplet_volume,
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
                        henrysConstant=self.HENRY_CONST.HENRY_CONST[compound].at(env_T[cell_id]),
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
                        dissociation_factor=dissociation_factors[compound].data,
                        radius=self.formulae.trivia.radius
                    )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def dissolution_body(super_droplet_ids, mole_amounts, env_mixing_ratio, henrysConstant, env_p, env_T, env_rho_d, dt, dv,
                    droplet_volume, multiplicity, system_type, specific_gravity, alpha, diffusion_constant, dissociation_factor, radius):
        mole_amount_taken = 0
        for i in super_droplet_ids:
            Mc = specific_gravity * Md
            Rc = R_str / Mc
            cinf = env_p / env_T / (Rd / env_mixing_ratio[0] + Rc) / Mc
            r_w = radius(volume=droplet_volume[i])
            v_avg = np.sqrt(8 * R_str * env_T / (np.pi * Mc))
            dt_over_scale = dt / (4 * r_w / (3 * v_avg * alpha) + r_w ** 2 / (3 * diffusion_constant))
            A_old = mole_amounts[i] / droplet_volume[i]
            H_eff = henrysConstant * dissociation_factor[i]
            A_new = (A_old + dt_over_scale * cinf) / (1 + dt_over_scale / H_eff / R_str / env_T)
            new_mole_amount_per_real_droplet = A_new * droplet_volume[i]
            assert new_mole_amount_per_real_droplet >= 0

            mole_amount_taken += multiplicity[i] * (new_mole_amount_per_real_droplet - mole_amounts[i])
            mole_amounts[i] = new_mole_amount_per_real_droplet
        delta_mr = mole_amount_taken * specific_gravity * Md / (dv * env_rho_d)
        assert delta_mr <= env_mixing_ratio
        if system_type == 'closed':
            env_mixing_ratio -= delta_mr

    def oxidation(self, n_sd, cell_ids, do_chemistry_flag,
                  k0, k1, k2, k3, K_SO2, K_HSO3, dt, droplet_volume, pH, dissociation_factor_SO2,
                  # output
                  moles_O3, moles_H2O2, moles_S_IV, moles_S_VI):
        ChemistryMethods.oxidation_body(
            n_sd, cell_ids.data, do_chemistry_flag.data, self.formulae.trivia.explicit_euler,
            self.formulae.trivia.pH2H,
            k0.data, k1.data, k2.data, k3.data, K_SO2.data, K_HSO3.data,
            dt, droplet_volume.data, pH.data, dissociation_factor_SO2.data,
            # output
            moles_O3.data, moles_H2O2.data, moles_S_IV.data, moles_S_VI.data
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def oxidation_body(n_sd, cell_ids, do_chemistry_flag, explicit_euler, pH2H,
                  k0, k1, k2, k3,
                  K_SO2, K_HSO3, dt, droplet_volume, pH, dissociation_factor_SO2,
                  # output
                  moles_O3, moles_H2O2, moles_S_IV, moles_S_VI):
        for i in numba.prange(n_sd):
            if not do_chemistry_flag[i]:
                continue

            cid = cell_ids[i]
            H = pH2H(pH[i])
            SO2aq = moles_S_IV[i] / droplet_volume[i] / dissociation_factor_SO2[i]

            # NB: This might not be entirely correct
            # https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JD092iD04p04171
            # https://www.atmos-chem-phys.net/16/1693/2016/acp-16-1693-2016.pdf

            ozone = (k0[cid] + (k1[cid] * K_SO2[cid] / H) + (k2[cid] * K_SO2[cid] * K_HSO3[cid] / H**2)) * (moles_O3[i] / droplet_volume[i]) * SO2aq
            peroxide = k3[cid] * K_SO2[cid] / (1 + k4 * H) * (moles_H2O2[i] / droplet_volume[i]) * SO2aq
            dt_times_volume = dt * droplet_volume[i]

            dconc_dt_O3 = -ozone
            dconc_dt_S_IV = -(ozone + peroxide)
            dconc_dt_H2O2 = -peroxide
            dconc_dt_S_VI = ozone + peroxide

            if (
                moles_O3[i] + dconc_dt_O3 * dt_times_volume < 0 or
                moles_S_IV[i] + dconc_dt_S_IV * dt_times_volume < 0 or
                moles_S_VI[i] + dconc_dt_S_VI * dt_times_volume < 0 or
                moles_H2O2[i] + dconc_dt_H2O2 * dt_times_volume < 0
            ):
                continue

            moles_O3[i] = explicit_euler(moles_O3[i], dt_times_volume, dconc_dt_O3)
            moles_S_IV[i] = explicit_euler(moles_S_IV[i], dt_times_volume, dconc_dt_S_IV)
            moles_S_VI[i] = explicit_euler(moles_S_VI[i], dt_times_volume, dconc_dt_S_VI)
            moles_H2O2[i] = explicit_euler(moles_H2O2[i], dt_times_volume, dconc_dt_H2O2)

    def chem_recalculate_drop_data(self, dissociation_factors, equilibrium_consts, cell_id, pH):
        for i in range(len(pH)):
            H = self.formulae.trivia.pH2H(pH.data[i])
            for key in DIFFUSION_CONST:
                dissociation_factors[key].data[i] = DISSOCIATION_FACTORS[key](H, equilibrium_consts, cell_id.data[i])

    def chem_recalculate_cell_data(self, equilibrium_consts, kinetic_consts, T):
        for i in range(len(T)):
            for key in equilibrium_consts:
                equilibrium_consts[key].data[i] = self.EQUILIBRIUM_CONST.EQUILIBRIUM_CONST[key].at(T.data[i])
            for key in kinetic_consts:
                kinetic_consts[key].data[i] = self.KINETIC_CONST.KINETIC_CONST[key].at(T.data[i])

    def equilibrate_H(self, equilibrium_consts, cell_id, N_mIII, N_V, C_IV, S_IV, S_VI, do_chemistry_flag, pH,
                      H_min, H_max, ionic_strength_threshold, rtol):
        ChemistryMethods.equilibrate_H_body(within_tolerance=self.formulae.trivia.within_tolerance,
                                            pH2H=self.formulae.trivia.pH2H,
                                            H2pH=self.formulae.trivia.H2pH,
                                            cell_id=cell_id.data,
                                            N_mIII=N_mIII.data,
                                            N_V=N_V.data,
                                            C_IV=C_IV.data,
                                            S_IV=S_IV.data,
                                            S_VI=S_VI.data,
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
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
    def equilibrate_H_body(within_tolerance,
                           pH2H, H2pH,
                           cell_id,
                           N_mIII, N_V, C_IV, S_IV, S_VI,
                           K_NH3, K_SO2, K_HSO3, K_HSO4, K_HCO3, K_CO2, K_HNO3,
                           do_chemistry_flag, pH,
                           # params
                           H_min, H_max, ionic_strength_threshold, rtol
    ):
        for i in range(len(pH)):
            cid = cell_id[i]
            args = (
                N_mIII[i], N_V[i], C_IV[i], S_IV[i], S_VI[i],
                K_NH3[cid], K_SO2[cid], K_HSO3[cid], K_HSO4[cid], K_HCO3[cid], K_CO2[cid], K_HNO3[cid]
            )
            a = pH2H(pH[i])
            fa = pH_minfun(a, *args)
            if abs(fa) < _realy_close_threshold:
                continue
            b = np.nan
            fb = np.nan
            use_default_range = False
            if abs(fa) < _quite_close_threshold:
                b = a * _quite_close_multiplier
                fb = pH_minfun(b, *args)
                if fa * fb > 0:
                    b = a
                    fb = fa
                    a = b / _quite_close_multiplier / _quite_close_multiplier
                    fa = pH_minfun(a, *args)
                    if fa * fb > 0:
                        use_default_range = True
            else:
                use_default_range = True
            if use_default_range:
                a = H_min
                b = H_max
                fa = pH_minfun(a, *args)
                fb = pH_minfun(b, *args)
                max_iter = _max_iter_default
            else:
                max_iter = _max_iter_quite_close
            H, _iters_taken = toms748_solve(pH_minfun, args, a, b, fa, fb, rtol=rtol, max_iter=max_iter,
                                            within_tolerance=within_tolerance)
            assert _iters_taken != max_iter
            pH[i] = H2pH(H)
            ionic_strength = calc_ionic_strength(H, *args)
            do_chemistry_flag[i] = (ionic_strength <= ionic_strength_threshold)


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
def pH_minfun(H, N_mIII, N_V, C_IV, S_IV, S_VI, K_NH3, K_SO2, K_HSO3, K_HSO4, K_HCO3, K_CO2, K_HNO3):
    ammonia = (N_mIII * H * K_NH3) / (K_H2O + K_NH3 * H)
    nitric = N_V * K_HNO3 / (H + K_HNO3)
    sulfous = S_IV * K_SO2 * (H + 2 * K_HSO3) / (H * H + H * K_SO2 + K_SO2 * K_HSO3)
    water = K_H2O / H
    sulfuric = S_VI * (H + 2 * K_HSO4) / (H + K_HSO4)
    carbonic = C_IV * K_CO2 * (H + 2 * K_HCO3) / (H * H + H * K_CO2 + K_CO2 * K_HCO3)
    zero = H + ammonia - (nitric + sulfous + water + sulfuric + carbonic)
    return zero
